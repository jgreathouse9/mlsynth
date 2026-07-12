# Reference run for the `sbc_hongkong` benchmark case.
#
# Independent R implementation of Shi, Xi & Xie's Synthetic Business Cycle (SBC)
# estimator, following their replication script SBC_HK/SBC_HK.R
# (github.com/jinxi-atlas/Synthetic-business-cycle-code, "A Synthetic Business
# Cycle Approach...", arXiv:2505.22388): linear-projection detrending (Hamilton
# horizon h = 4, p = 2 lags), then the Abadie synthetic-control optimizer
# (Synth::synth, ipop) applied to the cyclical component for SBC and to the
# levels for classical SC. The SBC weights and the post-treatment forecast gap
# are what the Python case pins against.
#
# This reads basedata/hong_kong_handover.csv -- the SAME Hong Kong handover panel
# mlsynth fits (FRED/World Bank PPP GDP per capita, 1961-2010, HK treated at the
# 1997 handover, the authors' 11-donor pool) -- and uses the established Synth
# solver, so it is an independent cross-check of mlsynth's SBC, not a re-run of
# the same code. (The upstream repo carries no licence, so its method is
# reproduced here on the in-repo public FRED data rather than vendored.)
#
# Run from the repository root:  Rscript benchmarks/reference/sbc_hongkong/reference.R
suppressMessages({library(Synth); library(kernlab)})

b <- read.csv("basedata/hong_kong_handover.csv")
years <- sort(unique(b$year))
countries <- unique(b$country)
M <- matrix(NA_real_, length(years), length(countries), dimnames = list(years, countries))
M[cbind(match(b$year, years), match(b$country, countries))] <- b$gdp

treated_name <- "Hong Kong"
# The authors' SBC_HK.R donor order (New_Zealand labelled "New Zealand").
donor_names <- c("Australia", "Austria", "Korea", "Canada", "Denmark", "France",
                 "Germany", "Italy", "Netherlands", "New Zealand", "US")
HK <- data.frame(Date = years, HK = M[, treated_name],
                 M[, donor_names], check.names = FALSE)

h <- 4; p <- 2; Fh <- h
# Treatment convention matches the in-repo panel (Handover = 1 from 1997), i.e.
# 1997 is the FIRST post-treatment period, so the pre-period is 1961-1996 and the
# forecast window is 1997-2000. (The authors' SBC_HK.R instead counts 1997 as the
# last pre period; aligning T0 to the data's encoding makes mlsynth and this
# reference solve the identical estimand -- the point of a cross-validation.)
T0 <- sum(HK$Date < 1997)
predata <- HK[1:T0, ]

# --- linear-projection detrending (SBC_HK.R) ---
lsq <- function(z) {
  tt <- length(z); y <- z[(h + p):tt]; X <- embed(z[1:(tt - h)], p)
  y <- ifelse(is.na(y), 0, y); X <- ifelse(is.na(X), 0, X)
  lm(y ~ ., data = data.frame(cbind(y, X)))
}
extract_residuals <- function(lm_list) do.call(cbind, lapply(lm_list, residuals))
trend_predict <- function(z, bhat) {
  tt <- length(z); pr <- numeric(h)
  for (i in 1:h) pr[i] <- sum(c(1, z[(tt - h + i):(tt - h - p + i + 1)]) * bhat)
  pr
}

list_detrended_Y0 <- apply(HK[, 3:ncol(HK)], 2, function(col) lsq(col))
detrended_donor <- extract_residuals(list_detrended_Y0)[1:(T0 + Fh - h - p + 1), ]
detrended_HK_pre <- lsq(predata[, 2])$residuals
bhat_HK <- lsq(predata[, 2])$coefficients
trend_HK_pre <- tail(predata[, 2], length(detrended_HK_pre)) - detrended_HK_pre
trend_HK_post <- trend_predict(predata[, 2], bhat_HK)

# --- classical SC on levels ---
X1 <- as.matrix(predata$HK)
X0 <- as.matrix(subset(predata[, 2:ncol(predata)], select = -HK))
synth.sc <- synth(X1 = X1, X0 = X0, Z0 = X0, Z1 = X1, custom.v = rep(1, T0),
                  optimxmethod = c("Nelder-Mead", "BFGS"), genoud = FALSE,
                  quadopt = "ipop", Margin.ipop = 5e-4, Sigf.ipop = 7,
                  Bound.ipop = 6, verbose = FALSE)
weights.sc <- synth.sc$solution.w
donor_predict_df <- HK[(h + p):(T0 + Fh), 3:ncol(HK)]
Y1hat.SC <- tail(as.matrix(donor_predict_df), (T0 + Fh - h - p + 1)) %*% weights.sc

# --- SBC on the cyclical component ---
X1 <- as.matrix(detrended_HK_pre)
X0 <- as.matrix(detrended_donor[1:length(detrended_HK_pre), ])
synth.sbc <- synth(X1 = X1, X0 = X0, Z0 = X0, Z1 = X1,
                   custom.v = rep(1, length(detrended_HK_pre)),
                   optimxmethod = c("Nelder-Mead", "BFGS"), genoud = FALSE,
                   quadopt = "ipop", Margin.ipop = 5e-4, Sigf.ipop = 7,
                   Bound.ipop = 6, verbose = FALSE)
weights.sbc <- synth.sbc$solution.w
cychat.sbc <- as.matrix(detrended_donor) %*% weights.sbc
Y1hat.sbc <- c(trend_HK_pre, trend_HK_post) + cychat.sbc

HK_compare <- HK$HK[(h + p):(T0 + Fh)]
# Treatment effect = actual - synthetic counterfactual, averaged over the Fh
# post-treatment forecast periods (1998-2001).
att_sbc <- sum(tail(HK_compare - Y1hat.sbc, Fh)) / Fh
att_sc  <- sum(tail(HK_compare - Y1hat.SC, Fh)) / Fh
# Cyclical pre-period SSE the authors' ipop attains (so the case can show mlsynth
# reaches a weakly-lower objective on the same strictly-convex program).
sbc_cyc_sse <- sum((detrended_HK_pre - as.matrix(detrended_donor[1:length(detrended_HK_pre), ]) %*% weights.sbc)^2)

wn <- rownames(weights.sbc)
gw <- function(name) if (name %in% wn) weights.sbc[name, 1] else 0.0

cat("== REFERENCE VALUES ==\n")
cat(sprintf("att\t%.6f\n", att_sbc))
cat(sprintf("att_sc\t%.6f\n", att_sc))
cat(sprintf("sbc_cyc_sse\t%.6f\n", sbc_cyc_sse))
for (i in seq_along(wn)) {
  if (weights.sbc[i, 1] > 1e-4) cat(sprintf("weight\t%s\t%.6f\n", wn[i], weights.sbc[i, 1]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
