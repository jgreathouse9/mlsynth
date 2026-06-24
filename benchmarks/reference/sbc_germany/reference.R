# Reference run for the `sbc_germany` benchmark case.
#
# Independent R implementation of Shi, Xi & Xie's Synthetic Business Cycle (SBC)
# estimator, following their replication script SBC_Germany/Germany.R
# (github.com/jinxi-atlas/Synthetic-business-cycle-code, "A Synthetic Business
# Cycle Approach...", arXiv:2505.22388): linear-projection detrending (Hamilton
# horizon h = 4, p = 2 lags), then the Abadie synthetic-control optimizer
# (Synth::synth, ipop) applied to the cyclical component for SBC and to the
# levels for classical SC. The SBC weights and the post-treatment forecast gap
# are what the Python case pins against.
#
# This reads basedata/german_reunification.csv -- the SAME canonical Abadie
# German-reunification panel mlsynth fits -- and uses the established Synth
# solver, so it is an independent cross-check of mlsynth's SBC, not a re-run of
# the same code. (The upstream repo carries no licence and its shipped wide CSV
# has mismatched column labels, so its method is reproduced here on the canonical
# long-format data rather than vendored.)
#
# Run from the repository root:  Rscript benchmarks/reference/sbc_germany/reference.R
suppressMessages({library(Synth); library(kernlab)})

b <- read.csv("basedata/german_reunification.csv")
years <- sort(unique(b$year))
countries <- unique(b$country)
M <- matrix(NA_real_, length(years), length(countries), dimnames = list(years, countries))
M[cbind(match(b$year, years), match(b$country, countries))] <- b$gdp

treated_name <- "West Germany"
donor_names <- setdiff(countries, treated_name)
Germany <- data.frame(Date = years, Germany = M[, treated_name],
                      M[, donor_names], check.names = FALSE)

h <- 4; p <- 2; Fh <- h
T0 <- which(Germany$Date == 1990)
predata <- Germany[1:T0, ]

# --- linear-projection detrending (Germany.R) ---
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

list_detrended_Y0 <- apply(Germany[, 3:ncol(Germany)], 2, function(col) lsq(col))
detrended_donor <- extract_residuals(list_detrended_Y0)[1:(T0 + Fh - h - p + 1), ]
detrended_Germany_pre <- lsq(predata[, 2])$residuals
bhat_Germany <- lsq(predata[, 2])$coefficients
trend_Germany_pre <- tail(predata[, 2], length(detrended_Germany_pre)) - detrended_Germany_pre
trend_Germany_post <- trend_predict(predata[, 2], bhat_Germany)

# --- classical SC on levels ---
X1 <- as.matrix(predata$Germany)
X0 <- as.matrix(subset(predata[, 2:ncol(predata)], select = -Germany))
synth.sc <- synth(X1 = X1, X0 = X0, Z0 = X0, Z1 = X1, custom.v = rep(1, T0),
                  optimxmethod = c("Nelder-Mead", "BFGS"), genoud = FALSE,
                  quadopt = "ipop", Margin.ipop = 5e-4, Sigf.ipop = 5,
                  Bound.ipop = 10, verbose = FALSE)
weights.sc <- synth.sc$solution.w
donor_predict_df <- Germany[(h + p):(T0 + Fh), 3:ncol(Germany)]
Y1hat.SC <- tail(as.matrix(donor_predict_df), (T0 + Fh - h - p + 1)) %*% weights.sc

# --- SBC on the cyclical component ---
X1 <- as.matrix(detrended_Germany_pre)
X0 <- as.matrix(detrended_donor[1:length(detrended_Germany_pre), ])
synth.sbc <- synth(X1 = X1, X0 = X0, Z0 = X0, Z1 = X1,
                   custom.v = rep(1, (T0 - h - p + 1)),
                   optimxmethod = c("Nelder-Mead", "BFGS"), genoud = FALSE,
                   quadopt = "ipop", Margin.ipop = 5e-4, Sigf.ipop = 5,
                   Bound.ipop = 10, verbose = FALSE)
weights.sbc <- synth.sbc$solution.w
cychat.sbc <- as.matrix(detrended_donor) %*% weights.sbc
Y1hat.sbc <- c(trend_Germany_pre, trend_Germany_post) + cychat.sbc

Germany_compare <- Germany$Germany[(h + p):(T0 + Fh)]
# Treatment effect = actual - synthetic counterfactual, averaged over the Fh
# post-treatment forecast periods (1991-1994).
att_sbc <- sum(tail(Germany_compare - Y1hat.sbc, Fh)) / Fh
att_sc  <- sum(tail(Germany_compare - Y1hat.SC, Fh)) / Fh

wn <- rownames(weights.sbc)
gw <- function(name) if (name %in% wn) weights.sbc[name, 1] else 0.0

cat("== REFERENCE VALUES ==\n")
cat(sprintf("att\t%.6f\n", att_sbc))
cat(sprintf("att_sc\t%.6f\n", att_sc))
cat(sprintf("greece_weight\t%.6f\n", gw("Greece")))
cat(sprintf("netherlands_weight\t%.6f\n", gw("Netherlands")))
cat(sprintf("italy_weight\t%.6f\n", gw("Italy")))
for (i in seq_along(wn)) {
  if (weights.sbc[i, 1] > 1e-4) cat(sprintf("weight\t%s\t%.6f\n", wn[i], weights.sbc[i, 1]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
