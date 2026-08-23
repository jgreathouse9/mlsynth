#!/usr/bin/env Rscript
# Golden per-step values for the SBC Hong Kong unit tests, captured from the
# authors' own detrending functions (Shi-Xi-Xie SBC_HK.R: the lsq linear
# projection and the trend_predict forecast) run on the in-repo handover panel,
# basedata/hong_kong_handover.csv -- the same panel mlsynth fits and the same
# setup reference.R uses, so the two captures describe one run.
#
# The bundle's reference.R records what the authors' Synth::synth produces: the
# ATT, the cyclical SSE and the weights. It records nothing about the step
# before that, which left the case's claim that mlsynth's Hamilton filter
# matches the authors' lsq resting on a numpy rebuild of the same design instead
# of on R. This script closes that: it emits the detrending itself, so
# mlsynth/tests/test_sbc_hongkong_reference.py can pin each step against R the
# way test_sbc_reference.py does on the German panel.
#
# Base R only. The weight solve needs Synth and kernlab; the detrending does
# not, so this capture regenerates on a bare R.
#
# Every quantity is emitted twice: as %.8f, and under a "hi:" prefix at %.17g.
# The tests read the "hi:" line and compare at one unit in its last recorded
# place, so the capture -- not a hand-set constant -- decides how tight the
# comparison is. The %.8f form resolves 1e-8, three orders coarser than the
# 1e-11 difference between the two implementations, so on its own it cannot
# measure their agreement.
#
# Run from the repository root:
#   Rscript benchmarks/reference/sbc_hongkong/golden_steps.R \
#     > benchmarks/reference/sbc_hongkong/golden_steps.txt

b <- read.csv("basedata/hong_kong_handover.csv")
years <- sort(unique(b$year)); countries <- unique(b$country)
M <- matrix(NA_real_, length(years), length(countries), dimnames=list(years, countries))
M[cbind(match(b$year, years), match(b$country, countries))] <- b$gdp

treated_name <- "Hong Kong"
donor_names <- c("Australia", "Austria", "Korea", "Canada", "Denmark", "France",
                 "Germany", "Italy", "Netherlands", "New Zealand", "US")
HK <- data.frame(Date=years, HK=M[, treated_name], M[, donor_names], check.names=FALSE)

h <- 4; p <- 2; Fh <- h
T0 <- sum(HK$Date < 1997)          # 1997 is the first post-treatment period
predata <- HK[1:T0, ]

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
fit_HK <- lsq(predata[, 2])
detrended_HK_pre <- fit_HK$residuals
bhat_HK <- fit_HK$coefficients
trend_HK_post <- trend_predict(predata[, 2], bhat_HK)

arr <- function(v) paste(sprintf("%.8f", v), collapse=",")
hi  <- function(v) paste(sprintf("%.17g", v), collapse=",")
emit <- function(key, v) {
  cat(key, "\t", arr(v), "\n", sep=""); cat("hi:", key, "\t", hi(v), "\n", sep="")
}

cat("== REFERENCE VALUES ==\n")
emit("treated_trend_coef", bhat_HK)
emit("treated_cycle_pre", detrended_HK_pre)
emit("trend_forecast", trend_HK_post)
for (nm in c("Italy", "Germany", "US", "Korea"))
  emit(paste0("donor_cycle_full:", nm), detrended_donor[, nm])
cat("== END ==\n")
