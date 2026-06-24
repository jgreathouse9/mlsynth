# Reference run for the `cwz_ttest` benchmark case.
#
# Runs the authors' scinference R package (Chernozhukov, Wuthrich & Zhu;
# kwuthrich/scinference) -- the debiased synthetic-control t-test (t-DISCo, K=3,
# 90% level) -- on three outcome-only SC panels and records the ATT (and, for
# the headline carbon-tax study, the 90% CI). These are the genuine package
# outputs the Python case pins against, not transcribed Table-5 constants.
#
# scinference is installed from github.com/kwuthrich/scinference (CVXR dropped
# from Imports -- it is only used by the unused constrained-lasso path; the SC
# t-test needs only limSolve). See benchmarks/R/install_scinference.sh.
#
# Run from the repository root:  Rscript benchmarks/reference/cwz_ttest/reference.R
suppressMessages({library(foreign); library(scinference)})

ttest <- function(p) {
  scinference(Y1 = p$Y1, Y0 = p$Y0, T1 = p$T1, T0 = p$T0,
              inference_method = "ttest", estimation_method = "sc",
              K = 3, alpha = 0.1, lsei_type = 1)
}

pivot <- function(d, unit, time, val, treated_unit, t_int) {
  d[[unit]] <- as.character(d[[unit]])
  years <- sort(unique(d[[time]]))
  u <- unique(d[[unit]])
  W <- sapply(u, function(x) { s <- d[d[[unit]] == x, ]; s[[val]][match(years, s[[time]])] })
  list(Y1 = W[, treated_unit], Y0 = W[, u != treated_unit, drop = FALSE],
       T0 = sum(years < t_int), T1 = sum(years >= t_int))
}

ct <- read.dta("basedata/carbontax_data.dta")
rc <- ttest(pivot(ct, "country", "year", "CO2_transport_capita", "Sweden", 1990))

b <- read.csv("basedata/basque_jasa.csv")
b <- b[b$regionname != "Spain (Espana)", ]
rb <- ttest(pivot(b, "regionname", "year", "gdpcap", "Basque Country (Pais Vasco)", 1975))

p99 <- read.csv("basedata/augmented_cali_long.csv")
rp <- ttest(pivot(p99, "state", "year", "cigsale", "California", 1989))

cat("== REFERENCE VALUES ==\n")
cat(sprintf("carbontax_att\t%.6f\n", rc$att))
cat(sprintf("carbontax_ci_lower\t%.6f\n", rc$lb))
cat(sprintf("carbontax_ci_upper\t%.6f\n", rc$ub))
cat(sprintf("basque_att\t%.6f\n", rb$att))
cat(sprintf("prop99_att\t%.6f\n", rp$att))
cat("== SESSION INFO ==\n")
print(sessionInfo())
