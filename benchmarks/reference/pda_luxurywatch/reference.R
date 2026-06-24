# Reference run for the `pda_luxurywatch` benchmark case.
#
# Runs Shi & Huang's own luxury-watch application script (fsPDA_app.R, vendored
# verbatim from the app1_luxury_watch/ folder of github.com/zhentaoshi/fsPDA,
# MIT) on the China luxury-watch import panel. This is the script the paper's
# Section 5 / Example 2 result is produced by: forward selection by BIC, then a
# post-selection t-test using the PREWHITENED Newey-West long-run variance
# (sandwich::lrvar(..., prewhite = TRUE)) -- which is exactly what mlsynth's PDA
# "fs" method with fs_intercept ports, and what makes the effect significant
# (the packaged est.fsPDA, which does not prewhiten, instead reports t ~ -1.15).
#
# The panel is read from basedata/china_watches_long.csv -- the SAME file mlsynth
# fits -- and pivoted to the treated vector + control matrix the script expects,
# so the two sides solve an identical problem. These are the genuine fsPDA app
# outputs the Python case pins against, not numbers transcribed from the paper.
#
# Run from the repository root:  Rscript benchmarks/reference/pda_luxurywatch/reference.R
source("benchmarks/reference/pda_luxurywatch/fsPDA_app.R")

d <- read.csv("basedata/china_watches_long.csv", check.names = FALSE)
d$yyyymm <- as.character(d$yyyymm)

times <- unique(d$yyyymm[order(d$time)])
units <- unique(d$unit)
M <- matrix(NA_real_, nrow = length(times), ncol = length(units),
            dimnames = list(times, units))
M[cbind(match(d$yyyymm, times), match(d$unit, units))] <- d$y

treated <- M[, "watches"]
control <- M[, setdiff(units, "watches"), drop = FALSE]

res <- fsPDA(treated, control, intervention_time = "201301")

sel <- res$control_group
t1  <- which(names(treated) == "201301") - 1
pre_treated <- treated[1:t1]
Xpre <- cbind(1, control[1:t1, sel, drop = FALSE])
b <- MASS::ginv(t(Xpre) %*% Xpre) %*% t(Xpre) %*% pre_treated
e <- pre_treated - as.vector(Xpre %*% b)
pre_r2 <- 1 - sum(e^2) / sum((pre_treated - mean(pre_treated))^2)

ate   <- res$t_test[["average treatment effect"]]
tstat <- res$t_test[["t-statistic"]]
pval  <- res$t_test[["p-value"]]

cat("== REFERENCE VALUES ==\n")
cat(sprintf("fs_ate_pct\t%.6f\n", ate * 100))
cat(sprintf("fs_pre_r2\t%.6f\n", pre_r2))
cat(sprintf("fs_n_controls\t%d\n", length(sel)))
cat(sprintf("fs_abs_tstat\t%.6f\n", abs(tstat)))
cat(sprintf("fs_p_value\t%.6f\n", pval))
# selected-control regression coefficients (intercept dropped), for the comparison
for (i in seq_along(sel)) {
  cat(sprintf("weight\t%s\t%.6f\n", sel[i], b[i + 1]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
