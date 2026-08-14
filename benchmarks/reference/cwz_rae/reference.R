# Reference run for the `cwz_rae` benchmark case.
#
# Table 1 of Chernozhukov, Wuthrich & Zhu's t-test paper: the relative
# asymptotic efficiency of the K-fold debiased confidence interval, as a
# function only of K, the pre-to-post ratio c0 = T0/T1, and the level. It is the
# formula behind mlsynth's ttest_K="auto" rule, which climbs K until the RAE
# meets a target, and until now that rule was exercised without ever being
# checked against the table it comes from.
#
# The two functions are verbatim from the authors' `code/auxiliary scripts/
# RAE.R` in the JPE replication package. No data and no packages: the table is
# a closed form, so this reference is the formula and not a fit.
#
# Run from the repository root:  Rscript benchmarks/reference/cwz_rae/reference.R

# Expected length of the K-fold interval
E_length <- function(K, c0, alpha) {
  g <- (c0 < 1) * K + (1 <= c0) * (c0 <= K) * (K / c0) + (c0 > K)
  2 * qt(1 - alpha / 2, df = K - 1) / (sqrt(K) * sqrt(K - 1)) *
    sqrt(1 + min(c0, K)) * sqrt(g) * sqrt(2) * (gamma(K / 2) / gamma((K - 1) / 2))
}

# Its limit as K grows
lim_E_length <- function(c0, alpha) {
  2 * qnorm(1 - alpha / 2) * sqrt(min((1 / c0), 1)) * sqrt(1 + c0)
}

alpha <- 0.1
K_vec <- seq(2, 10, 1)
c0    <- 30 / 16          # the carbon tax panel's T0 / T1

cat("== REFERENCE VALUES ==\n")
lim_leng <- lim_E_length(c0, alpha)
for (K in K_vec) {
  cat(sprintf("rae_K%d\t%.10f\n", K, lim_leng / E_length(K, c0, alpha)))
}
# The three differences the paper quotes in the text after Table 1, in
# percentage points against K = 3.
for (K in c(4, 5, 6)) {
  cat(sprintf("gain_K%d_over_K3\t%.10f\n", K,
              100 * (lim_leng / E_length(K, c0, alpha) -
                     lim_leng / E_length(3, c0, alpha))))
}
cat(sprintf("c0\t%.10f\n", c0))
cat("== SESSION INFO ==\n")
print(sessionInfo())
