# Reference run for the `cwz_conformal` benchmark case.
#
# Runs the authors' scinference R package (Chernozhukov, Wuthrich & Zhu;
# kwuthrich/scinference) -- the conformal inference procedure of their 2021 JASA
# paper -- on three outcome-only SC panels, and records the p-value for the joint
# null of no post-treatment effect under BOTH permutation schemes: moving block
# ("mb", their default, for serially dependent errors) and iid ("iid", the exact
# one). These are the genuine package outputs the Python case pins against, not
# numbers transcribed from the paper.
#
# scinference is installed at a pinned SHA from github.com/kwuthrich/scinference;
# see benchmarks/R/install_scinference.sh. CVXR is dropped from its Imports because
# only the constrained-lasso estimator needs it, and this run uses estimation_method
# = "sc".
#
# Run from the repository root:  Rscript benchmarks/reference/cwz_conformal/reference.R
suppressMessages({library(foreign); library(scinference)})

set.seed(20260814)   # only the "iid" scheme is stochastic (it subsamples permutations)

N_PERM <- 5000       # scinference's own default for the iid scheme
ALPHA  <- 0.1

conf <- function(p, permutation_method) {
  scinference(Y1 = p$Y1, Y0 = p$Y0, T1 = p$T1, T0 = p$T0,
              inference_method = "conformal", estimation_method = "sc",
              permutation_method = permutation_method,
              n_perm = N_PERM, alpha = ALPHA, lsei_type = 1)
}

# Wide-pivot a long panel into the (Y1, Y0, T0, T1) shape scinference expects.
pivot <- function(d, unit, time, val, treated_unit, t_int) {
  d[[unit]] <- as.character(d[[unit]])
  years <- sort(unique(d[[time]]))
  u <- unique(d[[unit]])
  W <- sapply(u, function(x) { s <- d[d[[unit]] == x, ]; s[[val]][match(years, s[[time]])] })
  list(Y1 = W[, treated_unit], Y0 = W[, u != treated_unit, drop = FALSE],
       T0 = sum(years < t_int), T1 = sum(years >= t_int))
}

panels <- list()
ct <- read.dta("basedata/carbontax_data.dta")
panels$carbontax <- pivot(ct, "country", "year", "CO2_transport_capita", "Sweden", 1990)

b <- read.csv("basedata/basque_jasa.csv")
b <- b[b$regionname != "Spain (Espana)", ]
panels$basque <- pivot(b, "regionname", "year", "gdpcap",
                       "Basque Country (Pais Vasco)", 1975)

p99 <- read.csv("basedata/augmented_cali_long.csv")
panels$prop99 <- pivot(p99, "state", "year", "cigsale", "California", 1989)

cat("== REFERENCE VALUES ==\n")
for (nm in names(panels)) {
  p <- panels[[nm]]
  cat(sprintf("%s_T0\t%d\n", nm, p$T0))
  cat(sprintf("%s_T1\t%d\n", nm, p$T1))
  cat(sprintf("%s_J\t%d\n", nm, ncol(p$Y0)))
  for (pm in c("mb", "iid")) {
    r <- conf(p, pm)
    cat(sprintf("%s_%s_pvalue\t%.6f\n", nm, pm, r$p_val))
  }
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
