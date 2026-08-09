#!/usr/bin/env Rscript
# Capture the DiSCos reference for benchmarks/cases/dsc_disco_xval.py.
#
# The reference is stochastic. DiSCo_weights_reg draws its quadrature points
# with runif, so a single run is not a target: at M = 10,000 the fitted weights
# still move by up to 0.016 (simplex) and 0.030 (sum-to-one) from one seed to
# the next. What is stable is their mean, so this captures the mean over
# N_SEEDS runs along with the across-seed standard deviation, and the case
# scores mlsynth against the mean while reading the spread as the yardstick.
#
# Both feasible sets are captured. simplex = TRUE is the set mlsynth defaults
# to; simplex = FALSE is the package's own default, sum-to-one with an upper
# bound of one and negatives allowed, which mlsynth reaches through
# weight_constraint = "sum_to_one".
#
# Donor identity travels with the weights. DiSCo returns control_ids alongside
# the weight vector and the case joins on it, so neither side depends on the
# other's ordering.
#
#   python benchmarks/reference/dsc_disco_xval/export_panel.py
#   Rscript benchmarks/reference/dsc_disco_xval/reference.R
suppressMessages({library(DiSCos); library(data.table)})

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
M <- 10000
G <- 1000
N_SEEDS <- 40
T0 <- 2003
TARGET <- 2

panel <- fread(file.path(here, "dube_panel.csv"))
setnames(panel, c("time_col", "id_col", "y_col"))

out <- NULL
for (spx in c(TRUE, FALSE)) {
  W <- NULL
  ids <- NULL
  for (s in seq_len(N_SEEDS)) {
    fit <- DiSCo(copy(panel), id_col.target = TARGET, t0 = T0, G = G, M = M,
                 num.cores = 1, seed = s, simplex = spx)
    W <- rbind(W, fit$weights)
    # control_ids is a plain numeric vector of the 33 donor fips codes, not a
    # per-period list; [[1]] would silently take its first element and recycle.
    this_ids <- fit$control_ids
    stopifnot(length(this_ids) == length(fit$weights))
    if (is.null(ids)) ids <- this_ids
    stopifnot(identical(ids, this_ids))     # donor order must not move by seed
  }
  out <- rbind(out, data.frame(
    mode = ifelse(spx, "simplex", "sum_to_one"),
    donor_id = ids,
    mean = colMeans(W),
    sd = apply(W, 2, sd),
    n_seeds = N_SEEDS
  ))
  cat(sprintf("simplex=%-5s  max across-seed sd = %.5f\n", spx, max(apply(W, 2, sd))))
}
write.csv(out, file.path(here, "gold_weights.csv"), row.names = FALSE)
cat(sprintf("wrote gold_weights.csv (%d rows, M = %d, %d seeds)\n",
            nrow(out), M, N_SEEDS))
