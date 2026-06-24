# Outcome-only reference for the original CRAN ``Synth`` solver. Runs
# Synth::synth with the pre-period outcomes serving as both the predictors (X)
# and the dependent fit (Z) -- the canonical outcome-only specialisation -- and
# writes the donor weights in control-row order. This is the same nested
# predictor/donor optimization MSCMT (Becker & Klossner 2018) and Malo et al.
# (2024) show can stop short of the global optimum; we run it directly so the
# Python side can cross-check VanillaSC against the genuine reference.
#
# The Python side dumps:
#   <dir>/Y.csv    units x periods outcome matrix, treated row LAST (no header)
#   <dir>/meta.csv columns T0, T, trt_row (1-indexed)
# Usage:  Rscript benchmarks/R/synth_outcome_ref.R <dir>
suppressMessages(library(Synth))
args <- commandArgs(trailingOnly = TRUE)
dir  <- if (length(args) >= 1) args[1] else "benchmarks/_scratch"
Y    <- as.matrix(read.csv(file.path(dir, "Y.csv"), header = FALSE))
meta <- read.csv(file.path(dir, "meta.csv"))
T0 <- meta$T0[1]; b <- meta$trt_row[1]
ctrl <- setdiff(seq_len(nrow(Y)), b)
sc <- Synth::synth(
  X1 = t(Y[b, 1:T0, drop = FALSE]),   X0 = t(Y[ctrl, 1:T0, drop = FALSE]),
  Z1 = t(Y[b, 1:T0, drop = FALSE]),   Z0 = t(Y[ctrl, 1:T0, drop = FALSE]),
  verbose = FALSE)
write.csv(data.frame(w = as.numeric(sc$solution.w)),
          file.path(dir, "w_ref.csv"), row.names = FALSE)
cat("wrote", file.path(dir, "w_ref.csv"), "\n")
