# Cross-validation reference: run R's Synth on a panel dumped by Python and
# write the per-period treated-vs-synthetic gap to CSV for cell-by-cell
# comparison. The Python side dumps:
#   <dir>/Y.csv       (units x periods outcome matrix; treated row last)
#   <dir>/meta.csv    (columns: T0, T, trt_row  -- 1-indexed)
# Usage:  Rscript benchmarks/R/synth_crosscheck.R <dir>
suppressMessages(library(Synth))
args <- commandArgs(trailingOnly = TRUE)
dir  <- if (length(args) >= 1) args[1] else "benchmarks/_scratch"
Y    <- as.matrix(read.csv(file.path(dir, "Y.csv"), header = FALSE))
meta <- read.csv(file.path(dir, "meta.csv"))
T0 <- meta$T0[1]; T <- meta$T[1]; b <- meta$trt_row[1]
ctrl <- setdiff(seq_len(nrow(Y)), b)
sc <- Synth::synth(
  X1 = t(Y[b, 1:T0, drop = FALSE]),   X0 = t(Y[ctrl, 1:T0, drop = FALSE]),
  Z1 = t(Y[b, 1:T0, drop = FALSE]),   Z0 = t(Y[ctrl, 1:T0, drop = FALSE]),
  verbose = FALSE)
w   <- sc$solution.w
gap <- as.numeric(Y[b, ] - t(w) %*% Y[ctrl, ])
write.csv(data.frame(period = seq_len(T), gap = gap),
          file.path(dir, "ref_gap.csv"), row.names = FALSE)
cat("wrote", file.path(dir, "ref_gap.csv"), "\n")
