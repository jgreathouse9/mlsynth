# Live pensynth reference: Abadie & L'Hour's penalized SC solver on Prop 99.
#
# Runs the authors' own ``wsoll1`` (penalized synthetic-control QP, solved with
# LowRankQP) over a lambda grid on a predictor matrix supplied by the Python
# benchmark, and writes the resulting donor-weight matrix back. The Python case
# (benchmarks/cases/pensynth_prop99.py) builds X0/X1 from mlsynth's vendored
# P99data.csv and cross-checks mlsynth's ``penalized_weights`` against this dump.
#
# The solver source (wsoll1.R, TZero.R) is sourced from a commit-pinned clone of
# jeremylhour/pensynth (see benchmarks/reference/clone_pensynth.py); LowRankQP is
# installed by benchmarks/R/install_pensynth.sh. Feeding both implementations the
# identical X0/X1 makes this a byte-level solver cross-check: for lambda > 0 the
# penalized QP is strictly convex (Theorem 1), so the optimum is unique.
#
# Usage: Rscript pensynth_prop99.R <funcdir> <X0.csv> <X1.csv> <grid.csv> <out.csv>
suppressMessages(library(LowRankQP))

args <- commandArgs(trailingOnly = TRUE)
funcdir <- args[1]; X0path <- args[2]; X1path <- args[3]
gridpath <- args[4]; outpath <- args[5]

source(file.path(funcdir, "wsoll1.R"))
source(file.path(funcdir, "TZero.R"))

X0 <- as.matrix(read.csv(X0path, header = FALSE))             # p x n0
X1 <- as.numeric(as.matrix(read.csv(X1path, header = FALSE))) # p
grid <- as.numeric(as.matrix(read.csv(gridpath, header = FALSE)))
V <- diag(nrow(X0))                                           # Gamma = I, as in EXA/EXB

Wref <- t(sapply(grid, function(l) {
  w <- wsoll1(X0, X1, V, l)
  as.numeric(TZero(w, 1e-6))
}))
write.table(Wref, outpath, sep = ",", row.names = FALSE, col.names = FALSE)
cat(sprintf("OK grid %d donors %d\n", length(grid), ncol(X0)))
