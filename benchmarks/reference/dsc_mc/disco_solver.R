# Reproduce DiSCo's weight solve verbatim on a supplied matrix.
#
# DiSCo_weights_reg() draws its own runif ranks and evaluates quantiles itself,
# so it cannot accept a pre-built design matrix. Everything after that sampling
# step is the block below, copied from R/DiSCo_weights_reg.R at ed2b3d94:
# the all-ones equality row, the spectral-norm rescale, and the pracma call
# with lb = 0, ub = 1.
suppressMessages(library(pracma))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
cells <- commandArgs(trailingOnly = TRUE)
for (cell in cells) {
  A <- as.matrix(read.csv(file.path(here, paste0("cell_", cell, "_A.csv")), header = FALSE))
  b <- as.vector(as.matrix(read.csv(file.path(here, paste0("cell_", cell, "_b.csv")), header = FALSE)))
  Aequ <- matrix(rep(1, ncol(A)), nrow = 1, ncol = ncol(A))
  sc <- base::norm(A, type = "2")
  w <- pracma::lsqlincon(A / sc, b / sc, A = NULL, b = NULL,
                         Aeq = Aequ, beq = 1, lb = 0, ub = 1)
  write.table(w, file.path(here, paste0("cell_", cell, "_w_disco.csv")),
              row.names = FALSE, col.names = FALSE)
  cat(sprintf("%s: sum(w)=%.10f  min(w)=%.3e  ||Aw-b||^2=%.10f\n",
              cell, sum(w), min(w), sum((A %*% w - b)^2)))
}
