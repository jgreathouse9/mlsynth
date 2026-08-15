# Reference run for the `lpca_mc` benchmark case.
#
# Runs Feng's own simulation code -- `knn.index` / `lpca` / `sel.r` / `compute`
# / `comstat` from Simulation/Feng-2024_LPCA_simul_suppfuns.R in
# yingjieum/Replication_NonlinearFactorModel_2023, pinned at ca34fba -- on the
# three Section 5 data-generating processes, and records the error statistics
# Table 1 reports.
#
# The panels are fixed, not drawn here. R and NumPy cannot be made to
# share a random stream, so drawing separately would leave a sampling channel
# between the two implementations and any comparison would be bounded by Monte
# Carlo error instead of by the estimators' actual agreement. The panels come
# from make_panels.py and both sides read the same bytes, so every number below
# is a cell-by-cell claim.
#
# The global-PCA baseline is included because Table 1's comparison is against
# it, and because it exercises HDRFA::PCA_FN, the eigenvalue-ratio factor count
# Feng's script uses.
#
# Install: benchmarks/R/install_feng_lpca.sh
# Run from the repository root:
#   Rscript benchmarks/reference/lpca_mc/reference.R
suppressMessages({library(Rfast); library(irlba); library(HDRFA)})

REPL <- Sys.getenv("LPCA_REFERENCE_DIR", "/tmp/feng_lpca")
source(file.path(REPL, "replication", "Simulation",
                 "Feng-2024_LPCA_simul_suppfuns.R"))

HERE <- "benchmarks/reference/lpca_mc"
NLAM <- 3
PROP <- 0.5

cat("== REFERENCE VALUES ==\n")
for (m in 1:3) {
  x <- as.matrix(read.csv(gzfile(file.path(HERE, sprintf("panel_model%d.csv.gz", m))),
                          header = FALSE))
  lat <- read.csv(file.path(HERE, sprintf("latent_model%d.csv", m)))
  ev  <- as.integer(readLines(file.path(HERE, sprintf("evaluated_model%d.csv", m))))
  n <- nrow(x); p <- ncol(x)

  # The noise-free surface, recomputed from the latent vectors (see make_panels.py).
  truth <- outer(lat$alpha, lat$eta, FUN = funlist[[m]])

  K <- n^(2 * 1 / (2 * 1 + 1))
  Kseq <- floor(K * c(0.5, 1, 1.5))
  p1 <- round(p * PROP); p2 <- p - p1
  mean.true <- truth[, (p1 + 1):p]

  # Feng's own driver over every unit and every K in the grid.
  fit <- compute(1:n, data = x, Kseq, NLAM, n, p, PROP)
  for (l in seq_along(Kseq)) {
    stat <- comstat(fit[, ((l - 1) * p2 + 1):(l * p2)], mean.true, ev)
    cat(sprintf("model%d_K%d_mae\t%.10f\n", m, Kseq[l], stat[1]))
    for (q in 1:3) {
      cat(sprintf("model%d_K%d_q%d\t%.10f\n", m, Kseq[l], q, stat[2 + q]))
    }
  }

  # Global PCA on doubly demeaned data, as the script's second baseline does.
  col.mean <- colmeans(x); g1 <- sweep(x, 2, col.mean)
  row.mean <- rowmeans(g1); g2 <- g1 - row.mean
  est.r <- PCA_FN(g2, 9)
  f <- irlba(g2, nu = est.r, nv = est.r)
  recon <- if (est.r == 1) tcrossprod(f$u, f$v * f$d) else
    tcrossprod(f$u, f$v %*% diag(f$d))
  gpca <- sweep(recon + row.mean, 2, col.mean, "+")
  stat <- comstat(gpca[, (p1 + 1):p], mean.true, ev)
  cat(sprintf("model%d_gpca_rank\t%d\n", m, est.r))
  cat(sprintf("model%d_gpca_mae\t%.10f\n", m, stat[1]))
  for (q in 1:3) cat(sprintf("model%d_gpca_q%d\t%.10f\n", m, q, stat[2 + q]))

  cat(sprintf("model%d_K_grid_1\t%d\n", m, Kseq[1]))
  cat(sprintf("model%d_K_grid_2\t%d\n", m, Kseq[2]))
  cat(sprintf("model%d_K_grid_3\t%d\n", m, Kseq[3]))
  # Realised neighbourhood for the first evaluated unit at the middle K: the
  # selection rule is a threshold, so ties widen it, and model 3 is binary.
  km <- knn.index(A = x[, 1:p1], Kseq = Kseq)
  cat(sprintf("model%d_realised_neighbourhood\t%d\n", m,
              sum(km[(n + 1):(2 * n), ev[1]])))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
