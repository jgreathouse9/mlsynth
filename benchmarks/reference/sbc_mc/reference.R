#!/usr/bin/env Rscript
# Panel capture for the SBC Monte Carlo (Shi, Xi & Xie 2025, Sec. 4, Table 1).
#
# Draws the simulated panels the Python case estimates on, from the authors' own
# data-generating processes: simulation/simulation_nonnegative.R in
# github.com/jinxi-atlas/Synthetic-business-cycle-code, the script that produced
# Table 1's non-negative-weights columns. Generation is theirs, estimation is
# mlsynth's, so a disagreement with Table 1 is the estimator's and cannot be a
# re-implemented DGP.
#
# The authors' estimation legs (Synth / kernlab ipop) are not run here, so this
# script needs no CRAN package and reproduces on a bare R.
#
# The draw order inside each replication is the authors' own, including the two
# Model 1 designs (drift 0 and drift 0.5) whose panels are discarded: they
# advance the RNG stream, so keeping them means these panels are the ones their
# loop produces at this seed. Model 3 reuses Model 2's `eps`, as their code does.
#
# Cell: T0 = 100, phi = 0.5, N + 1 = 12, h = p = Fh = 2 -- Table 1's middle
# pre-treatment length and middle persistence. Model 1 is their DGP 1.3, the
# mu ~ N(0, 1/4) column.
#
# Usage:  Rscript benchmarks/reference/sbc_mc/reference.R
# Writes: panels.tsv.gz, versions.txt

here <- file.path("benchmarks", "reference", "sbc_mc")

N   <- 12          # number of units, incl. the treated one (paper: N + 1 = 12)
T0  <- 100         # pre-treatment length
h   <- 2           # detrending horizon
Fh  <- h           # post-treatment periods
phi <- 0.5         # AR coefficient of the stationary factors
TT  <- T0 + Fh
REPS <- 60
SEED <- 20250522

cumulate <- function(eps) {                      # y_t = y_{t-1} + eps_t, y_1 = eps_1
  out <- matrix(0, nrow = nrow(eps), ncol = ncol(eps))
  for (t in seq_len(nrow(eps))) {
    out[t, ] <- if (t == 1) eps[t, ] else out[t - 1, ] + eps[t, ]
  }
  out
}

set.seed(SEED)
rows <- vector("list", 3 * REPS)
k <- 0
for (j in seq_len(REPS)) {
  # ---- Model 1: independent random walks with drift ----------------------
  invisible(replicate(N, cumsum(rnorm(TT) + 0)))      # DGP 1.1, discarded
  invisible(replicate(N, cumsum(rnorm(TT) + 0.5)))    # DGP 1.2, discarded
  y1 <- replicate(N, { mu <- rnorm(1, 0, sqrt(0.25)); cumsum(rnorm(TT) + mu) })

  # ---- Model 2: idiosyncratic unit roots, common stationary AR factors ----
  r <- 2
  Lambda <- matrix(rnorm(N * r), nrow = N, ncol = r)
  Fmat <- matrix(0, nrow = r, ncol = TT)
  Fmat[, 1] <- rnorm(r)
  for (t in 2:TT) Fmat[, t] <- phi * Fmat[, t - 1] + rnorm(r, sd = 1)
  U <- matrix(rnorm(TT * N, sd = 1), nrow = TT, ncol = N)
  eps <- t(Lambda %*% Fmat) + U
  y2 <- cumulate(eps)

  # ---- Model 3: partial cointegration (half the units share RW factors) --
  s <- 2
  Lambda_trend <- matrix(rnorm((.5 * N) * s, sd = T0^{-1/3}), nrow = .5 * N, ncol = s)
  eta_trend <- matrix(rnorm(TT * s, sd = 1), nrow = TT, ncol = s)
  f_trend <- apply(eta_trend, 2, cumsum)
  trend_mat <- t(Lambda_trend %*% t(f_trend))
  y3 <- cbind(trend_mat + eps[, 1:(.5 * N)],
              cumulate(eps[, (.5 * N + 1):N, drop = FALSE]))

  for (m in 1:3) {
    Y <- list(y1, y2, y3)[[m]]
    k <- k + 1
    rows[[k]] <- data.frame(
      model = m, rep = j, t = seq_len(TT),
      matrix(Y, nrow = TT, dimnames = list(NULL, sprintf("y%02d", 1:N))),
      check.names = FALSE)
  }
}
panels <- do.call(rbind, rows)

num <- vapply(panels[, sprintf("y%02d", 1:N)], function(col) sprintf("%.17g", col),
              character(nrow(panels)))
out <- cbind(panels[, c("model", "rep", "t")], num)
gz <- gzfile(file.path(here, "panels.tsv.gz"), "w")
write.table(out, gz, sep = "\t", quote = FALSE, row.names = FALSE)
close(gz)

writeLines(c(
  paste("R:", R.version.string),
  paste("source: jinxi-atlas/Synthetic-business-cycle-code,",
        "simulation/simulation_nonnegative.R (arXiv:2505.22388)"),
  sprintf("cell: T0=%d phi=%.1f N=%d h=%d Fh=%d reps=%d seed=%d",
          T0, phi, N, h, Fh, REPS, SEED),
  sprintf("panels: %d rows x %d units", nrow(out), N)
), file.path(here, "versions.txt"))

cat(sprintf("wrote %d panel rows to %s\n", nrow(out), file.path(here, "panels.tsv.gz")))
