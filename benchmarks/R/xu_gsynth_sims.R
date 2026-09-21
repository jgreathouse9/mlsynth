#!/usr/bin/env Rscript
# Xu (2017) "Generalized Synthetic Control Method", Political Analysis 25(1) --
# the R-side reference for mlsynth's GSYNTH on the author's own simulations.
#
# The replication archive (Dataverse) ships gsynth 1.0 plus seven simulation
# scripts, each targeting a table in the Online Appendix:
#
#   sim_TN.R        Table A1   finite-sample bias
#   sim_DID.R       Table A2   against difference-in-differences
#   sim_inter.R     Table A3   against interactive fixed effects
#   sim_adh.R       Table A3   against ADH synthetic control
#   sim_factor.R    Table A5   does cross-validation find the factor count
#   sim_coverage.R  --         parametric bootstrap CI coverage
#   sim_sampling.R  --         the shared data-generating process
#
# This script covers the two that exercise machinery mlsynth has and nothing in
# the suite pins: the factor-count choice (Algorithm 1) and the parametric
# bootstrap (Algorithm 2). The Online Appendix is not in the archive, so these
# are cross-validated against gsynth itself -- the author's own implementation,
# run on the author's own data-generating process, on panels shared with the
# Python side so the comparison is paired, not distributional.
#
# The data-generating process is sim_sampling.R's `simulate`, reproduced here so
# the script stands alone: a Bai (2009) interactive fixed-effects panel with
# `r` factors, `p` covariates, unit and time effects, loadings whose treated /
# control overlap is set by `w`, and a treatment effect of 1..10 over the ten
# post periods.
#
# Writes, into --out:
#   <case>.csv       the panels, long, one file per case (all replications)
#   reference.csv    per replication: gsynth's chosen r, its ATT, and its MSPE path
#
# Usage:
#   Rscript benchmarks/R/xu_gsynth_sims.R --out DIR [--sims N] [--cases a,b]

suppressWarnings(suppressMessages({library(gsynth)}))

args <- commandArgs(trailingOnly = TRUE)
argval <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
outdir <- argval("--out", "xuref")
sims <- as.integer(argval("--sims", "50"))
want <- argval("--cases", "")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# sim_sampling.R's generator, restricted to the settings these two cases use
# (no fixed factors or loadings, no AR(1), no observed Z).
# ---------------------------------------------------------------------------
simulate <- function(Ntr, Nco, T0, p, r, w = 1, D.sd = 1, att = c(1:10),
                     beta = NULL, mu = 0, fsize = 1, FE = 0) {
  N <- Ntr + Nco
  T <- T0 + length(att)
  ss <- sqrt(3)                                   # so the loadings have variance 1
  lambda <- matrix(runif(N * r, min = -ss, max = ss), N, r)
  lambda[1:Ntr, ] <- lambda[1:Ntr, ] + (1 - w) * 2 * ss   # treated/control overlap
  factor <- matrix(rnorm(T * r), T, r)
  if (FE == 1) {
    alpha <- runif(N, min = -ss, max = ss)
    alpha[1:Ntr] <- alpha[1:Ntr] + (1 - w) * 2 * ss
    xi <- rnorm(T, 0, 1)
  }
  e <- matrix(rnorm(T * N), T, N)
  if (p != 0) {
    X <- array(0, dim = c(T, N, p))
    for (j in 1:p) {
      X[, , j] <- matrix(rnorm(T * N), T, N) +
        0.5 * factor[, 1:2] %*% t(lambda[, 1:2]) +
        0.25 * matrix(1, T, 2) %*% t(lambda[, 1:2]) +
        0.25 * factor[, 1:2] %*% matrix(1, 2, N) + 1
    }
  }
  D <- cbind(rbind(matrix(0, T0, Ntr), matrix(1, (T - T0), Ntr)), matrix(0, T, Nco))
  eff <- matrix(c(rep(0, T0), att), T, N) +
    rbind(matrix(0, T0, N), matrix(rnorm((T - T0) * N, 0, D.sd), (T - T0), N))
  Y0 <- e + matrix(mu, T, N)
  if (r > 0) Y0 <- Y0 + fsize * factor %*% t(lambda)
  if (FE == 1) Y0 <- Y0 + matrix(alpha, T, N, byrow = TRUE) + matrix(xi, T, N)
  if (p != 0) for (k in 1:p) Y0 <- Y0 + X[, , k] * beta[k]
  Y1 <- Y0 + eff
  Y <- (matrix(1, T, N) - D) * Y0 + D * Y1
  panel <- data.frame(id = rep(101:(100 + N), each = T), time = rep(1:T, N),
                      Y = c(Y), D = c(D), eff = c(eff))
  for (i in 1:p) panel[[paste0("X", i)]] <- c(X[, , i])
  # the realised average effect on the treated, which is what a coverage check
  # compares the interval against (sim_coverage.R's `effect` under "parametric")
  attr(panel, "effect") <- apply(as.matrix(matrix(eff, T, N)[, 1:Ntr]), 1, sum) / Ntr
  panel
}

# The cases: sim_factor.R's grid restricted to Ntr = 5, plus the wider donor
# pools, which is where the factor-count choice is under most pressure.
CASES <- list(
  list(tag = "T10_Nco40",  T0 = 10, Nco = 40,  Ntr = 5),
  list(tag = "T30_Nco40",  T0 = 30, Nco = 40,  Ntr = 5),
  list(tag = "T15_Nco80",  T0 = 15, Nco = 80,  Ntr = 5),
  list(tag = "T15_Nco120", T0 = 15, Nco = 120, Ntr = 5)
)

selected <- if (nzchar(want)) strsplit(want, ",")[[1]] else NULL
set.seed(123)
rows <- list()
for (cs in CASES) {
  if (!is.null(selected) && !(cs$tag %in% selected)) next
  panels <- list()
  t0 <- Sys.time()
  for (i in 1:sims) {
    panel <- simulate(Ntr = cs$Ntr, Nco = cs$Nco, T0 = cs$T0, p = 2, r = 2,
                      w = 0.5, D.sd = 1, beta = c(1, 3), mu = 5,
                      att = c(1:10), fsize = 1, FE = 1)
    eff <- attr(panel, "effect")
    out <- tryCatch(
      gsynth(Y ~ D + X1 + X2, data = panel, index = c("id", "time"),
             se = 0, r = c(0, 5), CV = 1, force = "two-way"),
      error = function(e) NULL)
    mspe <- rep(NA_real_, 6)
    if (!is.null(out)) mspe[seq_len(nrow(out$CV.out))] <- out$CV.out[, "MSPE"]
    rows[[length(rows) + 1]] <- data.frame(
      case = cs$tag, rep = i, T0 = cs$T0, Nco = cs$Nco, Ntr = cs$Ntr,
      r_cv = if (is.null(out)) NA_integer_ else out$r.cv,
      att_avg = if (is.null(out)) NA_real_ else out$att.avg,
      true_att = mean(eff[(cs$T0 + 1):length(eff)]),
      mspe0 = mspe[1], mspe1 = mspe[2], mspe2 = mspe[3],
      mspe3 = mspe[4], mspe4 = mspe[5], mspe5 = mspe[6],
      stringsAsFactors = FALSE)
    panel$rep <- i
    panels[[i]] <- panel
  }
  write.csv(do.call(rbind, panels), file.path(outdir, paste0(cs$tag, ".csv")),
            row.names = FALSE)
  tbl <- do.call(rbind, rows)
  tbl <- tbl[tbl$case == cs$tag, ]
  cat(sprintf("%-11s T0=%2d Nco=%3d Ntr=%2d  sims=%3d  correct r  %.3f   over %.3f   under %.3f   (%.0fs)\n",
              cs$tag, cs$T0, cs$Nco, cs$Ntr, sims,
              mean(tbl$r_cv == 2, na.rm = TRUE),
              mean(tbl$r_cv > 2, na.rm = TRUE),
              mean(tbl$r_cv < 2, na.rm = TRUE),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
write.csv(do.call(rbind, rows), file.path(outdir, "reference.csv"), row.names = FALSE)
cat(sprintf("wrote %s\n", file.path(outdir, "reference.csv")))
