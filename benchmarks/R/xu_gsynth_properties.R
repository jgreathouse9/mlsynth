#!/usr/bin/env Rscript
# Xu (2017) "Generalized Synthetic Control Method", Political Analysis 25(1) --
# the R-side reference for the sampling properties of the ATT and for the
# Algorithm 2 parametric bootstrap.
#
# Two of the archive's simulation designs, both run through the gsynth 1.0 the
# archive ships:
#
#   sim_TN.R        Table A1   bias, standard deviation and RMSE of the ATT at
#                              the fifth post period, with the rank fixed at its
#                              true value so the estimator alone is measured
#   sim_coverage.R  --         share of periods whose 95% parametric-bootstrap
#                              interval covers the realised average effect on
#                              the treated
#
# The Online Appendix is not in the archive, so Table A1 is not available as a
# target. What is available is the author's own implementation at the version
# the paper ran, and both sides here see the same panels, so the residual is
# implementation difference and not Monte Carlo noise.
#
# The companion script benchmarks/R/xu_gsynth_sims.R covers sim_factor.R, where
# the rank is chosen, not given. The generator below is the same
# transcription of sim_sampling.R; each script carries it so either can be run
# on its own.
#
# Writes, into --out:
#   tn_<case>.csv    / coverage.csv   the panels, long, all replications
#   reference_tn.csv / reference_coverage.csv   gsynth's per-replication output
#
# Usage:
#   Rscript benchmarks/R/xu_gsynth_properties.R --out DIR [--sims N] [--boots N]

suppressWarnings(suppressMessages({library(gsynth)}))

args <- commandArgs(trailingOnly = TRUE)
argval <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
outdir <- argval("--out", "xuprop")
sims <- as.integer(argval("--sims", "30"))
boots <- as.integer(argval("--boots", "200"))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# sim_sampling.R's generator, restricted to the settings these designs use.
# ---------------------------------------------------------------------------
simulate <- function(Ntr, Nco, T0, p, r, w = 1, D.sd = 1, att = c(1:10),
                     beta = NULL, mu = 0, fsize = 1, FE = 0) {
  N <- Ntr + Nco
  T <- T0 + length(att)
  ss <- sqrt(3)
  lambda <- matrix(runif(N * r, min = -ss, max = ss), N, r)
  lambda[1:Ntr, ] <- lambda[1:Ntr, ] + (1 - w) * 2 * ss
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
  attr(panel, "effect") <-
    apply(as.matrix(matrix(eff, T, N)[, 1:Ntr]), 1, sum) / Ntr
  panel
}

# ---------------------------------------------------------------------------
# sim_TN.R: bias / sd / RMSE of the ATT with the rank given.
# Its grid is 36 cells, crossing T0, Nco and Ntr. Three are run here, holding
# T0 and Nco fixed so the only thing that moves is the size of the treated
# group; otherwise a change in dispersion could not be attributed to it.
# ---------------------------------------------------------------------------
TN_CASES <- list(
  list(tag = "Ntr1",  Ntr = 1,  Nco = 40, T0 = 15),
  list(tag = "Ntr5",  Ntr = 5,  Nco = 40, T0 = 15),
  list(tag = "Ntr20", Ntr = 20, Nco = 40, T0 = 15)
)

set.seed(123)
rows <- list()
for (cs in TN_CASES) {
  panels <- list()
  t0 <- Sys.time()
  for (i in 1:sims) {
    panel <- simulate(Ntr = cs$Ntr, Nco = cs$Nco, T0 = cs$T0, p = 2, r = 2,
                      w = 0.8, D.sd = 1, beta = c(1, 3), mu = 5,
                      att = c(1:10), fsize = 1, FE = 1)
    eff <- attr(panel, "effect")
    out <- tryCatch(gsynth(Y ~ D + X1 + X2, data = panel, index = c("id", "time"),
                           force = "two-way", se = FALSE, r = 2, CV = FALSE),
                    error = function(e) NULL)
    # sim_TN.R reports the fifth post period, its ATT_15
    k <- cs$T0 + 5
    rows[[length(rows) + 1]] <- data.frame(
      case = cs$tag, rep = i, Ntr = cs$Ntr, Nco = cs$Nco, T0 = cs$T0,
      att_k = if (is.null(out)) NA_real_ else out$att[k],
      true_k = eff[k],
      beta2 = if (is.null(out)) NA_real_ else out$beta[2],
      stringsAsFactors = FALSE)
    panel$rep <- i
    panels[[i]] <- panel
  }
  write.csv(do.call(rbind, panels), file.path(outdir, paste0("tn_", cs$tag, ".csv")),
            row.names = FALSE)
  tbl <- do.call(rbind, rows); tbl <- tbl[tbl$case == cs$tag, ]
  b <- tbl$att_k - tbl$true_k
  cat(sprintf("TN %-6s Ntr=%2d Nco=%3d  bias %+.4f  sd %.4f  rmse %.4f  (%.0fs)\n",
              cs$tag, cs$Ntr, cs$Nco, mean(b, na.rm = TRUE),
              sd(tbl$att_k, na.rm = TRUE), sqrt(mean(b^2, na.rm = TRUE)),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
write.csv(do.call(rbind, rows), file.path(outdir, "reference_tn.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# sim_coverage.R: the Algorithm 2 parametric bootstrap.
# The archive's own cell, at a reduced bootstrap count.
# ---------------------------------------------------------------------------
Ntr <- 40; Nco <- 120; T0 <- 30; T <- T0 + 10
set.seed(123)
rows <- list(); panels <- list()
t0 <- Sys.time()
for (i in 1:sims) {
  panel <- simulate(Ntr = Ntr, Nco = Nco, T0 = T0, p = 2, r = 2, w = 1,
                    D.sd = 1, beta = c(1, 3), mu = 5, att = c(1:10),
                    fsize = 1, FE = 1)
  eff <- attr(panel, "effect")
  out <- tryCatch(gsynth(Y ~ D + X1 + X2, data = panel, index = c("id", "time"),
                         se = 1, r = 2, CV = 0, force = "two-way",
                         nboots = boots, inference = "parametric",
                         parallel = FALSE),
                  error = function(e) NULL)
  if (is.null(out)) next
  covered <- as.integer(eff >= out$est.att[, 3] & eff <= out$est.att[, 4])
  rows[[length(rows) + 1]] <- data.frame(
    rep = i,
    cover_post = mean(covered[(T0 + 1):T]),
    cover_all = mean(covered),
    att_avg = out$att.avg,
    true_avg = mean(eff[(T0 + 1):T]),
    stringsAsFactors = FALSE)
  panel$rep <- i
  panels[[length(panels) + 1]] <- panel
}
write.csv(do.call(rbind, panels), file.path(outdir, "coverage.csv"), row.names = FALSE)
tbl <- do.call(rbind, rows)
write.csv(tbl, file.path(outdir, "reference_coverage.csv"), row.names = FALSE)
cat(sprintf("coverage  Ntr=%d Nco=%d T0=%d boots=%d sims=%d  post %.3f  all %.3f  (%.0fs)\n",
            Ntr, Nco, T0, boots, nrow(tbl), mean(tbl$cover_post),
            mean(tbl$cover_all),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
cat(sprintf("wrote %s\n", outdir))
