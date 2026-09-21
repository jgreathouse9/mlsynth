#!/usr/bin/env Rscript
# Xu (2017) Table A4 -- the factor model against the convex hull.
#
# The Online Appendix's "Comparison with the Synthetic Control Estimator
# (ADH 2010)": one treated unit, forty donors, fifteen pre-periods, ten post,
# and eight cells. Cases 1-4 hold the loading overlap w at one and give gsynth a
# rank of 1, 2, 3, 4 against a process that always has two factors, so only case
# 2 is correctly specified. Cases 5-8 keep the rank right and pull the treated
# unit's loadings away from the donors' until, at w = 0, the supports do not
# overlap and no convex combination of donors can reach the treated unit.
#
# The archive's sim_adh.R does not reproduce Table A4, in two ways, and this
# script follows the appendix and the published table where they differ. Both
# departures were settled by measurement; the reasoning is in the Python case.
#
# First, the panel. The script draws one panel per cell outside the replication
# loop and then redraws only the outcome, `panel$Y <- panel$Ybar + rnorm(N*T)`,
# which holds the treatment effect fixed. The appendix says the opposite: "for
# each set of simulations, the factors are drawn only once, while the treatment
# effect, regressors, factor loadings, and error terms are drawn repeatedly."
# The published table settles it. Its SD column is the dispersion of the ATT and
# its RMSE column the root mean square of the ATT minus the realised effect, so
# if the effect were held the two would coincide. Across the eight cells
# SD^2 - RMSE^2 is 1.01, 1.05, 1.01, 1.02, 1.02, 0.99, 1.05, 1.01 -- which is
# D.sd^2 for an effect redrawn every replication. So the panel is regenerated
# per replication here, at fixF = TRUE and fixL = FALSE.
#
# Second, the covariates. The script's own header sets `p <- 0  # no covariates`
# and then calls simulate() with p = 2 and beta = c(1, 3) while leaving the
# covariates out of gsynth's formula. Table A4 was generated at p = 0, and the
# arithmetic says so twice over. With p = 2 the independent part of X carries
# variance 1 + 9 into the outcome with nothing to absorb it, and gsynth's SD
# comes out near 4 against the table's 1.5 to 1.8. And X contains
# 0.5 * F %*% t(lambda), so at beta = (1, 3) the outcome's factor term is
# tripled and with it the loading-mismatch bias: Synth's bias at w = 0 measures
# 6.05 under p = 2 against the table's 2.13, almost exactly three times. At
# p = 0 both arms land on the table -- Synth's bias path over w = 0.75, 0.50,
# 0.25, 0.00 measures 0.70, 1.21, 1.66, 2.15 against the published 0.71, 1.33,
# 1.63, 2.13 -- so the covariate call is a later edit the table predates.
#
# The Synth arm keeps the script's own sparse specification: three special
# predictors, the outcome at periods 1, 8 and 15, optimised on 2-6 and 9-14.
# That is what produces the table's small but non-zero Synth failure rate.
#
# Writes, into --out:
#   case<k>.csv     the panels, long, all replications
#   reference.csv   per replication: each estimator's ATT at the fifth post
#                   period, the realised effect, and whether Synth solved
#
# Usage:
#   Rscript benchmarks/R/xu_gsynth_vs_scm.R --out DIR [--sims N] [--fl PATH]

suppressWarnings(suppressMessages({library(gsynth); library(Synth)}))

args <- commandArgs(trailingOnly = TRUE)
argval <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
outdir <- argval("--out", "xuadh")
sims <- as.integer(argval("--sims", "60"))
fl <- argval("--fl", "basedata/xu_gsynth_FLSource.RData")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
if (!file.exists(fl)) stop("FLSource.RData not found; pass --fl")
load(fl)   # F.source, F.u.source, L.source

Ntr <- 1; Nco <- 40; N <- Ntr + Nco; T0 <- 15; T <- T0 + 10

# sim_sampling.R's generator at p = 0, fixF = TRUE (factors read from the file)
# and fixL = FALSE (loadings redrawn), which is the appendix's description. The
# draw order is the original's: loadings, then the unit effects, then the error,
# then the treatment effect.
draw_panel <- function(w, mu = 5, att = c(1:10), D.sd = 1) {
  r <- 2
  ss <- sqrt(3)
  lambda <- matrix(runif(N * r, min = -ss, max = ss), N, r)
  lambda[1:Ntr, ] <- lambda[1:Ntr, ] + (1 - w) * 2 * ss
  factor <- F.source[(nrow(F.source) - T + 1):nrow(F.source), 1:r]
  alpha <- runif(N, min = -ss, max = ss)
  alpha[1:Ntr] <- alpha[1:Ntr] + (1 - w) * 2 * ss
  xi <- F.source[(nrow(F.source) - T + 1):nrow(F.source), 20]
  e <- matrix(rnorm(T * N), T, N)
  D <- cbind(rbind(matrix(0, T0, Ntr), matrix(1, (T - T0), Ntr)), matrix(0, T, Nco))
  eff <- matrix(c(rep(0, T0), att), T, N) +
    rbind(matrix(0, T0, N), matrix(rnorm((T - T0) * N, 0, D.sd), (T - T0), N))
  Y0 <- e + matrix(mu, T, N) + factor %*% t(lambda) +
    matrix(alpha, T, N, byrow = TRUE) + matrix(xi, T, N)
  Y <- Y0 + D * eff
  panel <- data.frame(id = rep(101:(100 + N), each = T), time = rep(1:T, N),
                      Y = c(Y), D = c(D))
  attr(panel, "effect") <- apply(as.matrix(eff[, 1:Ntr]), 1, sum) / Ntr
  panel
}

RR <- c(1, 2, 3, 4, 2, 2, 2, 2)
WW <- c(1, 1, 1, 1, 0.75, 0.50, 0.25, 0)

set.seed(1234)
rows <- list()
for (case in seq_along(RR)) {
  r <- RR[case]; w <- WW[case]
  panels <- list(); t0 <- Sys.time(); n_fail <- 0
  for (i in 1:sims) {
    panel <- draw_panel(w)
    eff <- attr(panel, "effect")
    g <- tryCatch(gsynth(Y ~ D, data = panel, index = c("id", "time"),
                         se = FALSE, r = r, CV = 0, force = "two-way"),
                  error = function(e) NULL)
    att_g <- if (is.null(g)) NA_real_ else g$att[T0 + 5]
    att_s <- NA_real_
    ok <- tryCatch({
      dp <- dataprep(foo = panel, dependent = "Y", unit.variable = "id",
                     time.variable = "time",
                     special.predictors = list(list("Y", 1, "mean"),
                                               list("Y", 8, "mean"),
                                               list("Y", 15, "mean")),
                     treatment.identifier = 101,
                     controls.identifier = 102:(100 + N),
                     time.predictors.prior = 1:T0,
                     time.optimize.ssr = c(2:6, 9:14),
                     time.plot = 1:T)
      sy <- synth(dp, verbose = FALSE)
      att_s <- (dp$Y1plot - (dp$Y0plot %*% sy$solution.w))[T0 + 5]
      TRUE
    }, error = function(e) FALSE)
    if (!ok) n_fail <- n_fail + 1
    rows[[length(rows) + 1]] <- data.frame(
      case = case, rep = i, r = r, w = w,
      att_gsynth = att_g, att_synth = att_s, true_k = eff[T0 + 5],
      synth_failed = as.integer(!ok), stringsAsFactors = FALSE)
    panel$rep <- i
    panels[[i]] <- panel
  }
  write.csv(do.call(rbind, panels), file.path(outdir, paste0("case", case, ".csv")),
            row.names = FALSE)
  tbl <- do.call(rbind, rows); tbl <- tbl[tbl$case == case, ]
  bg <- tbl$att_gsynth - tbl$true_k; bs <- tbl$att_synth - tbl$true_k
  cat(sprintf("case %d r=%d w=%.2f | GSC bias %+6.3f SD %5.3f RMSE %5.3f | Synth bias %+6.3f SD %5.3f RMSE %5.3f | fail %d (%.0fs)\n",
              case, r, w, mean(bg, na.rm = TRUE), sd(tbl$att_gsynth, na.rm = TRUE),
              sqrt(mean(bg^2, na.rm = TRUE)), mean(bs, na.rm = TRUE),
              sd(tbl$att_synth, na.rm = TRUE), sqrt(mean(bs^2, na.rm = TRUE)),
              n_fail, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  flush.console()
}
write.csv(do.call(rbind, rows), file.path(outdir, "reference.csv"), row.names = FALSE)
cat(sprintf("wrote %s\n", file.path(outdir, "reference.csv")))
