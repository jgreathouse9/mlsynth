#!/usr/bin/env Rscript
# Wan, Xie & Hsiao (2018) "Panel data approach vs synthetic control method",
# Economics Letters 164, 121-123 -- the R-side reference for mlsynth's PDA and
# SCM on every simulation design the authors' replication archive ships.
#
# The archive (Simspda.zip, the paper's supplementary code) contains eight
# scripts covering six design variants:
#
#   sim1b_c9.R     Design 1b   (J,T0) = (20,20)  seed 12345   needs rgdpl1980
#   sim2d_c4.R     Design 2d   (J,T0) = (10, 5)  seed  2017   needs rgdpl1980
#   sim2d_c7.R     Design 2d   (J,T0) = (20, 5)  seed  2017   needs rgdpl1980
#   sim5b_c5.R     Design 5b   (J,T0) = (10,10)  seed  1234
#   sim6a_c3ii.R   Design 6a   (J,T0) = ( 5,40)  seed  1234
#   sim6a_c9.R     Design 6a   (J,T0) = (20,20)  seed  1234
#   sim6c_c3ii.R   Design 6c   (J,T0) = ( 5,40)  seed  1234
#   sim6d_c3ii.R   Design 6d   (J,T0) = ( 5,40)  seed  1234
#
# Only Design 6a has a printed Table 2 column; the other five are variants whose
# results the paper leaves to its Supplementary Note. So 6a is validated against
# the paper (benchmarks/cases/wan_pda_vs_scm.py, Path B) and all six are
# cross-validated against this script (benchmarks/cases/wan_pda_vs_scm_ref.py).
#
# The data-generating blocks below are transcribed verbatim from those scripts,
# including their quirks -- Design 6c's `rep(a, t)` recycles a length-(J+1)
# intercept down a length-T*(J+1) column-major matrix, so the intercept a unit
# receives rotates with the period, and Design 5b overwrites the random walk it
# just built for column 2. Both are reproduced as written: the target is the
# authors' experiment, not a corrected version of it.
#
# Estimation is the authors' own: pampe(select = "AICc", nvmax = t0 - 4) for PDA
# and Synth::dataprep + synth(method = "BFGS") with each pre-period outcome as
# its own predictor for SCM, wrapped in their tryCatch/skipError so a
# non-converging replication drops out exactly as it does for them.
#
# Writes, into --out:
#   <design>.bin      the panels, float64, (rep, time, unit) row-major
#   reference.csv     one row per replication: the R-side MSE/MAE of both methods
#
# Usage:
#   Rscript benchmarks/R/wan_pda_vs_scm.R --out DIR [--reps N] [--designs a,b]
#                                         [--lgdp basedata/gvb_rgdpl1980.csv]

suppressWarnings(suppressMessages({
  library(Synth); library(leaps); library(pampe)
}))

args <- commandArgs(trailingOnly = TRUE)
argval <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
outdir  <- argval("--out", "wanref")
reps    <- as.integer(argval("--reps", "200"))
lgdp_p  <- argval("--lgdp", "basedata/gvb_rgdpl1980.csv")
want    <- argval("--designs", "")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

skipError <- function(e) return(NA)

# The Gardeazabal & Vega-Bayo (2017) donor pool: 1980 log real GDP per capita
# from PWT, the file their replication archive ships as rgdpl1980.txt. Designs
# 1b and 2d sample it; the other four need no external data.
lgdp <- if (file.exists(lgdp_p)) as.matrix(read.csv(lgdp_p)$rgdpl1980) else NULL

# ---------------------------------------------------------------------------
# The six data-generating processes, verbatim from the archive scripts.
# ---------------------------------------------------------------------------

gen_1b <- function(j, t) {                                    # sim1b_c9.R
  lam1 <- matrix(0, nrow = t, ncol = 1)
  eps1 <- rnorm(t, mean = 0, sd = 0.5)
  for (s in 2:t) lam1[s] <- 0.5 * lam1[s - 1] + eps1[s]
  lam <- cbind(lam1)
  miu.norm <- matrix(rnorm((j + 1), mean = 1, sd = 1), nrow = (j + 1), 1)
  gam <- sample(lgdp, (j + 1), replace = TRUE)
  ga.norm <- as.matrix((gam - mean(gam)) / sd(gam))
  z <- matrix(sample(lgdp, (j + 1), replace = TRUE))
  theta <- matrix(seq(from = 0.3, to = 0.9, length.out = t))
  zt <- theta %*% t(z)
  zt.norm <- matrix(0, nrow = t, ncol = (j + 1))
  for (iz in 1:(j + 1)) zt.norm[, iz] <- (zt[, iz] - mean(zt[, iz])) / sd(zt[, iz])
  y <- matrix(0, nrow = t, ncol = (j + 1))
  miulam <- lam %*% t(miu.norm)
  for (i in 1:(j + 1))
    y[, i] <- ga.norm[i] + zt.norm[, i] + miulam[, i] + rnorm(t, mean = 0, sd = 0.25)
  y
}

gen_2d <- function(j, t) {                                    # sim2d_c4.R / _c7.R
  lam <- matrix(0, nrow = t, ncol = 1)
  eps <- matrix(rnorm(t, mean = 0, sd = 0.5), nrow = t, ncol = 1)
  lam[1] <- sample(lgdp, 1, 1)
  eta <- rchisq(1, 1)
  for (s in 2:t) lam[s] <- eta + lam[s - 1] + eps[s]
  miu.norm <- as.matrix(rnorm((j + 1), mean = 1, sd = 2), nrow = (j + 1), 1)
  ga.norm <- rnorm((j + 1), -1, 1)
  z <- matrix(sample(lgdp, (j + 1), replace = TRUE))
  theta <- matrix(0, nrow = t, ncol = 1)
  eps1 <- matrix(rnorm(t, mean = 0, sd = 0.1), nrow = t, ncol = 1)
  theta[1] <- eps1[1]
  for (s in 2:t) theta[s] <- theta[s - 1] + eps1[s]
  zt.norm <- theta %*% t(z)
  y <- matrix(0, nrow = t, ncol = (j + 1))
  miulam <- lam %*% t(miu.norm)
  for (i in 1:(j + 1))
    y[, i] <- ga.norm[i] + zt.norm[, i] + miulam[, i] + rnorm(t, mean = 0, sd = 0.25)
  y
}

gen_5b <- function(j, t) {                                    # sim5b_c5.R
  y0 <- matrix(0, nrow = (t + 20), ncol = (j + 1))
  eps <- matrix(rnorm((t + 20) * (j + 1), mean = 0, sd = 0.5),
                nrow = (t + 20), ncol = (j + 1))
  for (i in 2:(j + 1)) for (s in 3:(t + 20)) y0[s, i] <- y0[(s - 1), i] + eps[s, i]
  beta1 <- seq(from = 1, to = 2, length.out = (j - 1))
  beta2 <- matrix(1, nrow = (j - 1), ncol = 1)
  y0[, 1] <- y0[, 3:(j + 1)] %*% beta1 + eps[, 1]
  y0[, 2] <- y0[, 3:(j + 1)] %*% beta2 + eps[, 2]
  y0[21:(t + 20), ]
}

gen_6a <- function(j, t) {                                    # sim6a_c9.R / _c3ii.R
  eta <- rchisq(1, 1)
  lam <- matrix(0, nrow = t, ncol = 1)
  for (s in 2:t) lam[s] <- eta + lam[s - 1] + rnorm(1, mean = 0, sd = 0.5)
  miu <- matrix(1, nrow = 1, ncol = (j + 1))
  eps <- matrix(rnorm(t * (j + 1), mean = 0, sd = 0.25), nrow = t, ncol = (j + 1))
  lam %*% miu + eps
}

gen_6c <- function(j, t) {                                    # sim6c_c3ii.R
  eta <- rchisq(1, 1)
  lam <- matrix(0, nrow = t, ncol = 1)
  for (s in 2:t) lam[s] <- eta + lam[s - 1] + rnorm(1, mean = 0, sd = 0.5)
  miu <- matrix(rnorm((j + 1), 1, 1), nrow = 1, ncol = (j + 1))
  eps <- matrix(rnorm(t * (j + 1), mean = 0, sd = 0.25), nrow = t, ncol = (j + 1))
  a <- matrix(rnorm((j + 1), 0, 1), nrow = 1, ncol = (j + 1))
  rep(a, t) + lam %*% miu + eps
}

gen_6d <- function(j, t) {                                    # sim6d_c3ii.R
  eta <- rchisq(1, 1)
  lam <- matrix(0, nrow = t, ncol = 1)
  for (s in 2:t) lam[s] <- eta + lam[s - 1] + rnorm(1, mean = 0, sd = 0.5)
  miu <- matrix(rnorm((j + 1), 1, 1), nrow = 1, ncol = (j + 1))
  eps <- matrix(rnorm(t * (j + 1), mean = 0, sd = 0.25), nrow = t, ncol = (j + 1))
  lam %*% miu + eps
}

DESIGNS <- list(
  list(tag = "d1b_j20_t20",  gen = gen_1b, seed = 12345, j = 20, t0 = 20, t1 = 10, lgdp = TRUE),
  list(tag = "d2d_j10_t5",   gen = gen_2d, seed =  2017, j = 10, t0 =  5, t1 = 10, lgdp = TRUE),
  list(tag = "d2d_j20_t5",   gen = gen_2d, seed =  2017, j = 20, t0 =  5, t1 = 10, lgdp = TRUE),
  list(tag = "d5b_j10_t10",  gen = gen_5b, seed =  1234, j = 10, t0 = 10, t1 = 10, lgdp = FALSE),
  list(tag = "d6a_j20_t20",  gen = gen_6a, seed =  1234, j = 20, t0 = 20, t1 = 10, lgdp = FALSE),
  list(tag = "d6a_j5_t40",   gen = gen_6a, seed =  1234, j =  5, t0 = 40, t1 = 20, lgdp = FALSE),
  list(tag = "d6c_j5_t40",   gen = gen_6c, seed =  1234, j =  5, t0 = 40, t1 = 20, lgdp = FALSE),
  list(tag = "d6d_j5_t40",   gen = gen_6d, seed =  1234, j =  5, t0 = 40, t1 = 20, lgdp = FALSE)
)

# ---------------------------------------------------------------------------
# The authors' estimation, unchanged: pampe for PDA, Synth for SCM.
# ---------------------------------------------------------------------------

fit_pda <- function(y, j, t0, t) {
  d.pda <- as.data.frame(y)
  pda <- tryCatch(pampe(time.pretr = 1:t0, time.tr = (t0 + 1):t, treated = 1,
                        nvmax = (t0 - 4), data = d.pda, select = "AICc"),
                  error = skipError)
  yhat <- tryCatch(as.matrix(pda$counterfactual[(t0 + 1):t, 2]), error = function(e) NA)
  a <- tryCatch(as.matrix(y[(t0 + 1):t, 1] - yhat), error = function(e) NA)
  x.pda <- tryCatch(as.matrix(d.pda[1:t0, pda$controls]), error = function(e) NA)
  y.pda <- as.matrix(d.pda[1:t0, 1])
  lm.pda <- tryCatch(lm(y.pda ~ x.pda), error = skipError)
  list(mse = tryCatch(mean(a^2), error = function(e) NA_real_),
       mae = tryCatch(mean(abs(y.pda - cbind(1, x.pda) %*%
                               as.matrix(lm.pda$coefficients))),
                      error = function(e) NA_real_),
       nk = tryCatch(length(pda$controls), error = function(e) NA_real_))
}

fit_scm <- function(ydata, j, t0, t) {
  nam <- as.matrix(seq(1, (j + 1), 1)) %x% rep(1, t)
  yr0 <- as.matrix(seq(1, t, 1)) %*% rep(1, (j + 1))
  yr <- matrix(yr0, nrow = t * (j + 1), ncol = 1)
  xx <- matrix(0, nrow = t * (j + 1), ncol = t0)
  for (ix in 1:t0) xx[, ix] <- as.matrix(ydata[ix, ] %x% rep(1, t))
  ycol <- matrix(ydata, nrow = t * (j + 1), ncol = 1)
  dat <- as.data.frame(cbind(nam, yr, ycol, xx))
  colnames(dat) <- c("country", "time", "ycol", paste0("p", 1:t0))
  s.data <- tryCatch(dataprep(foo = dat, predictors = paste0("p", 1:t0),
                              predictors.op = "mean", time.predictors.prior = 1:t0,
                              dependent = "ycol", unit.variable = "country",
                              time.variable = "time", treatment.identifier = 1,
                              controls.identifier = c(2:(j + 1)),
                              time.optimize.ssr = 1:t0, time.plot = 1:t),
                     error = skipError)
  scm <- tryCatch(synth(data.prep.obj = s.data, method = "BFGS"), error = skipError)
  w <- tryCatch(as.matrix(scm$solution.w), error = function(e) NA)
  a <- tryCatch(as.matrix(ydata[(t0 + 1):t, 1] -
                          as.matrix(ydata[(t0 + 1):t, 2:(j + 1)]) %*% w),
                error = function(e) NA)
  pre <- tryCatch(ydata[1:t0, 1] - as.matrix(ydata[1:t0, 2:(j + 1)]) %*% w,
                  error = function(e) NA)
  list(mse = tryCatch(mean(a^2), error = function(e) NA_real_),
       mae = tryCatch(mean(abs(pre)), error = function(e) NA_real_),
       # the pre-period SSR is Synth's own outer objective (time.optimize.ssr),
       # so it is the quantity on which the two solvers are directly comparable
       ssr = tryCatch(sum(pre^2), error = function(e) NA_real_))
}

# ---------------------------------------------------------------------------

selected <- if (nzchar(want)) strsplit(want, ",")[[1]] else NULL
rows <- list()
for (d in DESIGNS) {
  if (!is.null(selected) && !(d$tag %in% selected)) next
  if (d$lgdp && is.null(lgdp)) {
    cat(sprintf("SKIP %s (needs %s)\n", d$tag, lgdp_p)); next
  }
  t <- d$t0 + d$t1
  set.seed(d$seed)
  con <- file(file.path(outdir, paste0(d$tag, ".bin")), "wb")
  for (it in 1:reps) {
    y <- d$gen(d$j, t)
    writeBin(as.vector(t(y)), con, size = 8)       # row-major: time outer
    p <- fit_pda(y, d$j, d$t0, t)
    s <- fit_scm(as.matrix(y), d$j, d$t0, t)
    rows[[length(rows) + 1]] <- data.frame(
      design = d$tag, rep = it, j = d$j, t0 = d$t0, t1 = d$t1,
      mse_pda = p$mse, mae_pda = p$mae, nk_pda = p$nk,
      mse_scm = s$mse, mae_scm = s$mae, ssr_scm = s$ssr,
      ybar = abs(mean(y[1:d$t0, 1])), stringsAsFactors = FALSE)
  }
  close(con)
  tbl <- do.call(rbind, rows)
  tbl <- tbl[tbl$design == d$tag, ]
  cat(sprintf("%-13s (J,T0)=(%2d,%2d) reps=%4d  pampe MSE %10.4f  Synth MSE %9.4f  dropped %d/%d\n",
              d$tag, d$j, d$t0, reps, mean(tbl$mse_pda, na.rm = TRUE),
              mean(tbl$mse_scm, na.rm = TRUE),
              sum(is.na(tbl$mse_pda)) + sum(is.na(tbl$mse_scm)), 2 * reps))
}
write.csv(do.call(rbind, rows), file.path(outdir, "reference.csv"), row.names = FALSE)
cat(sprintf("wrote %s\n", file.path(outdir, "reference.csv")))
