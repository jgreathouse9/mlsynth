# Reference run for the `cwz_conformal` benchmark case.
#
# The conformal test-inversion procedure of Chernozhukov, Wuthrich & Zhu (JASA
# 2021), run from the authors' own R, on the panel their paper uses as its
# empirical application: Rhode Island's 2003 decriminalization of indoor
# prostitution and log female gonorrhea incidence (Cunningham & Shah 2018),
# T0 = 19 (1985-2003), T1 = 6 (2004-2009), 50 control states.
#
# Two implementations of the same method exist and both are the authors':
#
#   * the scinference package (kwuthrich/scinference at v1.0.0, 567c688), whose
#     `scinference(inference_method = "conformal")` is the packaged form; and
#   * the JASA online supplement's `functions_conformal_final.R`, the code that
#     produced the published tables and figures.
#
# This script runs the package and re-derives the supplement's three functions
# beside it, so the two are checked against each other on the same panel instead
# of being assumed identical. They are reproduced here from the supplement (about
# forty lines total) so the comparison needs no file the replication package
# does not already publish; the supplement is the authoritative text for them.
#
# lsei_type = 2 throughout. The panel has J = 50 donors against T01 = 25
# periods, and limSolve's type 1 reports "inequalities contradictory" and
# returns weights off the simplex -- so type 2 is not a preference here, it is
# the only feasible solve, and it is what the supplement uses
# (functions_conformal_final.R:29). The carbon tax panel in `cwz_ttest` is the
# other shape (J = 14, T01 = 46) and uses type 1 there.
#
# Run from the repository root:  Rscript benchmarks/reference/cwz_conformal/reference.R
suppressMessages(library(scinference))
set.seed(12345)

emit <- function(k, v) cat(sprintf("%s\t%.10f\n", k, v))

LT     <- 2
ALPHA  <- 0.1
Q_NORM <- 1                      # L1, the norm the application uses
T0     <- 19
T1     <- 6
NPERM  <- 5000                   # scinference's default for iid permutations

lfr <- read.delim("basedata/logfemrate.txt", header = TRUE)
Y1  <- as.matrix(lfr[, 1])
Y0  <- as.matrix(lfr[, 2:ncol(lfr)])

## ------------------------------------------------------------------------
## The supplement's functions, reproduced from
## online_supplements/replication_package_final/functions_conformal_final.R
## ------------------------------------------------------------------------

sup.sc <- function(Y1, Y0) {
  J <- dim(Y0)[2]
  w <- limSolve::lsei(A = Y0, B = Y1, E = matrix(1, 1, J), F = 1,
                      G = diag(x = 1, J, J), H = matrix(0, J, 1), type = 2)$X
  list(u.hat = Y1 - Y0 %*% w, w.hat = w)
}

sup.moving.block.q <- function(Y1, Y0, T0, T1, q) {
  T01 <- T0 + T1
  u.hat <- sup.sc(Y1, Y0)$u.hat
  u.hat.c <- c(u.hat, u.hat)
  S.vec <- matrix(NA, T01, 1)
  for (s in 1:T01) {
    S.vec[s, 1] <- if (q == 1) sum(abs(u.hat.c[s:(s + T1 - 1)])) else
      sqrt(sum((u.hat.c[s:(s + T1 - 1)])^2))
  }
  mean(S.vec >= S.vec[T0 + 1])
}

sup.all.q <- function(Y1, Y0, T0, T1, nperm, q) {
  T01 <- T0 + T1
  u.hat <- sup.sc(Y1, Y0)$u.hat
  post.ind <- ((T0 + 1):T01)
  Sq <- if (q == 1) sum(abs(u.hat[post.ind])) else sqrt(sum((u.hat[post.ind])^2))
  S.vec <- matrix(NA, nperm, 1)
  for (r in 1:nperm) {
    u.p <- u.hat[sample(1:T01, replace = FALSE)]
    S.vec[r, 1] <- if (q == 1) sum(abs(u.p[post.ind])) else
      sqrt(sum((u.p[post.ind])^2))
  }
  1 / (nperm + 1) * (1 + sum(S.vec >= Sq))
}

sup.ci <- function(Y1, Y0, alpha, thetagrid) {
  T01 <- dim(Y0)[1]
  p.vec <- rep(NA, length(thetagrid))
  for (ind in seq_along(thetagrid)) {
    Y1_0 <- Y1
    Y1_0[T01] <- Y1[T01] - thetagrid[ind]
    u.hat <- sup.sc(Y1_0, Y0)$u.hat
    p.vec[ind] <- mean(abs(u.hat) >= abs(u.hat[T01]))
  }
  thetagrid[p.vec > alpha]
}

cat("== REFERENCE VALUES ==\n")

## ------------------------------------------------------------------------
## Joint null of no effect, both permutation schemes
## ------------------------------------------------------------------------

p_mb <- scinference(Y1 = Y1, Y0 = Y0, T1 = T1, T0 = T0,
                    inference_method = "conformal", estimation_method = "sc",
                    permutation_method = "mb", alpha = ALPHA, lsei_type = LT)$p_val
emit("p_mb", p_mb)

set.seed(12345)
p_iid <- scinference(Y1 = Y1, Y0 = Y0, T1 = T1, T0 = T0,
                     inference_method = "conformal", estimation_method = "sc",
                     permutation_method = "iid", alpha = ALPHA,
                     n_perm = NPERM, lsei_type = LT)$p_val
emit("p_iid", p_iid)

# The supplement's own versions of the same two quantities. mb is deterministic
# and must agree to the last digit; iid resamples, so it agrees up to its own
# Monte-Carlo error and is emitted for the record.
emit("p_mb_supplement", sup.moving.block.q(Y1, Y0, T0, T1, Q_NORM))
set.seed(12345)
emit("p_iid_supplement", sup.all.q(Y1, Y0, T0, T1, NPERM, Q_NORM))

## ------------------------------------------------------------------------
## Pointwise confidence intervals, on the application's own grid
## ------------------------------------------------------------------------

GRID <- seq(-5, 2, 0.01)          # app_cunninghamshah_final.R:135
rci <- scinference(Y1 = Y1, Y0 = Y0, T1 = T1, T0 = T0,
                   inference_method = "conformal", estimation_method = "sc",
                   permutation_method = "mb", alpha = ALPHA, ci = TRUE,
                   ci_grid = GRID, lsei_type = LT)
for (t in 1:T1) {
  emit(sprintf("lb_%d", 2003 + t), rci$lb[t])
  emit(sprintf("ub_%d", 2003 + t), rci$ub[t])
}

# Same intervals from the supplement's `ci`, which the application calls one
# post period at a time on the pre-period plus that period.
for (t in 1:T1) {
  idx <- c(1:T0, T0 + t)
  s <- sup.ci(Y1[idx], Y0[idx, ], ALPHA, GRID)
  emit(sprintf("lb_%d_supplement", 2003 + t), min(s))
  emit(sprintf("ub_%d_supplement", 2003 + t), max(s))
}

## ------------------------------------------------------------------------
## Placebo specification tests (app_cunninghamshah_final.R:76-84)
##
## The last t pre-treatment periods are held out and tested as if they were the
## post-period, on a panel where no effect exists. A p-value near the level is
## the check that the procedure is calibrated on this panel before it is used
## on the real post-period.
## ------------------------------------------------------------------------

Y1pre <- Y1[1:T0]
Y0pre <- Y0[1:T0, ]
for (t in 1:3) {
  emit(sprintf("placebo_mb_T1_%d", t),
       scinference(Y1 = Y1pre, Y0 = Y0pre, T1 = t, T0 = T0 - t,
                   inference_method = "conformal", estimation_method = "sc",
                   permutation_method = "mb", alpha = ALPHA,
                   lsei_type = LT)$p_val)
}

## ------------------------------------------------------------------------
## Fit intermediates, so a disagreement can be localised to a seam
## ------------------------------------------------------------------------

fit_pre <- sup.sc(Y1[1:T0], Y0[1:T0, ])
for (j in seq_along(fit_pre$w.hat)) {
  cat(sprintf("weight\tdonor%02d\t%.10f\n", j, fit_pre$w.hat[j]))
}

cat("== SESSION INFO ==\n")
print(sessionInfo())
