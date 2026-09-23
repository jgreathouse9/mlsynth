# Reference run for the `cwz_conformal_mc` benchmark case.
#
# The simulation study of Chernozhukov, Wuthrich & Zhu (JASA 2021), Section 4,
# driven from the authors' own design as published in the online supplement
# (replication_package_final/simulations_conformal_final.R and its
# functions_conformal_final.R). The reported quantity is the empirical rejection
# rate of the moving-block conformal test under a true null, which is its size
# and should sit near alpha.
#
# The design, read off `sim()`:
#
#   Y0[t, j] = lambda1[j] + F1[t] + lambda2[j] * F2[t] + eps[t, j]
#   Y1[t]    = (Y0 %*% w)[t] + u[t]        (+ the alternative on the post block)
#
# with lambda1 = lambda2 = (1..J)/J, F1 ~ N(0, 1), F2 ~ N(0, 1) when stationary
# and N(0, 1) + t when not, eps[, j] and u unit-variance AR(1) with coefficients
# rho1 and rho2, and w the DGP:
#
#   DGP 1  w = 1/J each              uniform, inside the hull
#   DGP 2  w = (1/3, 1/3, 1/3, 0..)  sparse
#   DGP 3  w = -1/J each             negative uniform, outside the hull
#   DGP 4  w = (1, -1, 0, ..)        differencing, outside the hull
#
# T1 = 1 throughout, alpha = 0.1.
#
# Two things are emitted, answering two different questions.
#
#   size_*  -- the rejection rate over NREPS draws per cell. This is the
#              published quantity, and it validates the DGP port: mlsynth draws
#              its own panels from its own port of this design and has to land
#              on the same rate up to Monte-Carlo error.
#
#   draw_*  -- the panels themselves for the first N_DUMP draws of two cells,
#              with the p-value R computed on each. This validates the
#              inference: handed the identical panel, mlsynth has to return the
#              identical p-value, exactly, with no Monte-Carlo error anywhere in
#              the comparison. The rho = 0.6 cell is the one that matters:
#              serially correlated errors are the assumption boundary the
#              moving-block scheme exists for.
#
# lsei_type = 2, as in the supplement.
#
# Run from the repository root:
#   Rscript benchmarks/reference/cwz_conformal_mc/reference.R
suppressMessages(library(scinference))

emit <- function(k, v) cat(sprintf("%s\t%.10f\n", k, v))

ALPHA  <- 0.1
LT     <- 2
T0     <- 20
T1     <- 1
J      <- 20
NREPS  <- 500      # the paper uses 5000; the tolerance below carries the gap
N_DUMP <- 10
DUMP_DIR <- "benchmarks/reference/cwz_conformal_mc"

# The supplement's AR(1) generator, verbatim in behaviour: unit unconditional
# variance, started from a draw of the stationary distribution.
generate.AR.series <- function(T01, rho) {
  u <- rep(NA, T01)
  epsl <- rnorm(T01) * sqrt(1 - rho^2)
  startvalue <- rnorm(1)
  u[1] <- rho * startvalue + epsl[1]
  for (t in 2:T01) u[t] <- rho * u[t - 1] + epsl[t]
  u
}

draw_panel <- function(stationary, T0, T1, J, rho1, rho2, DGP, alternative) {
  T01 <- T0 + T1
  F1 <- rnorm(T01)
  F2 <- if (stationary == 1) rnorm(T01) else rnorm(T01) + c(1:T01)
  lambda1 <- c(seq(1, J, 1) / J)
  lambda2 <- lambda1
  w1 <- cbind(rep(1 / J, J))
  w2 <- cbind(c(1 / 3, 1 / 3, 1 / 3, rep(0, J - 3)))
  w3 <- (-1) * cbind(rep(1 / J, J))
  w4 <- cbind(c(1, -1, rep(0, J - 2)))
  w <- w1 * (DGP == 1) + w2 * (DGP == 2) + w3 * (DGP == 3) + w4 * (DGP == 4)
  y <- kronecker(lambda1, matrix(1, T01, 1)) + kronecker(matrix(1, J, 1), F1) +
    kronecker(cbind(lambda2), cbind(F2))
  eps <- matrix(NA, T01, J)
  for (j in 1:J) eps[, j] <- generate.AR.series(T01, rho1)
  Y0 <- matrix(y, T01, J) + eps
  u <- generate.AR.series(T01, rho2)
  Y1 <- Y0 %*% w + u
  Y1[(T0 + 1):T01] <- Y1[(T0 + 1):T01] + alternative
  list(Y1 = Y1, Y0 = Y0)
}

p_mb <- function(p)
  scinference(Y1 = p$Y1, Y0 = p$Y0, T1 = T1, T0 = T0,
              inference_method = "conformal", estimation_method = "sc",
              permutation_method = "mb", alpha = ALPHA, lsei_type = LT)$p_val

cat("== REFERENCE VALUES ==\n")

## ------------------------------------------------------------------------
## Size, stationary design, iid and serially correlated errors
## ------------------------------------------------------------------------

for (rho in c(0, 0.6)) {
  for (dgp in 1:4) {
    set.seed(12345)
    rej <- numeric(NREPS)
    for (r in 1:NREPS) {
      rej[r] <- p_mb(draw_panel(1, T0, T1, J, rho, rho, dgp, 0)) <= ALPHA
    }
    emit(sprintf("size_dgp%d_rho%s", dgp, sub("\\.", "", format(rho))), mean(rej))
  }
}

## ------------------------------------------------------------------------
## Seed-matched panels, for the exact per-draw comparison
## ------------------------------------------------------------------------

for (rho in c(0, 0.6)) {
  tag <- sprintf("rho%s", sub("\\.", "", format(rho)))
  set.seed(20260814)
  rows <- NULL
  for (r in 1:N_DUMP) {
    pan <- draw_panel(1, T0, T1, J, rho, rho, 1, 0)
    emit(sprintf("draw_%s_%02d_p", tag, r), p_mb(pan))
    rows <- rbind(rows, cbind(r, 1:(T0 + T1), pan$Y1, pan$Y0))
  }
  colnames(rows) <- c("draw", "t", "Y1", paste0("Y0_", 1:J))
  write.table(rows, file.path(DUMP_DIR, sprintf("draws_%s.tsv", tag)),
              sep = "\t", row.names = FALSE, quote = FALSE)
}

emit("n_reps", NREPS)
emit("n_dump", N_DUMP)
emit("T0", T0)
emit("J", J)

cat("== SESSION INFO ==\n")
print(sessionInfo())
