# Reference run for the `cwz_ttest_mc` benchmark case.
#
# The application-based simulation study behind Table 3 of Chernozhukov,
# Wuthrich & Zhu's t-test paper, driven from the authors' own code as published
# in their JPE replication package: `code/auxiliary scripts/calibration_dgps.R`
# for the calibration and `common_functions.R::sim_one_sample` for the draw and
# the estimators.
#
# The calibration is fitted to the Andersson (2019) carbon tax panel, which is
# already vendored at basedata/carbontax_data.dta and is the panel the paper's
# empirical section uses. From it:
#
#   w0_sc                    simplex SC weights on the pre-period
#   rho_u, var_u             AR(1) fit to the SC prediction errors
#   Lambda, var_factors      a 4-factor SVD of the detrended controls
#   rho_vec, var_epsl_vec    AR(1) per control unit on the factor residuals
#
# and a draw is
#
#   Y0 = nonstat + Factors %*% t(Lambda) + epsl
#   Y1 = const + Y0 %*% w + u
#
# with (w, const, nonstat) the DGP. The nine DGPs vary whether the SC is
# correctly specified and whether the panel is stationary; the treatment effect
# is zero throughout, so what is measured is the coverage of a nominal 90%
# interval and the bias of the debiased estimator.
#
# `detrend` comes from pracma in the authors' script; it removes a per-column
# linear trend, which is done here with lm() so the reference needs one package
# fewer. `li2020` and `synthdid` supply two further columns of their Table 3 and
# are not reproduced: mlsynth implements neither, and both would pull in
# dependencies the rest of this reference does not need.
#
# Two things are emitted, answering two different questions -- the same split as
# cwz_conformal_mc:
#
#   cov_*, leng_*, bias_*  the aggregate cells, which validate the DGP port;
#   draw_*                 seed-matched panels with the ATT and standard error
#                          R computed on each, which validate the estimator.
#
# Run from the repository root:
#   Rscript benchmarks/reference/cwz_ttest_mc/reference.R
suppressMessages({library(foreign); library(scinference)})

emit <- function(k, v) cat(sprintf("%s\t%.10f\n", k, v))

SIG    <- 0.1
K      <- 3
# lsei_type = 2. The authors' sim_one_sample passes 1, and on the real carbon
# tax panel that is fine -- cwz_ttest cross-validates it there to 4e-7. On these
# simulated panels it is not: limSolve's type 1 warns "inequalities
# contradictory" on the K fold refits and returns a solution that ignores the
# non-negativity constraint. Over the ten dumped DGP 8 draws it does so on five,
# with min(w) reaching -32.4, and those are exactly the draws where the R and
# Python answers separate. Weights off the simplex are not a synthetic control,
# so what type 1 returns there is not a different answer to the same problem, it
# is an answer to a different one. Type 2 solves the stated program on every
# draw. The count is emitted below so the claim is checkable and not asserted.
LT     <- 2
NREPS  <- 300      # the paper uses 5000; the tolerance carries the gap
N_DUMP <- 10
DUMP_DIR <- "benchmarks/reference/cwz_ttest_mc"

## ------------------------------------------------------------------------
## Calibration (calibration_dgps.R)
## ------------------------------------------------------------------------

carbontax <- read.dta("basedata/carbontax_data.dta")
carbontax <- carbontax[c("Countryno", "year", "CO2_transport_capita")]
wide <- reshape(carbontax, timevar = "year", idvar = c("Countryno"),
                direction = "wide")

Y1_app <- as.matrix(cbind(wide[13, -1]))          # row 13 is Sweden
Y0_app <- t(as.matrix(wide[-13, -1]))
T0_app <- 30                                      # 1960-1989
T1_app <- length(Y1_app) - T0_app
J_app  <- dim(Y0_app)[2]

w0_sc  <- as.matrix(scinference:::sc(Y1_app[1:T0_app], Y0_app[1:T0_app, ],
                                     lsei_type = LT)$w.hat)
u_hat  <- Y1_app[1:T0_app] - Y0_app[1:T0_app, ] %*% w0_sc
ar_u   <- ar(u_hat, order.max = 1)
rho_u  <- ar_u$ar
var_u  <- ar_u$var.pred

detrend_cols <- function(M) {
  tt <- seq_len(nrow(M))
  apply(M, 2, function(x) residuals(lm(x ~ tt)))
}
Y0dt <- detrend_cols(Y0_app)

sv  <- svd(Y0dt)
num_factor <- 4
dd <- sv$d
dd[(num_factor + 1):length(sv$d)] <- 0
Y0_resid <- Y0dt - sv$u %*% diag(dd) %*% t(sv$v)

Factors <- sv$u[, 1:num_factor]
Lambda  <- (sv$v %*% diag(dd))[, 1:num_factor]
var_factors <- apply(Factors, 2, var)

rho_vec <- rep(0, J_app)
var_epsl_vec <- rep(NA, J_app)
for (j in 1:J_app) {
  a <- ar(Y0_resid[, j], order.max = 1)
  if (a$order > 0) rho_vec[j] <- a$ar
  var_epsl_vec[j] <- a$var.pred
}

## ------------------------------------------------------------------------
## Draw and estimators (common_functions.R)
## ------------------------------------------------------------------------

ar_series_var <- function(T01, rho, var.error) {
  epsl <- rnorm(T01) * sqrt(var.error)
  start <- rnorm(1) * sqrt(var.error / (1 - rho^2))
  u <- rep(NA, T01)
  u[1] <- rho * start + epsl[1]
  for (t in 2:T01) u[t] <- rho * u[t - 1] + epsl[t]
  u
}

dgp_spec <- function(DGP, T0, T1, J) {
  T01 <- T0 + T1
  w_did <- matrix(1, J, 1) / J
  w_mis <- c(-3, 3, 1, rep(0, J - 3))
  const <- 0
  nonstat <- matrix(0, T01, J)
  if (DGP == 1) { w <- w0_sc }
  if (DGP == 2) { w <- w_did }
  if (DGP == 3) { w <- w_mis; const <- 2 }
  if (DGP == 4) { w <- w_mis; const <- 2
                  nonstat <- matrix(rep(1:T01, J), T01, J) }
  if (DGP == 5) { w <- w_mis; const <- 2
                  nonstat <- matrix(rep(cumsum(rnorm(T01)), J), T01, J) }
  if (DGP == 6) { w <- w0_sc
                  nonstat <- matrix(rep(1:T01, J), T01, J)
                  nonstat[, 1] <- nonstat[, 1] + (1:T01) }
  if (DGP == 7) { w <- w0_sc
                  nonstat <- matrix(rep(cumsum(rnorm(T01)), J), T01, J)
                  nonstat[, 1] <- nonstat[, 1] + cumsum(rnorm(T01)) }
  if (DGP == 8) { w <- w_mis
                  nonstat <- matrix(NA, T01, J)
                  for (j in 1:J) nonstat[, j] <- j + j * (1:T01) }
  if (DGP == 9) { w <- w0_sc
                  nonstat <- matrix(NA, T01, J)
                  for (j in 1:J) {
                    th <- rep(NA, T01); wn <- rnorm(T01)
                    th[1] <- j + wn[1]
                    for (t in 2:T01) th[t] <- j + th[t - 1] + wn[t]
                    nonstat[, j] <- th
                  } }
  list(w = w, const = const, nonstat = nonstat)
}

draw_panel <- function(DGP, T0, T1, J) {
  T01 <- T0 + T1
  spec <- dgp_spec(DGP, T0, T1, J)
  u_sim <- ar_series_var(T01, rho_u, var_u)
  Factors_sim <- matrix(NA, T01, 4)
  for (i in 1:4) Factors_sim[, i] <- rnorm(T01) * sqrt(var_factors[i])
  epsl_sim <- matrix(NA, T01, J)
  for (i in 1:J) epsl_sim[, i] <- ar_series_var(T01, rho_vec[i], var_epsl_vec[i])
  Y0 <- spec$nonstat + Factors_sim %*% t(Lambda) + epsl_sim
  Y1 <- spec$const + Y0 %*% spec$w + u_sim
  list(Y1 = Y1, Y0 = Y0, w = spec$w)
}

# The oracle of common_functions.R::oracle.cf -- known weights, no estimation.
oracle_cf <- function(y1, yJ, T0, T1, w.true, K) {
  T01 <- T0 + T1
  r <- min(floor(T0 / K), T1)
  y1.pre <- y1[1:T0]; yJ.pre <- yJ[1:T0, ]
  y1.post <- y1[(T0 + 1):T01]; yJ.post <- yJ[(T0 + 1):T01, ]
  tau <- rep(NA, K)
  for (k in 1:K) {
    Hk <- (T0 - (r * K)) + seq((k - 1) * r + 1, k * r, 1)
    tau[k] <- mean(y1.post - yJ.post %*% w.true) -
      mean(y1.pre[Hk] - yJ.pre[Hk, ] %*% w.true)
  }
  list(att = mean(tau), se = sqrt(1 + ((K * r) / T1)) * sd(tau) / sqrt(K))
}

crit <- qt(1 - SIG / 2, df = K - 1)

cat("== REFERENCE VALUES ==\n")

## ------------------------------------------------------------------------
## Table 3: coverage, length and bias of the debiased t-test, nine DGPs
## ------------------------------------------------------------------------

for (DGP in 1:9) {
  set.seed(12345)
  cov_sc <- leng_sc <- bias_sc <- rep(NA, NREPS)
  cov_or <- rep(NA, NREPS)
  for (r in 1:NREPS) {
    pan <- draw_panel(DGP, T0_app, T1_app, J_app)
    rs <- scinference(pan$Y1, pan$Y0, T1 = T1_app, T0 = T0_app,
                      inference_method = "ttest", estimation_method = "sc",
                      K = K, lsei_type = LT)
    cov_sc[r]  <- abs(rs$att / rs$se) <= crit
    leng_sc[r] <- 2 * crit * rs$se
    bias_sc[r] <- rs$att
    ro <- oracle_cf(pan$Y1, pan$Y0, T0_app, T1_app, pan$w, K)
    cov_or[r] <- abs(ro$att / ro$se) <= crit
  }
  emit(sprintf("cov_sc_dgp%d", DGP), mean(cov_sc))
  emit(sprintf("leng_sc_dgp%d", DGP), mean(leng_sc))
  emit(sprintf("bias_sc_dgp%d", DGP), mean(bias_sc))
  emit(sprintf("cov_oracle_dgp%d", DGP), mean(cov_or))
}

## ------------------------------------------------------------------------
## Seed-matched panels, for the exact per-draw comparison
##
## DGP 1 is the correctly specified case and DGP 8 the heterogeneous-trend
## stress case the paper singles out, so the pair brackets what the estimator
## has to handle.
## ------------------------------------------------------------------------

for (DGP in c(1, 8)) {
  set.seed(20260814)
  rows <- NULL
  for (r in 1:N_DUMP) {
    pan <- draw_panel(DGP, T0_app, T1_app, J_app)
    rs <- scinference(pan$Y1, pan$Y0, T1 = T1_app, T0 = T0_app,
                      inference_method = "ttest", estimation_method = "sc",
                      K = K, lsei_type = LT)
    emit(sprintf("draw_dgp%d_%02d_att", DGP, r), rs$att)
    emit(sprintf("draw_dgp%d_%02d_se", DGP, r), rs$se)
    rows <- rbind(rows, cbind(r, 1:(T0_app + T1_app), pan$Y1, pan$Y0))
  }
  colnames(rows) <- c("draw", "t", "Y1", paste0("Y0_", 1:J_app))
  write.table(rows, file.path(DUMP_DIR, sprintf("draws_dgp%d.tsv", DGP)),
              sep = "\t", row.names = FALSE, quote = FALSE)
}

## ------------------------------------------------------------------------
## The calibration itself, so a disagreement can be localised
## ------------------------------------------------------------------------

# The fitted calibration, written out so the Python side draws from the same
# structure instead of re-deriving it. R's `ar()` selects its order by AIC and
# solves Yule-Walker; re-implementing that in Python would put a second
# estimator between the two sides for no gain, since the calibration is an input
# to the design and not the thing under test.
calib <- data.frame(donor = 1:J_app, w0_sc = as.numeric(w0_sc),
                    rho = rho_vec, var_epsl = var_epsl_vec)
write.table(calib, file.path(DUMP_DIR, "calibration_units.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
lam <- data.frame(donor = 1:J_app, Lambda)
colnames(lam) <- c("donor", paste0("lambda_", 1:num_factor))
write.table(lam, file.path(DUMP_DIR, "calibration_lambda.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(factor = 1:num_factor, var = as.numeric(var_factors)),
            file.path(DUMP_DIR, "calibration_factors.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

## ------------------------------------------------------------------------
## How often lsei type 1 leaves the simplex on the fold refits
##
## Recorded so the reason this reference uses type 2 is a measurement and not
## a claim. A weight vector is counted as off the simplex when any entry
## falls below -1e-6.
## ------------------------------------------------------------------------

r_fold <- min(floor(T0_app / K), T1_app)
for (DGP in c(1, 8)) {
  set.seed(20260814)
  n_bad <- 0
  n_fold <- 0
  for (rep in 1:N_DUMP) {
    pan <- draw_panel(DGP, T0_app, T1_app, J_app)
    Y1p <- pan$Y1[1:T0_app]; Y0p <- pan$Y0[1:T0_app, ]
    for (k in 1:K) {
      Hk <- (T0_app - (r_fold * K)) + seq((k - 1) * r_fold + 1, k * r_fold, 1)
      w <- scinference:::sc(Y1p[-Hk], Y0p[-Hk, ], 1)$w.hat
      n_fold <- n_fold + 1
      if (min(w) < -1e-6) n_bad <- n_bad + 1
    }
  }
  emit(sprintf("lsei1_offsimplex_folds_dgp%d", DGP), n_bad)
  emit(sprintf("lsei1_total_folds_dgp%d", DGP), n_fold)
}

emit("calib_rho_u", rho_u)
emit("calib_var_u", var_u)
emit("calib_T0", T0_app)
emit("calib_T1", T1_app)
emit("calib_J", J_app)
emit("n_reps", NREPS)

cat("== SESSION INFO ==\n")
print(sessionInfo())
