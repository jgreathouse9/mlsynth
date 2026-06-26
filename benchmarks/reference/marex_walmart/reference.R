# Reference run for the `marex_walmart` benchmark case.
#
# Runs Abadie & Zhao's own synthetic-control experimental-design routine
# (jinglongzhao2/SCDesign -- the Walmart application, Section 4 / Figures 2-3) on
# the SAME full 45-store weekly Walmart panel, with the SAME four store-level
# covariates, that mlsynth's MAREX is fitted to, and prints the design quantities
# the Python case pins against. These are genuine SCDesign outputs, not
# transcribed constants.
#
# What is reproduced verbatim from SCDesign's `Walmart_LazyRun.R`:
#   * `Synthetic_Control`            -- the constrained-least-squares SC weight QP
#                                       (the quadprog::solve.QP path; their Gurobi
#                                       path is licence-gated and commented out in
#                                       the source -- this open path is the one
#                                       they ship runnable).
#   * `Synthetic_Experiment_Cardinality_Constraint` -- the paper's *constrained*
#                                       (cardinality-K) design: enumerate every
#                                       partition of size <= K, solve the treated
#                                       and control SC weights for each, and pick
#                                       the partition minimising the combined fit
#                                       loss. With K = 2 it selects exactly two
#                                       treated stores -- MAREX's `m_eq = 2`.
#   * `permutation.test` / `quantile_blank` -- their placebo permutation p-value
#                                       and conformal interval half-width.
# Gurobi's non-convex `Synthetic_Experiment` (the unconstrained MIQP) is NOT used:
# it needs a commercial licence, and the constrained routine above is the exact
# design MAREX's `m_eq` solves -- on the open quadprog backend, no Gurobi. SCDesign
# ships as loose R source (not a package); the fetch is documented in
# benchmarks/R/install_scdesign.sh.
#
# Time structure matches the paper and the case exactly: weeks 1..143;
# experimental period weeks 129..143 (T.naught = 128 pre-periods, 15 post -- the
# placebo intervention at week 129); fit weeks 1..100 (T.prime = 100), blank weeks
# 101..128 (28 -- MAREX's blank_periods = 28). Four store-level covariates
# (Temperature, Fuel_Price, CPI, Unemployment), each aggregated to its per-store
# pre-period (fit-window) mean exactly as MAREX does, are stacked onto the
# pre-period outcomes as predictors. Uniform population weights; per-predictor
# standardisation (the paper's Walmart normalisation -- MAREX's standardize=TRUE;
# centering cancels on the simplex and the n/(n-1) factor is uniform, so it does
# not affect the selected design).
#
# Run from the repository root:  Rscript benchmarks/reference/marex_walmart/reference.R
suppressMessages({library(Matrix); library(quadprog)})

# --- SCDesign Synthetic_Control: constrained least squares via quadprog -------
# (verbatim quadprog path from SCDesign's Walmart_LazyRun.R)
Synthetic_Control <- function(target.vector_, X.matrix_) {
  X.effective = X.matrix_ - matrix(data = target.vector_, nrow = nrow(X.matrix_),
                                   ncol = ncol(X.matrix_), byrow = FALSE)
  Dmat = 2 * (t(X.effective) %*% X.effective)
  pd_Dmat = as.matrix(nearPD(Dmat)$mat)
  avg.Dmat = mean(pd_Dmat)
  Dmat.SizeReduced = pd_Dmat / avg.Dmat
  dvec = rep(0, ncol(X.effective))
  A.eq = matrix(1, nrow = 1, ncol = ncol(X.effective))
  b.eq = c(1)
  A.nonneg = diag(1, ncol(X.effective))
  b.nonneg = rep(0, ncol(X.effective))
  result = solve.QP(Dmat = Dmat.SizeReduced, dvec = dvec,
                    Amat = t(rbind(A.eq, A.nonneg)),
                    bvec = c(b.eq, b.nonneg), meq = nrow(A.eq))
  list(Weights = result$solution,
       Objval = (0.5 * t(result$solution) %*% Dmat %*% result$solution))
}

# --- SCDesign constrained (cardinality-K) experiment design -------------------
# (verbatim from SCDesign's Walmart_LazyRun.R)
Synthetic_Experiment_Cardinality_Constraint <- function(T.prime_, r.ob.covariates.dim_,
    N.Regions_, Y.N.matrix_, Z.ob.covariates.matrix_, f.vector_, K.cardinality_ = -1) {
  K.cardinality = if (K.cardinality_ == -1) floor(N.Regions_ / 2) else K.cardinality_
  f.vec = f.vector_
  M.dim = T.prime_ + r.ob.covariates.dim_
  N.dim = N.Regions_
  X.matrix = matrix(NA, nrow = M.dim, ncol = N.dim)
  for (i in 1:M.dim) for (j in 1:N.dim) {
    if (i <= T.prime_) X.matrix[i, j] = Y.N.matrix_[j, i]
    else X.matrix[i, j] = Z.ob.covariates.matrix_[(i - T.prime_), j]
  }
  # rescale each row (the paper's Walmart normalisation -- MAREX standardize=TRUE)
  row.means = apply(X.matrix, 1, mean); row.stdevs = apply(X.matrix, 1, sd)
  X.matrix = (X.matrix - matrix(row.means, nrow = M.dim, ncol = N.dim, byrow = FALSE)) / row.stdevs
  center.vector = X.matrix %*% f.vec
  candidate.partition = list()
  for (pc in 1:K.cardinality)
    candidate.partition = c(candidate.partition, combn(1:N.Regions_, pc, simplify = FALSE))
  loss = c()
  for (tc in 1:length(candidate.partition)) {
    cand = candidate.partition[[tc]]
    X.t = if (length(cand) == 1) matrix(X.matrix[, cand], ncol = 1) else X.matrix[, cand]
    X.c = X.matrix[, -cand]
    St = Synthetic_Control(center.vector, X.t); Sc = Synthetic_Control(center.vector, X.c)
    lt = (X.t %*% St$Weights - center.vector)^2; lc = (X.c %*% Sc$Weights - center.vector)^2
    loss = c(loss, mean(lt) + mean(lc))
  }
  fm = which.min(loss); cand = candidate.partition[[fm]]
  X.t = if (length(cand) == 1) matrix(X.matrix[, cand], ncol = 1) else X.matrix[, cand]
  X.c = X.matrix[, -cand]
  St = Synthetic_Control(center.vector, X.t); Sc = Synthetic_Control(center.vector, X.c)
  tw = rep(0, N.Regions_); for (i in seq_along(cand)) tw[cand[i]] = St$Weights[i]
  comp = setdiff(1:N.Regions_, cand)
  cw = rep(0, N.Regions_); for (i in seq_along(comp)) cw[comp[i]] = Sc$Weights[i]
  list(Treatment.weights = round(tw, 4), Control.weights = round(cw, 4), selected = cand)
}

# SCDesign placebo permutation test (verbatim, type 0 = all permutations)
permutation.test <- function(pre, post, permutation.SAMPLES_ = 100000, seed_ = 123456) {
  set.seed(seed_)
  v = c(abs(pre), abs(post)); ts = sum(tail(v, length(post)))
  if (choose(length(v), length(post)) <= permutation.SAMPLES_) {
    cm = combn(1:length(v), length(post))
    ps = apply(cm, 2, function(idx) sum(v[idx]))
    sum(ps >= ts) / ncol(cm)
  } else {
    ps = replicate(permutation.SAMPLES_, sum(v[sample(length(v), length(post))]))
    sum(ps >= ts) / permutation.SAMPLES_
  }
}
# SCDesign conformal-interval half-width (verbatim)
quantile_blank <- function(r, phi_ = 0.95) sort(abs(r))[ceiling(length(r) * phi_)]

# --- Data: the case's walmart_weekly_sales_covariates.csv, all 45 stores -------
df <- read.csv("basedata/walmart_weekly_sales_covariates.csv")
covcols <- c("Temperature", "Fuel_Price", "CPI", "Unemployment")
stores <- sort(unique(df$store)); weeks <- sort(unique(df$week))
Y <- matrix(NA, nrow = length(stores), ncol = length(weeks))
for (i in seq_along(stores)) {
  s <- df[df$store == stores[i], ]
  Y[i, ] <- s$sales[order(s$week)]
}
N.Regions <- nrow(Y); T.total <- ncol(Y)
mean_sales <- mean(df$sales)

f.vector <- rep(1 / N.Regions, N.Regions)   # uniform population weights
T.naught <- 128                              # experimental period = weeks 129..143 (15)
T.prime  <- 100                              # fit 1..100; blank 101..128 (MAREX blank=28)

# Covariates: per-store mean over the fitting window (weeks 1..T.prime), exactly
# as MAREX aggregates them (pre_mean over pre_periods = T0). Z is r.cov x N.
r.cov <- length(covcols)
Z <- matrix(NA, nrow = r.cov, ncol = N.Regions)
for (j in seq_along(stores)) {
  sj <- df[df$store == stores[j] & df$week <= T.prime, ]
  for (k in seq_along(covcols)) Z[k, j] <- mean(sj[[covcols[k]]])
}

res <- Synthetic_Experiment_Cardinality_Constraint(
  T.prime, r.cov, N.Regions, Y, Z, f.vector, K.cardinality_ = 2)
tw <- res$Treatment.weights; cw <- res$Control.weights

# Placebo design: no real effect, so Y_I == Y_N.
T.fit.N <- as.numeric(t(tw) %*% Y[, 1:T.naught])
C.fit.N <- as.numeric(t(cw) %*% Y[, 1:T.naught])
T.after <- as.numeric(t(tw) %*% Y[, (T.naught + 1):T.total])
C.after <- as.numeric(t(cw) %*% Y[, (T.naught + 1):T.total])
blank.resid <- T.fit.N[(T.prime + 1):T.naught] - C.fit.N[(T.prime + 1):T.naught]
est.ATE <- T.after - C.after

prefit.resid   <- T.fit.N - C.fit.N
prefit_rmse_pct <- sqrt(mean(prefit.resid^2)) / mean_sales
abs_ate_pct     <- abs(mean(est.ATE)) / mean_sales
placebo_p_value <- permutation.test(blank.resid, est.ATE)
qhat   <- quantile_blank(blank.resid, 0.95)
ci.lo  <- mean(est.ATE) - qhat; ci.hi <- mean(est.ATE) + qhat
ci_covers_zero <- as.numeric(ci.lo <= 0 & 0 <= ci.hi)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("n_treated\t%.6f\n", sum(tw != 0)))
cat(sprintf("prefit_rmse_pct\t%.6f\n", prefit_rmse_pct))
cat(sprintf("abs_ate_pct\t%.6f\n", abs_ate_pct))
cat(sprintf("placebo_p_value\t%.6f\n", placebo_p_value))
cat(sprintf("ci_covers_zero\t%.6f\n", ci_covers_zero))
for (i in which(tw != 0)) cat(sprintf("weight\tstore_%d\t%.6f\n", stores[i], tw[i]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
