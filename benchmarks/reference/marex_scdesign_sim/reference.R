# SCDesign's own cardinality-constrained design, run on the simulation panels of
# Abadie & Zhao's Section 5, as the reference for MAREX.
#
# Two pieces of the authors' code, both reproduced verbatim from
# github.com/jinglongzhao2/SCDesign:
#
#   * the Section 5 generation block (SCdesign_LazyRun.R lines 19-176), which
#     draws the panels; and
#   * `Synthetic_Control` + `Synthetic_Experiment_Cardinality_Constraint`
#     ("3. Walmart Data Simulations/Walmart_LazyRun.R"), the constrained design.
#
# The authors' unconstrained routine is a non-convex Gurobi MIQP and is not used.
# The cardinality routine enumerates every partition of size 1..K, solves the
# treated and control synthetic-control weights for each by `quadprog::solve.QP`,
# and keeps the min-loss partition -- so it needs no commercial solver, and it is
# the design MAREX solves with m_min = 1, m_max = K.
#
# Run from the repository root:
#   Rscript benchmarks/reference/marex_scdesign_sim/reference.R
suppressMessages({library(Matrix); library(quadprog)})

M.REPS <- 6          # panels; each enumerates sum_{p<=K} C(15,p) partitions
K.CARD <- 2
set.seed(123456)

# --- Section 5 DGP (SCdesign_LazyRun.R lines 19-176) --------------------------
N.Regions <- 15; T.naught <- 25; T.prime <- 20; T.total <- 30
r.ob <- 7; F.unob <- 11
range.intercept.max <- 20; range.covariates.max <- 1; range.coefficients.max <- 10
noise.variance <- 1

draw_panel <- function() {
  delta <- sort(c(range.intercept.max * runif(T.naught),
                  range.intercept.max * runif(T.total - T.naught)))
  upsilon <- c(rep(NA, T.naught), sort(range.intercept.max * runif(T.total - T.naught)))
  Z <- matrix(NA, r.ob, N.Regions)
  for (j in 1:N.Regions) Z[, j] <- range.covariates.max * runif(r.ob)
  mu <- matrix(NA, F.unob, N.Regions)
  for (j in 1:N.Regions) mu[, j] <- range.covariates.max * runif(F.unob)
  theta <- matrix(NA, r.ob, T.total)
  for (t in 1:T.naught) theta[, t] <- range.coefficients.max * runif(r.ob)
  for (t in (T.naught+1):T.total) theta[, t] <- range.coefficients.max * runif(r.ob)
  gamma <- matrix(NA, r.ob, T.total)
  for (t in 1:T.naught) gamma[, t] <- rep(NA, r.ob)
  for (t in (T.naught+1):T.total) gamma[, t] <- range.coefficients.max * runif(r.ob)
  lambda <- matrix(NA, F.unob, T.total)
  for (t in 1:T.naught) lambda[, t] <- range.coefficients.max * runif(F.unob)
  for (t in (T.naught+1):T.total) lambda[, t] <- range.coefficients.max * runif(F.unob)
  eta <- matrix(NA, F.unob, T.total)
  for (t in 1:T.naught) eta[, t] <- rep(NA, F.unob)
  for (t in (T.naught+1):T.total) eta[, t] <- range.coefficients.max * runif(F.unob)
  eps <- matrix(NA, N.Regions, T.total)
  for (t in 1:T.naught) eps[, t] <- rnorm(N.Regions, 0, noise.variance)
  for (t in (T.naught+1):T.total) eps[, t] <- rnorm(N.Regions, 0, noise.variance)
  xi <- matrix(NA, N.Regions, T.total)
  for (t in 1:T.naught) xi[, t] <- rep(NA, N.Regions)
  for (t in (T.naught+1):T.total) xi[, t] <- rnorm(N.Regions, 0, noise.variance)
  Y.N <- matrix(NA, N.Regions, T.total); Y.I <- matrix(NA, N.Regions, T.total)
  for (j in 1:N.Regions) for (t in 1:T.total) {
    Y.N[j,t] <- delta[t] + theta[,t] %*% Z[,j] + lambda[,t] %*% mu[,j] + eps[j,t]
    Y.I[j,t] <- upsilon[t] + gamma[,t] %*% Z[,j] + eta[,t] %*% mu[,j] + xi[j,t]
  }
  list(Y.N = Y.N, Y.I = Y.I, Z = Z)
}

# --- SCDesign Synthetic_Control (verbatim quadprog path) ----------------------
Synthetic_Control <- function(target.vector_, X.matrix_) {
  X.effective = X.matrix_ - matrix(data = target.vector_, nrow = nrow(X.matrix_),
                                   ncol = ncol(X.matrix_), byrow = FALSE)
  Dmat = 2 * (t(X.effective) %*% X.effective)
  pd_Dmat = as.matrix(nearPD(Dmat)$mat)
  Dmat.SizeReduced = pd_Dmat / mean(pd_Dmat)
  result = solve.QP(Dmat = Dmat.SizeReduced, dvec = rep(0, ncol(X.effective)),
                    Amat = t(rbind(matrix(1, nrow = 1, ncol = ncol(X.effective)),
                                   diag(1, ncol(X.effective)))),
                    bvec = c(1, rep(0, ncol(X.effective))), meq = 1)
  list(Weights = result$solution)
}

# --- SCDesign constrained (cardinality-K) design (verbatim) -------------------
Synthetic_Experiment_Cardinality_Constraint <- function(T.prime_, r.ob_, N.Regions_,
    Y.N.matrix_, Z.matrix_, f.vector_, K.cardinality_) {
  M.dim = T.prime_ + r.ob_; N.dim = N.Regions_
  X.matrix = matrix(NA, nrow = M.dim, ncol = N.dim)
  for (i in 1:M.dim) for (j in 1:N.dim) {
    if (i <= T.prime_) X.matrix[i, j] = Y.N.matrix_[j, i]
    else X.matrix[i, j] = Z.matrix_[(i - T.prime_), j]
  }
  row.means = apply(X.matrix, 1, mean); row.stdevs = apply(X.matrix, 1, sd)
  X.matrix = (X.matrix - matrix(row.means, nrow = M.dim, ncol = N.dim, byrow = FALSE)) / row.stdevs
  center.vector = X.matrix %*% f.vector_
  candidate.partition = list()
  for (pc in 1:K.cardinality_)
    candidate.partition = c(candidate.partition, combn(1:N.Regions_, pc, simplify = FALSE))
  loss = c()
  for (tc in seq_along(candidate.partition)) {
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
  list(selected = sort(cand), tw = round(tw, 6), loss = min(loss))
}

f.vector <- rep(1 / N.Regions, N.Regions)
cat("== REFERENCE VALUES ==\n")
cat(sprintf("m_reps\t%.6f\n", M.REPS))
cat(sprintf("k_cardinality\t%.6f\n", K.CARD))
for (rep in 1:M.REPS) {
  p <- draw_panel()
  r <- Synthetic_Experiment_Cardinality_Constraint(T.prime, r.ob, N.Regions,
         p$Y.N, p$Z, f.vector, K.CARD)
  # the panel itself, so the Python side scores the identical draw
  for (j in 1:N.Regions) for (t in 1:T.total)
    cat(sprintf("weight\ty_r%d_u%d_t%d\t%.10f\n", rep, j, t, p$Y.N[j, t]))
  for (k in 1:r.ob) for (j in 1:N.Regions)
    cat(sprintf("weight\tz_r%d_k%d_u%d\t%.10f\n", rep, k, j, p$Z[k, j]))
  cat(sprintf("n_treated_rep%d\t%.6f\n", rep, length(r$selected)))
  cat(sprintf("loss_rep%d\t%.10f\n", rep, r$loss))
  for (j in r$selected) cat(sprintf("weight\tsel_r%d_u%d\t%.6f\n", rep, j, r$tw[j]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
