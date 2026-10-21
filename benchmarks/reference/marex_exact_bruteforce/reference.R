# MAREX's exact backend against brute force: every treated set, solved in R.
#
# The exact backend (mlsynth/utils/marex_helpers/exact.py) finds MAREX's design
# by a bounded search: a closed-form lower bound on the treated fit discards
# candidates, an incumbent cutoff stops the weakly-targeted fits early, and an
# active-set solver fits the survivors. A port of that search to R would share
# those ideas, and a wrong bound would discard the optimum in both languages at
# once. This script shares none of them. For every treated set S of the stated
# size it solves the fixed-S program with quadprog::solve.QP and keeps the
# minimum, so its answer is the optimum by enumeration.
#
# Each design is the fixed-S program of the corresponding MAREX objective, with
# x_j a market's predictor vector and A the mean of the x_j:
#
#   standard         min ||A - B_S w||^2 + ||A - B_C v||^2
#   penalized        standard + lambda1 D1'w + lambda2 D1'v,   D1_j = ||x_j - A||^2
#   unit_penalized   standard + xi r'w + lambda1_unit D1'w,
#                    r_j = min over the simplex of ||x_j - sum_{i != j} a_i x_i||^2
#   weakly_targeted  min ||A - B_S w||^2 + beta ||B_S w - B_C v||^2
#                    + beta lam sum_j v_j ||B_S w - x_j||^2   (lam = control_penalty)
#
# The weakly-targeted programs are solved jointly over (w, v) as one quadratic
# program: on sum v = 1 the last term is ||B_S w||^2 - 2 w'B_S'B_C v +
# sum_j v_j ||x_j||^2, so the program is a convex QP while
# beta lam (1 + lam) <= 1. The others split into two simplex fits.
#
# Predictors are the fitting-window outcomes, and for the covariate cases the
# covariates appended after them. With standardisation every column is divided
# by its population standard deviation across markets (as numpy's std does)
# and the covariate columns are then multiplied by their weights, MAREX's
# order (Vives-i-Bastida 2022, Algorithm 1).
#
# quadprog needs a positive definite matrix; a ridge of 1e-10 of the diagonal's
# scale is added, which moves the weights by far less than the case's
# tolerance. The panels are drawn here and printed in full, so the Python case
# scores the identical numbers.
#
# Run from the repository root:
#   Rscript benchmarks/reference/marex_exact_bruteforce/reference.R
suppressMessages(library(quadprog))

N.PANELS <- 6
N.UNITS <- 8
T.FIT <- 10
T.POST <- 2
RIDGE <- 1e-10

draw_panel <- function(p) {
  set.seed(1000 + p)
  T.all <- T.FIT + T.POST
  level <- rnorm(N.UNITS, sd = 2)
  slope <- rnorm(N.UNITS)
  common <- cumsum(rnorm(T.all))
  Y <- outer(rep(1, T.all), level) + outer((1:T.all) / T.all, slope) +
       common + matrix(rnorm(T.all * N.UNITS, sd = 0.5), T.all, N.UNITS)
  Z <- cbind(rnorm(N.UNITS), 40 * rnorm(N.UNITS))
  list(Y = Y, Z = Z)                 # Y: T x N (column j is market j); Z: N x 2
}

pop_sd <- function(x) sqrt(mean((x - mean(x))^2))

predictors <- function(Y, Z, cfg) {
  X <- t(Y[1:T.FIT, , drop = FALSE])                   # N x T.FIT
  n.out <- ncol(X)
  if (!is.null(cfg$cov_weight)) X <- cbind(X, Z)
  if (isTRUE(cfg$standardize)) {
    s <- apply(X, 2, pop_sd)
    flat <- s <= 1e-12 * apply(abs(X), 2, max)
    X <- sweep(X, 2, ifelse(flat, 1, s), "/")
  }
  if (!is.null(cfg$cov_weight)) {
    idx <- (n.out + 1):ncol(X)
    X[, idx] <- sweep(X[, idx, drop = FALSE], 2, cfg$cov_weight, "*")
  }
  t(X)                                                 # predictors x N
}

simplex_qp <- function(B, A, linear = NULL) {
  # min ||A - B w||^2 + linear'w  over w >= 0, sum w = 1
  k <- ncol(B)
  D <- 2 * crossprod(B)
  D <- D + RIDGE * max(1, mean(abs(diag(D)))) * diag(k)
  d <- 2 * drop(crossprod(B, A))
  if (!is.null(linear)) d <- d - linear
  sol <- solve.QP(Dmat = D, dvec = d, Amat = cbind(rep(1, k), diag(k)),
                  bvec = c(1, rep(0, k)), meq = 1)$solution
  sol <- pmax(sol, 0); sol <- sol / sum(sol)
  r <- A - B %*% sol
  list(w = sol, value = sum(r^2) + if (is.null(linear)) 0 else sum(linear * sol))
}

weakly_qp <- function(Bs, Bc, A, beta, lam) {
  Bs <- Bs - A; Bc <- Bc - A                 # every term is unchanged by the shift
  ks <- ncol(Bs); kc <- ncol(Bc)
  Gss <- crossprod(Bs); Gsc <- crossprod(Bs, Bc); Gcc <- crossprod(Bc)
  sq <- colSums(Bc^2)
  H <- rbind(cbind((1 + beta + beta * lam) * Gss, -beta * (1 + lam) * Gsc),
             cbind(-beta * (1 + lam) * t(Gsc), beta * Gcc))
  D <- 2 * H
  D <- D + RIDGE * max(1, mean(abs(diag(D)))) * diag(ks + kc)
  d <- c(rep(0, ks), -beta * lam * sq)
  Amat <- cbind(c(rep(1, ks), rep(0, kc)), c(rep(0, ks), rep(1, kc)), diag(ks + kc))
  sol <- solve.QP(Dmat = D, dvec = d, Amat = Amat,
                  bvec = c(1, 1, rep(0, ks + kc)), meq = 2)$solution
  w <- pmax(sol[1:ks], 0); w <- w / sum(w)
  v <- pmax(sol[(ks + 1):(ks + kc)], 0); v <- v / sum(v)
  t.syn <- drop(Bs %*% w); c.syn <- drop(Bc %*% v)
  pen <- sum(v * colSums((Bc - t.syn)^2))
  list(w = w, v = v,
       value = sum(t.syn^2) + beta * sum((t.syn - c.syn)^2) + beta * lam * pen)
}

reproducibility <- function(X) {
  sapply(seq_len(ncol(X)), function(j) simplex_qp(X[, -j, drop = FALSE], X[, j])$value)
}

design <- function(X, cfg) {
  n <- ncol(X); A <- rowMeans(X)
  D1 <- colSums((X - A)^2)
  a.lin <- NULL; b.lin <- NULL
  if (cfg$design == "penalized") { a.lin <- cfg$lambda1 * D1; b.lin <- cfg$lambda2 * D1 }
  if (cfg$design == "unit_penalized")
    a.lin <- cfg$xi * reproducibility(X) + cfg$lambda1_unit * D1
  best <- list(value = Inf)
  for (S in combn(n, cfg$m, simplify = FALSE)) {
    C <- setdiff(seq_len(n), S)
    if (cfg$design == "weakly_targeted") {
      f <- weakly_qp(X[, S, drop = FALSE], X[, C, drop = FALSE], A, cfg$beta, cfg$lam)
      val <- f$value; w <- f$w; v <- f$v
    } else {
      g <- simplex_qp(X[, S, drop = FALSE], A, if (is.null(a.lin)) NULL else a.lin[S])
      h <- simplex_qp(X[, C, drop = FALSE], A, if (is.null(b.lin)) NULL else b.lin[C])
      val <- g$value + h$value; w <- g$w; v <- h$w
    }
    if (val < best$value) best <- list(value = val, S = S, C = C, w = w, v = v)
  }
  best
}

CONFIGS <- list(
  list(name = "standard", design = "standard", m = 2),
  list(name = "standard_m3", design = "standard", m = 3),
  list(name = "penalized", design = "penalized", m = 2, lambda1 = 0.05, lambda2 = 0.02),
  list(name = "unit_penalized", design = "unit_penalized", m = 2, xi = 0.5,
       lambda1_unit = 0.05),
  list(name = "weakly_b001", design = "weakly_targeted", m = 2, beta = 0.01, lam = 0),
  list(name = "weakly_b1", design = "weakly_targeted", m = 2, beta = 1, lam = 0),
  list(name = "weakly_b100", design = "weakly_targeted", m = 2, beta = 100, lam = 0),
  list(name = "penalty_b1", design = "weakly_targeted", m = 2, beta = 1, lam = 0.1),
  list(name = "penalty_b1_near_bound", design = "weakly_targeted", m = 2, beta = 1,
       lam = 0.6),
  list(name = "covariates_std", design = "standard", m = 2, standardize = TRUE,
       cov_weight = c(0.1, 0.1)),
  list(name = "covariates_std_h", design = "standard", m = 2, standardize = TRUE,
       cov_weight = c(0.3, 2.0)),
  list(name = "covariates_weakly", design = "weakly_targeted", m = 2, beta = 1,
       lam = 0.1, standardize = TRUE, cov_weight = c(0.5, 0.1))
)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("n_panels\t%d\n", N.PANELS))
cat(sprintf("n_units\t%d\n", N.UNITS))
cat(sprintf("t_fit\t%d\n", T.FIT))
cat(sprintf("t_post\t%d\n", T.POST))
cat(sprintf("n_configs\t%d\n", length(CONFIGS)))
for (p in seq_len(N.PANELS)) {
  pn <- draw_panel(p)
  for (j in seq_len(N.UNITS)) {
    for (t in seq_len(T.FIT + T.POST))
      cat(sprintf("weight\ty_p%d_u%d_t%d\t%.15g\n", p, j, t, pn$Y[t, j]))
    for (k in 1:2) cat(sprintf("weight\tz_p%d_u%d_k%d\t%.15g\n", p, j, k, pn$Z[j, k]))
  }
  for (cfg in CONFIGS) {
    b <- design(predictors(pn$Y, pn$Z, cfg), cfg)
    cat(sprintf("obj_%s_p%d\t%.15g\n", cfg$name, p, b$value))
    for (k in seq_along(b$S))
      cat(sprintf("weight\tw_%s_p%d_u%d\t%.15g\n", cfg$name, p, b$S[k], b$w[k]))
    for (k in seq_along(b$C))
      cat(sprintf("weight\tv_%s_p%d_u%d\t%.15g\n", cfg$name, p, b$C[k], b$v[k]))
  }
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
