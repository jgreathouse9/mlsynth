# Reference run for the `scd_cps` benchmark case.
#
# Reproduces the Synthetic Control with Differencing (SCD) estimator of Rincon &
# Song (2026), "Synthetic Control with Differencing" (arXiv:2510.26106), together
# with the repeated-cross-section (RC) variance and confidence-set inference of
# Canen & Song (2025), on the authors' Arizona LAWA CPS application.
#
# The method is reproduced from scratch (the upstream package ratzanyelrincon/scd
# is GPL; it is reproduced on public-domain CPS microdata rather than vendored,
# as with the sbc_germany / lto_refined_placebo / proximal_germany_oid cases).
#
# Pipeline (data_type = repeated cross-sections):
#   * survey-weighted group means  m_{j,t} = sum_i pi_i Y_i 1{G_i=j,t} / sum pi_i 1{...}
#   * within-group differencing set by lambda:
#       DID     lambda = (0,...,0,1)   -> difference off the last pre-period
#       Uniform lambda = (1/T0,...)    -> difference off the pre-period average
#       SC      lambda = 0             -> no differencing (classic SC on levels)
#   * simplex fit  w = argmin_{w in Delta} || g_pre - G_pre w ||^2  (SLSQP)
#   * effect path  theta_t = g_t - G_t' w
#   * RC pointwise variance  sigma^2_t  (individual influence functions, sqrt-n)
#   * confidence-set membership in_C(w) via the projection QP + chi^2 dof
#
# NOTE ON THE RC VARIANCE: the upstream gen_hat_sigma_squared_RC builds its
# treated/donor multiplier as `ifelse(G_idx==1, 1, -w[G_idx-1])`, which indexes w
# at 0 for treated rows; R drops zero indices, shortening the vector so ifelse
# recycles it and misaligns the donor weights. This reference uses the corrected
# length-(K+1) lookup `c(1,-w)[G_idx]` (matching the sibling gen_hat_V_RC), which
# mlsynth implements. The bug only affects the RC pointwise SE, not the point
# estimator, V_RC, or the confidence set.
#
# Run from the repository root:
#   Rscript benchmarks/reference/scd_cps/reference.R
suppressMessages({library(nloptr); library(osqp); library(Matrix); library(nanoparquet)})

d <- as.data.frame(nanoparquet::read_parquet("basedata/cps_lawa_arizona.parquet"))
treated <- "Arizona"
Tstar <- min(d$period[d$state_name == treated & d$D == 1]); T0 <- Tstar - 1
Ttot <- max(d$period); T1 <- Ttot - T0; L <- Ttot - Tstar
states <- unique(d$state_name); donors <- setdiff(states, treated); K <- length(donors)
groups <- c(treated, donors); d$G <- match(d$state_name, groups) - 1L
d$t <- d$period; d$Y <- d$wklyearn; n <- nrow(d)

# survey-weighted group means m_{j,t} and weighted cell totals n_jt  ((K+1) x Ttot)
gm <- matrix(0, K + 1, Ttot); nj <- matrix(0, K + 1, Ttot)
for (tt in 1:Ttot) for (g in 0:K) {
  s <- d$G == g & d$t == tt; de <- sum(d$weight[s])
  nj[g + 1, tt] <- de; gm[g + 1, tt] <- sum(d$weight[s] * d$Y[s]) / de
}
M <- t(gm)

lambda_of <- function(scheme) {
  if (scheme == "did") { v <- rep(0, T0); v[T0] <- 1; v }
  else if (scheme == "uniform") rep(1 / T0, T0)
  else rep(0, T0)                                   # sc: no differencing
}
fit_w <- function(lam) {
  pre <- M[1:T0, , drop = FALSE]; base <- as.vector(lam %*% pre)
  hg <- pre[, 1] - base[1]; hG <- pre[, -1] - matrix(base[-1], T0, K, byrow = TRUE)
  obj <- function(w) { dv <- hg - hG %*% w; list(objective = crossprod(dv), gradient = -2 * crossprod(hG, dv)) }
  eqc <- function(w) list(constraints = sum(w) - 1, jacobian = rep(1, length(w)))
  w <- nloptr(x0 = rep(1 / K, K), eval_f = obj, eval_g_eq = eqc, lb = rep(0, K), ub = rep(1, K),
              opts = list(algorithm = "NLOPT_LD_SLSQP", xtol_rel = 1e-10, ftol_abs = 1e-14, maxeval = 1e5))$solution
  list(w = w, base = base, hG = hG, hg = hg)
}

emit <- function(k, v) cat(sprintf("%s\t%.8f\n", k, v))
cat("== REFERENCE VALUES ==\n")

# ---- point estimator + effect path, all three schemes ----
for (scheme in c("did", "uniform", "sc")) {
  lam <- lambda_of(scheme); f <- fit_w(lam)
  gfull <- M[, 1] - f$base[1]; Gfull <- M[, -1] - matrix(f$base[-1], Ttot, K, byrow = TRUE)
  theta <- as.numeric(gfull - Gfull %*% f$w)
  emit(paste0(scheme, "_att"), mean(theta[Tstar:Ttot]))
  emit(paste0(scheme, "_theta_post1"), theta[Tstar])
  emit(paste0(scheme, "_wsum"), sum(f$w))
  emit(paste0(scheme, "_wmax"), max(f$w))
}
# DID weights by state (the paper's default scheme)
lam <- lambda_of("did"); f <- fit_w(lam); hat_w <- f$w
ord <- order(-hat_w)
for (i in head(ord, 10)) emit(paste0("did_w[", donors[i], "]"), hat_w[i])

# ---- corrected RC pointwise variance (DID) ----
lam_did <- lambda_of("did"); delta <- c(lam_did, rep(0, Ttot - T0))
gi <- d$G + 1L; li <- gi + (d$t - 1L) * (K + 1L)
psi <- d$weight * (n / nj[li]) * (d$Y - gm[li]); wtm <- c(1, -hat_w)[gi]
base2 <- (psi * wtm)^2
sv <- numeric(Ttot); ag <- tapply(base2, d$t, sum); sv[as.integer(names(ag))] <- ag
tc <- sum(base2 * delta[d$t]^2); s2 <- (1 / n) * (tc + sv * (1 - 2 * delta))
se <- sqrt(pmax(s2, 0) / n)
emit("sigma2_post1", s2[Tstar]); emit("se_post1", se[Tstar]); emit("se_post2", se[Tstar + 1])

# ---- confidence-set machinery (DID) ----
hG <- f$hG; hg <- f$hg
hat_H <- (1 / T0) * crossprod(hG); hat_h <- (1 / T0) * crossprod(hG, hg)
Mm <- diag(K) - matrix(1, K, K) / K; e <- eigen(Mm, symmetric = TRUE)
B2 <- e$vectors[, -which(abs(e$values) < 1e-10), drop = FALSE]
bar_mu <- colMeans(hG); wv <- c(1, -hat_w); Vsum <- matrix(0, K - 1, K - 1)
for (pt in 1:T0) {
  it <- which(d$t == pt); gI <- d$G[it] + 1L; l <- gI + (pt - 1L) * (K + 1L)
  ps <- d$weight[it] * (n / nj[l]) * (d$Y[it] - gm[l]); pw <- ps * wv[gI]; vit <- sum(pw^2) / n
  mu <- hG[pt, ]; Mt <- tcrossprod(mu) / T0 - lam_did[pt] * (tcrossprod(bar_mu, mu) + tcrossprod(mu, bar_mu)) +
    T0 * lam_did[pt]^2 * tcrossprod(bar_mu)
  Vsum <- Vsum + vit * crossprod(B2, Mt) %*% B2
}
hatV <- Vsum / T0; precomp <- B2 %*% solve(hatV) %*% t(B2)
emit("hatV_trace", sum(diag(hatV)))

# in_C membership (reproduced from scratch): projection QP + chi^2 dof.
#   min_r (phi - r)' P (phi - r)  s.t. w'r = 0, r >= 0,  phi = hat_H w - hat_h
# reject when  n (phi-r)'P(phi-r) > qchisq(1-kappa, K-1-#binding).
kappa <- 0.05; tol <- 1e-6
in_C <- function(w) {
  phi <- as.numeric(hat_H %*% w - hat_h)
  if (all(w > 1e-8)) { r <- rep(0, K) } else {
    P <- 2 * precomp; q <- -2 * as.numeric(precomp %*% phi)
    A <- rbind(w, diag(K)); lo <- c(0, rep(0, K)); up <- c(0, rep(Inf, K))
    st <- osqpSettings(verbose = FALSE, eps_abs = 1e-8, eps_rel = 1e-8, max_iter = 10000, polish = TRUE)
    r <- pmax(osqp(P = Matrix(P, sparse = TRUE), q = q, A = Matrix(A, sparse = TRUE), l = lo, u = up, pars = st)$Solve()$x, 0)
  }
  dv <- phi - r; Ts <- as.numeric(n * (t(dv) %*% precomp %*% dv))
  gv <- precomp %*% dv; hd <- sum(abs(gv) < tol & w == 0); hk <- max(K - 1 - hd, 1)
  as.numeric(Ts <= qchisq(1 - kappa, df = hk))
}
u <- rep(1 / K, K)
v1 <- rep(0, K); v1[1] <- 1                       # a vertex (boundary)
ed <- rep(0, K); ed[1] <- 0.5; ed[5] <- 0.5       # an edge midpoint (boundary)
emit("inC_hatw", in_C(hat_w))                     # optimum -> in set
emit("inC_uniform", in_C(u))                      # interior
emit("inC_vertex1", in_C(v1))                     # boundary -> excluded
emit("inC_edge1_5", in_C(ed))                     # boundary -> excluded

cat("== SESSION INFO ==\n")
print(sessionInfo())
