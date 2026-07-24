#!/usr/bin/env Rscript
# Reference implementation of RRSC (He, Li, Shi & Miao 2026, "A robust
# regression approach to synthetic control with interference", arXiv:2411.01249),
# distilled from the authors' released reproducibility package for cross-
# validating mlsynth's port. Computes the per-unit average direct/interference
# effects for one panel and regime.
#
#   Rscript rrsc_reference.R <Ycsv> <Xcsv> <regime> <r> <update_dif> <out_csv>
#
# <Ycsv>  : N x T outcome matrix (no header), treated unit in row 1.
# <Xcsv>  : length-T 0/1 intervention indicator (single column, no header).
# <regime>: "large_n" or "fixed_n".
# Fixed-N uses stats::factanal + robustbase::ltsReg (the real reference
# routines); large-N uses the authors' EM factor analysis + Huber-IRLS.

suppressMessages({ library(robustbase); library(expm) })

## --- large-N: Bai-Li (2012) EM factor analysis (authors' fa.em) -------------
fa.pc <- function(Y, r) {                      # Y: n x p
  n <- nrow(Y); p <- ncol(Y)
  S <- (Y %*% t(Y)) / n
  e <- eigen(S, symmetric = TRUE)
  U <- e$vectors[, 1:r, drop = FALSE]
  Z <- sqrt(n) * U
  Gamma <- t(Y) %*% U / sqrt(n)
  Sigma <- apply(Y - Z %*% t(Gamma), 2, function(x) mean(x^2))
  list(Gamma = Gamma, Sigma = Sigma)
}
fa.em <- function(Y, r, tol = 1e-6, maxiter = 1000) {
  n <- nrow(Y); p <- ncol(Y)
  init <- fa.pc(Y, r); Gamma <- init$Gamma; invSigma <- 1 / init$Sigma
  sv <- colMeans(Y^2)
  tG <- sqrt(invSigma) * Gamma
  eM <- eigen(diag(r) + t(tG) %*% tG, symmetric = TRUE)
  YSG <- Y %*% (invSigma * Gamma); llh <- -Inf
  for (it in 1:maxiter) {
    varZ <- eM$vectors %*% ((1 / pmax(eM$values, .Machine$double.eps)) * t(eM$vectors))
    EZ <- YSG %*% varZ; EZZ <- n * varZ + t(EZ) %*% EZ
    eE <- eigen(EZZ, symmetric = TRUE); YEZ <- t(Y) %*% EZ
    G <- sqrt(pmax(eE$values, .Machine$double.eps)) * t(eE$vectors)
    GG <- Gamma %*% t(G)
    denom <- sv - 2 / n * rowSums(YEZ * Gamma) + 1 / n * rowSums(GG^2)
    invSigma <- 1 / pmax(denom, .Machine$double.eps)
    Gamma <- YEZ %*% (eE$vectors * (1 / pmax(eE$values, .Machine$double.eps))) %*% t(eE$vectors)
    tG <- pmin(sqrt(invSigma), 1 / sqrt(.Machine$double.eps)) * Gamma
    M <- t(tG) %*% tG + diag(r); eM <- eigen((M + t(M)) / 2, symmetric = TRUE)
    YSG <- Y %*% (invSigma * Gamma)
    old <- llh
    logdetY <- -sum(log(invSigma)) + sum(log(pmax(eM$values, .Machine$double.eps)))
    B <- (1 / sqrt(pmax(eM$values, .Machine$double.eps))) * (t(eM$vectors) %*% t(YSG))
    llh <- -logdetY - (sum(invSigma * sv) - sum(B^2) / n)
    if (abs(llh - old) < tol * abs(llh)) break
  }
  list(Gamma = Gamma, sigma = sqrt(1 / invSigma))
}
robust.f <- function(Yt, Lambda, sigma, delta = 1.345) {
  w <- rep(1, length(Yt))
  for (i in 1:50) {
    W <- diag(w / sigma^2); A <- t(Lambda) %*% W %*% Lambda
    e <- eigen(A, symmetric = TRUE)
    Ainv <- e$vectors %*% ((1 / pmax(e$values, .Machine$double.eps)) * t(e$vectors))
    f <- Ainv %*% t(Lambda) %*% W %*% Yt
    resid <- (Yt - Lambda %*% f) / sigma
    wn <- ifelse(abs(resid) <= delta, 1, delta / abs(resid))[, 1]
    if (max(abs(w - wn)) < 1e-6) break
    w <- wn
  }
  as.numeric(f)
}

effects_large_n <- function(Y, T0, r) {
  Ypre <- Y[, 1:T0]; Ypost <- Y[, (T0 + 1):ncol(Y)]
  tYc <- scale(t(Ypre), center = TRUE, scale = FALSE)
  fit <- fa.em(tYc, r); Lambda <- fit$Gamma; sigma <- fit$sigma
  Yreg <- rowMeans(Ypost)
  as.numeric(Yreg - Lambda %*% robust.f(Yreg, Lambda, sigma))
}

effects_fixed_n <- function(Y, T0, r, update_dif) {
  N <- nrow(Y); T <- ncol(Y); Ypre <- Y[, 1:T0]; Ypost <- Y[, (T0 + 1):T]
  fac <- factanal(t(Ypre), r, rotation = "none")
  FL <- sqrtm(var(t(Ypre))) %*% fac$loadings
  Yreg <- if (update_dif) rowMeans(Ypost) - rowMeans(Ypre) else rowMeans(Ypost)
  coef <- ltsReg(x = FL, y = Yreg, intercept = FALSE, alpha = 0.5)$coefficients
  est1 <- Yreg - FL %*% coef
  ev <- sort(eigen(cov(t(Y)))$values)
  sigma2 <- sum(ev[1:(floor(N / 2) + r)])
  sel <- abs(est1) <= sqrt(2 * log(N * T) * sigma2 / T)
  Ls <- FL[sel, ]; ah <- solve(t(Ls) %*% Ls + diag(1e-8, r)) %*% t(Ls) %*% Yreg[sel]
  as.numeric(Yreg - FL %*% ah)
}

args <- commandArgs(trailingOnly = TRUE)
Y <- as.matrix(read.csv(args[1], header = FALSE))
X <- as.integer(read.csv(args[2], header = FALSE)[, 1])
regime <- args[3]; r <- as.integer(args[4]); update_dif <- as.logical(args[5])
T0 <- sum(X == 0)
eff <- if (regime == "large_n") effects_large_n(Y, T0, r) else
       effects_fixed_n(Y, T0, r, update_dif)
write.csv(data.frame(effect = eff), args[6], row.names = FALSE)
