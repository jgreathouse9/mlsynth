# Reference generator for the ferman_manyperiods Path-B benchmark.
#
# Regenerates the SC-estimator columns (1-4) of Ferman (2021, JASA
# 116(536):1764-1772) Table 1 -- the Monte-Carlo demonstration that, in a linear
# factor model, the synthetic-control unit asymptotically recovers the treated
# unit's factor structure (weights concentrate on the treated unit's own factor
# group) and its one-period-ahead effect SD shrinks, as BOTH the number of
# control units J and pre-treatment periods T0 grow -- even with imperfect
# pretreatment fit.
#
# This is a clean-room reimplementation of the paper's own simulation
# (supplementary main.R + aux.R): a K=2-factor Gaussian AR(1) DGP and the plain
# simplex SC estimator solved by solve.QP (Goldfarb-Idnani) -- i.e. aux.R's
# synth_control_est. It uses ONLY base R + quadprog (no data, no Synth package),
# so it needs no CRAN network beyond quadprog.
#
#   Rscript benchmarks/R/ferman_manyperiods.R [nreps]
#
# Prints the SC columns of Table 1 (E[mu01], se[mu01], E[mu02], se(alpha)) for
# J in {4,10,50,100} x {Panel A: T0=J+5, Panel B: T0=2J}. With nreps=5000 and
# set.seed(47) it reproduces the published Table 1 to ~0.001. The benchmark case
# pins the *published* Table 1 numbers (independently cross-checked against the
# authors' supplementary results.csv); this script documents how to regenerate
# them from scratch.

suppressMessages({
  if (!require("quadprog", character.only = TRUE)) {
    install.packages("quadprog"); require("quadprog", character.only = TRUE)
  }
})

args <- commandArgs(trailingOnly = TRUE)
nreps <- if (length(args) >= 1) as.integer(args[1]) else 5000

rho <- 0.5
var_u <- 1 - rho^2          # -> var(lambda_kt) = 1
var_eps <- 1
K <- 2

# AR(1) started from its stationary distribution (aux.R simulate_ar1).
simulate_ar1 <- function(T0) {
  y <- rnorm(1, 0, sqrt(var_u / (1 - rho^2)))
  for (t in 1:T0) y <- c(y, rho * y[t] + rnorm(1, 0, sqrt(var_u)))
  y[-1]
}

# Plain simplex SC via solve.QP (aux.R synth_control_est): min ||y - Xw||^2
# s.t. sum(w)=1, w>=0.
synth_control_est <- function(y_before, y_after) {
  y <- y_before[, 1]; X <- y_before[, -1]
  Dmat <- t(X) %*% X + 1e-8 * diag(ncol(X))
  dvec <- t(X) %*% y
  Amat <- t(rbind(rep(1, ncol(X)), diag(ncol(X))))
  bvec <- c(1, rep(0, ncol(X)))
  w <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
  w <- pmax(w, 0)
  list(w = w, eff = as.numeric(y_after %*% c(1, -w)))
}

set.seed(47)
cat(sprintf("%-4s %4s %4s | %6s %6s %6s | %7s\n",
            "Pan", "J", "T0", "Emu01", "semu01", "Emu02", "sealpha"))
for (panel in c("A", "B")) for (J in c(4, 10, 50, 100)) {
  T0 <- if (panel == "A") J + 5 else 2 * J
  wl1 <- wl2 <- eff <- numeric(nreps)
  h <- J %/% 2
  grp1 <- c(rep(TRUE, h), rep(FALSE, J - h))   # donor in treated unit's factor group
  for (r in 1:nreps) {
    Factors <- sapply(1:K, function(k) simulate_ar1(T0 + 1))
    Mu <- matrix(0, J + 1, K)
    Mu[1:(1 + h), 1] <- 1; Mu[(2 + h):(J + 1), 2] <- 1
    y <- Factors %*% t(Mu) + matrix(rnorm((T0 + 1) * (J + 1), 0, sqrt(var_eps)),
                                    T0 + 1, J + 1)
    fit <- synth_control_est(y[1:T0, ], y[T0 + 1, ])
    wl1[r] <- sum(fit$w[grp1]); wl2[r] <- sum(fit$w[!grp1]); eff[r] <- fit$eff
  }
  cat(sprintf("%-4s %4d %4d | %6.3f %6.3f %6.3f | %7.3f\n",
              panel, J, T0, mean(wl1), sd(wl1), mean(wl2), sqrt(mean(eff^2))))
}
