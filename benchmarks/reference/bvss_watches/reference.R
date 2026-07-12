# Reference run for the `bvss_watches` benchmark case.
#
# The authors' own two-coordinate Gibbs sampler for Bayesian Synthetic Control
# with a soft simplex constraint (Xu & Zhou 2025, arXiv:2503.06454), taken
# verbatim from their fsPDA replication script (example2_fspda_2.R -- the China
# anti-corruption empirical example). mlsynth's BVSS is a direct port of this
# sampler, so this is a genuine cross-check of the port against the original R.
#
# Two blocks are emitted:
#   (A) DETERMINISTIC ENGINE -- the sampler's math primitives (VM log-det, RSS,
#       RSS2, AM, loglike) on a FIXED (gamma, tau, mu, phi). These are exact and
#       the Python case pins mlsynth's primitives against them value-for-value.
#   (B) POSTERIOR SUMMARY -- a full Gibbs run (seed 1) on the same panel, giving
#       the posterior-mean ATT, tau, phi and model size |gamma| with a 95%
#       credible set, for the distributional (within-Monte-Carlo-error) check.
#
# Data: basedata/china_watches_long.csv -- the fsPDA-style p>n anti-corruption
# panel (1 treated series `watches`, 87 donor product-category import series, 35
# pre-treatment months), the same panel mlsynth's BVSS fits. Only `truncnorm` is
# needed (the sampler's only non-base dependency); the fsPDA package is used in
# the authors' script solely to load their own data and is not required here.
#
# Run from the repository root:  Rscript benchmarks/reference/bvss_watches/reference.R
suppressMessages(library(truncnorm))

# ---- authors' sampler primitives (verbatim from example2_fspda_2.R) ----
VM <- function(gamma, tau) Gram[gamma == 1, gamma == 1] + diag(sum(gamma)) / tau
RSS <- function(gamma, tau, z) {
  V <- VM(gamma, tau); Xg <- X[, gamma == 1]; Xz <- t(Xg) %*% z
  as.numeric(t(z) %*% z - t(Xz) %*% solve(V, Xz))
}
RSS2 <- function(gamma, tau, z1, z2) {
  V <- VM(gamma, tau); Xg <- X[, gamma == 1]
  Xz1 <- t(Xg) %*% z1; Xz2 <- t(Xg) %*% z2
  as.numeric(t(z1) %*% z2 - t(Xz1) %*% solve(V, Xz2))
}
loglike <- function(gamma, tau, mu, phi) {
  z <- Y - X %*% mu; V <- VM(gamma, tau)
  M / 2 * log(phi) - sum(gamma) / 2 * log(tau) -
    0.5 * as.numeric(determinant(V, log = TRUE)$mod) - 0.5 * phi * RSS(gamma, tau, z)
}
AM <- function(gamma, tau, theta) {
  p <- sum(gamma) * log(theta) + (N - sum(gamma)) * log(1 - theta)
  V <- VM(gamma, tau)
  p + lfactorial(sum(gamma) - 1) - 0.5 * as.numeric(determinant(V, log = TRUE)$mod)
}
MH_tau <- function(tau, gamma, mu, phi, a = 0.01, b = 0.1) {
  nrep <- 11; tau_M <- rep(NA, nrep); tau_M[1] <- tau
  for (i in 2:nrep) {
    x <- tau_M[i - 1]; logy <- log(x) + rnorm(1)
    if (logy < log(tau_min)) logy <- 2 * log(tau_min) - logy
    y <- exp(logy)
    r <- dgamma(y, shape = a, rate = b, log = TRUE) - dgamma(x, shape = a, rate = b, log = TRUE) +
      loglike(gamma, y, mu, phi) - loglike(gamma, x, mu, phi) + log(y) - log(x)
    tau_M[i] <- if (runif(1) < exp(r)) y else x
  }
  tau_M[nrep]
}
gibbs_BVS <- function(M, N, size, kappa1, kappa2, tau_min = 1e-6) {
  musample <- matrix(NA, N, size); wsample <- matrix(0, N, size)
  phisample <- rep(NA, size); tausample <- rep(NA, size)
  musample[, 1] <- rep(1 / N, N); phisample[1] <- 0.8; tausample[1] <- 1
  combinations <- combn(1:N, 2)
  for (h in 2:size) {
    mutemp <- musample[, h - 1]
    for (j in 1:dim(combinations)[2]) {
      id <- combinations[, j]; mutemp[id] <- c(0, 0)
      s <- 1 - sum(mutemp[-id]); z <- Y - X %*% mutemp
      g00 <- ifelse(mutemp != 0, 1, 0); g11 <- g01 <- g10 <- g00
      g10[id[1]] <- 1; g01[id[2]] <- 1; g11[id] <- 1
      if (abs(s) > 1e-12) {
        L <- RSS(g11, tausample[h - 1], X[, id[2]] - X[, id[1]])
        O <- RSS2(g11, tausample[h - 1], X[, id[1]] - X[, id[2]], z - s * X[, id[2]]) / L
        A11 <- AM(g11, tausample[h - 1], theta); A10 <- AM(g10, tausample[h - 1], theta)
        A01 <- AM(g01, tausample[h - 1], theta)
        p10 <- A10 - phisample[h - 1] * RSS(g10, tausample[h - 1], z - s * X[, id[1]]) / 2
        p01 <- A01 - phisample[h - 1] * RSS(g01, tausample[h - 1], z - s * X[, id[2]]) / 2
        NC <- pnorm((s - O) * sqrt(phisample[h - 1] * L)) - pnorm(-O * sqrt(phisample[h - 1] * L))
        p11 <- A11 + log(NC) - phisample[h - 1] * (RSS(g11, tausample[h - 1], z - s * X[, id[2]]) - O^2 * L) / 2
        ptemp <- c(p10, p01, p11); pbar <- max(ptemp)
        post_p <- exp(ptemp - pbar) / sum(exp(ptemp - pbar))
        gamma <- sample(c(0, 1, 2, 3), size = 1, prob = c(0, post_p))
        if (gamma == 0) mutemp[id] <- c(0, 0)
        else if (gamma == 1) mutemp[id] <- c(s, 0)
        else if (gamma == 2) mutemp[id] <- c(0, s)
        else { mutemp[id[1]] <- rtruncnorm(1, mean = O, sd = 1 / sqrt(phisample[h - 1] * L), a = 0, b = s); mutemp[id[2]] <- s - mutemp[id[1]] }
      }
    }
    musample[, h] <- mutemp
    zmu <- Y - X %*% musample[, h]; gtemp <- ifelse(musample[, h] != 0, 1, 0)
    Xg <- X[, gtemp == 1]
    w <- solve(VM(gtemp, tausample[h - 1]), t(Xg) %*% Y + mutemp[gtemp == 1] / tausample[h - 1])
    wsample[gtemp == 1, h] <- w
    phisample[h] <- rgamma(1, shape = (M + kappa1) / 2, rate = (kappa2 + RSS(gtemp, tausample[h - 1], zmu)) / 2)
    tausample[h] <- MH_tau(tausample[h - 1], gtemp, musample[, h], phisample[h], a = 0.01, b = 0.1)
  }
  ws <- wsample[, (size / 2 + 1):size]
  Y1_control <- apply(sweep(X1, 2, mean_X, "-") %*% ws, 1, mean)
  treatments <- Y1 - Y1_control - mean_Y
  ATT_CS <- colMeans(as.vector(Y1) - mean_Y - sweep(X1, 2, mean_X, "-") %*% ws)
  list(ATT = mean(treatments), ATT_CS = sort(ATT_CS), tausample = tausample,
       phisample = phisample, musample = musample)
}

# ---- data (same panel mlsynth fits) ----
d <- read.csv("basedata/china_watches_long.csv")
units <- unique(d$unit); times <- sort(unique(d$time))
W <- matrix(NA_real_, length(times), length(units), dimnames = list(times, units))
W[cbind(match(d$time, times), match(d$unit, units))] <- d$y
treated <- "watches"; donor_names <- setdiff(units, treated)
tr <- d[d$unit == treated, ]; M <- sum(tr$treat == 0)            # pre-period count
Mall <- length(times)
Y0 <- matrix(W[1:M, treated], ncol = 1); X0 <- W[1:M, donor_names]
Y1 <- matrix(W[(M + 1):Mall, treated], ncol = 1); X1 <- W[(M + 1):Mall, donor_names]
mean_X <- colMeans(X0); mean_Y <- mean(Y0)
Y <- sweep(Y0, 2, mean_Y, "-"); X <- sweep(X0, 2, mean_X, "-")
N <- ncol(X); Gram <- t(X) %*% X
theta <- 0.25; tau_min <- 1e-6

# ---- (A) DETERMINISTIC ENGINE on a fixed (gamma, tau, mu, phi) ----
gamma_fix <- rep(0, N); gamma_fix[1:5] <- 1
tau_fix <- 0.5; phi_fix <- 1.3
mu_fix <- rep(0, N); mu_fix[1:5] <- c(0.4, 0.25, 0.15, 0.1, 0.1)
z_fix <- Y - X %*% mu_fix
Vf <- VM(gamma_fix, tau_fix)
vm_logdet <- as.numeric(determinant(Vf, log = TRUE)$mod)
rss_val <- RSS(gamma_fix, tau_fix, z_fix)
rss2_val <- RSS2(gamma_fix, tau_fix, X[, 2] - X[, 1], z_fix)
am_val <- AM(gamma_fix, tau_fix, theta)
ll_val <- loglike(gamma_fix, tau_fix, mu_fix, phi_fix)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("det_vm_logdet\t%.9f\n", vm_logdet))
cat(sprintf("det_rss\t%.9f\n", rss_val))
cat(sprintf("det_rss2\t%.9f\n", rss2_val))
cat(sprintf("det_am\t%.9f\n", am_val))
cat(sprintf("det_loglike\t%.9f\n", ll_val))

# ---- (B) POSTERIOR SUMMARY (full Gibbs, seed 1) ----
size <- 400
set.seed(1)
res <- gibbs_BVS(M, N, size, kappa1 = 1, kappa2 = 1, tau_min = 1e-6)
GM <- res$musample; GM[GM > 0] <- 1; ms <- colSums(GM)
keep <- (size / 2 + 1):size
cat(sprintf("post_att\t%.6f\n", res$ATT))
cat(sprintf("post_att_lo\t%.6f\n", quantile(res$ATT_CS, 0.025)))
cat(sprintf("post_att_hi\t%.6f\n", quantile(res$ATT_CS, 0.975)))
cat(sprintf("post_tau_mean\t%.6f\n", mean(res$tausample[keep])))
cat(sprintf("post_phi_mean\t%.6f\n", mean(res$phisample[keep])))
cat(sprintf("post_modelsize_mean\t%.6f\n", mean(ms[keep])))
cat("== SESSION INFO ==\n")
print(sessionInfo())
