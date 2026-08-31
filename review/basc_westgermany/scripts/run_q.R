# Does the fit gap close if the post-treatment effect is given the basis the
# paper itself describes?
#
# Section 3.1 sets q = 1 with D_1t = 1 "in the simplest specification used for
# comparison with standard SCM", and offers D_2t = t - T_0 as an alternative.
# The released sampler implements only the constant case: `Dt * alpha.v` relies
# on scalar recycling. Generalising it is one substitution, `Dt %*% alpha.v`,
# applied at the five sites where that expression appears (basc_funcs_q.R).
# Everything else is the authors' code.
suppressMessages(library(MASS))
rinvgamma <- function(n, shape, scale) 1 / rgamma(n, shape = shape, rate = scale)

# Upper-truncated multivariate normal. For q = 1 this is the exact inverse-CDF
# draw used for every earlier run, so the control reproduces bit for bit. For
# q > 1 the covariance W.a is not diagonal, so components are drawn from their
# truncated conditionals; sweeping repeatedly targets the same distribution.
rtmvnorm <- function(n, mean, sigma, upper, nsweep = 40, ...) {
  S <- as.matrix(sigma); qd <- length(mean)
  draw1 <- function(m, s, up) {
    p <- pnorm((up - m) / s)
    m + s * qnorm(runif(1, 0, max(p, 1e-12)))
  }
  if (qd == 1) return(draw1(mean[1], sqrt(S[1, 1]), upper[1]))
  x <- pmin(mean, upper - 1e-6)
  for (sw in seq_len(nsweep)) {
    for (i in seq_len(qd)) {
      j <- setdiff(seq_len(qd), i)
      Sjj_inv <- solve(S[j, j, drop = FALSE])
      cm <- mean[i] + S[i, j, drop = FALSE] %*% Sjj_inv %*% (x[j] - mean[j])
      cv <- S[i, i] - S[i, j, drop = FALSE] %*% Sjj_inv %*% S[j, i, drop = FALSE]
      x[i] <- draw1(as.numeric(cm), sqrt(max(as.numeric(cv), 1e-12)), upper[i])
    }
  }
  x
}
rdist <- function(x1, x2 = NULL) {
  x1 <- as.matrix(x1); if (is.null(x2)) x2 <- x1 else x2 <- as.matrix(x2)
  outer(seq_len(nrow(x1)), seq_len(nrow(x2)),
        Vectorize(function(i, j) sqrt(sum((x1[i, ] - x2[j, ])^2))))
}
source("basc_funcs_q.R")

args  <- commandArgs(trailingOnly = TRUE)
N     <- as.integer(args[1]); nburn <- as.integer(args[2]); seed <- as.integer(args[3])

y <- as.numeric(read.csv("y.csv", header = FALSE)[, 1])
x <- as.matrix(read.csv("x.csv", header = FALSE)); colnames(x) <- NULL
donor_names <- readLines("donors.csv")
years <- 1960:2003
vt <- seq_len(length(y)); T0 <- 30; j <- ncol(x); t <- length(vt)
pre <- vt <= T0; post <- vt > T0

# D_mt exactly as Section 3.1 writes it: the indicator, then the linear term.
D_const <- matrix(as.numeric(post), ncol = 1)
D_lin   <- cbind(as.numeric(post), as.numeric(post) * (vt - T0))

configs <- list(
  list(tag = "q1_const", D = D_const, q = 1),
  list(tag = "q2_linear", D = D_lin,  q = 2)
)

out <- NULL
for (cf in configs) {
  el <- system.time(invisible(capture.output(
    ch <- run_basc_chain(seed = seed, y = y, x = x, vt = vt, Dt = cf$D,
                         N = N, nburn = nburn, q = cf$q)
  )))[["elapsed"]]

  g <- ch$gamma.sample; u <- ch$u.sample
  beta <- matrix(0, N, j)
  for (i in 1:N) beta[i, ] <- (g[i, ] * u[i, ]) / sum(g[i, ] * u[i, ])
  y.samp <- matrix(0, N, t)
  for (i in 1:N) y.samp[i, ] <- as.numeric(x %*% beta[i, ] + ch$f.sample[i, ])
  path <- colMeans(y.samp)
  a <- matrix(ch$a.sample, nrow = N)

  row <- data.frame(
    config = cf$tag, q = cf$q, N = N, nburn = nburn, seed = seed,
    rmse_pre = sqrt(mean((y[pre] - path[pre])^2)),
    att_post = mean(y[post] - path[post]),
    alpha1 = mean(a[, 1]),
    alpha2 = if (cf$q > 1) mean(a[, 2]) else NA_real_,
    n_donors = sum(colMeans(g) > 0.5),
    secs = round(el, 1))
  out <- rbind(out, row)
  cat(sprintf("%-10s q=%d  rmse_pre=%8.2f  att=%9.2f  alpha=(%s)  donors=%2d  (%.0fs)\n",
              cf$tag, cf$q, row$rmse_pre, row$att_post,
              if (cf$q > 1) sprintf("%.1f, %.1f", row$alpha1, row$alpha2) else sprintf("%.1f", row$alpha1),
              row$n_donors, el))
  write.csv(data.frame(year = years, observed = y, synthetic = path),
            sprintf("path_%s_%d_%d.csv", cf$tag, N, seed), row.names = FALSE)
  write.csv(data.frame(donor = donor_names, weight = colMeans(beta), incl_prob = colMeans(g)),
            sprintf("weights_%s_%d_%d.csv", cf$tag, N, seed), row.names = FALSE)
}
write.csv(out, sprintf("q_results_%d_%d.csv", N, seed), row.names = FALSE)
