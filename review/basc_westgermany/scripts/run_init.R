# Does the sampler stay at a good fit if it is started at one?
#
# The pre-1990 criterion that standard SCM minimises is attained on this panel at
# RMSE 60.84 by a simplex vector inside BASC's own support (mlsynth VanillaSC,
# outcome-only). Starting the chain there separates two explanations for the
# published fit of 169: a posterior genuinely centred away from the good region,
# or a chain that never reaches it. If the chain holds near 61, the second; if it
# migrates to ~200, the first.
suppressMessages(library(MASS))
rinvgamma <- function(n, shape, scale) 1 / rgamma(n, shape = shape, rate = scale)
rtmvnorm <- function(n, mean, sigma, upper, ...) {
  sdv <- sqrt(diag(as.matrix(sigma))); qd <- length(mean); out <- numeric(qd)
  for (i in seq_len(qd)) {
    p <- pnorm((upper[i] - mean[i]) / sdv[i])
    out[i] <- mean[i] + sdv[i] * qnorm(runif(1, 0, max(p, 1e-12)))
  }
  matrix(out, nrow = 1)
}
rdist <- function(x1, x2 = NULL) {
  x1 <- as.matrix(x1); if (is.null(x2)) x2 <- x1 else x2 <- as.matrix(x2)
  outer(seq_len(nrow(x1)), seq_len(nrow(x2)),
        Vectorize(function(i, j) sqrt(sum((x1[i, ] - x2[j, ])^2))))
}
source("basc_funcs_g1.R")

args <- commandArgs(trailingOnly = TRUE)
N <- as.integer(args[1]); nburn <- as.integer(args[2]); seed <- as.integer(args[3])

y <- as.numeric(read.csv("y.csv", header = FALSE)[, 1])
x <- as.matrix(read.csv("x.csv", header = FALSE)); colnames(x) <- NULL
w_opt <- as.numeric(read.csv("w_opt.csv", header = FALSE)[, 1])   # mlsynth VanillaSC weights
years <- 1960:2003
vt <- seq_len(length(y)); Dt <- as.integer(vt >= 31); j <- ncol(x); t <- length(vt)
pre <- vt < 31; post <- vt >= 31

sig0 <- mean((y[pre] - (x %*% w_opt)[pre])^2)   # residual variance at that vector
cat(sprintf("start: pre-RMSE %.2f, sigma^2 %.0f\n", sqrt(sig0), sig0))

configs <- list(
  list(tag = "basc_init",  g1 = FALSE, au = NULL),
  list(tag = "g1_a1_init", g1 = TRUE,  au = 1),
  list(tag = "g1_a2.5_init", g1 = TRUE, au = NULL)
)

out <- NULL
for (cf in configs) {
  invisible(capture.output(
    chain <- run_basc_chain(seed = seed, y = y, x = x, vt = vt, Dt = Dt,
                            N = N, nburn = nburn, q = 1,
                            force_gamma1 = cf$g1, alpha_u_override = cf$au,
                            init_u = w_opt, init_sigsq = sig0)
  ))
  g <- chain$gamma.sample; u <- chain$u.sample
  beta <- matrix(0, N, j)
  for (i in 1:N) beta[i, ] <- (g[i, ] * u[i, ]) / sum(g[i, ] * u[i, ])
  y.samp <- matrix(0, N, t)
  for (i in 1:N) y.samp[i, ] <- as.numeric(x %*% beta[i, ] + chain$f.sample[i, ])
  path <- colMeans(y.samp)

  # fit implied by each retained draw, to see where the chain sat over time
  per_draw <- apply(y.samp, 1, function(v) sqrt(mean((y[pre] - v[pre])^2)))

  row <- data.frame(config = cf$tag, N = N, nburn = nburn, seed = seed,
                    rmse_pre = sqrt(mean((y[pre] - path[pre])^2)),
                    att_post = mean(y[post] - path[post]),
                    draw_rmse_first = per_draw[1],
                    draw_rmse_last = per_draw[N],
                    draw_rmse_median = median(per_draw),
                    sigma_post = sqrt(mean(chain$sig.sample)),
                    n_donors = sum(colMeans(g) > 0.5))
  out <- rbind(out, row)
  cat(sprintf("%-13s rmse_pre=%8.2f  att=%9.2f  per-draw rmse first/median/last = %.0f/%.0f/%.0f  sigma=%.0f\n",
              cf$tag, row$rmse_pre, row$att_post,
              row$draw_rmse_first, row$draw_rmse_median, row$draw_rmse_last, row$sigma_post))
  write.csv(data.frame(draw = 1:N, rmse = per_draw), sprintf("draws_%s_%d.csv", cf$tag, N), row.names = FALSE)
  write.csv(data.frame(year = years, observed = y, synthetic = path), sprintf("path_%s_%d.csv", cf$tag, N), row.names = FALSE)
}
write.csv(out, sprintf("init_results_%d.csv", N), row.names = FALSE)
