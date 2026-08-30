# Condition 5: run the authors' own BASC sampler with gamma fixed to 1.
#
# Section 3.1 states that when every donor is selected the weight prior reduces
# to Eq (11), and that at alpha_u = 1 this coincides with the uniform Dirichlet
# of Martinez and Vives-i-Bastida (Eq 5). Three configurations are run from the
# same seed and the same chain length:
#
#   basc      the sampler as published (gamma sampled, alpha_u = 2.5)
#   g1_a2.5   gamma fixed to 1, alpha_u = 2.5   (their prior, selection off)
#   g1_a1     gamma fixed to 1, alpha_u = 1     (the Section 3.1 configuration)
#
# Everything below the three-line patch in basc_funcs_g1.R is the authors' code.
suppressMessages(library(MASS))
rinvgamma <- function(n, shape, scale) 1 / rgamma(n, shape = shape, rate = scale)
rtmvnorm <- function(n, mean, sigma, upper, ...) {   # q=1, diagonal, upper-truncated
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

args  <- commandArgs(trailingOnly = TRUE)
N     <- as.integer(args[1]); nburn <- as.integer(args[2])
seed  <- if (length(args) >= 3) as.integer(args[3]) else 200

y  <- as.numeric(read.csv("y.csv", header = FALSE)[, 1])
x  <- as.matrix(read.csv("x.csv", header = FALSE)); colnames(x) <- NULL
donor_names <- readLines("donors.csv")
years <- 1960:2003
vt <- seq_len(length(y)); Dt <- as.integer(vt >= 31); j <- ncol(x); t <- length(vt)
pre <- vt < 31; post <- vt >= 31

configs <- list(
  list(tag = "basc",    g1 = FALSE, au = NULL),
  list(tag = "g1_a2.5", g1 = TRUE,  au = NULL),
  list(tag = "g1_a1",   g1 = TRUE,  au = 1)
)

out <- NULL
for (cf in configs) {
  el <- system.time(invisible(capture.output(
    chain <- run_basc_chain(seed = seed, y = y, x = x, vt = vt, Dt = Dt,
                            N = N, nburn = nburn, q = 1,
                            force_gamma1 = cf$g1, alpha_u_override = cf$au)
  )))[["elapsed"]]

  g <- chain$gamma.sample; u <- chain$u.sample
  beta <- matrix(0, N, j)
  for (i in 1:N) beta[i, ] <- (g[i, ] * u[i, ]) / sum(g[i, ] * u[i, ])

  # posterior counterfactual, exactly as the authors build it: donor combination
  # plus the GP term, with the treatment effect zeroed and no epsilon
  y.samp <- matrix(0, N, t)
  for (i in 1:N) y.samp[i, ] <- as.numeric(x %*% beta[i, ] + chain$f.sample[i, ])
  path <- colMeans(y.samp)

  # the donor combination on its own, i.e. the synthetic control without the GP
  y.don <- matrix(0, N, t)
  for (i in 1:N) y.don[i, ] <- as.numeric(x %*% beta[i, ])
  don <- colMeans(y.don)

  row <- data.frame(
    config      = cf$tag,
    N           = N, nburn = nburn, seed = seed,
    rmse_pre    = sqrt(mean((y[pre]  - path[pre])^2)),
    att_post    = mean(y[post] - path[post]),
    rmse_pre_donor_only = sqrt(mean((y[pre] - don[pre])^2)),
    mean_incl   = mean(colMeans(g)),
    n_donors    = sum(colMeans(g) > 0.5),
    sigma_post  = sqrt(mean(chain$sig.sample)),
    gp_sd       = sd(colMeans(chain$f.sample)),
    secs        = round(el, 1)
  )
  out <- rbind(out, row)
  cat(sprintf("%-8s rmse_pre=%8.2f  att=%9.2f  donor-only rmse=%9.2f  donors=%2d  sigma=%7.2f  (%.0fs)\n",
              cf$tag, row$rmse_pre, row$att_post, row$rmse_pre_donor_only,
              row$n_donors, row$sigma_post, el))

  write.csv(data.frame(year = years, observed = y, synthetic = path, donor_only = don),
            sprintf("path_%s_%d.csv", cf$tag, N), row.names = FALSE)
  write.csv(data.frame(donor = donor_names, weight = colMeans(beta), incl_prob = colMeans(g)),
            sprintf("weights_%s_%d.csv", cf$tag, N), row.names = FALSE)
}
write.csv(out, sprintf("gamma1_results_%d.csv", N), row.names = FALSE)
