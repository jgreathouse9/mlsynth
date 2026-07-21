# Reproduce the BASC in-sample fit on West Germany (500/500 and 2000/2000) and
# save the posterior counterfactual path (+95% CI) and donor weights.
#
# This script deliberately does NOT vendor the authors' sampler. It supplies the
# few helpers their code calls from CRAN packages that are hard to install here
# (rinvgamma, rtmvnorm, rdist), then sources the authors' own function
# definitions. To run it:
#   1. clone github.com/sll-lee/paper-BASC
#   2. from BASC_realdata.R, keep only the function definitions
#      (safe_chol .. run_basc_chain) as `basc_funcs.R`
#   3. export the West Germany outcome y (44), donor matrix x (44x16, year x
#      donor) and donor names from data/repgermany.dta
#   4. set N/nburn below and run.
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
source("basc_funcs.R")                       # authors' safe_chol .. run_basc_chain

y  <- as.numeric(read.csv("y.csv", header = FALSE)[, 1])
x  <- as.matrix(read.csv("x.csv", header = FALSE)); colnames(x) <- NULL
donor_names <- readLines("donors.csv")
years <- 1960:2003
vt <- seq_len(length(y)); Dt <- as.integer(vt >= 31); j <- ncol(x); t <- length(vt)
N <- 2000; nburn <- 2000                     # (paper uses 500000/500000)

invisible(capture.output(
  chain <- run_basc_chain(seed = 200, y = y, x = x, vt = vt, Dt = Dt, N = N, nburn = nburn, q = 1)
))
g <- chain$gamma.sample; u <- chain$u.sample
beta <- matrix(0, N, j); for (i in 1:N) beta[i, ] <- (g[i, ] * u[i, ]) / sum(g[i, ] * u[i, ])
y.samp <- matrix(0, N, t)
for (i in 1:N) y.samp[i, ] <- as.numeric(x %*% beta[i, ] + chain$f.sample[i, ])
ci <- apply(y.samp, 2, quantile, probs = c(0.025, 0.5, 0.975)); cf <- colMeans(y.samp)
pre <- vt < 31
cat(sprintf("BASC %d/%d in-sample RMSE = %.3f\n", nburn, N, sqrt(mean((y[pre] - cf[pre])^2))))
write.csv(data.frame(year = years, observed = y, synthetic = cf, lo95 = ci[1, ], hi95 = ci[3, ]),
          "../data/basc_counterfactual.csv", row.names = FALSE)
write.csv(data.frame(donor = donor_names, weight = colMeans(beta), incl_prob = colMeans(g)),
          "../data/basc_weights.csv", row.names = FALSE)
