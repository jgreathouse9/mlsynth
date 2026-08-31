# With the effect basis corrected (q = 2), what is left of the fit gap?
# Vary the two remaining knobs: the Dirichlet dispersion alpha_u, and whether
# donor selection is active. Least squares under the same q = 2 specification,
# with no prior, no GP and no selection, reaches 74.6.
suppressMessages(library(MASS))
rinvgamma <- function(n, shape, scale) 1 / rgamma(n, shape = shape, rate = scale)
rtmvnorm <- function(n, mean, sigma, upper, nsweep = 40, ...) {
  S <- as.matrix(sigma); qd <- length(mean)
  d1 <- function(m, s, up) { p <- pnorm((up - m)/s); m + s*qnorm(runif(1, 0, max(p, 1e-12))) }
  if (qd == 1) return(d1(mean[1], sqrt(S[1,1]), upper[1]))
  x <- pmin(mean, upper - 1e-6)
  for (sw in seq_len(nsweep)) for (i in seq_len(qd)) {
    j <- setdiff(seq_len(qd), i); Sj <- solve(S[j, j, drop = FALSE])
    cm <- mean[i] + S[i, j, drop = FALSE] %*% Sj %*% (x[j] - mean[j])
    cv <- S[i, i] - S[i, j, drop = FALSE] %*% Sj %*% S[j, i, drop = FALSE]
    x[i] <- d1(as.numeric(cm), sqrt(max(as.numeric(cv), 1e-12)), upper[i])
  }
  x
}
rdist <- function(x1, x2 = NULL) { x1 <- as.matrix(x1); if (is.null(x2)) x2 <- x1 else x2 <- as.matrix(x2)
  outer(seq_len(nrow(x1)), seq_len(nrow(x2)), Vectorize(function(i,j) sqrt(sum((x1[i,]-x2[j,])^2)))) }
source("basc_funcs_q.R")

a <- commandArgs(trailingOnly = TRUE); N <- as.integer(a[1]); nb <- as.integer(a[2]); sd0 <- as.integer(a[3])
y <- as.numeric(read.csv("y.csv", header = FALSE)[,1])
x <- as.matrix(read.csv("x.csv", header = FALSE)); colnames(x) <- NULL
vt <- seq_len(length(y)); T0 <- 30; j <- ncol(x); t <- length(vt)
pre <- vt <= T0; post <- vt > T0
D <- cbind(as.numeric(post), as.numeric(post)*(vt - T0))       # q = 2

cfgs <- list(
  list(tag="au2.5_select",  au=NULL, g1=FALSE),
  list(tag="au1_select",    au=1,    g1=FALSE),
  list(tag="au2.5_nosel",   au=NULL, g1=TRUE),
  list(tag="au1_nosel",     au=1,    g1=TRUE))
out <- NULL
for (cf in cfgs) {
  invisible(capture.output(ch <- run_basc_chain(seed=sd0, y=y, x=x, vt=vt, Dt=D, N=N, nburn=nb, q=2,
                                                force_gamma1=cf$g1, alpha_u_override=cf$au)))
  g <- ch$gamma.sample; u <- ch$u.sample
  b <- matrix(0,N,j); for (i in 1:N) b[i,] <- (g[i,]*u[i,])/sum(g[i,]*u[i,])
  ys <- matrix(0,N,t); for (i in 1:N) ys[i,] <- as.numeric(x %*% b[i,] + ch$f.sample[i,])
  p <- colMeans(ys); pd <- colMeans(t(apply(b,1,function(w) as.numeric(x %*% w))))
  r <- data.frame(config=cf$tag, alpha_u=if(is.null(cf$au)) 2.5 else cf$au,
                  selection=!cf$g1, seed=sd0, N=N,
                  rmse_pre=sqrt(mean((y[pre]-p[pre])^2)),
                  att_post=mean(y[post]-p[post]),
                  gp_sd=sd(colMeans(ch$f.sample)), n_donors=sum(colMeans(g)>0.5))
  out <- rbind(out, r)
  cat(sprintf("%-14s alpha_u=%.1f selection=%-5s  rmse_pre=%7.2f  att=%9.2f  gp_sd=%5.2f donors=%2d\n",
              cf$tag, r$alpha_u, r$selection, r$rmse_pre, r$att_post, r$gp_sd, r$n_donors))
}
write.csv(out, sprintf("decomp_%d_%d.csv", N, sd0), row.names=FALSE)
