# RRMA on the Basque panel at m = 2, written in R from the paper's text.
#
# Lv, J. and Li, J. (2026), "Rank-based Model Average Estimator for Average
# Treatment Effects", arXiv:2610.15553v1. The authors released no code; this
# script uses the R functions the paper names -- cor.test (Section 4.1),
# MASS::cov.rob (Section 2.3), quantreg::rq (Remark 1) -- and lpSolve for the
# linear programme of Section 2.4, which the paper states without naming a
# solver. It is a second implementation of the same reading as
# benchmarks/studies/rrma_ablation/oracle.py, written separately, so the two
# can disagree.
#
# m = 2 is the one value of Algorithm 1's grid at which every cov.rob call on
# this panel enumerates all subsets (choose(20, 3) = 1140 for the models,
# choose(10, 9) = 10 for the predictions), so the path is deterministic in R.
#
# Usage: Rscript rrma_m2.R <basque_data.csv>. Prints "key<TAB>values" lines.

suppressMessages({library(MASS); library(quantreg); library(lpSolve)})
args <- commandArgs(TRUE)
d <- read.csv(args[1])
W <- with(d, tapply(gdpcap, list(year, regionno), identity))
years <- as.numeric(rownames(W))
y <- W[, "17"]
X <- W[, colnames(W) != "17"]
T1 <- sum(years <= 1974)
emit <- function(key, v) cat(key, paste(format(v, digits = 17), collapse = "\t"), "\n", sep = "\t")

emit("donors", colnames(X))
pv <- sapply(seq_len(ncol(X)), function(j) cor.test(y[1:T1], X[1:T1, j], method = "kendall")$p.value)
emit("kendall_p", pv)
ord <- order(pv)
keep <- ord[pv[ord] <= 0.05]
m <- 2
S <- floor(length(keep) / m)
groups <- lapply(seq_len(S), function(s)
  if (s < S) keep[((s - 1) * m + 1):(s * m)] else keep[((S - 1) * m + 1):length(keep)])
for (s in seq_len(S)) emit(paste0("group", s), groups[[s]])

mallows <- function(Z) {
  r <- cov.rob(Z, method = "mve")
  pmin(1, qchisq(0.95, ncol(Z)) / mahalanobis(Z, r$center, r$cov))
}
# Remark 1: augmented data over ordered pairs t != t', weights b_tt' = h_t h_t'.
rankfit <- function(yy, Z, h) {
  n <- length(yy)
  idx <- which(outer(seq_len(n), seq_len(n), "!="), arr.ind = TRUE)
  b <- h[idx[, 1]] * h[idx[, 2]]
  yt <- b * (yy[idx[, 1]] - yy[idx[, 2]])
  Zt <- b * (Z[idx[, 1], , drop = FALSE] - Z[idx[, 2], , drop = FALSE])
  beta <- coef(rq(yt ~ Zt - 1, tau = 0.5))
  c(median(yy - Z %*% beta), beta)
}

hs <- lapply(groups, function(g) mallows(X[1:T1, g, drop = FALSE]))
for (s in seq_len(S)) emit(paste0("h", s), hs[[s]])

# Equation (11): growing-window predictions for t = T1/2 + 1, ..., T1.
val <- (T1 / 2 + 1):T1
P <- matrix(NA_real_, length(val), S)
for (k in seq_along(val)) {
  t <- val[k]
  for (s in seq_len(S)) {
    g <- groups[[s]]
    th <- rankfit(y[1:(t - 1)], X[1:(t - 1), g, drop = FALSE], hs[[s]][1:(t - 1)])
    P[k, s] <- th[1] + sum(X[t, g] * th[-1])
  }
}
for (k in seq_along(val)) emit(paste0("P", k), P[k, ])

# Section 2.4: tau weights from cov.rob on the predictions, then the LP.
hbar <- mallows(P)
emit("hbar", hbar)
yv <- y[val]
idx <- which(outer(seq_along(val), seq_along(val), "!="), arr.ind = TRUE)
tau <- hbar[idx[, 1]] * hbar[idx[, 2]]
dY <- yv[idx[, 1]] - yv[idx[, 2]]
dP <- P[idx[, 1], , drop = FALSE] - P[idx[, 2], , drop = FALSE]
np <- nrow(idx)
# variables: w (S), e_plus (np), e_minus (np), all >= 0
obj <- c(rep(0, S), tau, tau)
A <- rbind(cbind(dP, diag(np), -diag(np)), c(rep(1, S), rep(0, 2 * np)))
lp_out <- lp("min", obj, A, rep("=", np + 1), c(dY, 1))
w <- lp_out$solution[1:S]
emit("lp_status", lp_out$status)
emit("w", w)
emit("Q", mean(abs(yv - P %*% w)))

# Equation (9)-(10): full pre-period fits, counterfactual, ATE.
thetas <- lapply(seq_len(S), function(s) rankfit(y[1:T1], X[1:T1, groups[[s]], drop = FALSE], hs[[s]]))
for (s in seq_len(S)) emit(paste0("theta", s), thetas[[s]])
cf <- Reduce(`+`, lapply(seq_len(S), function(s)
  w[s] * (thetas[[s]][1] + X[, groups[[s]], drop = FALSE] %*% thetas[[s]][-1])))
emit("att", mean(y[(T1 + 1):length(y)] - cf[(T1 + 1):length(y)]))
emit("versions", c(R.version.string, as.character(packageVersion("MASS")),
                   as.character(packageVersion("quantreg")), as.character(packageVersion("lpSolve"))))
