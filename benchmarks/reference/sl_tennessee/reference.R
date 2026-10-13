## SL on Viviano and Bradic's own Medicaid-expansion panel, in R.
##
## The expert constructions and the weighting are transcribed from their
## replication package's `libraries/library.R` -- `generate_experts` for the
## experts and `Exp_algorithm` for Equations 11-12 -- so this is their algorithm
## and not a second reading of the paper. Their package is not redistributed
## here, which is why the functions are transcribed instead of sourced; each
## line below that departs from theirs is marked.
##
## Two departures, both corrections mlsynth also makes, measured in issue #651:
##
##   1. the penalty's folds are contiguous and unshuffled (`foldid`), where they
##      leave `cv.glmnet(nfolds = 5)` to draw them from the RNG. Theirs puts
##      lambda.min in one of three places on this window, one keeping no donors,
##      which moves the effect 40 percent with the random stream.
##   2. the bootstrap refits the ensemble with the same eta as the point
##      estimate, where `library.R:214` hard-codes eta = 1.
##
## Without (1) neither implementation is deterministic and a cross-validation
## measures the coin flip instead of the port.
##
## The random-forest expert is left out of the pinned ensemble on purpose.
## randomForest and scikit-learn's RandomForestRegressor are different
## implementations, so its path cannot agree cell for cell in any language pair,
## and including it would bound the port's accuracy by that gap instead of by
## the port. It is reported separately for context.

suppressPackageStartupMessages({
  library(glmnet)
})

panel <- read.csv("basedata/sl_tennessee_medcost.csv", stringsAsFactors = FALSE)
## Sorted, which is the donor column order mlsynth's dataprep produces. The
## order is not cosmetic here: the lasso's chosen penalty sits at the grid's
## minimum, where the fit is nearly unregularized on a 30-by-6 design, and
## coordinate descent is column-order sensitive at that point. Comparing the two
## implementations on differently ordered matrices moves the factor expert's
## path agreement from 4.9e-11 to 2.7e-07 -- a comparison of two orderings, not
## of two implementations.
states <- sort(setdiff(unique(panel$state), "TN"))
TT  <- max(panel$quarter)
wide <- function(col) {
  m <- matrix(NA_real_, nrow = TT, ncol = length(states))
  for (j in seq_along(states)) {
    s <- panel[panel$state == states[j], ]
    m[s$quarter, j] <- s[[col]]
  }
  m
}
X   <- wide("medcost")                              # donor outcomes, (T, 6)
Emp <- cbind(wide("employment"),
             panel[panel$state == "TN", "employment"][order(
               panel[panel$state == "TN", "quarter"])])
tn  <- panel[panel$state == "TN", ]
y   <- tn$medcost[order(tn$quarter)]
T0  <- min(tn$quarter[tn$expansion == 1]) - 1       # 50

TRAIN  <- 1:30                                      # Algorithm 1's first split
WEIGHT <- 31:50                                     # ...and its second
## their two grids, which are not the same: generate_experts gives the lasso
## expert exp(-10)..exp(-1) and the factor expert exp(-10)..exp(2), the second
## regressing a well-conditioned factor, not the outcome.
LAM_LASSO  <- seq(from = exp(-10), to = exp(-1), length = 79)
LAM_FACTOR <- seq(from = exp(-10), to = exp(2),  length = 79)

## contiguous folds: departure (1). K blocks of consecutive periods.
contig_folds <- function(n, k = 5) {
  rep(seq_len(k), each = ceiling(n / k), length.out = n)
}

## --- the experts, in their column order: lasso, factor, forest, did ---------
fit_lasso <- function(X, y, tr) {
  X1 <- X[tr, , drop = FALSE]; y1 <- y[tr]
  cv <- cv.glmnet(x = X1, y = y1, lambda = LAM_LASSO,
                  foldid = contig_folds(length(tr)))          # departure (1)
  f <- glmnet(x = X1, y = y1, lambda = cv$lambda.min)
  list(path = as.vector(predict(f, newx = X)),
       lambda = cv$lambda.min,
       nz = sum(as.vector(coef(f))[-1] != 0))
}

fit_factor <- function(X, y, tr) {
  X1 <- X[tr, , drop = FALSE]
  ## their eigen(X X')$vectors[, 1]; the leading left singular vector is the
  ## same direction up to sign, and the sign cancels in the loading step below.
  fac <- svd(X1)$u[, 1]
  cv <- cv.glmnet(x = X1, y = fac, lambda = LAM_FACTOR,
                  foldid = contig_folds(length(tr)))          # departure (1)
  Fh <- as.vector(predict(cv, newx = X, s = cv$lambda.min))
  assign("FACTOR_LAMBDA", cv$lambda.min, envir = .GlobalEnv)
  A  <- cbind(1, Fh[tr])
  b  <- solve(t(A) %*% A, t(A) %*% y[tr])
  as.vector(cbind(1, Fh) %*% b)
}

fit_did <- function(X, y, tr) {
  mean(y[tr]) - mean(colMeans(X[tr, , drop = FALSE])) + rowMeans(X)
}

## --- Equations 11-12, verbatim from their Exp_algorithm ---------------------
exp_weights <- function(ssr, eta) {
  w <- exp(-eta * (ssr - min(ssr)))     # shifted; a softmax is shift invariant
  w / sum(w)
}

stat  <- function(pred, y, idx) sum((pred[idx] - y[idx])^2) / sqrt(length(idx))
adj   <- function(pred, y, w_idx, p_idx)
  -mean(pred[p_idx] - y[p_idx]) - (-mean(pred[w_idx] - y[w_idx]))

la <- fit_lasso(X, y, TRAIN)
E3 <- cbind(lasso = la$path, factor = fit_factor(X, y, TRAIN),
            did = fit_did(X, y, TRAIN))

eta <- 1 / (sqrt(TT) * var(y))          # their 1/(sqrt(T) var(y))
ssr <- colSums((E3[WEIGHT, ] - y[WEIGHT])^2)
w3  <- exp_weights(ssr, eta)
cf3 <- as.vector(E3 %*% w3)

## the bootstrap, Algorithm 2, with departure (2): eta is the estimate's own
set.seed(20260926)
block_idx <- function(n, l) {
  starts <- sample.int(n, size = ceiling(n / l), replace = TRUE)
  idx <- unlist(lapply(starts, function(s) ((s - 1):(s + l - 2)) %% n + 1))
  idx[seq_len(n)]
}
boot_null <- function(P, y, w_idx, p_idx, eta, R = 10000, l = 3) {
  pool <- c(w_idx, p_idx); nw <- length(w_idx)
  out <- numeric(R)
  for (b in seq_len(R)) {
    idx <- pool[block_idx(length(pool), l)]
    fit <- idx[seq_len(nw)]; ev <- idx[-seq_len(nw)]
    s <- colSums((P[fit, , drop = FALSE] - y[fit])^2)
    out[b] <- stat(as.vector(P %*% exp_weights(s, eta)), y, ev)
  }
  out
}

POST <- (T0 + 1):TT
draws <- boot_null(E3, y, WEIGHT, POST, eta)
ts3 <- stat(cf3, y, POST)
p3  <- (1 + sum(draws >= ts3)) / (1 + length(draws))

cat("== REFERENCE VALUES ==\n")
cat(sprintf("eta\t%.10f\n", eta))
cat(sprintf("lasso_lambda\t%.10f\n", la$lambda))
cat(sprintf("lasso_n_selected\t%d\n", la$nz))
cat(sprintf("factor_lambda\t%.10f\n", FACTOR_LAMBDA))
for (j in seq_len(ncol(E3)))
  cat(sprintf("ssr_%s\t%.10f\n", colnames(E3)[j], ssr[j]))
cat(sprintf("statistic_3\t%.10f\n", ts3))
cat(sprintf("att_3\t%.10f\n", adj(cf3, y, WEIGHT, POST)))
cat(sprintf("p_value_3\t%.10f\n", p3))
cat(sprintf("crit05_3\t%.10f\n", quantile(draws, 0.95, names = FALSE)))
cat(sprintf("crit10_3\t%.10f\n", quantile(draws, 0.90, names = FALSE)))
for (m in c(0, 4, 8, 12))
  cat(sprintf("att_3_skip%d\t%.10f\n", m,
              adj(cf3, y, WEIGHT, (T0 + 1 + m):TT)))
for (j in seq_len(ncol(E3)))
  cat(sprintf("weight\t%s\t%.10f\n", colnames(E3)[j], w3[j]))
## the three expert paths, so the port is comparable path by path and not only
## through the ensemble
for (j in seq_len(ncol(E3)))
  for (t in seq_len(TT))
    cat(sprintf("path\t%s\t%d\t%.10f\n", colnames(E3)[j], t, E3[t, j]))

cat("== SESSION INFO ==\n")
print(sessionInfo())
