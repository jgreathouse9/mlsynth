# Reference generator for benchmarks/cases/fspda_table1.py
#
# Shi & Huang (2023) Table 1: the dense ("nonsparse") Monte Carlo, run with the
# authors' own estimators so that mlsynth's cells can be compared against two
# things at once -- the table as published, and the same table as their code
# produces it at this benchmark's replication count.
#
# Everything between the VERBATIM markers is copied unchanged from
# zhentaoshi/fsPDA at 5f4542c: simulation/nonsparse/{FS.R, lasso.BIC.R} and the
# DGP of FS.simulation.dense.R. The loadings are their committed loading.RData,
# written out beside this script as loadings.tsv so the Python case draws its
# panels against the same design matrix.
#
# scm.R is not run. Run.simulation.dense.R calls it, but Table 1 has no SCM
# column, and CVXR is not installable in this environment (see the note in
# benchmarks/reference/fspda_dense_mc/reference.R).
#
# The seven rejection columns D1-D7 come from one fit per replication. The
# estimators use pre-treatment data only, so the post-period prediction error
# d.hat does not depend on the treatment effect; their FS.simulation.dense
# forms the effect for each DGP as Delta_j + d.hat and tests that. D1 is the
# fit's own phi (Delta = 0); D2-D7 are the six columns of DGP.Delta.

suppressMessages(library(glmnet))
suppressMessages(library(MASS))
suppressMessages(library(foreach))

ARGS <- commandArgs(trailingOnly = TRUE)
SRC <- if (length(ARGS) >= 1) ARGS[1] else Sys.getenv("FSPDA_SRC", "/workspace/zhentaoshi/fspda")
OUT <- if (length(ARGS) >= 2) ARGS[2] else "benchmarks/reference/fspda_table1"
M   <- as.integer(Sys.getenv("FSPDA_TABLE1_M", "2000"))

NONSPARSE <- file.path(SRC, "simulation", "nonsparse")
load(file.path(NONSPARSE, "loading.RData"))     # -> Lambda, 101 x 4

# ========================= VERBATIM: nonsparse/FS.R =========================
source(file.path(NONSPARSE, "FS.R"))
# ====================== VERBATIM: nonsparse/lasso.BIC.R ======================
source(file.path(NONSPARSE, "lasso.BIC.R"))
# ================ VERBATIM: nonsparse/FS.simulation.dense.R =================
# DGP.Lambda, DGP.Delta, lrvar and beta0.dense; FS.simulation.dense itself is
# not called because it also runs scm().
source(file.path(NONSPARSE, "FS.simulation.dense.R"))
# ============================ END VERBATIM ==================================

# The (T1, T2) grid and the per-DGP truncation lags, from Run.simulation.dense.R.
T.GRID <- c(50, 100, 200)
H.IID <- cbind(c(1,1,2,1,1,2,2), c(2,1,3,1,1,3,3), c(2,1,4,1,1,4,4))
H.NND <- cbind(c(1,1,3,1,1,3,3), c(2,1,4,1,1,4,4), c(4,1,7,1,1,7,7))
ARMA.NND <- list(list(ar = 0.9), list(ma = c(0.8, 0.4)), list(ar = 0.5, ma = 0.5))

# The panel, exactly as the body of FS.simulation.dense builds it.
make.panel <- function(T1, T2, ARMA, sigma.eta = 0.5, sigma.u = 1, burn = 1000) {
  Tn <- T1 + T2
  N <- nrow(Lambda) - 1; K <- ncol(Lambda)
  eta <- matrix(rnorm(Tn * (N + 1), sd = sigma.eta), nrow = Tn, ncol = N + 1)
  if (is.null(ARMA)) {
    f <- foreach(k = 1:K, .combine = cbind) %do% {rnorm(Tn, sd = k * sigma.u)}
  } else {
    f <- cbind(rnorm(Tn, sd = sigma.u),
               foreach(k = 1:length(ARMA), .combine = cbind) %do% {
                 as.vector(arima.sim(model = ARMA[[k]], n = Tn, n.start = burn,
                                     sd = sigma.u))})
  }
  Y <- f %*% t(Lambda) + eta
  list(y = Y[, 1], Y = Y[, -1])
}

one.rep <- function(T1, ARMA, h, seed) {
  T2 <- T1
  # DGP.Delta sets its own seed, so the panel's is set after it, not before.
  Delta <- DGP.Delta(T2 = T2, seed = seed + 500000L)
  set.seed(seed)
  p <- make.panel(T1, T2, ARMA)
  y <- p$y; Y <- p$Y
  out <- list()
  for (nm in c("fs", "lasso")) {
    fit <- if (nm == "fs") FS(y = y, Y = Y, T1 = T1, H = 1, h = h[1])
           else lasso.BIC(y = y, Y = Y, T1 = T1, H = 2, h = h[1])
    d <- fit$d
    phi <- fit$phi                                   # D1: Delta = 0
    for (j in 1:ncol(Delta)) {                       # D2-D7
      eff <- Delta[, j] + d
      Z <- mean(eff) / sqrt(lrvar(eff, lag.max = h[j + 1]) / T2)
      phi <- c(phi, as.numeric(abs(Z) > qnorm(0.975)))
    }
    out[[nm]] <- list(R = fit$R, rmse = sqrt(mean(d^2)), phi = phi)
  }
  out
}

emit <- function(k, v) cat(k, "\t", format(v, digits = 10), "\n", sep = "")

cat("fsPDA Table 1 reference --", M, "replications per cell\n")
cat("R", R.version.string, "| glmnet", as.character(packageVersion("glmnet")), "\n\n")

cat("== REFERENCE VALUES ==\n")
emit("M", M)
emit("n_donors", nrow(Lambda) - 1)
emit("H_fs", 1); emit("H_lasso", 2)
emit("sigma_eta", 0.5)
emit("irrelevant_loading_halfwidth", max(abs(Lambda[6:nrow(Lambda), ])))

base <- 20250812L
for (dgp in c("iid", "nnd")) {
  ARMA <- if (dgp == "iid") NULL else ARMA.NND
  H <- if (dgp == "iid") H.IID else H.NND
  for (i in seq_along(T.GRID)) {
    T1 <- T.GRID[i]; h <- H[, i]
    acc <- list(fs = list(R = numeric(M), rmse = numeric(M), phi = matrix(0, M, 7)),
                lasso = list(R = numeric(M), rmse = numeric(M), phi = matrix(0, M, 7)))
    for (m in 1:M) {
      r <- one.rep(T1, ARMA, h, base + m)
      for (nm in c("fs", "lasso")) {
        acc[[nm]]$R[m] <- r[[nm]]$R
        acc[[nm]]$rmse[m] <- r[[nm]]$rmse
        acc[[nm]]$phi[m, ] <- r[[nm]]$phi
      }
    }
    for (nm in c("fs", "lasso")) {
      tag <- paste(nm, dgp, T1, sep = "_")
      emit(paste0(tag, "_nsel"), median(acc[[nm]]$R))
      emit(paste0(tag, "_rmspe"), mean(acc[[nm]]$rmse))
      for (j in 1:7) emit(paste0(tag, "_D", j), mean(acc[[nm]]$phi[, j]))
    }
    cat("# done", dgp, T1, "\n")
  }
}

# The loadings ride with the bundle: the Python case draws its panels against
# the same design matrix their table was produced on.
write.table(format(Lambda, digits = 17), file.path(OUT, "loadings.tsv"),
            sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
