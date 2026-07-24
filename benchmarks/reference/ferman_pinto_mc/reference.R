# Ferman & Pinto (2021), "Synthetic controls with imperfect pretreatment fit",
# Quantitative Economics 12:1197-1221 -- the two synthetic-control estimators of
# the Monte Carlo, VERBATIM from the authors' replication code (_aux.R):
#
#   synth_control_est        -- original SC: donor weights >= 0 summing to 1
#                               (a quadprog QP).  == mlsynth VanillaSC.
#   synth_control_est_demean -- demeaned SC: the same simplex with a FREE
#                               intercept.  == mlsynth TSSC's MSCa variant.
#
# This is the reference for the value-for-value golden tie: Python simulates a
# handful of panels from the calibrated MC DGP, writes them as JSON, and this
# script solves the authors' two QPs on the identical panels. The benchmark then
# checks that mlsynth's estimators reproduce these weights and effects.
suppressMessages(library(quadprog))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
panels_path <- args[1]                      # JSON written by the Python case

# --- authors' _aux.R, unchanged -------------------------------------------
synth_control_est <- function(y_before, y_after) {
  y <- y_before[, 1]; X <- y_before[, -1, drop = FALSE]
  Dmat <- t(X) %*% X; dvec <- t(X) %*% y
  Amat <- t(rbind(rep(1, ncol(X)), diag(ncol(X))))
  bvec <- c(1, rep(0, ncol(X)))
  m <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
  w <- m$solution
  list(w = w, effects = y_after %*% c(1, -w))
}

synth_control_est_demean <- function(y_before, y_after) {
  y <- y_before[, 1]; X <- cbind(1, y_before[, -1, drop = FALSE])
  Dmat <- t(X) %*% X; dvec <- t(X) %*% y
  Amat <- t(rbind(c(0, rep(1, ncol(X) - 1)), cbind(0, diag(ncol(X) - 1))))
  bvec <- c(1, rep(0, ncol(X) - 1))
  m <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
  w <- m$solution[-1]; intercept <- m$solution[1]
  list(w = w, intercept = intercept,
       effects = -intercept + y_after %*% c(1, -w))
}
# --------------------------------------------------------------------------

panels <- fromJSON(panels_path, simplifyVector = FALSE)
out <- lapply(panels, function(p) {
  pre  <- do.call(rbind, lapply(p$pre,  function(r) as.numeric(unlist(r))))
  post <- do.call(rbind, lapply(p$post, function(r) as.numeric(unlist(r))))
  sc <- synth_control_est(pre, post)
  dm <- synth_control_est_demean(pre, post)
  list(
    sc_w = round(as.numeric(sc$w), 10),
    sc_effect = round(mean(sc$effects), 10),
    dm_w = round(as.numeric(dm$w), 10),
    dm_intercept = round(as.numeric(dm$intercept), 10),
    dm_effect = round(mean(dm$effects), 10)
  )
})
cat(toJSON(out, auto_unbox = TRUE, digits = 12))
