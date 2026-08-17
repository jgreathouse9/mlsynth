# Reference generator for benchmarks/cases/dr_proximal_scenarios.py
#
# Qiu, Shi, Miao, Dobriban & Tchetgen Tchetgen (2024), "Doubly robust proximal
# synthetic controls", Biometrics 80(2) ujae055. Their simulation/ directory
# holds eight scenarios, each a generate_data.R carrying the DGP and eight
# estimators, plus a simulation.R that preprocesses the panel before estimating.
#
# The eight estimators are the same everywhere: correct.DR, correct.h,
# correct.q, mis.h.DR, mis.q.DR, mis.h, mis.q, OLS. The preprocessing is not,
# and it is part of what the estimators see, so each scenario's block is copied
# verbatim below from its own simulation.R:
#
#   normal, nonstationary, short_post, small_T
#       raw-power quadratic expansion of W and Z, each column divided by its
#       largest absolute value.
#   detrend
#       the same, after regressing every proxy series on poly(t, 2) pooled and
#       subtracting the fitted trend.
#   just_identified
#       none. The estimators run on the raw W and Z.
#   over_identified
#       none, and the DGP itself is wider: nU = 3, nW = 5, nZ = 10.
#   select_donor
#       Synth picks which of the pooled columns act as W and which as Z, so the
#       proxy assignment is estimated instead of given.
#
# What this emits, per scenario and replication: the panel the estimators
# actually receive (Y and the preprocessed W, Z), and every estimator's
# phi.hat, SE, CI and convergence flag. Storing the preprocessed panel rather
# than the raw draw is deliberate -- it puts the whole of their preprocessing on
# the R side, so the Python case cannot disagree with it by porting it wrong.

suppressMessages({
  library(magrittr); library(dplyr); library(tidyr)
  library(mvtnorm); library(gmm); library(quadprog); library(Synth)
})

ARGS <- commandArgs(trailingOnly = TRUE)
SRC <- if (length(ARGS) >= 1) ARGS[1] else Sys.getenv("DRPSC_SRC",
          "/workspace/qiu-hongxiang-david/dr_proximal_sc")
OUT <- if (length(ARGS) >= 2) ARGS[2] else "benchmarks/reference/dr_proximal_scenarios"

SIM <- file.path(SRC, "simulation")
METHODS <- c("correct.DR", "correct.h", "correct.q",
             "mis.h.DR", "mis.q.DR", "mis.h", "mis.q", "OLS")

# Their smallest grid cell per scenario: Ts[1] = 500 everywhere except small_T,
# whose grid is c(80, 100). nU = 2 is nUs[1]; over_identified fixes its own.
CELL <- list(
  normal          = list(T = 500, nU = 2),
  detrend         = list(T = 500, nU = 2),
  nonstationary   = list(T = 500, nU = 2),
  just_identified = list(T = 500, nU = 2),
  over_identified = list(T = 500, nU = NA),
  select_donor    = list(T = 500, nU = 2),
  short_post      = list(T = 500, nU = 2),
  small_T         = list(T = 80,  nU = 2)
)
REPS <- 4

# ---- VERBATIM: the preprocessing block of each scenario's simulation.R -------
prep_poly <- function(data) {                      # normal / nonstationary / short_post / small_T
  data2 <- data
  data2$W <- poly(data2$W, 2, raw = TRUE, simple = TRUE)
  data2$W <- apply(data2$W, 2, function(x) x / max(abs(x)))
  data2$Z <- poly(data2$Z, 2, raw = TRUE, simple = TRUE)
  data2$Z <- apply(data2$Z, 2, function(x) x / max(abs(x)))
  data2$nU <- ncol(data2$W)
  data2
}

prep_detrend <- function(data) {                   # detrend
  nU <- data$nU
  d <- data.frame(cbind(data$t, data$W, data$Z))
  names(d) <- c("t", paste0("W", 1:nU), paste0("Z", 1:nU))
  d <- pivot_longer(d, cols = c(paste0("W", 1:nU), paste0("Z", 1:nU)),
                    names_to = "unit", values_to = "outcome")
  model <- lm(outcome ~ poly(t, 2), data = d)
  trend <- predict(model, newdata = data.frame(t = data$t))
  data$W <- data$W - trend
  data$Z <- data$Z - trend
  data$Y <- data$Y - trend
  prep_poly(data)
}

prep_none <- function(data) data                   # just_identified / over_identified

prep_select_donor <- function(data) {              # select_donor
  T0 <- data$T0; nU <- data$nU
  X <- cbind(data$W, data$Z)
  X1 <- as.matrix(data$Y[1:T0]); X0 <- X[1:T0, ]
  invisible(capture.output(
    synth.fit <- synth(X1 = X1, X0 = X0, Z1 = X1, Z0 = X0,
                       custom.v = rep(1, T0), verbose = FALSE)))
  w <- synth.fit$solution.w
  o <- order(w, decreasing = TRUE)
  data$W <- X[, o[1:nU]]
  data$Z <- X[, o[(nU + 1):(2 * nU)]]
  data
}
# ---- END VERBATIM -----------------------------------------------------------

PREP <- list(normal = prep_poly, nonstationary = prep_poly, short_post = prep_poly,
             small_T = prep_poly, detrend = prep_detrend,
             just_identified = prep_none, over_identified = prep_none,
             select_donor = prep_select_donor)

num <- function(x) {
  if (is.null(x) || length(x) == 0) return("nan")
  out <- sprintf("%.17g", as.numeric(x))
  out[is.na(as.numeric(x))] <- "nan"
  out
}
lines <- c(); panel.lines <- c("cell\tt\ty\tW\tZ")
emit <- function(k, v) lines <<- c(lines, paste(k, num(v), sep = "\t"))

emit("n_reps", REPS); emit("n_methods", length(METHODS))

for (scen in names(CELL)) {
  env <- new.env()
  sys.source(file.path(SIM, scen, "generate_data.R"), envir = env)
  cfg <- CELL[[scen]]
  prep <- PREP[[scen]]
  set.seed(1737209457)                      # their seed base, one stream per scenario

  emit(sprintf("true_ATE_%s", scen), get("true.ATE", envir = env))
  for (rep in seq_len(REPS)) {
    data <- if (is.na(cfg$nU)) env$generate.data(cfg$T) else env$generate.data(cfg$T, cfg$nU)
    d2 <- prep(data)
    tag <- sprintf("%s_r%d", scen, rep)

    Wm <- as.matrix(d2$W); Zm <- as.matrix(d2$Z)
    emit(sprintf("T_%s", tag), d2$T); emit(sprintf("T0_%s", tag), d2$T0)
    emit(sprintf("nW_%s", tag), ncol(Wm)); emit(sprintf("nZ_%s", tag), ncol(Zm))
    for (t in seq_len(d2$T)) {
      panel.lines <- c(panel.lines, paste(tag, t, num(d2$Y[t]),
                                          paste(num(Wm[t, ]), collapse = " "),
                                          paste(num(Zm[t, ]), collapse = " "),
                                          sep = "\t"))
    }
    for (m in METHODS) {
      r <- tryCatch(get(m, envir = env)(d2),
                    error = function(e) list(phi.hat = NA, SE = NA,
                                             CI = c(NA, NA), convergence = 1))
      key <- gsub("\\.", "", m)
      emit(sprintf("%s_phi_%s", key, tag), r$phi.hat)
      emit(sprintf("%s_se_%s", key, tag), r$SE)
      emit(sprintf("%s_conv_%s", key, tag), r$convergence)
    }
  }
  cat("done:", scen, "\n", file = stderr())
}

con <- gzfile(file.path(OUT, "panels.tsv.gz"), "wt")
writeLines(panel.lines, con); close(con)
cat(capture.output(sessionInfo()), sep = "\n")
cat("\n")
writeLines(c("== REFERENCE VALUES ==", lines))
