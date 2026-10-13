#!/usr/bin/env Rscript
# Botosaru & Ferman (2019), "On the role of covariates in the synthetic control
# method", Econometrics Journal 22(2), 117-130 -- the R-side reference for the
# two specifications their replication package estimates on the German
# reunification panel of Abadie, Diamond & Hainmueller (2015).
#
# The package ships a single Stata do-file ("Do-file - Covariates in SC
# method.do") with two `synth` calls:
#
#   1. The ADH (2015) specification -- six predictors (GDP per capita, trade
#      openness, inflation, industry share, schooling, investment rate) with the
#      predictor weights ADH report, passed as customV:
#      (0.442, 0.134, 0.072, 0.001, 0.107, 0.245).
#
#   2. All 31 pre-treatment GDP lags as predictors, with V the identity. Since
#      the predictors are then the pre-treatment outcomes themselves and V is
#      flat, this is the plain outcome-only convex fit on 1960-1990, which is
#      what mlsynth's VanillaSC solves.
#
# Table 1 of the paper contrasts column 1 (West Germany's own predictor means)
# with column 4 (the same means for the synthetic control of specification 2,
# which never saw a covariate). The do-file computes those two columns; its
# comment records that columns 2 and 3 are taken from ADH (2015) directly.
#
# The predictor windows are the do-file's: 1981-1990 for GDP, trade, inflation
# and industry; 1980 and 1985 for schooling, which is observed every five years;
# 1980 alone for the investment rate.
#
# Output: one "key=value" line per quantity, for a Python harness to parse.
#
# Usage:
#   Rscript benchmarks/R/botosaru_ferman_covariates.R [basedata/repgermany.csv]

suppressWarnings(suppressMessages(library(Synth)))

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "basedata/repgermany.csv"

d <- read.csv(data_path, stringsAsFactors = FALSE)
d$country <- as.character(d$country)

TREATED <- 7                       # West Germany, the do-file's trunit(7)
CONTROLS <- setdiff(sort(unique(d$index)), TREATED)
T0 <- 1990                         # the do-file's trperiod(1991)

emit <- function(key, value) cat(sprintf("%s=%.10g\n", key, value))

# ---------------------------------------------------------------------------
# Specification 2: all pre-treatment GDP lags, V = I.
# ---------------------------------------------------------------------------
lags <- lapply(1960:T0, function(y) list("gdp", y, "mean"))
prep_nocov <- dataprep(
  foo = d, dependent = "gdp", unit.variable = "index",
  unit.names.variable = "country", time.variable = "year",
  special.predictors = lags,
  treatment.identifier = TREATED, controls.identifier = CONTROLS,
  time.predictors.prior = 1960:T0,
  time.optimize.ssr = 1961:T0,     # the do-file's mspeperiod(1961(1)1990)
  time.plot = 1960:2003)
fit_nocov <- synth(data.prep.obj = prep_nocov,
                   custom.v = rep(1, length(lags)), verbose = FALSE)
w_nocov <- as.numeric(fit_nocov$solution.w)
names(w_nocov) <- d$country[match(CONTROLS, d$index)]

for (nm in names(w_nocov)) emit(paste0("w_nocov_", gsub("[^A-Za-z]", "", nm)), w_nocov[nm])
emit("w_nocov_sum", sum(w_nocov))
emit("w_nocov_nonzero", sum(w_nocov > 1e-6))

# ---------------------------------------------------------------------------
# Specification 1: the ADH (2015) predictors at ADH's own V.
# ---------------------------------------------------------------------------
prep_adh <- dataprep(
  foo = d, dependent = "gdp", unit.variable = "index",
  unit.names.variable = "country", time.variable = "year",
  special.predictors = list(
    list("gdp", 1981:1990, "mean"),
    list("trade", 1981:1990, "mean"),
    list("infrate", 1981:1990, "mean"),
    list("industry", 1981:1990, "mean"),
    list("schooling", c(1980, 1985), "mean"),
    list("invest80", 1980, "mean")),
  treatment.identifier = TREATED, controls.identifier = CONTROLS,
  time.predictors.prior = 1981:1990,
  time.optimize.ssr = 1960:1989,   # the do-file's mspeperiod(1960(1)1989)
  time.plot = 1960:2003)
fit_adh <- synth(data.prep.obj = prep_adh,
                 custom.v = c(0.442, 0.134, 0.072, 0.001, 0.107, 0.245),
                 verbose = FALSE)
w_adh <- as.numeric(fit_adh$solution.w)
names(w_adh) <- d$country[match(CONTROLS, d$index)]
for (nm in names(w_adh)) emit(paste0("w_adh_", gsub("[^A-Za-z]", "", nm)), w_adh[nm])

# ---------------------------------------------------------------------------
# Table 1, columns 1 and 4.
# ---------------------------------------------------------------------------
windows <- list(gdp = 1981:1990, trade = 1981:1990, infrate = 1981:1990,
                industry = 1981:1990, schooling = c(1980, 1985),
                invest80 = 1980)
for (v in names(windows)) {
  sub <- d[d$year %in% windows[[v]], ]
  means <- tapply(sub[[v]], sub$index, function(x) mean(x, na.rm = TRUE))
  emit(paste0("col1_", v), means[as.character(TREATED)])
  emit(paste0("col4_", v), sum(w_nocov * means[as.character(CONTROLS)]))
}

# Pre-period fit of specification 2, and the gap the figure shows.
Y <- prep_nocov$Y0plot %*% w_nocov
obs <- as.numeric(prep_nocov$Y1plot)
pre <- 1:(T0 - 1960 + 1)
emit("rmspe_pre_nocov", sqrt(mean((obs[pre] - Y[pre])^2)))
emit("gap_1990_nocov", obs[T0 - 1960 + 1] - Y[T0 - 1960 + 1])
emit("gap_2003_nocov", obs[length(obs)] - Y[length(Y)])

# ---------------------------------------------------------------------------
# The unweighted variant of specification 2.
#
# Synth divides every predictor row by its cross-unit standard deviation before
# applying V (`divisor <- sqrt(apply(big.dataframe, 1, var))` in synth()), so
# customV(1, ..., 1) is V = I on standardized predictors, which on the raw scale
# weights each pre-treatment year by the reciprocal of its cross-country GDP
# variance. Because that variance grows with the level of GDP, later years carry
# less weight than earlier ones, and the specification is not the plain
# outcome-only convex fit.
#
# That plain fit -- minimise ||y1 - Y0 w||^2 over the simplex on 1960-1990 -- is
# what mlsynth's VanillaSC solves, so it is computed here directly with
# quadprog, independently of Synth, to give that estimator a reference.
# ---------------------------------------------------------------------------
if (requireNamespace("quadprog", quietly = TRUE)) {
  Y <- prep_nocov$Y0plot[1:(T0 - 1960 + 1), , drop = FALSE]
  y1 <- as.numeric(prep_nocov$Y1plot)[1:(T0 - 1960 + 1)]
  J <- ncol(Y)
  Dmat <- t(Y) %*% Y
  Dmat <- Dmat + diag(1e-8 * mean(diag(Dmat)), J)      # strict positive definiteness
  Amat <- cbind(rep(1, J), diag(J))                    # sum(w) = 1, w >= 0
  sol <- quadprog::solve.QP(Dmat, t(Y) %*% y1, Amat, c(1, rep(0, J)), meq = 1)
  w_flat <- pmax(sol$solution, 0)
  w_flat <- w_flat / sum(w_flat)
  names(w_flat) <- names(w_nocov)
  for (nm in names(w_flat))
    emit(paste0("w_flat_", gsub("[^A-Za-z]", "", nm)), w_flat[nm])
  emit("w_flat_nonzero", sum(w_flat > 1e-6))
  emit("rmspe_pre_flat", sqrt(mean((y1 - Y %*% w_flat)^2)))
  for (v in names(windows)) {
    sub <- d[d$year %in% windows[[v]], ]
    means <- tapply(sub[[v]], sub$index, function(x) mean(x, na.rm = TRUE))
    emit(paste0("col4flat_", v), sum(w_flat * means[as.character(CONTROLS)]))
  }
}
