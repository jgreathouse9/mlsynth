# Reference run for the `mscmt_basque` benchmark case.
#
# Runs the authors' MSCMT R package (Becker & Klossner 2018, mabe0033/MSCMT) on
# the Abadie-Gardeazabal (2003) Basque study, using the predictor specification
# from the package vignette "Working with package MSCMT" -- the thirteen-
# predictor model fitting per-capita GDP over the optimisation window 1960-1969.
# It prints the synthetic-control donor weights and the average post-treatment
# gap (1970-1990, MSCMT's `did`/`average.post` range). These are the genuine
# package outputs the Python case pins against, not transcribed vignette
# constants.
#
# The optimisation uses outer.optim="DEoptim" with seed=42 (deterministic).
# MSCMT installs from github.com/mabe0033/MSCMT with all hard Imports built from
# source (lpSolve, lpSolveAPI, Rglpk, Rdpack, ggplot2, rlang); see
# benchmarks/R/install_mscmt.sh.
#
# The data (basedata/basque_mscmt.csv) is the MSCMT-transformed `basque` panel:
# schooling rescaled to per-unit percentage shares with
# school.higher = school.high + school.post.high (exactly the vignette's
# pre-processing). MSCMT consumes data as a list of (time x unit) matrices, one
# per variable; we assemble that list from the long CSV below.
#
# Run from the repository root:  Rscript benchmarks/reference/mscmt_basque/reference.R
suppressMessages(library(MSCMT))

d <- read.csv("basedata/basque_mscmt.csv", check.names = FALSE,
              stringsAsFactors = FALSE)

# Assemble MSCMT's 'list' data object: one (time x unit) matrix per variable.
years <- sort(unique(d$year))
units <- sort(unique(d$regionname))
vars  <- setdiff(colnames(d), c("year", "regionname"))
mklist <- function(v) {
  M <- matrix(NA_real_, length(years), length(units),
              dimnames = list(as.character(years), units))
  for (i in seq_len(nrow(d))) M[as.character(d$year[i]), d$regionname[i]] <- d[[v]][i]
  M
}
Basque <- setNames(lapply(vars, mklist), vars)

treatment.identifier <- "Basque Country (Pais Vasco)"
controls.identifier  <- setdiff(units, c(treatment.identifier, "Spain (Espana)"))

# Vignette specification reproducing Abadie & Gardeazabal (2003).
times.dep  <- cbind("gdpcap"                = c(1960, 1969))
times.pred <- cbind("school.illit"          = c(1964, 1969),
                    "school.prim"           = c(1964, 1969),
                    "school.med"            = c(1964, 1969),
                    "school.higher"         = c(1964, 1969),
                    "invest"                = c(1964, 1969),
                    "gdpcap"                = c(1960, 1969),
                    "sec.agriculture"       = c(1961, 1969),
                    "sec.energy"            = c(1961, 1969),
                    "sec.industry"          = c(1961, 1969),
                    "sec.construction"      = c(1961, 1969),
                    "sec.services.venta"    = c(1961, 1969),
                    "sec.services.nonventa" = c(1961, 1969),
                    "popdens"               = c(1969, 1969))
agg.fns <- rep("mean", ncol(times.pred))

res <- suppressWarnings(
  mscmt(Basque, treatment.identifier, controls.identifier,
        times.dep, times.pred, agg.fns,
        outer.optim = "DEoptim", seed = 42, verbose = FALSE))

w <- res$w
gap <- window(res$gaps$gdpcap, start = 1970, end = 1990)   # MSCMT 'did' range

cat("== REFERENCE VALUES ==\n")
cat(sprintf("weight\tCataluna\t%.6f\n",              w["Cataluna"]))
cat(sprintf("weight\tBaleares (Islas)\t%.6f\n",      w["Baleares (Islas)"]))
cat(sprintf("weight\tMadrid (Comunidad De)\t%.6f\n", w["Madrid (Comunidad De)"]))
cat(sprintf("avg_post_gap\t%.6f\n",                  mean(gap)))
cat("== SESSION INFO ==\n")
print(sessionInfo())
