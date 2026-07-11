#!/usr/bin/env Rscript
# Reference capture: synthdid (Arkhangelsky, Athey, Hirshberg, Imbens & Wager
# 2021) on the Abadie-Diamond-Hainmueller Proposition 99 smoking panel.
#
# Reads the exact panel mlsynth's benchmark uses (basedata/smoking_data.csv:
# 39 states x 31 years, 1970-2000, California treated from 1989), builds the
# balanced outcome matrix, and records the Synthetic DiD point estimate (plus
# the DiD and pure-SC estimates the same package produces) so the mlsynth case
# can cross-validate against the authors' own R.
#
# Reproduce:  Rscript benchmarks/reference/sdid_prop99_synthdid/reference.R
suppressMessages(library(synthdid))

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[[1]] else
  file.path("basedata", "smoking_data.csv")

d <- read.csv(data_path, check.names = FALSE)

# synthdid requires the balanced outcome matrix with the treated unit as the
# last row and the pre-treatment periods as the leading columns. California is
# the single treated unit (from 1989); order the 38 controls first, then CA,
# with years ascending, so N0 = 38 controls and T0 = 19 pre-periods (1970-1988).
years   <- sort(unique(d$year))
treated <- "California"
controls <- sort(setdiff(unique(d$state), treated))
ordered_states <- c(controls, treated)
Y <- matrix(NA_real_, length(ordered_states), length(years),
            dimnames = list(ordered_states, as.character(years)))
for (i in seq_len(nrow(d)))
  Y[as.character(d$state[i]), as.character(d$year[i])] <- d$cigsale[i]
stopifnot(!anyNA(Y))
N0 <- length(controls)
T0 <- sum(years < 1989)
setup <- list(N0 = N0, T0 = T0)

sdid <- as.numeric(synthdid_estimate(Y, N0, T0))
did  <- as.numeric(did_estimate(Y, N0, T0))
sc   <- as.numeric(sc_estimate(Y, N0, T0))

cat(sprintf("SDID ATT = %.6f\n", sdid))
cat(sprintf("DID  ATT = %.6f\n", did))
cat(sprintf("SC   ATT = %.6f\n", sc))

json <- sprintf(
  paste0('{\n  "values": {\n',
         '    "sdid_att": %.6f,\n',
         '    "did_att": %.6f,\n',
         '    "sc_att": %.6f,\n',
         '    "n0": %d,\n    "t0": %d\n  }\n}\n'),
  sdid, did, sc, setup$N0, setup$T0)
out <- if (length(args) >= 2) args[[2]] else
  "benchmarks/reference/sdid_prop99_synthdid/reference.json"
writeLines(json, out)
cat("wrote", out, "\n")
