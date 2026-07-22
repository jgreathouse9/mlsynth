#!/usr/bin/env Rscript
# Reference oracle for benchmarks/cases/mvbbsc_germany.py -- the authors' own
# bsynth package (Martinez & Vives-i-Bastida 2024, arXiv:2206.01779).
#
# mlsynth's MVBBSC is a clean-room NumPyro port of the bsynth `model1` (outcome-
# only, predictor_match = FALSE: uniform-Dirichlet simplex weights, HalfNormal
# scale, Gaussian likelihood on the pre-period-standardized series). This script
# runs the reference bsynth on the German reunification panel and dumps the
# posterior counterfactual, credible band, ATT, and in-sample RMSE so the port
# can be cross-checked cell-for-cell.
#
# Install (pinned): benchmarks/R/install_bsynth.sh. Needs R + rstan, so this is
# NOT run in CI; it is committed so the cross-check is reproducible on demand.
#
#   Rscript benchmarks/R/mvbbsc_bsynth_ref.R \
#     basedata/german_reunification.csv benchmarks/R/mvbbsc_bsynth_ref.out
suppressMessages({library(bsynth); library(dplyr)})

args   <- commandArgs(trailingOnly = TRUE)
infile <- if (length(args) >= 1) args[1] else "basedata/german_reunification.csv"
outfile <- if (length(args) >= 2) args[2] else "benchmarks/R/mvbbsc_bsynth_ref.out"

d <- read.csv(infile)
d$treat <- as.integer(d$Reunification)

# bayesianSynth with predictor_match = FALSE is model1: outcome-only, no
# covariates, no Gaussian-process term -- exactly the model MVBBSC ports.
gs <- bayesianSynth$new(data = d, time = year, id = country, treated = treat,
                        outcome = gdp, ci_width = 0.95, predictor_match = FALSE)
gs$fit(cores = 4)

pd <- as.data.frame(gs$plotData)     # year, gdp, y_synth, LB, UB, tau, tau_LB, tau_UB
pre  <- pd$year < 1990
post <- pd$year >= 1990
rmse <- sqrt(mean((pd$gdp[pre] - pd$y_synth[pre])^2))
att  <- mean(pd$gdp[post] - pd$y_synth[post])
band_pre  <- mean((pd$UB - pd$LB)[pre])
band_post <- mean((pd$UB - pd$LB)[post])

# machine-parseable dump (key=value), plus the full path as a CSV alongside.
lines <- c(
  sprintf("pre_rmse=%.4f", rmse),
  sprintf("mean_att=%.4f", att),
  sprintf("band_width_pre=%.4f", band_pre),
  sprintf("band_width_post=%.4f", band_post),
  sprintf("n_periods=%d", nrow(pd))
)
writeLines(lines, outfile)
write.csv(pd, sub("\\.out$", "_path.csv", outfile), row.names = FALSE)
cat(paste(lines, collapse = "\n"), "\n")
cat("wrote", outfile, "and", sub("\\.out$", "_path.csv", outfile), "\n")
