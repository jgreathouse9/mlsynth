#!/usr/bin/env Rscript
# Reference run for the `mvbbsc_germany` benchmark case.
#
# Runs the authors' own bsynth package (Martinez & Vives-i-Bastida 2024,
# arXiv:2206.01779, github.com/ignacio82/bsynth) on the West-German reunification
# panel and emits the posterior summaries the Python case cross-validates against:
# the pre-1990 in-sample RMSE, the mean post-1990 ATT, and the mean 95% credible-
# band width in the pre- and post-treatment windows. These are the genuine bsynth
# outputs mlsynth's MVBBSC (a clean-room NumPyro port of the same model) is pinned
# against, not numbers transcribed from a table.
#
# bsynth with predictor_match = FALSE is `model1`: outcome-only, no covariates and
# no Gaussian-process term -- exactly the generative model MVBBSC ports (uniform-
# Dirichlet simplex weights, HalfNormal scale, Gaussian likelihood on the
# pre-period-standardized series), ci_width = 0.95, four chains of 1000 warm-up +
# 1000 draws. This is the stdout-emitting twin of benchmarks/R/mvbbsc_bsynth_ref.R
# (same fit; this one prints the generate.py `== REFERENCE VALUES ==` block).
#
# Install the pinned toolchain with benchmarks/R/install_bsynth.sh (rstan from
# apt, bsynth + vizdraws from pinned GitHub SHAs). Needs R + rstan, so it is NOT
# run in CI; the captured bundle is committed so the cross-check is reproducible.
#
# Run from the repository root:
#   Rscript benchmarks/reference/mvbbsc_germany/reference.R
suppressMessages({library(bsynth); library(dplyr)})

d <- read.csv("basedata/german_reunification.csv")
# read.csv parses the "True"/"False" flag as character, not logical; go through
# as.logical so the treated indicator is a clean 1/0 (bsynth rejects non-binary).
d$treat <- as.integer(as.logical(d$Reunification))

gs <- bayesianSynth$new(data = d, time = year, id = country, treated = treat,
                        outcome = gdp, ci_width = 0.95, predictor_match = FALSE)
gs$fit(cores = 4)

pd   <- as.data.frame(gs$plotData)   # year, gdp, y_synth, LB, UB, tau, tau_LB, tau_UB
pre  <- pd$year < 1990
post <- pd$year >= 1990
rmse      <- sqrt(mean((pd$gdp[pre] - pd$y_synth[pre])^2))
att       <- mean(pd$gdp[post] - pd$y_synth[post])
band_pre  <- mean((pd$UB - pd$LB)[pre])
band_post <- mean((pd$UB - pd$LB)[post])

cat("== REFERENCE VALUES ==\n")
cat(sprintf("pre_rmse\t%.6f\n", rmse))
cat(sprintf("mean_att\t%.6f\n", att))
cat(sprintf("band_width_pre\t%.6f\n", band_pre))
cat(sprintf("band_width_post\t%.6f\n", band_post))
cat("== SESSION INFO ==\n")
print(sessionInfo())
