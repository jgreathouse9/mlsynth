# Reference generator for the synth_jhai_prop99 cross-validation benchmark.
#
# Runs Jens Hainmueller's R Synth (j-hai/Synth 1.2.0) on the California
# Proposition 99 panel under the canonical ADH (2010) predictor spec, and dumps
# the synthetic control (donor weights, per-period synthetic + gap, ATT,
# pre-RMSPE) and its split-conformal + parametric prediction bands from
# synth_inference(). mlsynth's VanillaSC(backend="mscmt") and
# inference="conformal_split" are cross-checked against these in
# benchmarks/cases/synth_jhai_prop99.py.
#
# Install route (CRAN firewalled; git clone the mirror -- see install_propsdid.sh):
#   apt-get install -y r-base r-base-dev r-cran-kernlab r-cran-optimx r-cran-rgenoud r-cran-numderiv
#   git clone --depth 1 https://github.com/j-hai/Synth && R CMD INSTALL Synth
#
# Run:  Rscript benchmarks/R/synth_jhai_prop99.R <cali_csv> <outdir>
suppressMessages(library(Synth))
cat("Synth (j-hai) version:", as.character(packageVersion("Synth")), "\n")
args <- commandArgs(trailingOnly = TRUE)
csv    <- ifelse(length(args) >= 1, args[1], "basedata/augmented_cali_long.csv")
outdir <- ifelse(length(args) >= 2, args[2], "benchmarks/reference/synth_jhai_prop99")

d <- read.csv(csv)                                  # pct15-24 -> pct15.24
ca <- 5L                                            # California's stateno
ctrl <- setdiff(sort(unique(d$stateno)), ca)

dp <- dataprep(foo = d,
  predictors = c("loginc", "p_cig", "pct15.24"), predictors.op = "mean",
  time.predictors.prior = 1980:1988,
  special.predictors = list(list("pc_beer", 1984:1988, "mean"),
                            list("cigsale", 1975, "mean"),
                            list("cigsale", 1980, "mean"),
                            list("cigsale", 1988, "mean")),
  dependent = "cigsale", unit.variable = "stateno", unit.names.variable = "state",
  time.variable = "year", treatment.identifier = ca, controls.identifier = ctrl,
  time.optimize.ssr = 1970:1988, time.plot = 1970:2000)
so <- synth(dp, verbose = FALSE)

w  <- as.numeric(so$solution.w)
nm <- dp$names.and.numbers$unit.names[match(rownames(so$solution.w), dp$names.and.numbers$unit.numbers)]
wdf <- data.frame(unit = as.character(nm), w = w)
wdf <- wdf[order(-wdf$w), ]
write.csv(wdf[wdf$w > 1e-5, ], file.path(outdir, "weights.csv"), row.names = FALSE)

years <- as.numeric(rownames(dp$Y1plot))
treated <- as.numeric(dp$Y1plot); synth_path <- as.numeric(dp$Y0plot %*% so$solution.w)
gap <- treated - synth_path
post <- years >= 1989
ci_c <- synth_inference(so, dp, method = "conformal", alpha = 0.05)
ci_p <- synth_inference(so, dp, method = "parametric", alpha = 0.05)
write.csv(data.frame(year = years, treated = treated, synth = synth_path, gap = gap,
  conf_lo = ci_c$intervals[, 1], conf_hi = ci_c$intervals[, 2],
  param_lo = ci_p$intervals[, 1], param_hi = ci_p$intervals[, 2]),
  file.path(outdir, "series.csv"), row.names = FALSE)

cat(sprintf("ATT(1989-2000)=%.6f  pre_RMSPE=%.6f  gap2000=%.6f  conformal_q=%.6f\n",
  mean(gap[post]), sqrt(mean(gap[!post]^2)), gap[years == 2000], ci_c$conformal_q))
cat("top weights:\n"); print(head(wdf[wdf$w > 1e-3, ], 6))
