# Reference R packages for mlsynth cross-validation benchmarks.
# Run once:  Rscript benchmarks/R/requirements.R
# rstan compiles the BFSC appendix Stan program live for `bfsc_prop99`; it is
# heavy to build (StanHeaders/RcppEigen), so if it is absent that one live
# cross-check simply skips.
# gmm + dplyr + tidyr back the DR-proximal Brazil cross-check
# (`dr_proximal_brazil`); its harness uses only those, not the full tidyverse.
pkgs <- c("Synth", "synthdid", "did", "rstan", "gmm", "dplyr", "tidyr")
inst <- rownames(installed.packages())
for (p in pkgs) if (!(p %in% inst)) install.packages(p, repos = "https://cloud.r-project.org")
cat("reference packages present:", paste(intersect(pkgs, rownames(installed.packages())), collapse=", "), "\n")
