# Reference R packages for mlsynth cross-validation benchmarks.
# Run once:  Rscript benchmarks/R/requirements.R
pkgs <- c("Synth", "synthdid", "did")
inst <- rownames(installed.packages())
for (p in pkgs) if (!(p %in% inst)) install.packages(p, repos = "https://cloud.r-project.org")
cat("reference packages present:", paste(intersect(pkgs, rownames(installed.packages())), collapse=", "), "\n")
