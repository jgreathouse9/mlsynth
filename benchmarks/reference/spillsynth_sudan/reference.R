#!/usr/bin/env Rscript
# Reference: the authors' spatial-autoregressive synthetic-control sampler
# (Sakaguchi & Tagawa 2026, The Econometrics Journal, doi:10.1093/ectj/utag006;
# replication archived at Zenodo doi:10.5281/zenodo.19066186) run on the 2011
# Sudan-secession panel. Their estimator ships as an Rcpp/RcppArmadillo package,
# not on CRAN, so point SAR_REF_DIR at the unpacked replication package (the dir
# containing R/ and src/). The DATA is the committed, nonproprietary basedata
# CSVs, so the only external input is the authors' code.
#
#   SAR_REF_DIR=/path/to/03-replication-package Rscript reference.R
#
# Writes reference.json (the pinned values the Python case compares against) and
# prints a human-readable dump (captured in reference.out). rho is weakly
# identified in this model (authors' reported ESS ~390 at 1e6 draws), so this
# uses a long run (M=1e5, burn=5e4) rather than a smoke setting.
suppressMessages({library(dplyr); library(tidyr); library(jsonlite)})

ref <- Sys.getenv("SAR_REF_DIR")
if (ref == "" || !dir.exists(file.path(ref, "R")))
  stop("set SAR_REF_DIR to the unpacked Sakaguchi-Tagawa replication package")
base <- Sys.getenv("MLSYNTH_BASEDATA",
                   normalizePath(file.path(dirname(sub("--file=", "",
                     grep("--file=", commandArgs(FALSE), value=TRUE)[1])), "..", "..", "..", "basedata")))

for (f in c("01_utils.R","02_utils_data_prep.R","10_sc_spillover.R","21_mcmc_alpha.R","22_mcmc_sar.R"))
  source(file.path(ref, "R", f), local = TRUE)
Rcpp::sourceCpp(file.path(ref, "src", "20_mcmc.cpp"))

panel <- read.csv(file.path(base, "sudan_panel.csv"), check.names = FALSE)
W <- as.matrix(read.csv(file.path(base, "sudan_W_matrix.csv"), row.names = 1, check.names = FALSE))
w <- as.matrix(read.csv(file.path(base, "sudan_w_vector.csv"), row.names = 1, check.names = FALSE))
panel$treatment <- ifelse(panel$country == "Sudan" & panel$year >= 2011, 1, 0)

fit <- sc_spillover(data = panel, treated_unit = "Sudan", w = w, W = W,
  treatment_dummy = "treatment", y = colnames(panel)[4], X = colnames(panel)[5:10],
  p_factors = 1, M = 100000L, burn = 50000L, seed = 20251022, step_rho = 0.02,
  unit_col = "country", time_col = "year")

post_years <- sort(unique(panel$year[panel$year >= 2011]))     # 2011..2015
sud <- panel %>% filter(country == "Sudan") %>% arrange(year)
obs_post <- sud[[colnames(panel)[4]]][match(post_years, sud$year)]
te <- fit$effects$te_point                                     # treated effect, levels
cf_post <- obs_post - te
pct <- 100 * te / cf_post                                      # (obs - cf)/cf, %
names(pct) <- post_years

spill <- colMeans(fit$effects$spill)                           # per-donor mean spillover
spill <- spill[order(spill)]                                   # most negative first
clean <- function(x) sub(",.*$", "", names(x))
shares <- abs(spill) / sum(abs(head(spill, 8)))                # share of top-8 |spillover|

vals <- list(
  rho            = unname(round(fit$rho_hat, 4)),
  rho_ci_lo      = unname(round(quantile(fit$rho_draws, 0.025), 4)),
  rho_ci_hi      = unname(round(quantile(fit$rho_draws, 0.975), 4)),
  effect_2012_pct = unname(round(pct["2012"], 3)),
  effect_2015_pct = unname(round(pct["2015"], 3)),
  ate_level      = unname(round(fit$effects$ate_point, 3)),
  top_spillover  = clean(spill)[1],
  egypt_share    = unname(round(shares[1], 4)),
  kenya_share    = unname(round(shares[2], 4)))

cat("rho:", vals$rho, "CI [", vals$rho_ci_lo, ",", vals$rho_ci_hi, "]\n")
cat("effect 2012:", vals$effect_2012_pct, "%  2015:", vals$effect_2015_pct, "%  ATE(level):", vals$ate_level, "\n")
cat("top spillover:", vals$top_spillover, " egypt_share:", vals$egypt_share, " kenya_share:", vals$kenya_share, "\n")
cat("spillover ranking:", paste(clean(spill)[1:5], collapse=" > "), "\n")

writeLines(toJSON(list(values = vals), auto_unbox = TRUE, pretty = TRUE),
           file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value=TRUE)[1])), "reference.json"))
