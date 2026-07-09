# Capture the Sakaguchi-Tagawa / Mendez California Proposition 99 Stage-3
# (Bayesian Spatial SAR SCM) reference from the tutorial's OWN helper code, so
# benchmarks/cases/spillsynth_prop99_sar.py can cross-check mlsynth against it.
#
# Source of the reference implementation (fetched at runtime, as the tutorial
# itself does -- the upstream replication package is third-party and gitignored):
#   https://github.com/cmg777/starter-academic-v501
#     content/post/r_sc_bayes_spatial/helpers/{01_utils.R, 21_mcmc_alpha.R,
#       22_mcmc_sar.R, 10_sc_spillover.R, 20_mcmc.cpp, california_smoking.rda}
#   Tutorial: https://carlos-mendez.org/post/r_sc_bayes_spatial/
#
# The helper C++ kernels (hs_alpha_gibbs_cpp, sar_full_sampler_cpp_step2) are the
# Rcpp reference; sc_spillover() is the tutorial's Stage-3 entry point. We run it
# in two configurations and emit reference.json:
#   * bare  : p_factors = 0, no covariate  -> the identified spatial core
#   * full  : p_factors = 1, X = retprice  -> the tutorial's headline config
#
# Verified on Ubuntu + R 4.3.3 with apt r-base + Rcpp/RcppArmadillo/Matrix and
# the tidyverse helpers (magrittr/dplyr/tidyr) the tutorial's data prep needs.
# See agents/agents_r_environment.md for the sandbox install recipe.
#
# Usage:
#   SAR_HELPERS_DIR=<dir with the fetched helpers> Rscript reference.R
#   (defaults to $PWD if unset)

suppressMessages({
  library(Rcpp); library(RcppArmadillo); library(Matrix)
  library(magrittr); library(dplyr); library(tidyr)
})

HERE <- Sys.getenv("SAR_HELPERS_DIR", unset = getwd())
for (h in c("01_utils.R", "21_mcmc_alpha.R", "22_mcmc_sar.R", "10_sc_spillover.R"))
  source(file.path(HERE, h))
Rcpp::sourceCpp(file.path(HERE, "20_mcmc.cpp"))
R_NilValue <- NULL   # the helper's X=NULL branch references this C++ symbol by name

TREAT_YEAR <- 1988L
SEED <- 20251022L
load(file.path(HERE, "california_smoking.rda"))

panel_df <- california_smoking$panel_df
panel_df$treatment <- as.integer(panel_df$state == "California" &
                                 panel_df$year >= TREAT_YEAR)
w <- as.matrix(california_smoking$w[, 2])
W <- as.matrix(california_smoking$W[, -1])
rownames(W) <- california_smoking$W$state
colnames(W) <- california_smoking$W$state

run_one <- function(X, p, M, burn, seed = SEED) {
  fit <- sc_spillover(
    data = panel_df, treated_unit = "California", w = w, W = W,
    treatment_dummy = "treatment", y = "cigsale", X = X, p_factors = p,
    M = M, burn = burn, seed = seed, step_rho = 0.01,
    unit_col = "state", time_col = "year", verbose = FALSE)
  spill <- fit$effects$spill
  times <- as.numeric(rownames(spill))
  post <- which(times >= TREAT_YEAR)
  savg <- colMeans(spill[post, , drop = FALSE])
  ord <- order(-abs(savg))
  list(rho = fit$rho_hat, att = fit$effects$ate_point,
       att_ci = as.numeric(fit$effects$ate_ci95),
       rho_ci = as.numeric(quantile(fit$rho_draws, c(.025, .975), names = FALSE)),
       n_active = sum(abs(fit$alpha_hat) > 0.01),
       te = as.numeric(fit$effects$te_point), years = times[post],
       spill_states = names(savg)[ord[1:2]], spill_vals = as.numeric(savg[ord[1:2]]))
}

bare <- run_one(NULL,          0L, 40000L, 20000L)
full <- run_one(c("retprice"), 1L, 40000L, 20000L)
# rho weak-identification: three seeds of the full config
full_rho_seeds <- vapply(c(20251022L, 1L, 7L),
  function(s) run_one(c("retprice"), 1L, 40000L, 20000L, seed = s)$rho, numeric(1))

cat(sprintf("BARE : rho=%.4f att=%.4f n_active=%d\n", bare$rho, bare$att, bare$n_active))
cat(sprintf("FULL : rho=%.4f att=%.4f CrI[%.4f,%.4f] n_active=%d\n",
            full$rho, full$att, full$att_ci[1], full$att_ci[2], full$n_active))
cat(sprintf("FULL rho across seeds: %s\n",
            paste(sprintf("%.4f", full_rho_seeds), collapse = ", ")))
cat(sprintf("top spillover: %s=%.4f, %s=%.4f\n",
            full$spill_states[1], full$spill_vals[1],
            full$spill_states[2], full$spill_vals[2]))
