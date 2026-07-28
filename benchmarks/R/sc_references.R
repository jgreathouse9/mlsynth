#!/usr/bin/env Rscript
# Reference capture for benchmarks/cases/vanillasc_xval_references.py
#
# Emits the two external references mlsynth's classic synthetic control and its
# in-space placebo inference are cross-validated against, on the canonical
# Abadie-Diamond-Hainmueller Proposition 99 panel:
#
#   A. Synth (Abadie's own package) driven with a UNIFORM custom.v. Fixing V
#      collapses Synth's inner problem to the same V-free quadratic program
#      mlsynth solves with backend="outcome-only", so any disagreement is
#      solver quality rather than the (non-identified, famously unstable)
#      predictor-weight search. Reports the California weights, the pre-period
#      SSR that both implementations are minimising, and the in-space placebo
#      RMSPE ratio for every unit.
#
#   B. tidysynth run the way its vignette intends -- the full ADH predictor
#      spec with the V search left on -- and its own generate_placebos /
#      grab_significance machinery. This is the practical comparison: does
#      mlsynth's placebo inference reach the same conclusion as the package a
#      user would otherwise reach for.
#
# Note on units: tidysynth's `mspe_ratio` is post_mspe / pre_mspe, i.e. the
# SQUARED scale. mlsynth reports the RMSPE ratio, its square root. The Python
# side reconciles this explicitly; do not compare them raw.
#
# Usage:  Rscript benchmarks/R/sc_references.R [outfile.json]
# Deps:   Synth, tidysynth, jsonlite (see install_sc_references.sh)

suppressMessages({library(Synth); library(tidysynth); library(dplyr); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
outfile <- if (length(args) >= 1) args[1] else "sc_references.json"
root <- normalizePath(file.path(dirname(sub("--file=", "",
          grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "..", ".."),
          mustWork = FALSE)
if (is.na(root) || !dir.exists(root)) root <- "."

pre <- 1970:1988; post <- 1989:2000; treated <- "California"

# ---------------------------------------------------------------- reference A
d <- read.csv(file.path(root, "basedata", "smoking_data.csv"))
Y <- reshape(d[, c("state", "year", "cigsale")], idvar = "year",
             timevar = "state", direction = "wide")
Y <- Y[order(Y$year), ]; rownames(Y) <- Y$year
M <- as.matrix(Y[, -1]); colnames(M) <- sub("^cigsale\\.", "", colnames(M))
units <- colnames(M)

fit_uniform_v <- function(tr) {
  dn <- setdiff(colnames(M), tr)
  X1 <- matrix(M[as.character(pre), tr], ncol = 1)
  X0 <- M[as.character(pre), dn, drop = FALSE]
  s <- try(synth(X1 = X1, X0 = X0, Z1 = X1, Z0 = X0,
                 custom.v = rep(1, length(pre)), quadopt = "ipop",
                 verbose = FALSE), silent = TRUE)
  if (inherits(s, "try-error")) return(NULL)          # ipop can fail outright
  w <- as.numeric(s$solution.w); names(w) <- dn
  gap <- M[, tr] - as.numeric(M[, dn, drop = FALSE] %*% w)
  names(gap) <- rownames(M)
  list(w = w, gap = gap, ssr = sum(gap[as.character(pre)]^2))
}
rmspe_ratio <- function(g) {
  sqrt(mean(g[as.character(post)]^2)) / sqrt(mean(g[as.character(pre)]^2))
}

fits <- list(); failed <- character(0)
for (u in units) {
  f <- fit_uniform_v(u)
  if (is.null(f)) { failed <- c(failed, u); next }
  fits[[u]] <- f
}
ratios <- vapply(fits, function(f) rmspe_ratio(f$gap), numeric(1))
ca <- fits[[treated]]
w_ca <- ca$w[ca$w > 1e-4]

refA <- list(
  spec = "Synth, uniform custom.v (V-free; same QP as mlsynth backend='outcome-only')",
  california_weights = as.list(round(w_ca, 8)),
  california_pre_ssr = unname(ca$ssr),
  california_rmspe_ratio = unname(ratios[treated]),
  rank = unname(sum(ratios >= ratios[treated])),
  n_units_fitted = length(ratios),
  p_value = unname(sum(ratios >= ratios[treated]) / length(ratios)),
  ipop_failures = as.list(failed),
  rmspe_ratios = as.list(round(ratios, 8))
)

# ---------------------------------------------------------------- reference B
data("smoking")
out <- smoking %>%
  synthetic_control(outcome = cigsale, unit = state, time = year,
                    i_unit = treated, i_time = 1988,
                    generate_placebos = TRUE) %>%
  generate_predictor(time_window = 1980:1988,
                     ln_income = mean(lnincome, na.rm = TRUE),
                     ret_price = mean(retprice, na.rm = TRUE),
                     youth = mean(age15to24, na.rm = TRUE)) %>%
  generate_predictor(time_window = 1984:1988,
                     beer_sales = mean(beer, na.rm = TRUE)) %>%
  generate_predictor(time_window = 1975, cigsale_1975 = cigsale) %>%
  generate_predictor(time_window = 1980, cigsale_1980 = cigsale) %>%
  generate_predictor(time_window = 1988, cigsale_1988 = cigsale) %>%
  generate_weights(optimization_window = pre) %>%
  generate_control()

sig <- as.data.frame(grab_significance(out))
ca_row <- sig[sig$unit_name == treated, ]
wts <- as.data.frame(grab_unit_weights(out))
wts <- wts[wts$weight > 1e-4, ]

refB <- list(
  spec = "tidysynth, full ADH predictor spec, V search on (package default)",
  california_weights = setNames(as.list(round(wts$weight, 8)), wts$unit),
  california_mspe_ratio_squared_scale = ca_row$mspe_ratio,
  california_rmspe_ratio_root_scale = sqrt(ca_row$mspe_ratio),
  rank = ca_row$rank,
  n_units = nrow(sig),
  p_value = ca_row$fishers_exact_pvalue
)

# ADH 2010 Table 2, for context on which implementation lands closer
published <- list(Utah = 0.334, Nevada = 0.234, Montana = 0.199,
                  Colorado = 0.164, Connecticut = 0.069)

payload <- list(
  generated_by = "benchmarks/R/sc_references.R",
  r_version = R.version.string,
  packages = list(Synth = as.character(packageVersion("Synth")),
                  tidysynth = as.character(packageVersion("tidysynth"))),
  panel = "basedata/smoking_data.csv (39 states x 1970-2000) / tidysynth::smoking",
  reference_A_synth_uniform_v = refA,
  reference_B_tidysynth_adh = refB,
  adh2010_table2_weights = published
)
write_json(payload, outfile, auto_unbox = TRUE, digits = 10, pretty = TRUE)
cat("wrote", outfile, "\n")
