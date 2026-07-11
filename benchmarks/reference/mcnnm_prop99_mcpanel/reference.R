#!/usr/bin/env Rscript
# Reference capture: MCPanel (Athey, Bayati, Doudchenko, Imbens & Khosravi 2021)
# matrix-completion (mcnnm_cv) on the Abadie-Diamond-Hainmueller Proposition 99
# smoking panel.
#
# Reads the exact panel mlsynth's benchmark uses (basedata/smoking_data.csv:
# 39 states x 31 years, 1970-2000, California treated from 1989), builds the
# outcome matrix and the observation mask (treated post-1989 California cells
# held out), runs mcnnm_cv at the package defaults, and records the ATT and the
# fitted California counterfactual path so the mlsynth case can cross-validate
# against the authors' own R.
#
# The cross-validation fold assignment is randomised; set.seed(1) fixes it.
# Reproduce:  Rscript benchmarks/reference/mcnnm_prop99_mcpanel/reference.R
suppressMessages(library(MCPanel))

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[[1]] else
  file.path("basedata", "smoking_data.csv")

d <- read.csv(data_path, check.names = FALSE)
states <- sort(unique(d$state))
years  <- sort(unique(d$year))
Y <- matrix(NA_real_, length(states), length(years),
            dimnames = list(states, as.character(years)))
for (i in seq_len(nrow(d)))
  Y[as.character(d$state[i]), as.character(d$year[i])] <- d$cigsale[i]

ca   <- which(states == "California")
post <- which(years >= 1989)
mask <- matrix(1, nrow(Y), ncol(Y)); mask[ca, post] <- 0   # 1 observed, 0 missing

set.seed(1)
est  <- mcnnm_cv(Y, mask, num_lam_L = 100L, num_folds = 5L)   # package defaults
Lhat <- est$L + replicate(ncol(Y), est$u) + t(replicate(nrow(Y), est$v))
ca_cf_post <- Lhat[ca, post]
att <- mean(Y[ca, post] - ca_cf_post)

cat(sprintf("MCPanel mcnnm ATT = %.6f\n", att))

years_post <- years[post]
json <- sprintf(
  paste0('{\n  "values": {\n',
         '    "mcnnm_att": %.6f,\n',
         '    "ca_counterfactual_post": [%s],\n',
         '    "years_post": [%s],\n',
         '    "num_lam_L": 100,\n    "num_folds": 5,\n    "seed": 1\n  }\n}\n'),
  att,
  paste(sprintf("%.6f", ca_cf_post), collapse = ", "),
  paste(years_post, collapse = ", "))
out <- if (length(args) >= 2) args[[2]] else
  "benchmarks/reference/mcnnm_prop99_mcpanel/reference.json"
writeLines(json, out)
cat("wrote", out, "\n")
