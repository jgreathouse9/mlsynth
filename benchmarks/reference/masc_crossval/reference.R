# Reference run for the `masc_crossval` benchmark case.
#
# Runs Kellogg, Mogstad, Pouliot & Torgovitsky's own MASC estimator (matching +
# synthetic control) on the Abadie-Gardeazabal Basque terrorism panel, under the
# authors' outcome-path specification: the nearest-neighbour match and the
# synthetic-control step both operate on the pre-treatment per-capita GDP path
# (no covariate/predictor block), with the mixing parameter phi chosen by the
# authors' rolling-origin cross-validation over m in 1..10 nearest neighbours.
#
# The synthetic-control QP is solved with `nogurobi = TRUE`, i.e. the open-source
# LowRankQP path rather than the commercial Gurobi default. The problem is a
# convex simplex-constrained least squares, so its optimum is solver-invariant;
# mlsynth solves the same QP with CLARABEL. This makes the case a genuine live
# cross-validation of two independent implementations on the same estimand.
#
# The MASC sources are the authors' own, vendored verbatim (MIT, (c) 2019 Maxwell
# Kellogg) under ../masc_basque/. The `sc_estimator` LowRankQP branch has an
# upstream latent bug on a *diagnostic* line (loss.w references an unset field);
# it is never read downstream. We source the vendored file with that one line
# neutralised at read-time (the file on disk stays pristine) so the nogurobi path
# runs; every number below comes from the authors' unmodified algorithm.
#
# Run from the repository root:
#   Rscript benchmarks/reference/masc_crossval/reference.R
suppressMessages({library(data.table); library(LowRankQP)})

ref_dir <- "benchmarks/reference/masc_basque"
# Source the vendored estimator with the diagnostic-only loss.w line neutralised.
est_src <- readLines(file.path(ref_dir, "masc_estimator.R"))
est_src <- gsub("params\\$loss.w<-mean\\(\\(treated-donors%\\*%params\\$weights.sc\\)\\^2\\)",
                "params$loss.w<-NA", est_src)
eval(parse(text = est_src), envir = globalenv())
source(file.path(ref_dir, "masc_crossvalidation.R"))

panel <- read.csv("basedata/basque_jasa.csv")
panel <- panel[panel$regionname != "Spain (Espana)", ]   # drop the national aggregate
years <- sort(unique(panel$year))
treated_name <- "Basque Country (Pais Vasco)"
donor_names  <- setdiff(unique(panel$regionname), treated_name)

col <- function(r) panel$gdpcap[panel$regionname == r][order(panel$year[panel$regionname == r])]
Y_treated <- matrix(col(treated_name), ncol = 1)
Y_donors  <- sapply(donor_names, col)                    # T x N0
Tn <- length(years)
treatment <- which(years == 1970)                        # first treated period index

fit <- masc(treated = Y_treated,
            donors  = Y_donors,
            treatment = treatment,
            match_est = NearestNeighbors,
            tune_pars_list = list(m = 1:10, min_preperiods = 5),
            nogurobi = TRUE)

w  <- as.vector(fit$weights)
cf <- as.vector(Y_donors %*% w)
pre  <- 1:(treatment - 1); post <- treatment:Tn
pre_rmse <- sqrt(mean((Y_treated[pre] - cf[pre])^2))
att      <- mean(Y_treated[post] - cf[post])

cat("== REFERENCE VALUES ==\n")
cat(sprintf("masc_phi_hat\t%.6f\n", fit$phi_hat))
cat(sprintf("masc_m_hat\t%d\n", fit$m_hat))
cat(sprintf("masc_pre_rmse\t%.6f\n", pre_rmse))
cat(sprintf("masc_att\t%.6f\n", att))
ord <- order(-w)
for (i in ord) cat(sprintf("weight\t%s\t%.6f\n", donor_names[i], w[i]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
