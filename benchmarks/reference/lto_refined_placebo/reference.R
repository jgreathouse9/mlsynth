# Reference run for the `lto_refined_placebo` benchmark case.
#
# Independent reproduction of the Leave-Two-Out (LTO) refined placebo test of
# Sudijono & Lei (2023/2025), "Inference for Synthetic Controls via Refined
# Placebo Tests" (arXiv:2401.07152), following the authors' own replication code
# (tsudijon/LeaveTwoOutSCI -- MIT, (c) 2023 Tim Sudijono). Their O(N^2) pair loop
# is reproduced verbatim in structure across all three of the paper's empirical
# applications: California Proposition 99, West German reunification, and the
# Basque Country. For every unordered pair {i, j} of control units the synthetic
# control is refit for the treated unit and for each of i, j on the donor pool
# with i and j removed; the treated unit "wins" the triple when its post/pre
# error ratio exceeds the max of the two donors', and the naive LTO p-value is
# the fraction of pairs the treated unit does NOT win.
#
# The synthetic control is outcome-only on both sides (simplex least squares on
# the pre-treatment outcome path, solved with LowRankQP), so the comparison
# isolates the LTO inference machinery rather than the SC solver (mlsynth
# cross-validates the solver in vanillasc_prop99 / synth_prop99 / mscmt_basque).
# The p-value is a rank statistic, so the MSE-ratio vs RMSPE-ratio convention is
# immaterial to it.
#
# Data (all in-repo, public):
#   basedata/smoking_data.csv          -- ADH (2010) Prop 99 (California, T0=19)
#   basedata/german_reunification.csv  -- West German reunification (T0=30)
#   basedata/basque_jasa.csv           -- Abadie-Gardeazabal Basque (T0=16)
#
# Run from the repository root:
#   Rscript benchmarks/reference/lto_refined_placebo/reference.R
suppressMessages(library(LowRankQP))

lto_pvalue <- function(csv, unitcol, timecol, outcome, treated, pre_cutoff,
                       drop_units = character(0)) {
  d <- read.csv(csv, check.names = FALSE)
  d <- d[!(d[[unitcol]] %in% drop_units), ]
  d <- d[!is.na(d[[outcome]]), ]
  units <- unique(d[[unitcol]]); years <- sort(unique(d[[timecol]]))
  Y <- sapply(units, function(s) {
    di <- d[d[[unitcol]] == s, ]; di[[outcome]][order(di[[timecol]])]
  })
  colnames(Y) <- units
  T0 <- sum(years <= pre_cutoff); Tn <- length(years)
  controls <- setdiff(units, treated)

  sc_ratio <- function(y, D) {
    Dp <- D[1:T0, , drop = FALSE]; yp <- y[1:T0]; G <- t(Dp) %*% Dp
    sol <- LowRankQP(Vmat = 2 * (G + diag(1e-8, ncol(G))), dvec = -2 * (t(Dp) %*% yp),
                     Amat = matrix(1, 1, ncol(G)), bvec = 1, uvec = rep(1, ncol(G)), method = "LU")
    res <- y - D %*% as.vector(sol$alpha)
    mean(res[(T0 + 1):Tn]^2) / mean(res[1:T0]^2)
  }

  cidx <- match(controls, colnames(Y)); nC <- length(controls)
  losses <- 0; npairs <- 0
  for (a in 1:(nC - 1)) for (b in (a + 1):nC) {
    i <- cidx[a]; j <- cidx[b]
    D <- Y[, setdiff(cidx, c(i, j)), drop = FALSE]
    R_I <- sc_ratio(Y[, which(colnames(Y) == treated)], D)
    R_a <- sc_ratio(Y[, i], D); R_b <- sc_ratio(Y[, j], D)
    npairs <- npairs + 1
    if (!(R_I > max(R_a, R_b))) losses <- losses + 1
  }
  Rt <- sc_ratio(Y[, which(colnames(Y) == treated)], Y[, cidx, drop = FALSE])
  list(p = losses / npairs, losses = losses, npairs = npairs, ratio = sqrt(Rt))
}

ca  <- lto_pvalue("basedata/smoking_data.csv", "state", "year", "cigsale",
                  "California", 1988)
ger <- lto_pvalue("basedata/german_reunification.csv", "country", "year", "gdp",
                  "West Germany", 1989)
bq  <- lto_pvalue("basedata/basque_jasa.csv", "regionname", "year", "gdpcap",
                  "Basque Country (Pais Vasco)", 1970, drop_units = "Spain (Espana)")

cat("== REFERENCE VALUES ==\n")
for (nm in c("ca", "ger", "bq")) {
  r <- get(nm)
  cat(sprintf("%s_p_value\t%.10f\n", nm, r$p))
  cat(sprintf("%s_treated_losses\t%d\n", nm, r$losses))
  cat(sprintf("%s_n_pairs\t%d\n", nm, r$npairs))
  cat(sprintf("%s_treated_rmspe_ratio\t%.6f\n", nm, r$ratio))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
