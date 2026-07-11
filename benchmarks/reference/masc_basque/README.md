# MASC reference (Kellogg-Mogstad-Pouliot-Torgovitsky)

Vendored R sources for Kellogg, Mogstad, Pouliot & Torgovitsky's MASC estimator
(matching + synthetic control), MIT-licensed (c) 2019 Maxwell Kellogg
(`maxkllgg/masc`).

Files
- `masc_estimator.R`, `masc_crossvalidation.R` — the authors' MASC estimator and
  its rolling-origin cross-validation of the match/SC mixing parameter.
- `masc_AG_replication.R` — the authors' Abadie-Gardeazabal (Basque) replication
  driver.
- `LICENSE.masc` — the upstream license.

Live cross-validation bundle
`benchmarks/reference/masc_crossval/` sources these files and runs
`masc(..., nogurobi = TRUE)` on the Basque outcome-path panel; its captured
`reference.json` is what `benchmarks/cases/masc_crossval.py` pins mlsynth's MASC
against, value for value (phi, m, ATT, pre-RMSE, donor weights all agree to
solver tolerance). Regenerate with
`python benchmarks/reference/generate.py masc_crossval`.

The synthetic-control step is a convex simplex-constrained least squares, so its
optimum is solver-invariant; the `nogurobi` LowRankQP path is used in place of
the commercial Gurobi default (`nogurobi = FALSE`) and reaches the same optimum
that mlsynth's CLARABEL solve does. The one upstream latent bug on the LowRankQP
branch is a diagnostic-only `loss.w` line that is never read downstream;
`masc_crossval/reference.R` neutralises just that line at read-time while leaving
the vendored file on disk pristine.

The separate `masc_basque.py` case remains the Path-A paper replication (the KMPT
Section 5 covariate/predictor specification); `masc_crossval` complements it by
cross-validating the outcome-path machinery against the authors' own code.
