# BASC vs the standard SC toolkit — West Germany (review material)

A short, self-contained analysis supporting a referee report on *Bayesian Donor
Set Selection in Synthetic Controls* (BASC). It compares BASC's in-sample
(pre-1990) fit on the Abadie, Diamond & Hainmueller (2015) West Germany
reunification study against `mlsynth`'s own estimators on the identical data
(original per-capita GDP scale):

- `VanillaSC` — the textbook Abadie estimator: outcome-only, and the ADH 2015
  covariate spec via the Malo global-optimum and MSCMT nested-DE bilevel solvers;
- `CLUSTERSC` — the robust low-rank RPCA-SC / PCR (RSC) family.

Headline: every standard-toolkit estimator (including the robust ones) attains a
materially tighter pre-treatment fit than BASC (BASC ~169 vs `VanillaSC` 61 and
`CLUSTERSC` 89–98), and BASC drops the canonical Austria/USA donors that every
standard analysis — and ADH 2015 — identifies as most important.

## Layout

    basc_westgermany_review.qmd    # the Quarto report (renders from data/ only)
    data/                          # pre-computed CSVs (no heavy deps to render)
      toolkit_counterfactuals.csv  #   observed + each method's synthetic path
      insample_rmse_toolkit.csv    #   in-sample RMSE by method
      weights_comparison.csv       #   donor weights: each method + ADH 2015
      basc_counterfactual.csv      #   BASC posterior path + 95% CI (2000/2000)
      basc_weights.csv             #   BASC posterior donor weights
      basc_insample_rmse.txt       #   BASC 2000/2000 in-sample RMSE
    scripts/
      gen_toolkit.py               # regenerate the mlsynth toolkit CSVs
      basc_run.R                   # regenerate the BASC CSVs (see header)

## Rendering

Pushing to the `claude/basc-review-westgermany` branch triggers
`.github/workflows/render-basc-review.yml`, which renders the `.qmd` to a
self-contained HTML and uploads it as a workflow artifact
(`basc-westgermany-review`). The render reads only the committed CSVs — it does
not re-run BASC or mlsynth — so it needs just Quarto + pandas + matplotlib.

Locally: `quarto render review/basc_westgermany/basc_westgermany_review.qmd`.

## Regenerating the data

- Toolkit: `python review/basc_westgermany/scripts/gen_toolkit.py` (needs
  `mlsynth` and the ADH `repgermany.dta`; set `REPGERMANY` to its path).
- BASC: see the header of `scripts/basc_run.R` — it uses the authors' own
  sampler from github.com/sll-lee/paper-BASC (not vendored here).
