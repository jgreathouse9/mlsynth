# BASC vs the standard SC toolkit: West Germany (review material)

A short, self-contained analysis supporting a referee report on *Bayesian Donor
Set Selection in Synthetic Controls* (BASC). It compares BASC's in-sample
(pre-1990) fit on the Abadie, Diamond & Hainmueller (2015) West Germany
reunification study against `mlsynth`'s own estimators on the identical data
(original per-capita GDP scale):

- `VanillaSC`, the textbook Abadie estimator: outcome-only, and the ADH 2015
  covariate spec via the Malo global-optimum and MSCMT nested-DE bilevel solvers;
- `CLUSTERSC`, the paper's own two two-stage comparison methods: `method="rpca"`
  is fPCA-SYNTH (Bayani 2021/2022), `method="PCR"` is ClusterSC (Rho et al. 2025).

Headline: every standard-toolkit estimator (including the robust ones) attains a
materially tighter pre-treatment fit than BASC (BASC ~169 vs `VanillaSC` 61 and
`CLUSTERSC` 89–98).

The cause of that gap is established by running the authors' own sampler with
the donor-selection indicators forced to 1 (`scripts/run_gamma1.R`, a four-line
patch to `BASC_realdata.R`). Selection is not what costs the fit: switching it
off makes the fit worse, and the Section 3.1 configuration lands at 193 across
three seeds. The likelihood is the cause. It scores residuals over all 44 years
with one constant absorbing the post-1990 divergence, so the donor weights are
fitted to the post-treatment outcomes. Constrained least squares under that
specification reproduces the paper's published Table 7 BASC weights to an L1
distance of 0.04, with no sampler, prior, Gaussian process or selection.

The report also records three things that do not depend on any `mlsynth` fit:

- three of the paper's five reported ATTs reproduce (BASC, fPCA-SYNTH -1,655
  vs -1,501, and ClusterSC -2,427 vs -2,039), while standard SCM (-159 vs
  -1,298) and B-MV (-218 vs -2,079) are an order of magnitude out, so a single
  mis-scaled outcome does not explain it;
- the Table 7 s-SCM weight vector fits the pre-1990 path worse than uniform
  weights (RMSE ~1,925 vs ~1,289) and implies an ATT of the opposite sign to the
  one the text reports, robust to the table's two-decimal rounding;
- BASC's 95% credible band excludes the observed series in 20 of 30 pre-treatment
  years, at a mean width of ~180.

What the authors get right, and the report says so: their public sampler is on
the original per-capita scale exactly as their response letter claims, and it
reproduces (same three donors as their Table 7). The gap is that the repository
contains only the BASC sampler, none of the four comparison methods.

Recommendation: major revision. The theory and the simulation design are not in
question; the concerns are confined to Section 5, Table 7, and what can be
verified. Six numbered conditions are listed at the end of the report, of which
the full replication script is the one the others depend on: of the four
comparison methods, two agree with the paper when checked (fPCA-SYNTH to 9%,
ClusterSC to 16%) and two do not (standard SCM, B-MV), and the same routines
generate the numerical study on which the competitiveness claim now rests.

## Layout

    basc_westgermany_review.qmd    # the Quarto report (renders from data/ only)
    data/                          # pre-computed CSVs (no heavy deps to render)
      counterfactuals.csv          #   every counterfactual in the report, by year
      basc_counterfactual.csv      #   BASC posterior path + 95% CI (2000/2000)
      basc_counterfactual_500.csv  #   BASC posterior path + 95% CI (500/500)
      bmv_counterfactual.csv       #   B-MV (bsynth) posterior path + 95% CI
      basc_weights.csv             #   BASC posterior donor weights
      basc_insample_rmse.txt       #   BASC 2000/2000 in-sample RMSE
      gamma1_diagnostic.csv        #   gamma = 1 runs, three configurations x three seeds
    scripts/
      basc_run.R                   # regenerate the BASC CSVs (see header)
      run_gamma1.R                 # the gamma = 1 diagnostic (see header)
      run_init.R                   # start the chain at the best simplex fit

`data/counterfactuals.csv` is a single transparency table: one row per year
(1960–2003) with the treatment indicator, observed West Germany, and every
counterfactual that goes into the report: the seven render-time `mlsynth` fits
(VanillaSC ×3, CLUSTERSC ×2, FDID, MVBBSC/B-MV), the offline `bsynth` and BASC
(500/500, 2000/2000) posterior paths, and the two dot-product weight vectors
(the Table 7 s-SCM weights and uniform 1/16). The BASC 25000/25000 chain is not
included: only its two summary numbers were saved, not a posterior path.

The `mlsynth` estimators (`VanillaSC`, `CLUSTERSC`) run at render time via
`compare_estimators` on `basedata/repgermany.dta`; only BASC is precomputed
(its Gibbs sampler needs R and is slow).

## Rendering

Pushing to the `claude/basc-review-westgermany` branch triggers
`.github/workflows/render-basc-review.yml`, which installs `mlsynth` and TinyTeX,
renders the `.qmd` to a self-contained HTML and a PDF, and uploads both as a
workflow artifact (`basc-westgermany-review`).

Locally: `quarto render review/basc_westgermany/basc_westgermany_review.qmd`
(needs `mlsynth` installed: `pip install -e .`; the PDF also needs a LaTeX engine,
e.g. `quarto install tinytex`). Add `--to html` or `--to pdf` for a single format.

## Regenerating the BASC data

See the header of `scripts/basc_run.R`; it uses the authors' own sampler from
github.com/sll-lee/paper-BASC (not vendored here). The `mlsynth` side needs no
regeneration; it runs in the document.
