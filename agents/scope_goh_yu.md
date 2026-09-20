# Goh & Yu (2022) Bayesian MAP synthetic control — replication spike

Status: validated, not built. This is the `/replicate` output for

> Goh, G. and Yu, J. (2022), "Synthetic control method with convex hull
> restrictions: a Bayesian maximum a posteriori approach", The Econometrics
> Journal 25(1), 215–232. doi:10.1093/ectj/utab015

The paper's replication package ships four R scripts: three for the Section 5
simulation (one per `theta0`) and one for the Section 6 Basque application.

## What the method is

The constraint set is the parallelly shiftable convex hull,

```
W_ps-conv = { w : w_1 in R, w_i >= 0 for i >= 2, sum_{i>=2} w_i = 1 }
```

a simplex over donors plus a free intercept. mlsynth already solves that
geometry: it is `TSSC`'s MSCa variant, documented there as the demeaned
synthetic control. What the paper adds is a Bayesian treatment of it.

The weighting matrix `V` is diagonal and estimated, not cross-validated. Write
the design as `T0` outcome rows stacked on `p` covariate rows, with the
intercept column set to one on the outcome rows and zero on the covariate rows.
Then `V = nu^{-1} diag(xi)` where `xi` is fixed at 1 on the outcome rows and
carries a spike-and-slab inclusion indicator on each covariate row. The MAP is
computed by Monte Carlo EM:

* M-step — the weighted QP over `W_ps-conv`;
* E-step — Gibbs over `(nu, xi)` holding `w` fixed, returning the posterior mean
  of `xi` and of `1/nu`.

Posterior inference is a second Gibbs sampler restricted to the MAP's active
donor set: the intercept from a normal, each active donor bar the last from a
truncated normal on `[0, U_i]`, the last absorbing the sum-to-one residual, then
`nu` and `xi`. HPD intervals come from the draws.

## What was validated

The authors' own script reproduces their Table 3 and Table 4 exactly. Their
`sec6_Empirical_Application.R` is reproduced here as
`agents/prototypes/goh_yu_basque_reference.R` with two substitutions and no
other change, because neither package installs in this environment:

* `invgamma::rinvgamma(1, shape=a, rate=b)` -> `1/rgamma(1, shape=a, rate=b)`;
* `HDInterval::hdi(x, credMass)` -> the narrowest window of the sorted draws
  holding `ceiling(credMass * n)` of them.

The first `hdi` substitution written here was wrong: it selected a window
holding `n - m + 1` draws where it needed `m`, which at 95% is 5% of the mass.
It returned `(-1.064, -1.028)`, an interval of width 0.036 that does not contain
the point estimate. The paper's printed interval is what exposed it. The
corrected version recovers `(-1.96, 1.96)` on 200,000 standard normal draws.

A Python port of the algorithm is `agents/prototypes/goh_yu_bayes_scm.py`. It
was run against the design matrices dumped from R, so the comparison isolates
the algorithm from the ingestion.

| quantity | paper | R reference | Python port |
| --- | --- | --- | --- |
| Bayes SCM ATE, 1981–1994 | −1.193 | −1.1930 | −1.1953 |
| 95% HPD | (−1.861, −0.559) | (−1.8620, −0.5601) | (−1.8693, −0.5928) |
| ADH-SCM ATE | −0.880 | −0.8797 | −0.8797 |
| per capita GDP loss | 15.26% | 15.259% | 15.288% |
| intercept | −0.176 | −0.1758 | −0.1769 |
| Baleares | 0.273 | 0.2734 | 0.2747 |
| Cataluna | 0.346 | 0.3460 | 0.3450 |
| Comunidad Valenciana | 0.010 | 0.0096 | 0.0089 |
| Madrid | 0.371 | 0.3711 | 0.3713 |
| active donors | 4 | 4 | 4 |
| EM iterations | — | 15 | 13 |

The port selects the same four regions, the same sparsity pattern and the same
intercept sign and magnitude. Remaining differences are in the third decimal and
are what independent RNG streams produce: both the E-step and the inference
sampler are stochastic, so neither the MAP nor the HPD is deterministic given
the data.

Runtime in Python is 8 s for the EM and 15 s for 20,000 inference draws, against
minutes for the R original.

## What a build needs that the repo does not have

The ADH predictor recipe folds `school.post.high` into `school.high` before
normalising the schooling block to sum to 100. None of the three Basque files in
`basedata/` can support it:

* `basque_data.csv` — no `school.post.high`, and its `school.high` is
  forward-filled (1.778114 across 1955–1958 where the source is missing), so it
  is not the raw panel;
* `basque_jasa.csv` — no `school.post.high`;
* `basque_mscmt.csv` — carries `school.higher`, which is the folded column after
  the 100-normalisation, a derived form; it does preserve the source's 666
  missing values.

So a build adds the raw `Synth::basque` (774 x 17) as its own file. The
alternative, deriving the predictor block from `basque_mscmt.csv`, means
reversing a normalisation, which is not worth the fragility.

## Recommendation

Build, `BMAPSC`, on its own branch, Path A first against the table above.

Two things to decide during the build:

1. The E-step runs 15,000 Gibbs draws per EM iteration and discards the first
   5,000. The chain is over one continuous and `p` binary coordinates with the
   residuals held fixed, so it mixes fast; the draw count looks like headroom
   and not a requirement. Measure the MAP's sensitivity to it before copying the
   number, because it sets the estimator's runtime.
2. `V` is not fully marginalised — the `T0` outcome rows are pinned at weight 1
   and only the `p` covariate rows get inclusion probabilities. On an
   outcome-only panel, which is the common case in this library, `V` collapses
   to a scalar and the Bayesian tuning story has nothing left to do. Whether the
   estimator then differs from `TSSC` MSCa beyond the inference is an open
   question the build should answer with a measurement.

Path B (Table 1's MSEs and Table 2's coverage) is feasible but expensive: 1,000
Monte Carlo replications of an EM that itself runs 17,000 Gibbs draws per
iteration. Size it at reduced `M` with tolerances set from the reduced run.
