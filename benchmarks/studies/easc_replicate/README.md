# EASC, demonstrated before it is built

Reference: Rok Spruk (2026), "Synthetic Control under Out-of-Span Common
Shocks: Diagnosis, Exposure Balance and Correction", arXiv:2610.05535v1
(4 October 2026). Proposition 1 is the leverage bound, Corollary 1 the
silent-factor diagnostic, Proposition 3 the exposure hull and frontier,
Proposition 4 the exposure-augmented estimator (EASC), Proposition 5 the
bias-aware intervals and breakdown value; Section 5 is the Monte Carlo and
Section 6 the Iran application.

This is a replication spike. It produces a recommendation, not estimator code.

```
python -m benchmarks.studies.easc_replicate.run               # Path A, MC cross-check, MC at 100 reps (~1 min)
python -m benchmarks.studies.easc_replicate.run --reps 500    # the paper's 500 reps (~3 min on 4 cores)
```

## The method

Synthetic control trusts pre-treatment fit to balance the factors behind
untreated outcomes. Spruk's case is an observed common shock (the oil price)
that moves with the latent factors before treatment and breaks away after: the
synthetic unit can fit the treated unit's pre-period path while carrying much
less exposure to the shock, and the post-period gap absorbs the difference.

- Diagnostics, none of which use the treated unit's post-period outcomes:
  the leverage `kappa_t` of each period's factor vector in the metric of the
  pre-period second moments; the partial leverage `ell_t` of the shock on
  latent factors estimated from the donors; the exposure imbalance
  `EI(w) = s_1 - sum_j w_j s_j` of any weights; the distance `d_min` from the
  treated unit's exposure to the donors' exposure hull; the exposure-fit
  frontier `R(d)`.
- Estimators: EASC subtracts `phi_t EI(w)` from the synthetic-control gap, with
  `phi_t` the coefficient on exposure in a cross-donor regression of
  `Y_jt - Y_jb` on `(1, s_j, c_j)`, `c_j` the donor's growth in each third of
  the pre-period. EBSC-min is the best-fitting simplex weights at the smallest
  attainable imbalance; EASC* applies the correction at a frontier point picked
  by a heuristic MSE criterion.
- Inference: Armstrong-Kolesar bias-aware intervals with bias bound
  `M kappa RMSPE`, a breakdown value `M*`, and a noise scale from standardised
  in-space placebos.

## Files

| file | role |
| --- | --- |
| `oracle.py` | the readable port; each function names the paper section and the `ebsc.py` function it ports |
| `simulate.py` | the Section 5.1 DGP and per-replication pipeline, ported from `mc.py` |
| `paper_values.py` | every number the study matches against: Tables 1-5 and 7, the appendix factor table |
| `activeset_frontier.py` | the frontier on mlsynth's active-set solver, for the solver choice below |
| `run.py` | ingestion through `dataprep`, the comparisons, `results/easc_replicate.json` |
| `../../reference/easc_iran/` | `capture.py`, which runs the author's package live, and the captured `reference.json`, `reference.out`, `provenance.json` |
| `basedata/iran_sanctions_oil.csv` | the application's panel; provenance in its README |

The author's package is a JAE submission, not public at the time of the study
and without a licence, so it is not vendored. `capture.py` runs it from a local
copy and records a sha256 of each file it ran; the captured numbers are what
the study compares against, and a run of `capture.py` on the same package
regenerates them.

## What matched

Path A ingests the 19-donor normaliser pool and the 21-donor exposure-feasible
pool through `dataprep`; the exposure index is each unit's 1996-2011 mean of
oil rents and the shock is log Brent.

Against the author's code, run live (largest absolute difference over each
group, both pools):

| quantity | difference |
| --- | --- |
| SC weights; SC and EASC gaps (29 periods) | 2.7e-08; 5.5e-09 |
| `kappa_t`, latent `kappa_t`, `ell_t` (29 periods each) | 4.2e-12 |
| `phi_t` and its jackknife standard error; phase averages; hat values; delete-one-donor `phi` | 4.2e-15 |
| frontier at 41 tolerances: RMSPE, EI, weights, EASC* criterion | 1.3e-09, 2.0e-07, 2.2e-08, 2.2e-07 |
| EBSC-min and EASC* gaps; EASC* frontier point | 1.9e-09; identical |
| inference, 2 estimators x 3 phases x 2 pools: estimate, noise s.d., leverage, p-value | 3.6e-09, 4.7e-07, 4.3e-12, 0 |
| interval bounds at `M` = 0, 0.5, 1; `M*`; robust `M*` | 9.3e-07; 9.0e-07; 4.8e-06 |
| leverage with 1, 2, 3 latent factors | 1.0e-11 |
| 13 robustness specifications (EASC, EASC*, EBSC-min, SC by phase; EI; RMSPE) | 4.1e-09, EI 2.1e-07 |

The remaining differences come from the synthetic-control solver (mlsynth's
active-set solver here, CLARABEL in the reference) and from Brent's method in
place of sixty bisections for `M*`. EI is a weight vector times exposures of
order 20, so a weight difference of 1e-8 shows as 2e-7.

Against the paper's printed tables: 270 of 270 cells reproduce at the
precision they are printed. That is Table 4 (all diagnostics), Table 5 (point
estimates; placebo p-values, noise s.d., leverage, both interval bounds at
`M` = 0 and 0.5, `M*` and robust `M*` for SC and EASC in both pools; the phase
averages of `phi` and their jackknife t), Table 7 (EI, the EASC estimates, and
which EBSC-min cells are infeasible, for all ten rows), the hat values of
Angola and Azerbaijan, and the appendix table of leverage, R^2 and `M*` for
one to three latent factors.

The headline: SC finds -0.127, -0.087, -0.089 log points in 2012-15, 2016-17
and 2018-24. EASC finds -0.122, -0.011, +0.109 in the exposure-feasible pool
and -0.152, -0.033, +0.086 in the normaliser pool. The 2012-15 loss survives
(SC `M*` 0.41 and 0.27); nothing after 2015 does.

mlsynth's `VanillaSC` with `backend="outcome-only"` gives the same SC fit on
the same panel: weights MEX 0.369, NGA 0.254, THA 0.244, KAZ 0.133,
pre-period RMSPE 0.0144, phase gaps -0.127, -0.087, -0.089.

Path B is seeded as `mc.py` is (`20261002 + 1000 * design + replication`).

| check | result |
| --- | --- |
| panels drawn by `simulate.draw` vs `mc.draw`, 12 designs x 3 seeds | 36 of 36 bitwise identical (sha256) |
| `one_rep` vs `mc.one_rep`, designs S0, S2, S3, S9, S11 x 3 seeds: errors, interval lengths, leverage, s.d., ranks, diagnostics | 1.7e-08; no coverage indicator differs |
| Tables 1-3 and the appendix MC tables at the paper's 500 replications: bias, RMSE (3 phases), coverage and length (2016-17, 2018-24; S8-S11 2018-24), for SC, EBSC-min, EASC, EASC* | 496 of 496 cells at printed precision |

Because the draws are bitwise identical and each replication agrees to 1e-8,
the 500-replication tables are the paper's tables, not an approximation to
them. The other estimators in those tables (augmented SC, synthetic DiD,
de-meaned SC, generalized SC, matrix completion) are left to their own
benchmarks.

## The solver

The synthetic-control weights everywhere (treated unit, placebos, every Monte
Carlo replication) come from mlsynth's active-set solver,
`mlsynth.utils.solvers.active_set.solve_simplex_qp`.

The frontier (EBSC-min, EASC*) stays on cvxpy with CLARABEL, as in the
reference. `solve_simplex_qp` takes a linear objective term but no further
constraints. With a scalar index, the constraint `|EI(w)| <= d` binds on one
side, so the frontier point can be found on the active-set solver by pricing
that side into the objective (`linear = -mu sign(EI) s0`) and bisecting on
`mu`. `activeset_frontier.py` does this, and over the 41 tolerances of each
pool it reproduces `R(d)` to 1e-8. It does not reproduce the weights: they
differ by up to 1.6e-4 and EI by up to 1.5e-3, because with 16 pre-periods and
19 or 21 donors the solution set is not a single point, and an active-set
vertex and an interior-point solution are different members of it. It also
takes about 5 s per pool against 0.1 s. A build would keep CLARABEL for the
frontier (cvxpy is already a dependency, used in 42 modules) and the active-set
solver for the weights.

## Where the text and the code differ

For the EASC interval, Section 4.3 and Online Appendix B add
`(|EI| se(phi_bar_T))^2` to the noise variance, the jackknife s.e. of the
phase average of `phi`. `app_inference.py` and `mc.py` add `|EI|` times the
phase mean of the per-period jackknife s.e.s, which is larger. Every number
above follows the code, because the tables do. With the text's version the
2012-15 noise s.d. falls from 0.064 to 0.063 (normalisers) and 0.052 to 0.050
(exposure-feasible), and EASC's `M*` rises from 0.41 to 0.43 and 0.21 to
0.22; no conclusion changes. `oracle.phase_inference(easc_se=...)` carries both.

## What the replication shows about the method

The arithmetic is reproduced in full. The following come out of the same
runs.

1. EASC costs precision when no silent factor exists. With no exposure
   channel (S0) or an in-span shock (S1), EASC's RMSE is 1.5 to 1.9 times
   SC's in every phase (2018-24: 0.058 against 0.030, 0.064 against 0.042).
   The correction is something to apply after the diagnostic flags a silent
   factor. Applied by default, it costs precision.
2. In the designs whose latent leverage matches the application, EASC is no
   better than SC. In S9 (latent leverage after 2015 at the application's
   level; the paper reports a mean 2018-24 `kappa` of 26.5) EASC's 2018-24 RMSE is 0.152 against SC's
   0.140; in S10 0.250 against 0.245, with EASC's bias twice SC's (0.074
   against 0.036). The application's 2018-24 leverage is 26.7 and 40.4. The
   gains in S2, S3 and S7 (2018-24 RMSE ratios of 0.35 to 0.65 against SC)
   are real, but they sit at moderate leverage, where the paper places them on
   purpose.
3. The correction rests on two donors. In the exposure-feasible pool the
   2018-24 `phi` has jackknife t of -3.6, with Angola and Azerbaijan at hat
   values 0.70 and 0.61; in the normaliser pool, where Iran lies outside the
   exposure range and the regression extrapolates, the t is -1.2. Dropping
   Angola moves EASC's 2018-24 estimate from 0.109 to 0.138.
4. The frontier machinery adds nothing to the application. The EASC* criterion
   selects the synthetic-control end in both pools, so EASC* equals EASC.
   Across the twelve designs EASC* and EASC differ by at most 0.0035 in bias
   and 0.008 in RMSE (S0). EBSC-min is feasible only in the exposure-feasible
   pool.
5. The breakdown value depends on the number of latent factors. SC's 2012-15
   `M*` is 0.96, 0.41, 0.30 with one, two and three factors (normalisers) and
   1.48, 0.27, 0.23 (exposure-feasible). EASC at synthetic-control weights
   does not depend on the factors; its inference does.
6. Without the bias term, SC intervals under-cover, as the paper claims
   (2018-24 coverage at `M = 0`: 0.70 in S2, 0.72 in S4). EASC's intervals at
   `M = 0` over-cover in the designs where the correction is large (1.00 in S2
   and S7, length 0.353 and 0.289), because the jackknife term added for the
   correction is conservative.
7. The paper's own conclusion stands: the long-run effect is not identified
   with these comparators. EASC's 2018-24 interval at `M = 0` is
   [-0.25, 0.47] and its `M*` is zero after 2015, as is SC's.

Data: the World Bank and FRED hosts were not reachable from this environment,
so the package's WDI and oil-rent extracts, which its README says were
transcribed from API responses, were not re-pulled. The checks that were
possible pass (`basedata/iran_sanctions_oil.README.md`).

## Recommendation

Build, scoped as follows.

- A new estimator `EASC` (the name `ESC` is taken by Ensemble SC). Default:
  the correction at synthetic-control weights, which is the paper's EASC and
  the only variant its application reports.
- The diagnostics as typed result fields: `kappa_t`, latent `kappa_t`,
  `ell_t` and its pre-period R^2, `phi_t` with jackknife s.e., EI, `d_min`,
  and the frontier `R(d)`. Of everything in the paper, these are the parts
  that are cheap, need no tuning and hold up in every design.
- Bias-aware intervals over a grid of `M` and the breakdown value `M*`, with
  the number of latent factors as a config field recorded on the result, and
  the EASC variance following the code, with the text's version available.
- EBSC-min and EASC* as an opt-in `weights=` choice, not the default.
- Inputs beyond `dataprep`: a per-unit exposure index (a column and a window to
  average over, or a mapping) and an optional observed-shock series for the
  diagnostics. A warning when the treated unit's exposure lies outside the
  exposure regression's support, the case where the correction extrapolates.

Next: `/new-estimator EASC` on its own branch. The durable case
`benchmarks/cases/easc_iran.py` follows once the estimator exists, reading
`benchmarks/reference/easc_iran/reference.json` through the estimator's result
accessors; the Monte Carlo DGP in `simulate.py` moves to
`utils/easc_helpers/simulation.py` as part of that build.
