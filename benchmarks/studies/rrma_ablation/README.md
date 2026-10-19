# RRMA, demonstrated before it is built

Reference: Jiaqi Lv and Jianqiang Li (2026), "Rank-based Model Average Estimator
for Average Treatment Effects", arXiv:2610.15553v1. Section 2 is the estimator,
Algorithm 1 the candidate models, Section 4.2 the variance, Section 5 the
simulations (Table 1), Section 6 the Basque application (Table 4).

This is a replication spike. It produces a recommendation, not estimator code.

```
python -m benchmarks.studies.rrma_ablation.run                 # both parts, about 40 minutes on 4 cores
python -m benchmarks.studies.rrma_ablation.run --only basque   # Path A, about 2 minutes
python -m benchmarks.studies.rrma_ablation.run --only mc --reps 100
```

Results land in `results/rrma_ablation.json`; the numbers below are from that
file.

## The method

RRMA predicts the treated unit's untreated path from donor units in four steps.

1. Screen. Each donor is kept if its pre-period series is associated with the
   treated unit's by Kendall's tau (a rank correlation), in order of p-value.
2. Candidate models. The kept donors are split, in screen order, into groups of
   `m`; each group is one candidate model, fitted by weighted Wilcoxon rank
   regression: the slope minimises `sum over pairs t < t' of b_tt' |e_t - e_t'|`,
   the weighted absolute differences of residuals, so a single outlying period
   moves the fit far less than in least squares. The weights `b_tt' = h_t h_t'`
   (Mallows weights) shrink periods whose donor values sit far from the bulk, as
   measured by a minimum-volume-ellipsoid (MVE) estimate of location and scatter.
3. Averaging. Each candidate is refitted on growing windows of the pre-period
   (growing-window cross-validation, GWCV), its out-of-window predictions are
   collected, and model weights on the simplex minimise the same kind of
   pairwise absolute loss over those predictions.
4. The group size `m` is chosen by the smallest average absolute prediction
   error over the grid `M_min .. M_max`. The variance is a moving-block
   bootstrap (MBB) over the pre-period.

The paper's claim is that steps 2 and 3 together, a rank loss and averaging,
beat least-squares synthetic control under heavy-tailed pre-period errors.

## Path A: the Basque application

The port ingests the Basque panel through `dataprep` and is checked, quantity by
quantity, against an independent R implementation of the `m = 2` path
(`benchmarks/reference/rrma_basque/`):

| quantity | largest difference from R |
| --- | --- |
| screen order and the 8 pairs | identical (and the pairs equal the paper's, p. 29) |
| Mallows weights `h` | 5.6e-13 |
| GWCV predictions | 1.2e-14 |
| model-weight pair weights | 3.1e-13 |
| model weights | 7.8e-13 |
| fitted coefficients | 3.0e-14 |
| ATE | -0.5739153 in both |

Table 4 prints symmetric intervals, so their midpoint is the paper's estimate,
-0.5976, and their half-widths imply a standard error of 0.1247. The port's
-0.5739 is 0.024 away. That gap is not rounding, and the paper leaves enough
choices open that the following readings of the same `m = 2` path are all
defensible:

| `m = 2` reading | ATE | donors with weight |
| --- | --- | --- |
| as specified (Mallows weights over the whole pre-period, pair weights on) | -0.5739 | 1, 4, 8 |
| pair weights off | -0.5719 | 1, 4, 8 |
| Mallows weights off (plain Wilcoxon) | -0.9360 | 1, 4, 5, 7 |
| both off | -0.9502 | 1, 4, 5, 7 |
| Mallows weights recomputed on each training window | -0.9755 | 1, 4, 5, 7, 8 |
| the same, pair weights off | -0.6798 | 1, 4, 5, 7, 8 |
| model weights on levels, not pairwise | -0.5534 | 1, 4, 8 |

The paper reports positive weight on models 1, 4, 5 and 8 (p. 30); no reading
reproduces that support.

What decides the number, beyond those readings:

- The MVE sample. `MASS::cov.rob` enumerates every subset below 5,000 and samples
  above, so any group large enough makes the Mallows weights, and through them
  the choice of `m`, depend on the seed. Over 200 seeds the full Algorithm 1
  (`m` from 2 to 7) chose `m = 2` 92 times, 4 21 times, 5 61, 6 5 and 7 21. At
  `m = 2` every seed gives -0.5739; across all seeds the ATE runs from -3.41 to
  +0.77 (interquartile range -0.65 to -0.57), and none lands within 0.005 of
  the paper's -0.5976.
- The donor order. Fourteen of the sixteen donors share one Kendall p-value,
  1.2e-15: over the 20 pre-periods their statistics are identical, so the
  screen cannot order them, their order is the input order, and the pairs
  follow it. Under 200 random donor orders the `m = 2` ATE takes 200
  distinct values, from -2.62 to +0.25 (5-95 percent: -1.79 to -0.23).
- The ingestion key. Keyed by region name instead of number, `dataprep` sorts
  the donors alphabetically, which moves Asturias, and the `m = 2` ATE is
  +0.734: the opposite sign.
- Unidentified fits. The partition gives the last group the remainder, so at
  `m = 6` it holds 10 donors, and GWCV's first training window has 10
  pre-periods. Their differences span 9 directions for 10 coefficients, any
  point of a face of minimisers is optimal, and the prediction depends on
  which point the solver returns. The study takes the minimum-norm point and
  counts these fits: one in each of the 400 seed-sweep runs.

The standard error. The MBB with `B = 1000` gives 0.51 to 0.83 over five seeds,
four to seven times the paper's 0.1247. Dropping the factor `T1` from
`Cov(sqrt(T1) theta)` in equation (15) gives 0.12 to 0.19, which brackets the
paper's figure. That the paper computed it this way is a hypothesis, not a
reading of its text.

## The ablation the paper does not run

Table 1 credits RRMA as a whole. The study separates its two ingredients on the
paper's own Example 1 (`T1 = 60`, `T2 = 30`, `N = 25`, three latent factors; 500
replications per case, so Monte Carlo standard errors are about 7 percent of an
MSE). Two of the arms form a 2 x 2:

| | one model (all screened donors) | averaged over groups |
| --- | --- | --- |
| rank loss | `RANK_one` | `RRMA` |
| least squares | `OLS_one` | `LS_avg` |

`LS_avg` is RRMA with every rank fit and the weight step replaced by least
squares, on the same candidate groups. It is not the paper's LSMA, whose case a
MSE in Table 1 is 9.25 / 8.67 against `LS_avg`'s 6.57. Beside them: `RRMA_level`
(model weights fitted on levels, not pairwise), `RANK_one_u` (no Mallows
weights), equal weights `EW`, `MSCM` (Li 2020), and the panel data approach with
forward selection or lasso (`PDA_fs`, `PDA_lasso`).

MSE x 100 (Monte Carlo standard error), with the paper's two columns where it
reports the arm:

| arm | case a: N(0,1) | paper | case b: t3 | paper | case c: 10% N(0,25) | paper |
| --- | --- | --- | --- | --- | --- | --- |
| RRMA | 7.17 (0.48) | 7.57 / 7.17 | 10.65 (0.74) | 10.43 / 9.53 | 10.14 (0.66) | 8.10 / 8.32 |
| RRMA_level | 7.35 (0.46) | | 11.20 (0.78) | | 10.34 (0.67) | |
| LS_avg | 6.57 (0.43) | | 13.93 (1.06) | | 15.02 (1.01) | |
| RANK_one | 11.39 (0.70) | | 14.39 (0.98) | | 12.75 (0.82) | |
| RANK_one_u | 10.58 (0.63) | | 13.30 (0.87) | | 12.48 (0.84) | |
| OLS_one | 9.05 (0.56) | | 16.15 (1.43) | | 16.09 (1.03) | |
| EW | 9.67 (0.67) | 10.15 / 10.66 | 13.53 (1.01) | 14.28 / 12.89 | 12.46 (0.81) | 11.64 / 10.40 |
| MSCM | 7.39 (0.46) | 8.30 / 7.23 | 13.75 (1.04) | 13.08 / 15.01 | 15.02 (0.94) | 10.66 / 12.27 |
| PDA_fs | 7.55 (0.45) | | 15.03 (1.34) | | 17.28 (1.10) | |
| PDA_lasso | 7.30 (0.46) | | 15.79 (1.25) | | 18.60 (1.41) | |

The paper's two columns are independent Monte Carlo runs of one quantity
(`simulate.py` explains why), so their spread is the paper's own noise. Cases a
and b match it. In case c every method the paper reports has a higher MSE here,
RRMA by about a quarter and MSCM by a third, and RRMA stays first; the
contamination is coded as the paper states it (10 percent of pre-period errors
from a normal with variance 25), and the difference is unexplained.

The 2 x 2, as paired MSE ratios with 95 percent bootstrap intervals over
replications (below 1 means the first arm wins):

| ratio | case a | case b | case c |
| --- | --- | --- | --- |
| averaging, given the rank loss (`RRMA / RANK_one`) | 0.63 [0.56, 0.71] | 0.74 [0.65, 0.84] | 0.80 [0.70, 0.90] |
| averaging, given least squares (`LS_avg / OLS_one`) | 0.73 [0.65, 0.80] | 0.86 [0.76, 0.97] | 0.93 [0.84, 1.05] |
| rank loss, given averaging (`RRMA / LS_avg`) | 1.09 [1.02, 1.17] | 0.76 [0.69, 0.84] | 0.68 [0.61, 0.74] |
| rank loss, given one model (`RANK_one / OLS_one`) | 1.26 [1.17, 1.36] | 0.89 [0.77, 1.02] | 0.79 [0.71, 0.88] |
| pairwise model weights (`RRMA_level / RRMA`) | 1.03 [0.98, 1.07] | 1.05 [1.01, 1.10] | 1.02 [0.98, 1.06] |

What the ablation says:

- Averaging over donor groups helps in every case and under either loss, by 20
  to 37 percent of MSE given the rank loss. It is the larger ingredient when
  the errors are normal.
- The rank loss helps only when the pre-period errors have heavy tails: given
  averaging it cuts MSE by 24 percent under t3 errors and 32 percent under
  contamination, and costs 9 percent under normal errors, where least-squares
  averaging (6.57) beats RRMA (7.17).
- Fitting the model weights on pairwise differences instead of levels gains
  2 to 5 percent, distinguishable from zero only in case b.
- Against MSCM, RRMA ties under normal errors (ratio 1.03 [0.92, 1.16]) and wins
  by 29 and 48 percent in cases b and c.
- Least-squares averaging on RRMA's own candidate groups is a stronger baseline
  than the paper's LSMA: 6.57 in case a against the paper's 9.25 / 8.67, and
  better than RRMA there. Table 1 compares RRMA with a weaker version of its
  own second ingredient.

So RRMA's advantage in Table 1 is real in the cases built to show it, and it is
two ingredients of different kinds: averaging, which any loss can use, and a
robust loss, which pays only when the pre-period has outliers.

## The solvers

Every linear programme in the study is solved by NumPy code written for it; the
estimator path imports no SciPy (pinned in `benchmarks/tests/`), and HiGHS
remains only as the independent check the tests compare against.

- `solver_pairfree.py` solves the weighted Wilcoxon fits without any array
  indexed by pairs of periods. The pair multipliers are flows on a complete
  graph, Hoffman's circulation theorem reduces dual feasibility to a cut
  condition per tie group, and the cut condition reduces to a prefix test in
  one sort. Batched over all models of one size and warm-started window to
  window, a replication's 600 or so fits take 1.7 to 2.1 seconds, 10 to 11
  pivots each, and agree with HiGHS to 4e-16 relative in objective
  (`results/solver_pairfree.json`).
- `lp.py` solves the simplex-constrained weight programmes, pairwise and on
  levels, by the same vertex walk with constraints.
- `special.py` replaces `scipy.stats`: the normal quantile (AS241), the
  chi-square distribution through the regularized incomplete gamma, and the
  exact Kendall test with a tail summed in integers, so p-values that R reports
  as `1 - P` (and once as -4.4e-16) keep their digits.

The solvers were audited against Nocedal and Wright's *Numerical Optimization*
and Russinoff's *Formal Verification of Floating-Point Hardware Design*, and the
rules that came out of it are now library doctrine in
`agents/agents_solvers.md`. Every solve is certified on the original data by its
duality gap, with dual feasibility re-checked, and a failure raises
`CertificateWarning`; every fit the data do not identify raises
`UnidentifiedFitWarning`. The study counts both: zero failed certificates in
Path A and in all 1,500 replications.

Measured before those fixes, and pinned by tests and mutants now: data at a
scale of 1e-7 left the weight solves 0.3 to 0.6 percent from the optimum,
weights scaled by 1e-6 left one 19 percent from it, and data at a level of 1e8
left them 18 to 26 percent from it. None of that touched the study's own
numbers, whose data are of order one; it would have touched anyone else's.

## Recommendation

Do not build RRMA as specified.

- Its Basque estimate is not a function of the data. The sampled MVE, the order
  of tied Kendall p-values and the ingestion key each move the ATE by more than
  its reported standard error, the last one across zero, and the paper's
  number is not recovered under any seed tried.
- Its standard error, as printed in equation (15), is four to seven times the
  paper's figure.
- Algorithm 1's grid makes fits the data cannot identify on the paper's own
  application.
- Its simulation advantage decomposes into averaging, which least squares can
  carry and which then wins under normal errors, and a robust loss whose value
  depends on outliers in the pre-period.

If anything is built, it should be narrower: a robust-loss option for
an existing averaging estimator, with a deterministic screen (ties broken by
the statistic, not the input order), a fixed or exhaustive MVE, and group sizes
capped below the shortest training window. The pair-free solver here is the
piece that makes the rank loss cheap enough to offer.

## Files

| file | role |
| --- | --- |
| `run.py` | Path A and the Monte Carlo; writes `results/rrma_ablation.json` |
| `oracle.py` | the port: MVE, Mallows weights, screen, partition, GWCV, weights, choice of `m`, MBB |
| `simulate.py` | Example 1's data and every arm of the ablation; counts solver diagnostics per replication |
| `solver_pairfree.py` | the pair-free batched Wilcoxon solver and its measurement |
| `lp.py` | the constrained LP solvers for the model weights |
| `special.py` | normal, chi-square and Kendall functions |
| `paper_values.py` | Table 1, Table 4 and Section 6 as printed |
| `highs_reference.py` | HiGHS, for the tests only |
| `solver_pivots.py` | the earlier pair-based pivot measurement |

The R references are in `benchmarks/reference/rrma_basque/` (the `m = 2` path)
and `benchmarks/reference/rrma_special/` (`qnorm`, `qchisq`, `pchisq`,
`cor.test`). The tests are `benchmarks/tests/test_rrma_*.py`.
