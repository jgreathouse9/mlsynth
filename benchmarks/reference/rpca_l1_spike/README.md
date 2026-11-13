# RPCA-SC with a rank-k l1 fit -- spike for Gong, Lai & Ma (2026)

Demonstrate-first spike. No estimator is added.

Paper: Gong, P., Lai, L. & Ma, J. (2026), "Certifying optimality in nonconvex
robust PCA", arXiv:2601.21333. It studies `min ||X Y^T - M||_1` over rank-k
factors and shows that under incoherence and randomly placed sparse
corruptions the true factorisation is a Clarke-critical point, a sharp local
minimum at k = rank(L), and a strict saddle at k > rank(L). The certificate is
Lemma 1: a sign matrix that matches the residual's signs on its support, is
orthogonal to the factors, and stays within +-1 elsewhere.

Question: does a rank-k l1 denoiser, with that certificate, improve
`CLUSTERSC`'s RPCA-SC over the denoisers it ships (PCP, HQF, HSVT)?

## Verdict

Park. The l1 fit wins only where the paper's model holds exactly: noiseless
low rank plus sparse corruption. Add Gaussian noise of any size and PCP is as
good or better, and the certificate never certifies. On Bayani's
West-Germany panel the l1 fit's effect moves by 800 USD over k = 1..6 and does
not certify at any rank. Nothing here earns a build.

## What was run

* `l1rpca.py` -- the rank-k l1 fit (the paper's subgradient method from
  several small random starts and from the SVD, each finished by alternating
  exact least-absolute-deviation regressions so the residual's zeros are exact)
  and the criticality certificate as a linear program (`t* <= 1` iff critical).
* `run_sim.py` -- 240 placebo panels: 20 donors, T = 40, T0 = 30, a rank-3
  factor model, the treated unit in the donors' convex hull, Gaussian noise
  sigma in {0, 0.2, 0.5}, outlier patterns {none, 5% sparse, 15% sparse,
  spells of 5 periods on 30% of donors}, 20 seeds each. Every method denoises
  all 20 donors (no clustering), fits RPCA-SC's non-negative weights, and is
  scored by post-period RMSE against the treated unit's untreated signal.
* `run_germany.py` -- Bayani's West-Germany cluster (the panel
  `benchmarks/cases/clustersc_rpca_germany.py` cross-validates), the real 1990
  design and an in-time placebo at 1980.

`l1_true` uses the true rank 3; `l1_usvt` and `hsvt_usvt` use the rank USVT
picks, which is what a user would get.

## Results

Median post-period RMSE (`sim.csv`, `results.json`):

| sigma | pattern | oracle | PCP | HQF | HSVT (USVT) | l1 (rank 3) | l1 (USVT rank) |
|---|---|---|---|---|---|---|---|
| 0 | 5% sparse | 0 | 0.000 | 0.715 | 0.371 | 0.000 | 0.119 |
| 0 | 15% sparse | 0 | 0.010 | 0.787 | 0.617 | 0.040 | 0.169 |
| 0 | spells | 0 | 0.000 | 0.477 | 0.126 | 0.000 | 0.063 |
| 0.2 | none | 0.097 | 0.106 | 0.177 | 0.105 | 0.108 | 0.111 |
| 0.2 | 5% sparse | 0.064 | 0.115 | 0.746 | 0.408 | 0.144 | 0.204 |
| 0.2 | 15% sparse | 0.050 | 0.121 | 0.732 | 0.539 | 0.161 | 0.232 |
| 0.2 | spells | 0.074 | 0.123 | 0.289 | 0.205 | 0.265 | 0.263 |
| 0.5 | 5% sparse | 0.222 | 0.280 | 0.647 | 0.453 | 0.332 | 0.297 |
| 0.5 | 15% sparse | 0.168 | 0.323 | 0.807 | 0.535 | 0.344 | 0.358 |
| 0.5 | spells | 0.147 | 0.249 | 0.326 | 0.363 | 0.261 | 0.307 |

1. With no dense noise, PCP already recovers the signal (0.000 on 5% sparse
   and spells). The l1 fit matches it at the true rank and loses at 15%.
2. With noise, PCP is the best robust denoiser in every outlier cell; the l1
   fit at the true rank is 5-116% worse. Head to head over all 240 panels the
   l1 fit at rank 3 beats PCP in 45%.
3. At the rank a user would get (USVT), the l1 fit is worse again. USVT picks
   the true rank in 1 of 11 outlier cells: outliers inflate the spectrum. A
   rank-k method inherits that, and PCP, which has no rank to choose, does not.
4. HQF is worst in every outlier cell, by a wide margin. That is a finding
   about `rpca_method="HQF"` in its own right.

The certificate certifies the rank-3 l1 fit in 95% of noiseless 5%-sparse
panels and in no panel with any Gaussian noise. With dense noise the residual
is nonzero almost everywhere, the sign pattern is fixed on every entry, and the
orthogonality constraints cannot be met. As a diagnostic it says "the data are
not low rank plus sparse", which the noise level already says.

### West Germany (`germany.json`)

| denoiser | ATT 1990 | pre RMSE | placebo RMSE (1980) |
|---|---|---|---|
| PCP (1,000 iterations, the benchmark) | -1,501 | 88.6 | 189.6 |
| PCP converged | -1,500 | 88.4 | 189.0 |
| HQF | -2,147 | 127.9 | 91.8 |
| HSVT, USVT rank 4 | -2,247 | 94.1 | 766.0 |
| l1, k = 1 | -1,792 | 107.7 | 151.8 |
| l1, k = 2 | -1,692 | 106.1 | 97.8 |
| l1, k = 3 | -1,947 | 93.8 | 325.5 |
| l1, k = 6 | -2,513 | 95.3 | 494.2 |

No l1 fit certifies at any k, in either window; the residual is nonzero on
43-90% of entries. The in-time placebo ranks the l1 fit at k = 2 near HQF and
ahead of PCP, but k = 3 is 3.3 times worse and nothing in the data picks k = 2
in advance. One placebo year split on 11 donors is too thin to rank denoisers.

## The theory's conditions on mlsynth's panels

Theorem 1 needs rank r <= c sqrt(min(m, n)) / log^2(mn) and corruption
probability p <= c / log(mn). At 11 donors and 44 years the rank bound is
about 0.09 c, below 1 for any small c. Its model has no dense noise. Donor
panels have both, so the guarantees do not reach them.

## Other findings

* PCP stops unconverged on the benchmark panel at its 1,000-iteration default
  (relative residual 2.9e-5); filed as #1054. Converging it changes the West
  Germany weights by up to 0.06 and the ATT by 1 USD, and the simulation's
  `pcp_conv` and `pcp_1000` agree to three decimals.
* NNLS weights sum to more than 1 in every West Germany fit (1.05-1.16), which
  is extrapolation outside the donors' hull.
* The first simulation run stopped at panel 48 on a HiGHS status 15 in one LAD
  linear program (`sim_first_attempt.log`). LAD is always feasible and
  bounded; `lad` now retries HiGHS's dual simplex and interior point.

## Papers assessed alongside

Read at the user's request while this ran, no runs made:

* Peng et al. (2025), modified nuclear norm (arXiv:2507.18327): `||D(L)||_*`
  with D a time-difference operator targets smooth, low-rank donor signals.
  The candidate to measure next, as one more arm here.
* Zhou, Li, Wright, Candes & Ma (2010), stable PCP (`||M - L - S||_F <= delta`),
  reached through Park, Lee & Park (2024): the model these panels actually
  follow. The other arm to measure.
* Dorabiala, Aravkin & Kutz (2024), ensemble PCA: estimates directions, not a
  denoised matrix, and loses to RPCA on sparse corruption. Pass.

## Files

| file | contents |
|---|---|
| `l1rpca.py` | the l1 fit and certificate |
| `run_sim.py`, `sim.csv`, `sim.log`, `sim_first_attempt.log` | the simulation |
| `run_germany.py`, `germany.json`, `germany.log` | the West Germany runs |
| `analyse.py`, `results.json` | summaries |
