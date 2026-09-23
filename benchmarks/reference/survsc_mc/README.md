# SSC Path-B targets — Han & Shah (2025), Section 4

A pre-build replication spike for Synthetic Survival Control
([arXiv:2511.14133v1](https://arxiv.org/abs/2511.14133)), the synthetic-control
estimator for censored time-to-event outcomes. No estimator has been added to
`mlsynth` yet; this bundle establishes that the paper's Monte Carlo reproduces
before one is built, and records the design details the paper leaves unstated.

The paper ships no code and no data. Its clinical application uses proprietary
multi-country T-cell-lymphoma records, so Table 1 is the only external check
that exists.

## What is here

```
survsc_oracle.py   readable port of the estimator + the Section 4 DGPs
run_grid.py        driver: the grid against Table 1
dump_results.py    regenerates results.json (every number below)
km_floor.py        how much of the Table 1 error is the Kaplan-Meier step
results.json       generated
```

```bash
python benchmarks/reference/survsc_mc/dump_results.py
python benchmarks/reference/survsc_mc/km_floor.py
```

## The estimator

Four steps (paper Section 3.4.2). Kaplan-Meier per unit-period; subsample the
curves onto a shared grid of `T0` points; principal-component regression of the
treated unit's pre-period curve on the donors' pre-period curves; apply those
weights to the donors' post-period curves.

Step 3 needs nothing new. The paper's closed form,
`w = sum_{i<=r0} (1/s_i) v_i u_i' S_0n`, is `mlsynth.utils.pcr.pcr_weights`
verbatim — they agree to 8e-17 across ranks 1, 3, 6 and 10.

## What reproduced

Table 1 is the sup-norm error against the true post-period control survival
trajectory. Both Section 4 DGPs fix the baseline hazard, so event times are
exponential and the estimand has the closed form `exp(-rate * t)`; the error is
measured against that, not against a simulated approximation.

Twenty replications, `r0` by USVT, horizon pooled, weights as the paper
specifies them:

| DGP | K | this port | Table 1 | ratio |
|---|---|---|---|---|
| Cox | 100 | 0.1061 +/- 0.1318 | 0.1177 +/- 0.0835 | 0.90 |
| Cox | 300 | 0.0949 +/- 0.1298 | 0.0652 +/- 0.0497 | 1.46 |
| Cox | 700 | 0.0724 +/- 0.1251 | 0.0542 +/- 0.0413 | 1.34 |
| Aalen | 100 | 0.0796 +/- 0.0394 | 0.0621 +/- 0.0307 | 1.28 |
| Aalen | 300 | 0.0436 +/- 0.0158 | 0.0507 +/- 0.0388 | 0.86 |
| Aalen | 700 | 0.0337 +/- 0.0177 | 0.0245 +/- 0.0102 | 1.38 |

Every cell lands within 1.5x, two of them on top of the published value, and the
Figure 4 claim — error falling in `K` — holds in both DGPs.

## What did not

The confounder-aware parametric comparator. This port fits the correctly
specified exponential model on the true latent factors and gets 0.0120 / 0.0068
/ 0.0033 on Cox against the paper's 0.0247 / 0.0204 / 0.0194. Two to six times
more accurate, and falling much faster in `K`. The likely cause is the baseline:
Section 3.3.2 fits a Cox model with a Breslow cumulative-hazard estimator, whose
nonparametric baseline converges more slowly than the parametric fit here. The
gap between SSC and the comparator is therefore wider in this port than in the
paper, which flatters neither method — it makes SSC look worse.

The Cox standard deviations are also wider here (0.1318 against 0.0835 at
K = 100), driven by the design draw, not the patient draw; see below.

## Three details the paper does not state

The rank `r0`. Decisive. Section 3.4.2 says "a gap rule, elbow, or
cross-validation as explained in Agarwal et al." and picks none of them. Of the
rules `mlsynth` ships, only USVT reproduces the paper; the other two get the
sign of the K-effect wrong, reporting error that grows as data is added:

| rule | mean r0 | Cox K=100 / 300 / 700 | Aalen K=100 / 300 / 700 |
|---|---|---|---|
| cumvar (0.95) | 1.0-2.5 | 0.158 / 0.205 / 0.223 | 0.247 / 0.255 / 0.259 |
| spectral | 1.6-2.1 | 0.274 / 0.281 / 0.286 | 0.132 / 0.129 / 0.125 |
| usvt | 5.5-6.9 | 0.106 / 0.095 / 0.072 | 0.080 / 0.044 / 0.034 |
| Table 1 | — | 0.118 / 0.065 / 0.054 | 0.062 / 0.051 / 0.025 |

Whether the latent design is redrawn per replication. Also decisive, for the
Figure 4 claim specifically. Under this DGP the latent draw moves the evaluation
horizon over two orders of magnitude — the pooled 90th percentile of observed
times has median 93 and range [20, 5743] across Cox draws — which swamps the
effect of `K`. Drawing an independent design at each `K` breaks the monotone
decrease (Cox runs 0.143 / 0.051 / 0.085); holding the design fixed across `K`,
as common random numbers, restores it. The grid above uses common random
numbers. The paper says the factors are "fixed throughout the simulation", which
is consistent with this but does not settle it.

Whether `tau_tilde` pools every cell or uses the treated unit. Not decisive.
Reading `quantile_0.90({T_{p,n,i}})` as the pooled sample gives the table above;
reading it as the treated unit's own times gives 0.1182 / 0.0999 / 0.0766 on Cox
and 0.0827 / 0.0573 / 0.0429 on Aalen, the same picture. Carried in
`results.json` as `grid_pcr_treated`.

## Two findings that bear on a build

The estimator's output is usually not a survival function. The PCR weights are
unconstrained — Assumption 6 asks only that the treated unit lie in the linear
span of the donors — so the counterfactual `sum_m w_m S_m(t)` need not be
monotone or stay inside [0, 1]. Across the grid it is monotone in 0-55 percent
of replications and inside [0, 1] in 15-45 percent. The paper does not raise
this; with nine donors its own figures happen to come out clean. The worst cell
is Aalen at K = 100, where no replication out of twenty returns a monotone
curve.

Abadie's convex-hull condition fixes it for free, because a convex combination
of monotone [0, 1] curves is monotone in [0, 1]. Replacing step 3 with the
simplex fit already in `mlsynth.utils.inferutils._outcome_only_simplex` gives
monotone, in-range output in 100 percent of replications at every cell, and
costs nothing on average — better than PCR on Cox at all three sample sizes
(0.0909 / 0.0786 / 0.0675), slightly worse on Aalen (0.0864 / 0.0560 / 0.0450),
with the K-monotonicity intact in both.

Table 1 is mostly measuring Kaplan-Meier. `km_floor.py` measures the sup-norm
distance between one unit-period Kaplan-Meier curve and its own truth, on the
estimator's grid:

| DGP | K | one KM curve | Table 1 SSC | ratio |
|---|---|---|---|---|
| Cox | 100 | 0.0658 | 0.1177 | 1.79 |
| Cox | 300 | 0.0376 | 0.0652 | 1.73 |
| Cox | 700 | 0.0235 | 0.0542 | 2.30 |
| Aalen | 100 | 0.0812 | 0.0621 | 0.76 |
| Aalen | 300 | 0.0457 | 0.0507 | 1.11 |
| Aalen | 700 | 0.0293 | 0.0245 | 0.84 |

The reported error is the same order as a single Kaplan-Meier curve's error, and
on Aalen it sits below one — the weighted average over nineteen donors averages
part of that noise away. The Kaplan-Meier error falls at 1/sqrt(K) (0.0658 ->
0.0376 -> 0.0235 tracks sqrt(1/3) and sqrt(3/7) to two digits), which is where
the K-dependence in Figure 4 comes from. This squares with Theorem 2, whose rate
contains no `K` at all: `K` enters the theorem only as a threshold for the PCR
stability argument, and the convergence the simulation displays lives in step 1,
which the theorem treats as a precondition.

## A robustness gap in mlsynth, found here

`mlsynth/utils/pcr/core.py:78` divides by `s_r` with no guard:

```python
return Vt_r.T @ ((U_r.T @ target) / s_r)
```

When the requested rank exceeds the design matrix's numerical rank this returns
NaN silently. It is reachable on this design: the Cox rates span `exp(+/-4)`, so
draws with a long horizon leave most donor curves flat at zero and the donor
matrix near rank one. `survsc_oracle.py` caps the rank at the numerical rank
before calling `pcr_weights`. The guard belongs upstream, on its own branch —
every PCR caller (`SI`, `CLUSTERSC`, `RSC`) shares the exposure.
