# Classic SC against Graph-SC, plain and debiased, on four simulation designs

References:

- Yanlin Zhang, Sungyong Chung and Alireza Talebpour (2026), "Graph-Regularized
  Synthetic Control for Stable Counterfactual Estimation under Donor
  Redundancy", arXiv:2610.24113v1. The estimator under test.
- Victor Chernozhukov, Kaspar Wuthrich and Yinchu Zhu (2025), "Debiasing and
  t-tests for synthetic control inference on average causal effects",
  arXiv:1812.10820. The cross-fitted debiasing and t-test (CWZ below).
- Chengwang Liao, Zhentao Shi and Yapeng Zheng (2026), "A relaxation approach to
  synthetic control", arXiv:2508.01793. The latent-group design; they pair it
  with the CWZ t-test at five blocks.
- Xiaomeng Zhang, Wendun Wang and Xinyu Zhang (2022), "Asymptotic properties of
  the synthetic control method", arXiv:2211.12095. Hsiao and Zhou's (2019) DGP6
  with an exact post-period risk.

This is a study. It produces a recommendation, not estimator code. The
replication of the paper's own numbers is in `../graphsc_replicate`.

```
python -m benchmarks.studies.graphsc_mc.run        # 21,000 replications, about 20 minutes on 4 workers
```

Results land in `results/graphsc_mc.json`: for every cell and arm, the bias,
RMSE and MSE ratio to SC (with a paired bootstrap interval), the debiased
interval's coverage and length, and the design's own measure of weight quality.
Replications are seeded by cell and index, so a rerun reproduces every number.
The tests are `benchmarks/tests/test_graphsc_mc.py`; the mutants are the
`graphsc-mc-*` targets in `tools/mutation/targets.toml`, all 21 killed.

## The arms

| arm | weights |
| --- | --- |
| SC | the simplex, outcome-only |
| Graph-SC signed | eq. (2) over `R^J` (Definition 3.4), the paper's `lambda_1 = 0.05`, `lambda_g = 10`, `lambda_sum = 50`, a `k = 5` graph learned from the rows the weights are fit on |
| Random graph signed | the same, on a uniformly random graph with as many edges |
| Ridge signed | the same, with `L` replaced by the identity |
| Graph-SC, random graph and ridge, `w >= 0` | the same three over `w >= 0` (Algorithm 2's reading) |
| Graph-SC signed, raw units | the signed arm on the data as given |

The paper's lambdas are fixed constants, and they are not free of units: the fit
term grows with the square of the data's scale and the penalties do not. Every
arm but the last divides the rows it is fit on by the root-mean-square level of
their donors, which puts each panel on the order-one scale of the paper's own
simulation. The last solves on the raw data, as the paper does on Prop 99.

For each arm the plain estimate is the mean post-period gap of weights fit on
the whole pre-period. The debiased estimate and its 90 percent interval are
`mlsynth.utils.inferutils.debiased_sc_ttest` with the arm as `weight_fn`: `K`
blocks at the end of the pre-period, the weights refit without each block, the
held-out gap subtracted, a `t_{K-1}` interval. The oracle row is the same t-test
on the true weights.

## The designs

| family | drawn by | cells | replications | `K` | what is known |
| --- | --- | --- | --- | --- | --- |
| CWZ calibrated | `benchmarks/cwz_common.py`, calibration from `benchmarks/reference/cwz_ttest_mc` | DGPs 1 to 9; `J = 14`, `T0 = 30`, `T1 = 16` | 1000 | 3 | no effect; the systematic part `c + Y0 w` |
| Latent groups | `mlsynth/utils/laxscm_helpers/simulation.py` | `J` in {50, 100}, `T0` in {J/2, J, 2J}, exact and perturbed groups; `J = 200` exact; `T1 = 50` | 500 | 5 | no effect; oracle weights equal within groups |
| Hsiao-Zhou DGP6 | `designs.hz_design`, which reproduces `../hsiao_zhou_counterfactuals` draw for draw | `J` in {30, 50}, `T0` in {50, 100, 200, 400}, `T1 = 10` | 500 | 5 | no effect; the exact risk `R(w)` and its simplex minimum |
| Graph-SC design | `../graphsc_replicate/simulate.py` | 200 donors in 10 clusters, `T0 = 50`, `T1 = 20` | 500 | 5 | effect 2; the noiseless counterfactual |

`K = 3` is CWZ's own choice at `T0 = 30`; `K = 5` is Liao, Shi and Zheng's.

The latent-group design is the case built for a donor graph: donors in a group
share their loadings, and the oracle synthetic control spreads weight equally
inside each group, the smoothness the graph penalty rewards.

## Results

ATT MSE relative to SC's, plain estimates, range over each family's cells:

| arm | CWZ, DGPs 1-5 | CWZ, DGPs 6-9 (trends) | Latent groups | Hsiao-Zhou DGP6 | Graph-SC design |
| --- | --- | --- | --- | --- | --- |
| Graph-SC signed | 0.09 to 1.08 | 25.3 to 795 | 1.04 to 1.50 | 0.36 to 0.96 | 5.51 |
| Random graph signed | 0.09 to 1.07 | 61.3 to 5760 | 1.02 to 1.54 | 0.34 to 0.93 | 3.07 |
| Ridge signed | 0.23 to 1.07 | 67.5 to 5120 | 0.99 to 1.51 | 0.34 to 0.93 | 3.04 |
| Graph-SC w>=0 | 0.09 to 1.08 | 25.3 to 795 | 0.91 to 1.01 | 0.92 to 1.04 | 1.03 |
| Random graph w>=0 | 0.09 to 1.07 | 61.3 to 5763 | 0.92 to 1.01 | 0.92 to 1.03 | 1.18 |
| Ridge w>=0 | 0.23 to 1.07 | 67.5 to 5120 | 0.92 to 1.01 | 0.92 to 1.03 | 1.18 |
| Graph-SC signed, raw units | 0.63 to 1.50 | 1.11 to 43.3 | 1.28 to 2.60 | 0.19 to 0.91 | 7.13 |

Coverage of the debiased 90 percent interval, range over each family's cells:

| arm | CWZ, DGPs 1-5 | CWZ, DGPs 6-9 (trends) | Latent groups | Hsiao-Zhou DGP6 | Graph-SC design |
| --- | --- | --- | --- | --- | --- |
| SC | 0.89 to 0.91 | 0.60 to 0.92 | 0.89 to 0.93 | 0.89 to 0.92 | 0.88 |
| Graph-SC signed | 0.28 to 0.90 | 0.42 to 0.99 | 0.88 to 0.93 | 0.87 to 0.93 | 0.85 |
| Random graph signed | 0.28 to 0.89 | 0.75 to 0.98 | 0.89 to 0.93 | 0.88 to 0.94 | 0.83 |
| Ridge signed | 0.35 to 0.89 | 0.75 to 1.00 | 0.88 to 0.93 | 0.87 to 0.93 | 0.84 |
| Graph-SC w>=0 | 0.28 to 0.91 | 0.42 to 0.99 | 0.88 to 0.93 | 0.88 to 0.92 | 0.87 |
| Random graph w>=0 | 0.28 to 0.91 | 0.76 to 0.98 | 0.88 to 0.93 | 0.88 to 0.92 | 0.85 |
| Ridge w>=0 | 0.35 to 0.91 | 0.75 to 1.00 | 0.87 to 0.92 | 0.88 to 0.92 | 0.86 |
| Graph-SC signed, raw units | 0.02 to 0.91 | 0.55 to 0.79 | 0.89 to 0.95 | 0.89 to 0.94 | 0.85 |
| oracle weights | 0.88 to 0.91 | 0.90 | 0.88 to 0.92 | 0.88 to 0.91 | 0.91 |

Weight quality where the design defines it: the median, over replications, of
the post-period squared error against the oracle synthetic control relative to
SC's (latent groups), and the mean exact risk relative to the best simplex
weights (Hsiao-Zhou):

| arm | Latent groups, path error / SC's | Hsiao-Zhou, risk / simplex minimum |
| --- | --- | --- |
| SC | 1 | 1.04 to 1.18 |
| Graph-SC signed | 0.85 to 9.54 | 0.41 to 1.32 |
| Random graph signed | 0.87 to 10.01 | 0.39 to 1.28 |
| Ridge signed | 0.79 to 9.83 | 0.39 to 1.27 |
| Graph-SC w>=0 | 0.49 to 0.96 | 0.99 to 1.23 |
| Random graph w>=0 | 0.52 to 0.97 | 0.99 to 1.22 |
| Ridge w>=0 | 0.51 to 0.97 | 0.99 to 1.22 |
| Graph-SC signed, raw units | 2.05 to 22.16 | 0.23 to 1.18 |

### The graph against its own controls

In the signed reading the learned graph's ATT MSE is 0.97 to 1.07 of the random
graph's and 0.99 to 1.07 of ridge's over the 23 latent-group and Hsiao-Zhou
cells, and it is ahead of the random graph in 9 of them and of ridge in 5. It separates from them only in CWZ's DGPs 8 and
9, where donor trends make nearest neighbours meaningful, and there all three
are 25 to 5,800 times worse than SC.

In the `w >= 0` reading on latent groups the learned graph does carry
information, and little of it. Paired over replications, its path error is
0.942 to 0.996 of the random graph's and 0.951 to 0.996 of ridge's, every 95
percent interval below one, the edge shrinking as `J` and `T0` grow. The ATT
MSE ratios agree to two decimals. Against SC the same arm gains 4 to 51 percent
in path error, so nearly all of the gain is shrinkage, not the graph. In the
Graph-SC paper's own design the `w >= 0` graph beats its controls (1.03 against
1.18 of SC's MSE) and still loses to SC.

### Where Graph-SC beats SC, and what does the work

Hsiao-Zhou DGP6. The signed arms have 0.34 to 0.96 of SC's MSE, an exact risk
0.39 to 1.32 of the best simplex weights against SC's 1.04 to 1.18, and debiased
intervals 0.57 to 1.10 of SC's length at coverage 0.87 to 0.94. The treated
unit's loadings are a standard normal draw and mostly lie outside the donor
hull; signed weights can follow it and simplex weights cannot. Ridge and the
random graph do as well as the learned graph, and the raw arm, which shrinks
least on this design's scale, does best (0.19 to 0.91). The gain is signed
weights, and shrinkage gives part of it back.

Latent groups. The `w >= 0` arms trace the oracle path more closely than SC
(0.49 to 0.97 of its error) at the same ATT MSE (0.91 to 1.01). This is
shrinkage toward equal weights, which the oracle has inside each group, and
ridge earns it as well as the graph. Liao, Shi and Zheng report their L2
relaxation, RESCM's `RELAX_L2`, at 0.15 to 0.53 of SC's error on this design.

CWZ DGPs 2, 4 and 5. DGP 2's true weights are uniform, which every shrunken arm
moves toward (0.77). In DGPs 3 to 5 the treated unit carries a constant 2 that
no weight vector reproduces, so every plain estimate is off by about 2; under
DGP 4's common trend a weight sum away from one absorbs part of it (0.09). The
debiased estimates show what that costs.

### The debiased t-test needs a bias that does not drift

The CWZ correction subtracts the gap on a held-out pre-period block from the
post-period gap, which removes the bias when it is the same in both windows.
SC fits a trend exactly and keeps it constant. A shrunken weight vector does
not: in DGP 4 Graph-SC's weights miss summing to one by 0.036 on average, and
over 40 draws its mean gap is +1.07 in the last pre-period block and +0.60 after
treatment. The correction subtracts the wrong amount, and coverage falls to
0.28 against SC's 0.89. In DGPs 6 to 9 the shrunken arms either undercover (0.42
in DGP 9, 0.75 to 0.77 in DGP 7) or reach nominal coverage only with intervals 14
to 34 times SC's (every shrunken arm in DGP 6, the random-graph and ridge arms in
DGP 8). In the stationary designs every arm covers between 0.87 and 0.95.

Classic SC reproduces CWZ's own pattern: 0.89 to 0.92 except DGP 6 (0.81) and
DGP 8 (0.60), the shape of their Table 3, which `benchmarks/cases/cwz_ttest_mc.py`
pins against the authors' code. The oracle row covers 0.88 to 0.92 throughout.

### The paper's lambdas have no scale

No single normalization makes the fixed lambdas right. Divided by the RMS level,
Graph-SC fails where a trend sets that level: in DGP 4 the variation across
donors is 1 percent of the scale, the penalties swamp the fit, and the MSE ratio
in DGPs 6 to 9 reaches 795. On raw data the same arm is 1.3 to 2.6 times worse
than SC on latent groups and the best arm on Hsiao-Zhou. A build would need
tuned lambdas, and the paper fixes them without tuning.

## Recommendation

Do not build Graph-SC, as an estimator or as a RESCM method.

- Its graph adds nothing over a random graph or ridge in the signed reading, and
  0.4 to 6 percent of path error in the `w >= 0` reading on the design built for
  it.
- Where it beats SC, signed weights or plain shrinkage do the work, and RESCM
  already holds or can hold both.
- Its fixed lambdas are not scale-free, and under trends its bias drifts, so the
  CWZ t-test, the inference mlsynth would pair it with, fails.

The study leaves four issues:
[#875](https://github.com/jgreathouse9/mlsynth/issues/875) (RESCM `LASSO` returns
the SC weights),
[#876](https://github.com/jgreathouse9/mlsynth/issues/876) (the active-set solver
below for RESCM's penalized branch),
[#877](https://github.com/jgreathouse9/mlsynth/issues/877) (spike: signed-weight
ridge as a RESCM method) and
[#878](https://github.com/jgreathouse9/mlsynth/issues/878) (the debiased t-test's
documentation overstates which weight functions it composes with).

## The solver

Every signed solve uses `solver.py`: an active-set (feature-sign) search,
warm-started across the cross-fitting folds, whose returned point is re-solved
by QR of the stacked design and certified by its KKT residual on the caller's
data. Its tolerances are measured, not chosen: an optimality condition holds at
`64 eps` of its own terms (worst observed `1.9 eps`), and a returned point at a
KKT residual of `1e-12` (worst observed `3.2e-16`). A capped search, a singular
system or a failed certificate falls back to CLARABEL and keeps the better
certified point; a point nothing certifies carries a `CertificateWarning`. Over
21,000 replications no solve fell back and none was uncertified. On five fold
refits it is 8 to 27 times faster than CLARABEL (2 ms against 54 at `J = 50`,
57 ms against 1073 at `J = 200`). The tests check it against an exact rational
optimum found by enumerating sign patterns, and require bitwise-identical
weights when the data and lambdas are rescaled by exact powers of two.
