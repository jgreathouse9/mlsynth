# MAREX and the debiased t-test

The #914 study (`marex_global_test_dependence`) measured MAREX's global test
under serially correlated factors. Eq. (17) rejects a true null 16.5% of the
time at an AR(1) coefficient of 0.8. The cyclic shifts with the signed statistic
bring that to 6.7-8.6%. Nothing it tried holds its level near a unit root: under
random-walk factors the cyclic test rejects 18-25% of the time.

Chernozhukov, Wuthrich and Zhu's debiased t-test (JPE 2026) tests the average
post-period effect differently, by cross-fitting. Its validity covers two cases
of nonstationarity. The first is drift shared by every unit (their
Assumption 4), which simplex weights cancel. The second is drift that differs
across units, provided the synthetic control is correctly specified and the
post period is short relative to the pre-period (Assumptions 5-6). This study
puts the test on the same MAREX designs, with the same seeds, beside eq. (17)
and the cyclic test.

## The test

The last `K r` pre-periods, `r = min(span // K, T1)`, form `K` consecutive
blocks. For block `k`, the synthetic control of the treated series is fitted on
the other pre-periods, and

    tau_k = mean_post(y - Y0 w_k) - mean_{block k}(y - Y0 w_k),

the post gap less the bias the fit shows on the block it did not see. The
estimate is the mean of the `tau_k`. Its standard error is
`sqrt(1 + K r / T1) sd(tau_k) / sqrt(K)`, and the statistic is referred to t with
`K - 1` degrees of freedom. On a MAREX design, `y` is the treated composite
`Y @ w` and `Y0` holds the units the design left untreated, the same pool
MAREX's inversion band refits on. The arms (`ttest.ARMS`):

- `refit_pre_K`: the library's `debiased_sc_ttest` as it ships, blocks over the
  whole pre-period (`scinference`'s placement);
- `refit_blank_K`: the blocks confined to the blank window. The design chose its
  treated set on the fitting window, so a block there shows an in-sample fit;
- `fixed_blank_K`: MAREX's own control weights, held fixed instead of refitted,
  which isolates what the refit contributes;

each at `K = 3` (the paper's starting point) and `K = 4`.

## Design

`#914`'s panel: Abadie and Zhao's Section 5 factor model under six regimes for
the factors, 104 and 156 pre-experiment weeks, MAREX's default blank window
(31 and 46 periods), 12 post weeks, `m_eq = 3`, the exact backend, and 2,000
draws per cell. The seeds are `#914`'s, and the eq. (17) and cyclic arms
reproduce its numbers exactly. An effect `tau` added to the treated units moves
every `tau_k` by `tau` and leaves the standard error, so the t-test's p-values at
every effect come from one fit per draw.

The cells below read "104 weeks / 156 weeks". The t-test at `K = 3` with
blocks confined to the blank window equals the unconfined test at 156 weeks:
its 36 block periods already lie inside the 46-period blank window.

## Results

Rejection rate of a true null at 5% (percent):

| Factors | eq. (17) | cyclic, signed | t, K=3 | t, K=4 | t, K=3, blank blocks | t, K=3, MAREX's control |
|---|---|---|---|---|---|---|
| independent | 5.2 / 5.2 | 4.7 / 2.7 | 5.0 / 4.4 | 5.6 / 5.0 | 4.0 / 4.4 | 4.6 / 5.0 |
| AR(1) 0.5 | 7.8 / 7.8 | 6.5 / 4.0 | 5.2 / 4.7 | 5.5 / 5.1 | 5.0 / 4.7 | 5.6 / 4.9 |
| AR(1) 0.8 | 16.9 / 16.5 | 8.6 / 6.7 | 5.5 / 5.9 | 6.7 / 6.1 | 5.7 / 5.9 | 6.6 / 5.7 |
| AR(1) 0.95 | 31.9 / 27.2 | 16.6 / 12.5 | 7.9 / 7.7 | 11.3 / 9.0 | 8.6 / 7.7 | 7.2 / 7.1 |
| random walk | 33.6 / 31.5 | 24.8 / 18.2 | 10.4 / 8.2 | 14.3 / 9.8 | 10.1 / 8.2 | 7.3 / 6.0 |
| season + AR(1) 0.5 | 7.4 / 8.2 | 6.0 / 3.9 | 5.1 / 4.9 | 5.2 / 4.9 | 5.0 / 4.9 | 6.9 / 5.5 |

The tests reject at different rates under the null, so their raw power is not
comparable. Each arm's size-adjusted power rejects at the largest cutoff that
rejects at most 5% of its own null draws. "n/a" marks an arm whose smallest
p-value already occurs in more than 5% of null draws, so no cutoff gives a 5%
test. Power at an effect of one error standard deviation (percent):

| Factors | eq. (17) | cyclic, signed | t, K=3 | t, K=4 | t, K=3, blank blocks | t, K=3, MAREX's control |
|---|---|---|---|---|---|---|
| independent | 22.2 / 24.6 | 39.7 / 48.4 | 28.9 / 32.2 | 39.4 / 43.2 | 31.1 / 32.2 | 23.5 / 26.5 |
| AR(1) 0.5 | 20.2 / 24.9 | 17.4 / 25.6 | 16.6 / 19.3 | 23.8 / 25.1 | 17.2 / 19.3 | 11.6 / 14.8 |
| AR(1) 0.8 | 14.4 / 14.3 | 15.8 / 14.0 | 11.8 / 11.6 | 14.9 / 13.7 | 11.2 / 11.6 | 8.9 / 9.8 |
| AR(1) 0.95 | n/a / n/a | n/a / n/a | 8.6 / 10.4 | 10.2 / 11.8 | 9.5 / 10.4 | 9.2 / 8.1 |
| random walk | n/a / n/a | n/a / n/a | 18.6 / 23.4 | 25.4 / 29.0 | 20.0 / 23.4 | 19.1 / 22.7 |
| season + AR(1) 0.5 | 26.1 / 29.0 | 21.3 / 27.0 | 16.4 / 19.0 | 24.6 / 28.0 | 15.6 / 19.0 | 12.5 / 14.0 |

At an effect of two:

| Factors | eq. (17) | cyclic, signed | t, K=3 | t, K=4 | t, K=3, blank blocks | t, K=3, MAREX's control |
|---|---|---|---|---|---|---|
| independent | 79.2 / 86.0 | 82.9 / 92.2 | 68.3 / 72.4 | 85.7 / 92.0 | 73.2 / 72.4 | 60.8 / 63.5 |
| AR(1) 0.5 | 65.7 / 71.5 | 52.8 / 68.4 | 45.0 / 48.9 | 62.7 / 64.9 | 44.2 / 48.9 | 31.4 / 37.8 |
| AR(1) 0.8 | 41.9 / 42.4 | 42.4 / 41.9 | 27.3 / 27.8 | 37.9 / 36.4 | 23.9 / 27.8 | 18.1 / 22.7 |
| AR(1) 0.95 | n/a / n/a | n/a / n/a | 20.5 / 24.0 | 27.9 / 28.7 | 21.6 / 24.0 | 20.0 / 19.1 |
| random walk | n/a / n/a | n/a / n/a | 46.2 / 57.0 | 64.3 / 72.2 | 50.3 / 57.0 | 48.4 / 56.4 |
| season + AR(1) 0.5 | 75.8 / 81.0 | 61.7 / 76.9 | 44.0 / 48.8 | 66.5 / 71.0 | 43.2 / 48.8 | 31.0 / 35.2 |

Median half-width of the 95% interval for the average effect, in error standard
deviations:

| Factors | refit, K=3 | refit, K=4 | blank blocks, K=3 | blank blocks, K=4 | MAREX's control, K=3 | MAREX's control, K=4 |
|---|---|---|---|---|---|---|
| independent | 1.48 / 1.41 | 1.11 / 1.08 | 1.55 / 1.41 | 1.20 / 1.08 | 1.70 / 1.61 | 1.35 / 1.25 |
| AR(1) 0.5 | 2.10 / 2.06 | 1.59 / 1.58 | 2.15 / 2.06 | 1.65 / 1.56 | 2.55 / 2.45 | 1.97 / 1.89 |
| AR(1) 0.8 | 2.97 / 2.95 | 2.21 / 2.24 | 3.01 / 2.95 | 2.19 / 2.20 | 3.60 / 3.49 | 2.57 / 2.63 |
| AR(1) 0.95 | 2.76 / 2.81 | 2.05 / 2.19 | 2.62 / 2.81 | 1.76 / 2.14 | 3.29 / 3.50 | 2.17 / 2.65 |
| random walk | 1.39 / 1.34 | 1.06 / 1.06 | 1.34 / 1.34 | 0.96 / 1.03 | 1.68 / 1.58 | 1.11 / 1.25 |
| season + AR(1) 0.5 | 2.15 / 2.02 | 1.52 / 1.47 | 2.18 / 2.02 | 1.63 / 1.47 | 2.45 / 2.53 | 1.79 / 1.87 |

The estimate of the average effect is centred: its mean under no effect lies
within 0.03 of zero in every cell and arm (`summary.json`, `*_bias`).

## Findings

1. The t-test holds its level where the permutation tests do not. At `K = 3` it
   rejects 4.4-5.9% of true nulls in every regime up to an AR(1) coefficient of
   0.8. At 0.8 that is 5.5% and 5.9%, against 8.6% and 6.7% for the cyclic test
   and 16.9% and 16.5% for eq. (17). At 0.95 the figures are 7.7-7.9% against
   12.5-16.6% and 27.2-31.9%. Under random-walk factors they
   are 8.2-10.4% against 18.2-24.8% and 31.5-33.6%. Near a unit root it is closer
   to its level than either permutation test, and still not at it.
2. Calibrated to the same size, the t-test at `K = 4` has the cyclic test's power
   under AR(1) 0.5 and 0.8 (23.8 / 25.1 against 17.4 / 25.6, and 14.9 / 13.7
   against 15.8 / 14.0, at an effect of one). With independent factors the
   cyclic test is ahead: 39.7 / 48.4 against 39.4 / 43.2. Under AR(1) 0.95 and
   random walks no cutoff gives either permutation test a 5% level. The t-test
   there has 8-12% power at an effect of one and 19-29% under random walks.
3. `K = 4` buys power for size. Against `K = 3` its intervals are a quarter
   shorter and its size-adjusted power at an effect of one is 13-50% higher. In
   return it rejects more true nulls under strong dependence: 9.0-11.3% at 0.95
   and 9.8-14.3% under random walks.
4. Refitting the control is what makes the test powerful. MAREX's own control
   held fixed gives intervals 10-26% wider than the refit with the same blocks,
   and less size-adjusted power in every regime. It has the best size under random walks, 7.3% and 6.0% at `K = 3`.
5. Confining the blocks to the blank window changes little. At 104 weeks the
   unconfined test puts five of its 36 block periods in the fitting window, where
   the design chose its treated set. Its size there is within a point of the
   confined test's in every regime, and neither placement is ahead across
   regimes. On these panels, the in-sample fit of the treated composite does not
   bias the debiasing step.

## Recommendation

The t-test is the most robust test of the average effect measured on MAREX
designs. It is also the only one that comes with an interval for that effect,
not only a p-value. It fits as a third `global_test` option, `"ttest"`, beside
`"subsets"` and `"cyclic"` (#916). It would refit the treated composite on the
untreated units with the library's `debiased_sc_ttest`, report the estimate,
standard error and interval on `MAREXInference`, and expose `K`. Which `K` to
default to is the author's call:

- `K = 3`: the paper's starting point, closest to the nominal level under
  strong dependence;
- `K = 4`: intervals a quarter shorter and about the cyclic test's power, at up
  to 4 points more size under strong dependence.

Under random walks every arm still rejects above 5%. A drift that the synthetic
control does not cancel is outside what any of these tests guarantee.

## Reproducing

```
python -m pytest benchmarks/studies/marex_ttest -q
python -m benchmarks.studies.marex_ttest.run 2000
python -m benchmarks.studies.marex_ttest.run report
python tools/mutation/run_mutants.py --target marex-ttest-study
```

## References

- Chernozhukov, V., Wuthrich, K. and Zhu, Y. (2026). Debiasing and t-tests for
  synthetic control inference on average causal effects. Journal of Political
  Economy 134(9). doi:10.1086/742424.
- Chernozhukov, V., Wuthrich, K. and Zhu, Y. (2021). An exact and robust
  conformal inference method for counterfactual and synthetic controls. JASA
  116(536), 1849-1864.
- Abadie, A. and Zhao, J. (2026). Synthetic controls for experimental design.
