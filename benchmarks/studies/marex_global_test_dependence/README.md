# MAREX's global test under serially dependent residuals (#914)

MAREX's global test is Abadie and Zhao's eq. (17): the mean absolute contrast
over the post periods, eq. (16), ranked among the same statistic over every
``H``-subset of the blank and post periods. Their Theorem 2 makes it exact when
the factor loadings ``lambda_t`` are exchangeable over those periods, and
Theorem OA.1 keeps it approximately valid for fixed ``lambda_t`` as the number
of fitting periods grows. On simulated weekly geo panels, #914 found it
rejecting a true null 8.5% of the time with 80 weeks of history and 16.5% with
random-walk demand, and traced most of it to serial correlation in the
contrast: a contiguous post block varies more than a random subset of periods.

This study asks which reference set and statistic restore the test's size with
two or more years of pre-treatment data, and what each costs in power.

## Design

- The panel is Abadie and Zhao's Assumption 1 at their Section 5 dimensions,
  ``J = 15`` units, observed and unobserved covariates on Uniform(0, 1), errors
  N(0, 1), the ``marex_cumulative_conformal`` study's panel. The period effects
  ``delta_t``, ``theta_t`` and ``lambda_t`` follow one of six regimes: drawn
  independently (``iid``, the exchangeable case), a stationary AR(1) at 0.5,
  0.8 and 0.95, a random walk, and an annual cycle plus an AR(1) at 0.5
  (``season``), each with the spread of the uniform.
- ``T0`` is 104 or 156 weeks; MAREX's default blank window is then 31 or 46
  periods; 12 post periods; MAREX's design is solved on the fitting periods
  (``m_eq = 3``, exact backend).
- A constant effect ``tau`` (0, 0.5, 1 and 2, in units of the errors' standard
  deviation) is added to the treated units in the post periods, which moves the
  contrast by ``tau``.
- 2,000 draws per regime and ``T0``, so a size of 5% carries a Monte Carlo
  standard error of 0.5 points.

The reference sets and statistics (``globaltest.py``):

- subsets, eq. (16): what MAREX ships, computed by the library's own
  ``compute_inference`` (``panels.shipped_pvalue``, checked against MAREX's
  p-value);
- cyclic shifts: the ``N = T_b + H`` rotations of the blank-then-post contrast,
  Chernozhukov, Wuthrich and Zhu's (2021) moving-block permutations, valid for
  stationary weakly dependent residuals. The observed block is one of the
  rotations, so across the rotations of any series at most ``floor(alpha N)``
  are rejected (the tests pin this). With 31 and 46 blank periods ``N`` is 43
  and 58, so the p-value's floor is 1/43 and 1/58;
- weighted cyclic shifts: the same, each rotation weighted ``0.97`` to its
  distance from the post window (Barber, Candes, Ramdas and Tibshirani 2023);
- blank windows: the post block against the windows of ``H`` consecutive blank
  periods only, so an effect cannot enter the reference;
- the signed statistic: the absolute value of the mean contrast over a block,
  CWZ's Remark 1 statistic for an average effect, in place of eq. (16).

## Results

| | iid 104 | iid 156 | ar5 104 | ar5 156 | ar8 104 | ar8 156 | ar95 104 | ar95 156 | rw 104 | rw 156 | season 104 | season 156 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Blank periods | 31 | 46 | 31 | 46 | 31 | 46 | 31 | 46 | 31 | 46 | 31 | 46 |
| Lag-1 autocorrelation, blank contrast | -0.04 | -0.03 | 0.31 | 0.34 | 0.50 | 0.52 | 0.48 | 0.54 | 0.18 | 0.20 | 0.31 | 0.33 |

### Size at 5%, no effect (percent)

| Reference, statistic | iid 104 | iid 156 | ar5 104 | ar5 156 | ar8 104 | ar8 156 | ar95 104 | ar95 156 | rw 104 | rw 156 | season 104 | season 156 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Subsets, eq. (16) (shipped) | 5.2 | 5.2 | 7.8 | 7.8 | 16.9 | 16.5 | 31.9 | 27.2 | 33.6 | 31.5 | 7.4 | 8.2 |
| Cyclic, eq. (16) | 4.8 | 3.8 | 4.7 | 3.0 | 6.9 | 5.8 | 14.5 | 10.5 | 21.2 | 14.7 | 4.8 | 3.8 |
| Cyclic, eq. (16), weighted | 2.5 | 2.5 | 2.0 | 1.4 | 4.3 | 3.9 | 9.5 | 6.5 | 13.7 | 10.2 | 2.2 | 2.5 |
| Blank windows, eq. (16) | 20.6 | 12.4 | 19.8 | 13.5 | 23.6 | 17.5 | 33.6 | 23.4 | 42.8 | 31.0 | 17.8 | 13.5 |
| Subsets, signed | 5.7 | 3.9 | 17.8 | 16.6 | 30.2 | 30.6 | 42.9 | 39.2 | 43.2 | 41.1 | 20.2 | 21.2 |
| Cyclic, signed | 4.7 | 2.7 | 6.5 | 4.0 | 8.6 | 6.7 | 16.6 | 12.5 | 24.8 | 18.2 | 6.0 | 3.9 |
| Cyclic, signed, weighted | 2.4 | 1.6 | 3.4 | 2.5 | 5.1 | 5.0 | 11.6 | 8.9 | 16.6 | 12.8 | 3.4 | 2.4 |

### Size at 10%, no effect (percent)

| Reference, statistic | iid 104 | iid 156 | ar5 104 | ar5 156 | ar8 104 | ar8 156 | ar95 104 | ar95 156 | rw 104 | rw 156 | season 104 | season 156 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Subsets, eq. (16) (shipped) | 9.8 | 10.4 | 13.9 | 13.4 | 21.5 | 22.0 | 36.3 | 32.3 | 40.2 | 38.0 | 12.2 | 13.7 |
| Cyclic, eq. (16) | 9.8 | 9.1 | 9.2 | 8.4 | 11.5 | 10.9 | 20.3 | 17.4 | 29.5 | 25.1 | 8.7 | 8.6 |
| Cyclic, eq. (16), weighted | 6.7 | 6.6 | 6.2 | 6.2 | 8.2 | 8.5 | 15.8 | 13.7 | 23.7 | 19.6 | 6.3 | 6.2 |
| Blank windows, eq. (16) | 24.2 | 17.3 | 23.4 | 17.8 | 26.6 | 20.4 | 36.2 | 26.9 | 45.6 | 35.3 | 21.9 | 18.0 |
| Subsets, signed | 10.8 | 8.9 | 24.9 | 23.4 | 35.9 | 36.5 | 47.3 | 44.6 | 50.5 | 47.9 | 27.3 | 28.7 |
| Cyclic, signed | 10.2 | 8.0 | 11.3 | 9.7 | 13.5 | 11.2 | 23.1 | 18.7 | 33.7 | 27.8 | 9.6 | 8.8 |
| Cyclic, signed, weighted | 7.2 | 5.7 | 8.4 | 7.2 | 10.3 | 9.1 | 18.9 | 16.2 | 27.0 | 23.1 | 7.1 | 6.6 |

### Power at 5%, constant effect 0.5 (percent)

| Reference, statistic | iid 104 | iid 156 | ar5 104 | ar5 156 | ar8 104 | ar8 156 | ar95 104 | ar95 156 | rw 104 | rw 156 | season 104 | season 156 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Subsets, eq. (16) (shipped) | 8.1 | 9.2 | 11.8 | 14.3 | 20.8 | 21.3 | 33.1 | 30.7 | 41.3 | 38.9 | 12.4 | 14.1 |
| Cyclic, eq. (16) | 7.4 | 5.3 | 7.0 | 5.5 | 9.3 | 8.2 | 17.2 | 13.1 | 27.9 | 22.9 | 6.1 | 6.5 |
| Cyclic, eq. (16), weighted | 3.8 | 3.4 | 3.3 | 3.3 | 6.0 | 5.0 | 11.5 | 9.8 | 20.1 | 17.1 | 3.5 | 4.5 |
| Blank windows, eq. (16) | 27.5 | 17.2 | 25.0 | 20.1 | 27.8 | 22.3 | 35.4 | 25.4 | 48.5 | 38.7 | 25.1 | 20.1 |
| Subsets, signed | 19.7 | 19.2 | 26.7 | 29.5 | 33.7 | 35.3 | 45.4 | 41.8 | 50.6 | 49.5 | 30.6 | 33.1 |
| Cyclic, signed | 14.6 | 11.8 | 12.2 | 9.6 | 12.4 | 9.7 | 19.7 | 15.3 | 32.0 | 27.6 | 12.1 | 8.2 |
| Cyclic, signed, weighted | 7.6 | 7.4 | 7.8 | 6.0 | 8.4 | 6.6 | 14.3 | 12.2 | 23.8 | 21.0 | 7.8 | 5.5 |

### Power at 5%, constant effect 1 (percent)

| Reference, statistic | iid 104 | iid 156 | ar5 104 | ar5 156 | ar8 104 | ar8 156 | ar95 104 | ar95 156 | rw 104 | rw 156 | season 104 | season 156 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Subsets, eq. (16) (shipped) | 23.8 | 25.0 | 25.9 | 31.2 | 32.1 | 32.5 | 40.6 | 37.5 | 55.8 | 55.9 | 31.0 | 35.8 |
| Cyclic, eq. (16) | 16.8 | 13.9 | 14.1 | 14.4 | 15.7 | 15.1 | 25.1 | 21.3 | 43.5 | 42.9 | 18.0 | 18.1 |
| Cyclic, eq. (16), weighted | 8.7 | 8.6 | 8.7 | 9.7 | 10.9 | 10.0 | 19.1 | 16.9 | 34.2 | 33.9 | 11.0 | 11.9 |
| Blank windows, eq. (16) | 47.5 | 38.0 | 42.1 | 36.8 | 38.3 | 31.8 | 43.5 | 33.4 | 60.2 | 56.3 | 46.2 | 41.4 |
| Subsets, signed | 58.0 | 61.7 | 50.9 | 55.0 | 47.3 | 50.4 | 51.9 | 49.3 | 65.0 | 65.4 | 57.1 | 61.9 |
| Cyclic, signed | 39.7 | 38.6 | 25.2 | 25.6 | 22.1 | 19.1 | 28.6 | 24.0 | 50.8 | 50.0 | 30.6 | 27.0 |
| Cyclic, signed, weighted | 25.1 | 24.7 | 17.4 | 18.3 | 15.8 | 14.5 | 22.9 | 20.3 | 41.4 | 40.9 | 21.5 | 19.1 |

### Power at 5%, constant effect 2 (percent)

| Reference, statistic | iid 104 | iid 156 | ar5 104 | ar5 156 | ar8 104 | ar8 156 | ar95 104 | ar95 156 | rw 104 | rw 156 | season 104 | season 156 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Subsets, eq. (16) (shipped) | 81.2 | 86.1 | 71.7 | 77.0 | 60.7 | 65.3 | 60.9 | 60.5 | 79.8 | 80.1 | 79.7 | 85.0 |
| Cyclic, eq. (16) | 60.0 | 58.4 | 47.9 | 50.3 | 39.6 | 41.5 | 49.5 | 46.3 | 74.0 | 74.5 | 58.7 | 61.7 |
| Cyclic, eq. (16), weighted | 38.8 | 39.2 | 33.8 | 37.8 | 31.1 | 33.1 | 42.4 | 41.0 | 66.8 | 67.2 | 43.9 | 46.9 |
| Blank windows, eq. (16) | 92.8 | 91.3 | 81.3 | 79.5 | 65.0 | 62.8 | 62.5 | 55.6 | 81.8 | 79.8 | 85.9 | 86.5 |
| Subsets, signed | 98.2 | 99.0 | 91.8 | 94.2 | 78.1 | 80.2 | 70.7 | 70.8 | 87.1 | 85.7 | 93.2 | 96.9 |
| Cyclic, signed | 82.9 | 84.8 | 66.7 | 68.4 | 49.7 | 50.2 | 53.4 | 49.5 | 80.6 | 80.0 | 74.0 | 76.9 |
| Cyclic, signed, weighted | 63.0 | 65.7 | 52.9 | 56.5 | 42.6 | 43.1 | 48.0 | 45.8 | 76.4 | 75.2 | 61.7 | 63.3 |


## Findings

1. The shipped test is exact with independent period effects (5.2% at both
   lengths) and over-rejects as soon as they carry memory: 7.4% to 8.2% where
   the contrast's lag-1 autocorrelation is about 0.3 (AR(1) at 0.5, the
   season), 16.5% to 16.9% at an AR(1) of 0.8, and 27% to 34% near and at a
   unit root. With stationary period effects a third year of history does not
   change this: the size at 156 weeks is within a point of the size at 104.
2. Cyclic shifts with the signed statistic restore the size where the period
   effects are stationary and weakly dependent: 2.7% to 6.5% with independent,
   AR(1) 0.5 and seasonal period effects, and 6.7% to 8.6% at an AR(1) of 0.8.
   Against a constant effect, where the shipped test holds its size (independent
   period effects), they have more power at an effect of 1 (39.7% and 38.6%
   against 23.8% and 25.0%) and about the same at an effect of 2 (82.9% and
   84.8% against 81.2% and 86.1%).
3. Cyclic shifts with eq. (16) hold the size about as well (6.9% and 5.8% at an
   AR(1) of 0.8) and lose a quarter to a third of the power: 58% to 60% at an
   effect of 2 with independent period effects, against 81% to 86%. The windows next to the post block hold most
   of its periods, effect included, and an absolute statistic reads them as
   extreme too; a signed mean averages the effect in as a level the observed
   block holds in full.
4. Recency weights bring the signed cyclic test to the level at an AR(1) of 0.8
   (5.1% and 5.0%) and make it conservative with independent, AR(1) 0.5 and
   seasonal period effects (1.6% to 3.4%), for about a quarter of the
   unweighted test's power at an effect of 2 (63.0% and 65.7% against 82.9% and
   84.8%).
5. Blank windows are not a valid test: 12% to 21% with independent period
   effects. The overlapping windows share most of their periods, so they act
   as two or three independent draws, and the observed block, independent of
   all of them, exceeds them all far more often than the reference's size
   implies. The cyclic shifts avoid this because the observed block is one of
   the rotations.
6. The signed statistic on random subsets is the worst arm under dependence,
   16.6% to 30.6% at AR(1) 0.5 to 0.8: a sum of correlated errors is what a
   random subset misjudges most.
7. No reference set holds the size with random-walk factors (10.2% to 43.2%).
   At an AR(1) of 0.95 the weighted cyclic test with eq. (16) comes closest, at
   6.5% and 9.5%, and the signed cyclic test reaches 12.5% and 16.6%. Moving
   blocks need stationary residuals; a random walk in the factors puts a drift
   in the contrast that every reference block sits earlier on than the post
   block. The ``marex_cumulative_conformal`` study's detrending arms are the
   route there.
8. The contrast's lag-1 autocorrelation does not identify the failing regimes:
   it is 0.18 to 0.20 under the random walk, lower than under the stationary
   AR(1) at 0.5. A diagnostic for the drift has to look at the contrast's
   level over time, not its first autocorrelation.

The lag-1 autocorrelation of real MAREX contrasts measured in the
``marex_cumulative_conformal`` study is -0.28 to -0.07 on GeoLift and 0.21 to
0.22 on Walmart, inside the range of the stationary regimes here.

Power is compared only where both tests hold their size; an arm that
over-rejects with no effect has power that counts its false rejections.

## Scope and limits

- Constant effects only. The signed statistic has less power than eq. (16)
  against effects that change sign over the post period, which a geo lift
  rarely does; a test against the sharp null of zero in every period with
  power in every direction keeps eq. (16).
- One design size (``J = 15``, three treated). The test reads only the contrast,
  so the panel enters through the contrast's dependence, which the lag-1 row
  reports.
- Sequential predictive conformal inference (Xu and Xie 2023) builds intervals
  one step ahead from the residuals observed so far; in an experiment the post
  residuals carry the effect, so it concerns MAREX's pointwise band, not the
  global test.

## Reproducing

```
python -m pytest benchmarks/studies/marex_global_test_dependence -q
python -m benchmarks.studies.marex_global_test_dependence.run 2000
python -m benchmarks.studies.marex_global_test_dependence.run report
python tools/mutation/run_mutants.py --target marex-global-test-dependence
```

``results/runs.jsonl.gz`` holds every draw's p-values (``gunzip -k`` it to
resume a run), ``results/summary.json`` every cell.

## References

- Abadie, A. and Zhao, J. (2026). Synthetic controls for experimental design.
  Section 3, Theorem 2; Online Appendix, Theorem OA.1.
- Chernozhukov, V., Wuthrich, K. and Zhu, Y. (2021). An exact and robust
  conformal inference method for counterfactual and synthetic controls. Journal
  of the American Statistical Association 116(536), 1849-1864. Section 2.2 and
  Assumption 2; Remark 1.
- Barber, R. F., Candes, E. J., Ramdas, A. and Tibshirani, R. J. (2023).
  Conformal prediction beyond exchangeability. Annals of Statistics 51(2).
- Xu, C. and Xie, Y. (2023). Sequential predictive conformal inference for time
  series. ICML.
