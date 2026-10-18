# Conformal intervals for MAREX's cumulative effect

MAREX reports a pointwise band and a permutation test for the post-period
effects. The question here is a band for the total over the post period,
`C = sum_t tau_t`, with finite-sample validity.

## What exists

| Source | Construction for the total | Validity |
| --- | --- | --- |
| MAREX `cumulative_band` | sign-flipped block paths of the blank-period contrast errors, one level per horizon-length window (`block_error_paths`) | none in finite samples; capped by the window count, `docs/cumulative_calibration.rst` |
| `conformal.cumulative` (VanillaSC, PDA) | split conformal on rolling-origin block sums, stride `H`, centred | exchangeable windows, before centring |
| `conformal.cumulative_inversion` | CWZ test inversion of a constant effect, refit on all periods, cyclic shifts | constant path, stationary mixing errors, asymptotic |
| TBR (Kerman, Wang and Vaver 2017, eqns 4 and 6) | t posterior for the cumulative effect; scale `T s (v_a + 2 xbar v_ab + v_b xbar^2 + 1/T)^(1/2)` | Gaussian iid regression errors, flat prior |
| SynthRTCI / SynthPower (Wheeler) | see below | empirical, long pre-periods |

TBR's scale splits into a parameter term that grows in `T^2` and a noise term
that grows in `T`. That is the level and fluctuation split the calibration study
found in MAREX's blank errors: the fit's own bias is common to every post
period, so a total sees it `H` times over.

### SynthRTCI

Read from `docs/py/synth_methods.py` (a copy of SynthPower's `methods.py`) and
`docs/py/analysis.py` at github.com/apwheele/SynthRTCI.

- Rolling-origin forecast errors, the site's default. An origin sits at every
  pre-period month from `min_train` (36) to `T0 - 1`; the model is refitted on
  the months before it and forecasts up to `H` months ahead inside the
  pre-period. The score for horizon `h` is the sum of the first `h` forecast
  errors, the band half-width is the `ceil((n + 1)(1 - alpha))`-th absolute
  score, and no band is drawn when fewer origins reach `h` than that rank
  needs. Origins step by one month, so the windows overlap and the score count
  is `T0 - min_train - h + 1`: a finite 95% band needs `T0 >= min_train + H +
  18`, not the `19 H` that disjoint windows need. Each origin is a fresh fit,
  so the fit's extrapolation bias is drawn once per origin.
- Jackknife: leave-one-month-out pre-period residuals. The cumulative band
  either draws a symmetrised residual independently per month and accumulates
  (Wheeler 2023; SynthPower reports 21-28% false positives by year 2 and 68-76%
  by year 10), or takes every cyclic `h`-sum of the residuals, gated to the
  `T0 - h + 1` sums that do not wrap.
- Placebos: each donor above a population cutoff is fitted as if treated; its
  cumulative gap over its own pre-period RMSPE gives a conformal quantile, which
  is scaled by the treated unit's pre-period RMSPE. Nineteen placebos for a 95%
  band; p-values carry the `+1`.

The MAREX analogue of the rolling origin keeps the design's treated set fixed
(one mixed-integer solve), places origins inside the blank window, and refits
both simplex weight vectors on the periods before each origin, which is a
quadratic program of about 33 ms (`cumulative_calibration/design_cost.py`).
The forecast windows then lie in periods the design never saw, and there are
`T_b - H + 1` scores: 35 at `T_b = 47`, `H = 13`.

## A bound for bands built from calibration errors alone

Proposition. Let `Z_1, ..., Z_N`, `N = n + H`, be real errors, the first `n`
observed (blank) and the last `H` not (post), and let `Gamma(Z_1, ..., Z_n)` be
an interval for `S = Z_(n+1) + ... + Z_N`, possibly randomised independently of
the post errors. If `P(S in Gamma) >= 1 - alpha` for every exchangeable law of
`Z`, then for every continuous distribution `F`,

    P_(F^n) (Gamma is unbounded above) >= 1 - alpha N / H = 1 - alpha (n / H + 1),

and the same below. A band that is bounded with probability one under some
continuous `F` therefore needs `n >= H (1 / alpha - 1)`: 247 blank periods at
`H = 13` and 95%, which is what disjoint windows require.

Proof. Fix `F` and `M > 0`, draw `xi_1, ..., xi_N` iid from `F` and `J`
uniform on `{1, ..., N}` independently, and set `Z = xi + M e_J`, which is
exchangeable. On `{J > n}`, of probability `H / N`, the observed errors are
`xi_1, ..., xi_n ~ F^n`, independent of `J`, and `S = M + V` with
`V = sum_(t > n) xi_t`. Coverage gives

    1 - alpha <= P(J <= n) + (H / N) P(M + V in Gamma(xi_1, ..., xi_n)),

so `P(M + V in Gamma(xi)) >= 1 - alpha N / H` for every `M`. On the event that
`Gamma(xi)` is bounded above by `u(xi) < infinity`, `M + V in Gamma` needs
`M + V <= u(xi)`, whose probability goes to zero as `M` grows since `V` is
finite. Hence `P(Gamma unbounded above) >= 1 - alpha N / H`. QED.

Under iid errors (not only exchangeable) the same argument runs with an
independent spike in each period with probability `p`: on the event that the
blank periods carry none and the post window at least one, the observed errors
are `F^n`, so `P_(F^n)(Gamma bounded above) <= alpha / q` with
`q = max_p (1 - (1 - p)^H) (1 - p)^n`, which is 0.090 at `n = 47`, 0.043 at
`n = 104` and 0.019 at `n = 247` for `H = 13`. At `n = 47` an iid-valid band is
unbounded with probability at least 0.44.

The proposition concerns bands built from the blank errors alone: MAREX's band,
the rolling-origin band and every split-conformal band are of that kind. It
does not reach a band that also reads the post outcomes, which in the causal
problem are observed (the post errors plus the effect). That is what the test
inversion does.

The bound does not reach a band that reads the post outcomes, which is what a
test inversion does.

## First look

`first_look.py`, 1000 draws per cell, `H = 13`; the full table is
`results/first_look.log`. Coverage of the post sum at 95%:

| Arm | law | `n = 47` | `n = 104` | `n = 247` |
| --- | --- | --- | --- | --- |
| shipped | gauss | 0.941 | 0.974 | 0.971 |
| shipped | spike_exch | 0.778 | 0.897 | 0.953 |
| shipped | spike_iid | 0.847 | 0.892 | 0.927 |
| batch (disjoint windows) | any | infinite | infinite | 0.941-0.953 |
| rolling (overlapping sums) | gauss | 0.893 | 0.955 | 0.943 |
| rolling (overlapping sums) | spike_exch | 0.787 | 0.893 | 0.950 |
| inversion, signed sum, constant path | gauss | 0.935 | 0.955 | 0.957 |
| inversion, signed sum, constant path | spike_exch | 0.952 | 0.955 | 0.955 |
| inversion, signed sum, constant path | ar5 | 0.797 | 0.777 | 0.779 |
| inversion, mean absolute, ramp path | gauss | 0.058 | 0.037 | 0.025 |

- The shipped band meets the exchangeable bound: `1 - 13 / 60 = 0.783` against
  0.778 measured at `n = 47`.
- Inverting the constant-effect sharp null over the blank and post periods,
  with Abadie and Zhao's all-subsets permutation and no refit (MAREX's weights
  come from the fitting periods alone), is exact for a constant path and needs
  no windows.
- The statistic decides whether a non-constant path is covered. Abadie and
  Zhao's mean absolute post error grows with the path's deviations, and
  coverage of the total falls to zero under decay. The signed post sum depends
  on the path only through `C`; under ramp and decay it stayed within Monte
  Carlo error of nominal or above for every exchangeable law, and fell to 0.903
  under the exchangeable spike with a one-week path at `n = 47`.
- Serial dependence breaks the all-subsets permutation at every `n`: 0.78 under
  an AR(1) with coefficient one half. Cyclic shifts (CWZ's `Pi_->`) are the
  candidate repair.

## Candidate 2: the signed-sum inversion

`inversion.py` inverts the constant-effect sharp null with the statistic
`|sum_{t in post} e_t|`. Each reference block holding a share `r_k` of the
post periods counts towards the p-value where `|a_k - r_k C| >= |G - C|`,
which is a sign condition on a product of two linear functions of `C`, so the
band is computed exactly at the breakpoints. It matched a grid of 40,001
points on 400 random cases. Far enough out only the observed block counts, so
the band is bounded once the reference set has more than `1 / alpha` blocks.

Two reference sets: `subsets`, every `H`-subset of the blank and post periods
(or 999 drawn uniformly), which is Abadie and Zhao's Theorem 2 scheme; and
`cyclic`, the `N` cyclic shifts of the blank-then-post series, CWZ's `Pi_->`,
which keeps time order inside a block.

### Error series

`candidate2.py`, 2000 draws a cell, `results/candidate2.log`. Coverage of the
total at 95%, and the median width of the band in standard deviations of the
post sum (a Gaussian oracle is 3.92):

| law | scheme | `n=12, H=4` | `n=26, H=13` | `n=47, H=13` | `n=104, H=13` |
| --- | --- | --- | --- | --- | --- |
| gauss | subsets | 0.948 (4.34) | 0.950 (4.35) | 0.951 (4.15) | 0.950 (3.96) |
| gauss | cyclic | unbounded | 0.983 (8.43) | 0.951 (5.11) | 0.958 (4.54) |
| skew | subsets | 0.950 | 0.944 | 0.939 | 0.953 |
| t2 | subsets | 0.952 | 0.950 | 0.951 | 0.950 |
| ar5 | subsets | 0.855 | 0.806 | 0.804 | 0.804 |
| ar5 | cyclic | unbounded | 0.961 | 0.936 | 0.963 |
| ar8 | subsets | 0.766 | 0.674 | 0.656 | 0.618 |
| ar8 | cyclic | unbounded | 0.943 | 0.920 | 0.942 |

- Under exchangeable errors the subsets scheme is exact for a constant path at
  every size, `n = 12` included, and within 11% of the oracle width. Disjoint
  windows would need 76 blank periods for a finite band at `H = 4`.
- Serial dependence defeats the subsets scheme at every size, and more blank
  periods do not help: an AR(1) at 0.8 holds 0.62 at `n = 104`. Cyclic shifts
  hold 0.92-0.96, but need `N > 1 / alpha` periods and are wide when `N` is
  near that limit.
- The skew cell at `n = 47` (0.939 at 95%, 0.770 at 80%) is Monte Carlo noise
  across 112 cells: 5000 fresh draws give levels of 0.046-0.052 at 5% and
  0.185-0.207 at 20%.

Paths with total `H` and deviations of spread 0.5, 2 and 8 about it: under
every continuous law the worst path's coverage is within Monte Carlo error of
the constant path's, and large deviations raise it. The exception is the
exchangeable spike (one error of size 200), where the subsets scheme falls to
0.914-0.918 at 95% and to 0.688 at 80% (`n = 26`, one-week path, spread 8).
`path_search.py` scores 150 random zero-mean paths on cases small enough to
enumerate every block (`results/path_search.log`); under Gaussian, skewed and
bimodal errors none raised the rejection rate above the null path's. Under the
spike it found paths that do: 0.087 to 0.125 at level 0.1 (subsets, `n = 5`,
`H = 3`) and 0.166 to 0.249 at level 0.17 (cyclic, `n = 9`, `H = 3`).

### MAREX designs

`designs.py`, 500 draws a cell, `results/designs.log`. A Section 5 panel
(`J = 15`, `m_eq = 3`, exact backend), the contrast of the solved design on the
blank and post periods, an effect added to the treated units. Coverage of the
total at 95% / 80%:

| periods | arm | constant | worst other path |
| --- | --- | --- | --- |
| `T_b = 5, H = 5`, iid | shipped | refused | refused |
| `T_b = 5, H = 5`, iid | subsets, signed sum | 0.948 / 0.796 | 0.986 / 0.892 |
| `T_b = 5, H = 5`, iid | subsets, mean absolute | 0.950 / 0.800 | 0.798 / 0.454 |
| `T_b = 26, H = 13`, iid | shipped | 0.908 / 0.768 | same |
| `T_b = 26, H = 13`, iid | subsets, signed sum | 0.942 / 0.802 | 0.982 / 0.878 |
| `T_b = 26, H = 13`, iid | cyclic, signed sum | 0.976 / 0.824 | 0.976 / 0.832 |
| `T_b = 26, H = 13`, ar | shipped | 0.756 / 0.636 | same |
| `T_b = 26, H = 13`, ar | subsets, signed sum | 0.680 / 0.564 | 0.750 / 0.604 |
| `T_b = 26, H = 13`, ar | cyclic, signed sum | 0.928 / 0.748 | 0.952 / 0.744 |
| `T_b = 26, H = 13`, sorted | shipped | 0.402 / 0.294 | same |
| `T_b = 26, H = 13`, sorted | cyclic, signed sum | 0.824 / 0.532 | 0.904 / 0.550 |

- At Section 5's own sizes the shipped band raises (a horizon-length block is
  longer than half the five blank periods); the signed-sum inversion is exact
  there under Theorem 2's assumption.
- With serially dependent factors the cyclic scheme is the only arm near
  nominal at 95%, and it is short at 80%.
- With trending factors every arm fails. The design's imbalance in the
  unobserved loadings times a trending `lambda_t` is a bias that grows across
  the post period, a cumulative total adds it `H` times, and the blank periods
  carry less of it than the post periods do. That is Theorem 1's bias, which no
  calibration on the blank periods can see.

### Block permutations and recency weights

`blocks_weights.py`, 1000 draws a cell, `H = 12`, `results/blocks_weights.log`.
The `blocks` scheme cuts the periods into contiguous blocks of length `L` and
permutes them as units: `L = 1` is all subsets, `L = H` is disjoint windows.
Recency weights are Barber, Candes, Ramdas and Tibshirani's (2023): reference
block `k` weighs `0.97` to the distance between its centre and the post
window's, and their Theorem 2 bounds the loss of coverage by the weighted total
variation between the data and its swap. Coverage of the total at 95%, constant
path, with the median width in standard deviations of the noise sum:

| law | `n` | subsets | `L = 3` | `L = 6` | cyclic | cyclic, weighted |
| --- | --- | --- | --- | --- | --- | --- |
| gauss | 48 | 0.952 (4.07) | 0.956 (4.22) | 0.958 (4.66) | 0.959 (4.91) | 0.985 (8.31) |
| ar5 | 48 | 0.794 (2.66) | 0.900 (3.56) | 0.947 (4.44) | 0.936 (4.47) | 0.964 (5.77) |
| ar8 | 48 | 0.653 (1.84) | 0.820 (2.68) | 0.918 (3.67) | 0.921 (3.73) | 0.950 (4.26) |
| ar8 | 96 | 0.627 (1.80) | 0.794 (2.64) | 0.890 (3.43) | 0.938 (3.99) | 0.971 (4.51) |
| drift | 48 | 0.786 | 0.802 | 0.847 | 0.838 | 0.931 |
| drift | 96 | 0.454 | 0.505 | 0.612 | 0.704 | 0.887 |

- Longer blocks recover coverage under autocorrelation monotonically, and at
  the same width cyclic shifts reach as much or more: blocks interpolate
  between subsets and cyclic shifts without beating the latter. Blocks earn a
  place only where `N` is too short for cyclic shifts at the chosen level.
- Under an AR(1) at 0.8 cyclic shifts hold 0.92-0.94 at 95% and 0.75-0.78 at
  80%; weighting lifts that to 0.95-0.97 and 0.79-0.82 for 15-20% more width.
- A drift defeats every scheme as the blank period lengthens, since every
  reference block sits earlier on the trend than the post window. Weighting
  slows the loss (0.887 against 0.704 at `n = 96`) and does not stop it: no
  reweighting of the past reaches the post window's position on a trend.

### What the reading adds

- Vovk, Gammerman and Shafer (2022), Section 4.5: the transductive conformal
  predictor for `k` labels at once ranks the observed subset among all
  `(l + k)! / l!` ordered subsets (eq. 4.27). The `subsets` scheme is that
  predictor with the signed sum as its transductive nonconformity measure, so
  its exactness under exchangeability is theirs. Their Table 11.1 puts the first
  informative step at `1 / epsilon` examples under exchangeability.
- The same book (Section 13.5.1) credits Chernozhukov, Wuthrich and Zhu (2018)
  with the theory for block-based conformal prediction on time series, and
  (Section 13.5.2) describes CWZ (2021) as exactly valid for IID data and
  approximately valid for time series, which is what the cyclic rows show.
- For trends, the first edition's detrending transformations (quoted in the
  second, Section 8.6.8): fix a family `F_theta` such that some `F_theta` makes
  the sequence exchangeable, reject the `theta` that do not, and predict with
  the rest. Remark 11.7 does this for a polynomial trend with IID Gaussian noise.
- Stocker, Malgorzewicz, Fontana and Ben Taieb (2025): split conformal stays
  approximately valid under beta-mixing with explicit slack terms, and
  non-stationarity is the failure it cannot absorb. Their block conformal arm
  under-covered when the block length was mis-set.
- Memmesheimer, Heuveline and Hesser (2025) review conformal methods for
  individual treatment effects; none treats panels or a total over time.
- Chernozhukov, Wuthrich and Zhu (2021), Remark 1: "when capturing deviations
  in the average effect", use `S(u) = T*^(-1/2) sum_t u_t`. Candidate 2's
  statistic is theirs. Their Theorem 1 bounds the size distortion by
  `C (delta~_T + ...)` with `delta~_T = (T*/T0)^(1/4) log T`, using all
  permutations under iid shocks (Assumption 2.1) and moving blocks under
  stationary strong mixing (Assumption 2.2); the subsets and cyclic rows above
  reproduce that split. Their test refits the counterfactual on every period
  under the null; MAREX's weights come from the fitting periods alone, so the
  blank and post errors need no refit. Remark 2 handles unconditional
  heteroscedasticity by filtering to standardized residuals, the same move as
  detrending.
- Barber and Pananjady (2026), Corollary 1: for a pretrained score with memory
  `L` on a stationary beta-mixing series, split conformal loses at most
  `min_tau {(tau + L)/(n - L + 1) + 2 beta(tau)}` coverage, tight at `L = 0`
  (Theorem 2). A sum over `H` consecutive errors is a score with memory
  `H - 1`, which puts the loss of the rolling, overlapping windows at order
  `H / n`, the order of the spike bound `H / (n + H)` above, and is why
  SynthRTCI's rolling origin needs long pre-periods. The post window's errors
  are unobserved, so the `H - 1` windows straddling the treatment date are
  missing from the calibration scores; whether their deletion operation covers
  that gap exactly is not checked here.
- Sun and Yu (2024), copula conformal prediction: finite-sample joint regions
  for `k` steps, from many independent trajectories. In MAREX the trajectories
  would be placebo designs, the cross-sectional route; for a total, split
  conformal on the placebo sums is the direct version, and a joint region
  projected onto the sum is wider than it.

## Spikes

### Detrending the contrast

`detrend.py`, 500 draws a cell, `H = 12`, `results/detrend.log`. Coverage of
the total at 95% with the median width in standard deviations of the noise sum:

| law | `n` | cyclic | subsets | subsets, plug-in slope | subsets, exact slope set |
| --- | --- | --- | --- | --- | --- |
| gauss | 48 | 0.946 (4.88) | 0.932 (4.07) | 0.786 (4.40) | 0.992 (15.08) |
| gauss | 96 | 0.954 (4.52) | 0.954 (3.97) | 0.876 (4.27) | 0.998 (11.68) |
| drift | 96 | 0.702 (7.85) | 0.474 (6.47) | 0.856 (4.24) | 1.000 (11.84) |
| steep | 96 | 0.180 (31.3) | 0.000 (21.1) | 0.862 (4.26) | 1.000 (11.88) |
| curved | 96 | 0.052 (22.0) | 0.000 (12.6) | 0.870 (7.15) | 1.000 (14.01) |
| drift_ar5 | 96 | 0.814 (5.51) | 0.504 (3.55) | 0.666 (2.80) | 0.966 (7.22) |

- The exact arm (a permutation confidence set for the slope from the blank
  periods at `alpha / 2`, then the union of the signed-sum bands over it at
  `alpha / 2`) holds at every law, the quadratic one included, and at 80%
  covers 0.97-0.99: the Bonferroni split and the union make it conservative.
  It is three to five times wider than the undetrended band when there is no
  trend, and its cyclic version is unbounded at `n = 24`, where the slope test's
  24 shifts cannot reach `alpha / 2`.
- Treating the estimated slope as known is not valid: 0.65-0.88 at 95%, the
  trend-free case included, since the slope's error is extrapolated across the
  post window and the band does not see it.
- Under an AR(1) at 0.5 the subsets slope test is not exact either, and the
  exact arm falls to 0.84-0.87 at 80%.

### The library's CWZ inversion, with the signed statistic

`designs_spike.py`, 300 draws a cell, the Section 5 panel through MAREX at 40
fitting, 26 blank and 13 post periods, `results/designs_spike.log`. Every
refit arm calls the library's
:func:`~mlsynth.utils.conformal.refit.conformal_refit_gaps` (the simplex
control refitted on all periods under each null); `cwz` is
:func:`~mlsynth.utils.conformal.cumulative_conformal_by_inversion` as it ships,
with the `S_1` statistic. Coverage at 95% / 80%, and the median width at 95%
for the constant path in outcome units:

| factors | arm | constant | ramp | width |
| --- | --- | --- | --- | --- |
| iid | blank-only cyclic, signed | 0.980 / 0.813 | 0.997 / 0.980 | 49.1 |
| iid | `cwz` (`S_1`) | 0.947 / 0.740 | 0.113 / 0.013 | 45.6 |
| iid | refit, signed | 0.950 / 0.773 | 0.997 / 0.907 | 22.0 |
| iid | refit, signed, weighted | 0.963 / 0.790 | 0.997 / 0.960 | 27.4 |
| ar | blank-only cyclic, signed | 0.917 / 0.757 | 0.970 / 0.853 | 58.1 |
| ar | `cwz` (`S_1`) | 0.900 / 0.697 | 0.280 / 0.077 | 80.9 |
| ar | refit, signed | 0.910 / 0.730 | 0.973 / 0.800 | 40.8 |
| ar | refit, signed, weighted | 0.927 / 0.747 | 0.977 / 0.850 | 45.6 |
| sorted | blank-only cyclic, signed | 0.833 / 0.537 | 0.997 / 0.980 | 34.3 |
| sorted | `cwz` (`S_1`) | 0.947 / 0.737 | 0.003 / 0.000 | 98.4 |
| sorted | refit, signed | 0.983 / 0.833 | 1.000 / 0.970 | 28.8 |
| sorted | refit, signed, weighted | 0.983 / 0.850 | 1.000 / 0.997 | 34.7 |

- The refit with the signed statistic is the narrowest band on every regime,
  half the blank-only band under iid factors and under a third of `S_1`'s under
  a trend, and it covers the ramp where `S_1` does not.
- The trend in the factors is shared with the controls, so refitting the
  control weights on every period tracks it: 0.983 at 95% against 0.833 for the
  blank-only band. A drift unique to the contrast has nothing for a refit to
  use, and only the detrending route covers it.
- Serially dependent factors leave every arm short at 80%. Recency weights
  (`0.97` to the circular distance of the shift) add 1.5-2 points for 12-24%
  more width.

### Which paths are covered

`path_search.py` with a symmetric spike (`+-200` at one period) and Cauchy
errors, `results/path_search_sym.log`. The symmetric spike still breaks the
inversion for some paths (0.089 to 0.101 at level 0.1 on all subsets; 0.166 to
0.208 at level 0.17 on cyclic shifts), so the asymmetry of the first spike was
not the cause. Cauchy errors, symmetric and unimodal with no variance, gave
none: the one cell where the worst path read above the null (0.1258 against
0.1220) was the null's smaller sample, and on common draws the path rejects
less (0.1255 against 0.1256, three seeds of 400,000). The failure needs a rare
error that dominates the block sum; heavy tails alone do not produce it.

### Serial dependence: the cause

How dependent real contrasts are, on pre-period data with no effect anywhere
(MAREX, exact backend; about 40 periods a series, so each estimate carries a
standard error near 0.16): GeoLift's contrast has lag-1 autocorrelation -0.07
in the fitting periods and -0.28 in the blank and pseudo-post ones, Walmart's
+0.21 and +0.22, and the CWZ refit residuals -0.12 and +0.07. The earlier
calibration study measured 0.06-0.07 on its panel. Factors at 0.8 are a stress
case well beyond these.

Whitening (`whiten.py`, after CWZ Section 4.4.2 and Lemma 7): fit AR(K) to the
null-imputed series, permute the innovations, rebuild the levels, keep the
signed post sum on the levels. On pure AR(1) error series it does what the
lemma says, AR(0.8) at 95% moving from 0.92-0.94 (cyclic) to 0.94-0.96, and the
exact inversion over the AR coefficient holds everywhere at 0.94-1.00. Through
MAREX designs it does not help: AR(1) 0.883 / 0.687 under factors at 0.8,
against 0.910 / 0.730 for plain cyclic shifts; AR(2), AR(p) by AIC, AR(6) and
ARMA(1, 1) do no better (`whiten_diag.py`).

The descent (`whiten_diag.py`, 300 draws), the CWZ refit residuals at the true
total:

| factors | mean squared residual: fitting / blank / post | post sum's spread over the reference's |
| --- | --- | --- |
| 0.8 | 0.84 / 1.29 / 1.30 | 1.59 |
| 0.5 | 1.00 / 1.43 / 1.44 | 1.35 |
| iid | 1.05 / 1.46 / 1.38 | 1.17 |

- The blank and post stretches behave alike. The fitting periods, where MAREX
  chose the treated set, carry residuals about 30% smaller in mean square: the
  same leak the calibration study measured as a spread ratio of 0.818 on its
  panel.
- Persistence amplifies the leak for a sum, because a 13-period sum is
  dominated by slow movements and choosing the treated set on a window fits
  those away there: the reference understates the post sum's spread by a factor
  of 1.17 under iid factors, 1.35 at 0.5 and 1.59 at 0.8. Against blocks inside
  the blank periods alone the understatement is 1.07-1.09.
- Ruled out: the cyclic seam (wrapping and contiguous blocks have the same
  spread, ratio 1.03-1.04); the order of the dependence model (every whitening
  model fails alike, since the gap is between stretches, not a stationary
  dependence); recency weights (at most one point).

Confirming it (`leak_fix.py`, 300 draws): keep the fitting periods out of the
reference. Coverage at 95% / 80%, constant path, median width at 95%:

| factors | refit, all periods referenced | refit, blank and post referenced | refit on blank and post | no refit, blank and post |
| --- | --- | --- | --- | --- |
| 0.8 | 0.910 / 0.730 (42.5) | 0.967 / 0.777 (60.0) | 0.977 / 0.833 (78.8) | 0.917 / 0.757 (52.5) |
| 0.5 | 0.927 / 0.767 (30.0) | 0.973 / 0.787 (45.0) | 0.977 / 0.813 (56.2) | 0.947 / 0.770 (45.0) |
| iid | 0.950 / 0.773 (25.0) | 0.967 / 0.827 (37.5) | 0.973 / 0.820 (45.0) | 0.980 / 0.813 (47.5) |
| sorted | 0.983 / 0.833 (32.5) | 0.983 / 0.807 (38.8) | 0.993 / 0.860 (52.5) | 0.833 / 0.537 (32.5) |

- With the fitting periods out of the reference, coverage holds under every
  regime, the persistent ones included, and the ramp path is covered at 0.91 or
  more throughout.
- The construction that results takes from both papers: CWZ's refit on every
  period, which tracks a trend the controls share; CWZ's Remark 1 signed
  statistic, which depends on the effect path only through the total; and
  Abadie and Zhao's reference set, the blank and post periods the design never
  saw. Its band is wider than the leaky one by about half, and narrower than
  the blank-only band without a refit under iid factors (37.5 against 47.5).
- What is left for pure serial dependence is the finite-sample term Barber and
  Pananjady describe: 0.777 at 80% under factors at 0.8, a standard error from
  nominal at 300 draws.

## Next

1. The library change: CWZ's Remark 1 statistic as an option of
   :func:`~mlsynth.utils.conformal.cumulative_conformal_by_inversion` and the
   refit machinery under it, with the permuted window settable apart from the
   refit window, test-first on its own branch; then a MAREX cumulative band
   that refits on every period and permutes over the blank and post periods.
2. The inference fixes recorded on #744.

## References

Abadie, A. and J. Zhao (2026). Synthetic controls for experimental design.

Chernozhukov, V., K. Wuthrich and Y. Zhu (2021). An exact and robust conformal
inference method for counterfactual and synthetic controls. Journal of the
American Statistical Association 116(536), 1849-1864.

Kerman, J., P. Wang and J. Vaver (2017). Estimating ad effectiveness using geo
experiments in a time-based regression framework. Google Technical Report.

Wheeler, A. P. (2026). SynthPower. github.com/apwheele/SynthPower.

Wheeler, A. P. (2026). Synthetic control for city crime [web application].
github.com/apwheele/SynthRTCI.
