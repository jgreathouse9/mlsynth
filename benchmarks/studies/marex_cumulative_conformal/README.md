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

Take `n` blank errors and `H` post errors, `N = n + H` exchangeable values, and
a band for the post sum computed from the blank errors only (MAREX's band, the
rolling-origin band and every split-conformal band are of this kind). Put one
spike of size `M` at a uniformly drawn period on top of iid continuous noise.
With probability `H / N` the spike lands in the post window, the blank errors
carry no trace of it, and as `M` grows the band covers only if it is unbounded.
So a band valid under exchangeability alone is unbounded with probability at
least `1 - alpha (n / H + 1)` on ordinary iid data. A finite band needs
`n >= H (1 / alpha - 1)`, 247 blank periods at `H = 13` and 95%, which is what
disjoint windows already require.

Under iid errors the same construction with an independent spike in each
period with probability `p` gives a worst-case miss of
`max_p (1 - (1 - p)^H) (1 - p)^n`: 0.090 at `n = 47`, 0.043 at `n = 104` and
0.019 at `n = 247` for `H = 13`.

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

## Next

1. The trend case by the detrending route: invert jointly over the total `C`
   and a linear slope `theta` in the contrast, the slope estimated on the blank
   periods, and project the accepted set onto `C`. Valid when the contrast's
   trend is in the family; the projection prices the slope's uncertainty into
   the band.
2. A statement of which paths the signed-sum inversion covers. The spike case
   shows it is not every path under every exchangeable law; continuous laws
   gave no counterexample.
3. A written proof of the bound for calibration-only bands.

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
