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

## Next

1. Signed-sum inversion with cyclic-shift permutations, against AR errors.
2. The rolling-origin arm with the treated set fixed and weights refitted per
   origin, on a factor-model panel run through MAREX designs.
3. Which effect paths the signed-sum inversion covers, as a statement with a
   proof or a counterexample.
4. A written proof of the bound above.

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
