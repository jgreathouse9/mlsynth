# Repairing a contaminated control market in a synthetic control design

A synthetic control design (MAREX, Abadie and Zhao 2026) picks which markets to
treat and which to use as controls before the experiment starts, using only
pre-experimental data. The treated weights `w` and the control weights `v` are
fixed at `T0` and the experiment then runs. Suppose that during the experiment
an exogenous event hits one control market -- a competitor launches there, a
store closes, a city changes a regulation. The control group is now measuring
something other than what the treated markets would have done untreated.

This study measures what that costs and what can be done about it, on six
panels drawn by the library's own DGPs.

## The arithmetic

Write the contaminated market's observed outcome as its untreated outcome plus
a shift, `Y[k*] = Y_N[k*] + pi` over the post-period. The design estimator is
`sum_j w_j Y_j - sum_j v_j Y_j`, so its estimation error is

    naive - oracle = -v[k*] * pi

This is algebra and the study confirms it to machine precision in all six DGPs.
Three consequences follow. The error scales with the weight the contaminated
market carries, not with the size of the event, so a contaminated market with
`v[k*] = 0` costs nothing. A competitor that depresses the market biases the
measured lift upward, so the failure mode is a false positive on your own
campaign. And an event that hits the treated markets in the same proportion
differences out, leaving only the differential.

## The arms

| arm | what it does | keeps `v`? |
| --- | --- | --- |
| `naive` | nothing | yes |
| `iterative` | Melnychuk (2024): rebuild `k*` from the clean controls, overwrite its post-period, leave `v` alone | yes |
| `iscm` | Di Stefano and Mellace (2024) with one affected unit: solve their 2x2 cross-weight system | yes |
| `renorm` | zero `v[k*]` and rescale the rest | no |

`iterative` repairs the data and `renorm` repairs the weights. That distinction
decides the question here, because a design's value is that `w` and `v` were
committed to before any post-period outcome was seen. Re-optimising the weights
after seeing the contamination spends that commitment.

In the library both corrections already exist as `SPILLSYNTH(method="iterative")`
and `SPILLSYNTH(method="iscm")`. For this use case the setting to use is
`iterative_replace_pre=False`, whose docstring states the reason: the treated
unit's fitting data is then bit-identical to the naive fit's, so the refit
returns the naive weights. The `True` setting, which Melnychuk's published
tables use, moves the weights and is the wrong choice for a locked design.

## `iterative` and `iscm` are the same estimator here

With one affected unit, the inclusive system is

    theta_hat = theta - v[k*] * gamma
    gamma_hat = -l_1 * theta + gamma

where `l_1` is the weight the treated aggregate receives in `k*`'s own
synthetic control. Keep the treated markets out of `k*`'s donor pool and
`l_1 = 0`, the system is triangular, and `theta = theta_hat + v[k*] * gamma_hat`
-- which is what `iterative` computes by substituting the rebuilt series.

Measured agreement is 1.07e-14 over 1920 replications spanning all six DGPs.
The library's own two methods also return bit-identical ATT on a
single-affected-unit panel, and the cross-weight there is 0.0 exactly, a
simplex corner with three of nine donors carrying weight.

So the inclusive system's extra machinery buys nothing unless the treated
markets sit in `k*`'s donor pool. They need not, and generally should not. It
earns its keep when the clean markets cannot fit `k*` without borrowing from
the treated ones: admitting them gave `l_1` of 0.24 and 0.66 in two of three
seeds, and there `iterative` would carry the treatment effect into the
supposedly clean donor.

## When repairing beats living with it

With `e` the error `k*`'s reconstruction makes over the post-window,

    naive     = oracle - v[k*] * pi
    corrected = oracle + v[k*] * e

so mean squared error favours repairing exactly when `pi^2 > E[e^2]`. The
weight cancels: how heavily the contaminated market is weighted does not enter
the decision, only whether the event is larger than your ability to rebuild
that market. At `pi = 0` repairing costs something and the rule matters.

The threshold is a quantity to estimate, and the obvious estimator is wrong.

## The threshold estimator, and why the first one failed

`k*`'s weights are fit on the pre-period, so its pre-period reconstruction
error is an in-sample residual and understates the out-of-sample error. Every
DGP shows this, with the ratio of predicted to realised threshold at or below
one (`threshold_calibration.py`, 25 seeds, every unit standing in as `k*`):

| DGP | realised | in-sample | ratio | blank-window | ratio |
| --- | --- | --- | --- | --- | --- |
| `pangeo_seasonal` | 1.722 | 1.170 | 0.679 | 1.622 | 0.942 |
| `marex_native` | 2.358 | 2.163 | 0.917 | 2.557 | 1.084 |
| `hsc_shared_trend` | 4.541 | 2.707 | 0.596 | 4.928 | 1.085 |
| `fdid_ar1_rho.7` | 0.391 | 0.309 | 0.790 | 0.462 | 1.182 |
| `rank_shift_dormant` | 0.267 | 0.247 | 0.928 | 0.356 | 1.333 |
| `rank_shift_ok` | 0.335 | 0.293 | 0.877 | 0.451 | 1.349 |

The fix is to measure the threshold on a blank window the fit never saw, which
is the device Abadie and Zhao already reserve for their inference. Ratios then
straddle one and lean high, and leaning high is the safe direction: repairing
when there is nothing to repair is the expensive error. The two rank-shift rows
overstate most because a post-period of 20 against a pre-period of 40 leaves
room for a single blank block.

That table averages over every unit standing in as `k*`, and so is optimistic
about the market the repair is actually applied to. The contaminated market in
the end-to-end arm is the one carrying the largest control weight, and it earned
that weight by doing work the other donors cannot do, which makes it harder to
reconstruct than an average market. For `pangeo_seasonal` the blank-window
ratio is 0.942 averaged over all units and 0.577 at the largest-weight market.
Calibration measured over all units is the cheap arm; calibration at the
selected `k*` is the one that governs the decision.

Phase mismatch under seasonality was the first explanation and it is wrong.
`seasonality_negative.py` holds that result: blocks drawn at the post window's
own seasonal phase do no better (0.63 against 0.65), and the understatement
survives with the season amplitude set to zero (0.70). Keeping the arm stops
the explanation being rediscovered.

A second prediction that measured out wrong: `rank_shift_dormant`, built so a
factor is dormant until `T0` and active after, was expected to be the hardest
case. It is among the mildest. The DGP sets the treated row's loading to a
combination of the donors', so the linear relation holds at every date -- and
that property carries over to rebuilding any unit from the others, so the
dormant factor does not break reconstruction the way it breaks the treated
unit's extrapolation.

## What is here

| file | what it measures |
| --- | --- |
| `dgps.py` | the six panels, all from the library's own simulation helpers |
| `repair.py` | the contamination, the four arms, and both thresholds |
| `run.py` | end to end: design, contaminate, repair, score both decision rules over two contamination grids |
| `threshold_calibration.py` | in-sample against blank-window threshold, needs no design solve |
| `seasonality_negative.py` | the wrong explanation, kept as a negative result |
| `analyze.py` | the tables |
| `results/` | the runs behind the tables above |

```bash
cd benchmarks/studies/contaminated_control
python threshold_calibration.py 25 results/threshold_calibration.csv   # seconds
python seasonality_negative.py                                         # seconds
python run.py 40 results/end_to_end.csv                                # ~30 min
python analyze.py results/end_to_end.csv
```

`threshold_calibration.py` uses every unit as `k*` in turn and solves no
mixed-integer design, so it is the cheap arm to re-run when a DGP is added.
`run.py` solves one design per replication, which is where the time goes.

Each replication in `run.py` is swept over two contamination grids off that one
design solve, because no single grid answers both questions. On the `thr_oos`
grid the contamination is a multiple of the replication's own out-of-sample
threshold, which is what places the crossover at 1.0 when the threshold is
calibrated. That grid cannot score the decision rules against each other: the
contamination is then a fixed multiple of `thr_oos`, so the out-of-sample rule
fires exactly when the multiple exceeds one and carries no per-replication
information, while the in-sample rule compares against a separate quantity and
keeps its variation. A comparison on that grid measures the grid. The `panel`
grid sets the contamination as a multiple of the median pre-period unit
standard deviation, which depends on neither threshold, and the decision
comparison uses it.

## The six panels

| DGP | source | what it stresses |
| --- | --- | --- |
| `marex_native` | `generate_marex_sample` | the design's own DGP; the only one supplying treated potential outcomes |
| `rank_shift_ok` | `simulate_rank_shift_panel(dormant_factor=False)` | ranks agree across the intervention |
| `rank_shift_dormant` | `simulate_rank_shift_panel(dormant_factor=True)` | a factor dormant until `T0`, active after |
| `fdid_ar1_rho.7` | `simulate_fdid_serial_sample(rho=0.7)` | serial correlation in the residual at the fitted optimum |
| `hsc_shared_trend` | `simulate_hsc_regime(rho_u=1.0)` | a shared stochastic trend |
| `pangeo_seasonal` | `make_seasonal_sales_panel` | 104 weeks of geo sales with a 52-week season |

`hsc_shared_trend` is the hardest regime by a distance: a realised threshold of
4.54 against 0.27 for the rank-shift panels, because a common drift makes any
market hard to rebuild out of sample. There, repairing pays only for large
contamination.

## Scope

What is measured assumes the contaminated market is known, and known from
outside the outcome data. Identifying it from the panel instead is a different
problem and is not addressed here: control markets looking unusual after the
intervention is often the estimator working, and dropping the ones that look
unusual is a specification search that biases toward finding effects.

The contamination is a level shift on one market over the post-period. Two or
more contaminated markets, and dynamic contamination, are not measured.
Melnychuk (2024) reports the methods converging and degrading as the affected
share of the donor pool rises.

MAREX's own design spreads control weight thinly, which limits the exposure: a
mean largest control weight of 0.08 to 0.19 across these panels, against 0.51
for a plain single-treated-unit synthetic control on comparable draws. The
exposure grows as the pool shrinks or as a penalised design concentrates the
weights.

## Status

Exploratory. Nothing here is wired into the library, and the repair is not a
new estimator: it is a post-hoc adjustment that applies to any design estimator
exposing weights, so it belongs with weight caps and concentration diagnostics
on the design estimators.
