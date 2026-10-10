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
| `closed_form_plot.py` | one draw, the three closed forms against what is measured |
| `partial_id.py` | stops assuming the spillovers away, bounding the effect from the pre-period weight set |
| `analyze_partial_id.py` | tables for that arm |
| `detection.py` | drops the assumption that the contaminated market is known, screening for it with SPOTSYNTH, and runs RRSC as a baseline |
| `analyze_detection.py` | tables for that arm |
| `spillover_pool.py` | admits the treated markets to the contaminated market's pool, where the cross-weight stops being zero |
| `analyze_spillover.py` | tables for that arm |
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

`closed_form_plot.py` draws the three error expressions on a single
replication, sweeping the contamination at zero cross-weight, sweeping the
cross-weight at fixed contamination, and plotting the formulas against the
measured values over 441 settings. The largest gap is 5.7e-15, so the
expressions are exact and not approximations that happen to fit.

## What a committed design keeps when it stops assuming the spillovers away

Every other arm removes the contamination and reports a number. Wei (2026)
declines to: with the interference pattern unknown, outcomes alone do not
separate the direct effect from the spillovers, and what the data support is a
set. The device is that synthetic-control weights are not unique. For a
committed treated aggregate ``w`` and any convex control weight ``v``,

    tau_hat(v) = [ sum w_j Y^N_j - sum v_j Y^N_j ] + theta - v'gamma

and the validity rule bounds that untreated discrepancy by ``M * d(v)``, with
``d(v)`` the weight's pre-treatment discrepancy. Every weight is one inequality

    | theta - v'gamma - tau_hat(v) |  <=  M * d(v)

in the same unknowns. One weight leaves the effect bounded only by the
spillover restriction; many weights, disagreeing in their post-period
estimates, pin the spillover vector in the directions their differences span.

This looked incompatible with an experimental design, which commits to one
``v`` before the experiment and gets its credibility from doing so, while the
identification consumes the set of them. It is not. Both the weights and their
discrepancies are pre-period objects, fixed at ``T0`` before any post-period
outcome exists, so a design can commit to one weight for running the experiment
and keep the rest for the identification afterwards. Nothing in the set was
chosen after seeing a result, so no specification search is involved.

With a box restriction on the spillovers the set is a pair of linear programs
in ``(theta, gamma)``. Sampling weights instead of representing the continuum
exactly gives an outer approximation, so every width reported is conservative
and the true set is at least that tight.

An infeasible program is a result and not a failure. It says no direct effect
and spillover vector satisfies every weight's inequality, so the envelope and
the spillover bound are together refuted by the data.

## Dropping the assumption that the market is known

Every other arm is told which market was contaminated. `detection.py` is not.
It screens with :class:`~mlsynth.SPOTSYNTH`, which forecasts each donor's
post-intervention values from pre-intervention donor data alone and returns the
donors the forecast misses, then repairs whatever the screen returns and
compares that against the repair told the right answer. The difference is what
detection error costs. A screen that misses the contaminated market leaves the
bias in place; a screen that over-flags rebuilds clean markets for nothing and
spends their contribution to the fit. Both of the screen's selection rules are run, because only one of them is a
test. ``S1`` keeps the ``n_donors`` donors with the smallest forecast error and
``n_donors`` defaults to half the pool, so it flags a fixed count whatever the
data look like. A first run used that default and found every panel flagging
50 to 54 percent of its controls, with the flagged count at zero contamination
identical to the count overall. That is arithmetic, not a property of the
method, and reporting it as one would have been a statement about the default.
``S2`` keeps the donors whose realised value falls inside the forecast's
posterior predictive interval, so its count responds to the data and ``ppi``
sets the false-positive rate. Neither rule is tuned against these results.

The same arm runs :class:`~mlsynth.RRSC` as a baseline that needs neither a
nominated market nor a clean pool. RRSC is reported only where it passes an
applicability gate: on a clean panel of the same shape it must recover a known
effect to within 15 percent of its size. An earlier version of this arm set that
tolerance at half, which admits an estimator whose errors run near 40 percent,
so the gate passed the configurations it existed to catch. Its two regimes assume dimensions these panels
do not all have, and its factor model assumes time-invariant loadings that a
shared stochastic trend violates, so an ungated number would read as a finding
about the method when it is a statement about the panel. An empty cell is a
result: it says the estimator's conditions do not hold there.

Both benchmarks are reported. Against the design's own oracle, which isolates
what the contamination costs and matches the other arms; and against the true
effect, which is the only fair benchmark for RRSC, since it never uses the
design's weights and does not inherit the design's fit error.

### What the two arms measured

For the screen, recall decides and precision costs little. The fixed-size rule
flags more and finds the contaminated market more often (0.735 to 0.915 against
0.785 to 0.855), and the repair built on it is the better one in five of six
panels. Missing the contaminated market leaves the whole bias in place, while
rebuilding a clean market spends only its contribution to the fit, so a screen
tuned for this use should err toward flagging. The test rule's false-positive
behaviour varies by panel more than its averages suggest: with nothing
contaminated it flags 1.5 markets on the serial-correlation and seasonal panels
and 6.1 on the dormant-factor panel, which is more than the fixed rule flags
there by construction.

Both are far behind the repair that is told the answer, and sometimes behind
doing nothing at all. Excess error over the known-market repair runs from 0.064
to 1.048, and on the seasonal panel under the test rule the repaired estimate
is worse than the uncorrected one (1.238 against 0.487). Detection quality, not
repair quality, is what binds.

RRSC, once the gate discriminates, beats the uncorrected estimate in four of the
five panels where it is admissible, and on the dormant-factor panel it matches
the repair that was told the right market (0.063 against 0.065, measured against
the true effect). It does not match it elsewhere. The seasonal panel passes the
gate in none of 40 replications, so it has no row.

The partial-identification arm behaves as its algebra predicts and is dear. The
weight set narrows the set by 35 percent on average at the tightest envelope,
by 16 percent at the next, and by 6 percent at the loosest, since a loose
envelope leaves the weight inequalities slack. The best single replication
narrows by 99.5 percent. Coverage among feasible programs is 0.973. At the
tightest envelope the programs on the shared-trend and seasonal panels are
infeasible about 70 percent of the time, which refutes the envelope and the
spillover bound together on those panels.

The cost is width. The sets run from 2.2 to 23.8 in outcome units against
effects of 0.8 to 8.4, so they are several times the quantity being estimated,
while the uncorrected estimate's own error is 0.089 to 0.708. The arm buys
freedom from the identifying assumptions at a price that would leave most
practical questions open, which is the same shape as Wei's own application,
where the inversion sets admit both signs.

## Where the two corrections stop agreeing

The identity above holds because the treated markets are kept out of `k*`'s
donor pool, which sets the cross-weight `l1` to zero. That exclusion is
available in the exogenous case, since nothing the treatment did caused the
contamination. Under spillover it may not be: Di Stefano and Mellace's
motivating example has Austria at 42 percent of synthetic West Germany, and
dropping West Germany from Austria's pool gives implausible spillover
estimates.

`spillover_pool.py` admits the treated markets and measures the consequence.
The exclusion is not a free choice when the clean donors fit `k*` poorly: with
the simplex free to use them, `l1` averages 0.36 across these panels and
reaches 0.79 on `marex_native`.

With the treated markets at total weight `l1`, the rebuild picks up `l1 * tau`
over the post-period and the arms part company:

    iterative error = v_k * e - v_k * l1 * tau
    iscm error      = v_k * (e + delta * l1) / (1 - v_k * l1)

for `e` the rebuild error and `delta` the design's own fit error. The leak in
`iterative` scales with the treatment effect; the inclusive system removes that
term and pays a `1 / (1 - v_k * l1)` inflation for it.

Two predictions were stated before the run. The first holds and the second was
wrong.

The gap between the arms tracks the leak: regressing the observed
`iscm - iterative` on `v_k * l1 * tau` gives slope 1.0075 and intercept
-0.0002 at a correlation of 0.9972, and the mean gap matches the mean leak at
0.1199 against 0.1183 and 0.3566 against 0.3549 for the two effect sizes.

The second prediction was that the arms agree at `tau = 0` however large `l1`
is. They do not: the largest disagreement is 0.202. The prediction contradicts
the decomposition above, which was read for its leak term alone. Setting
`tau = 0` in the two expressions leaves

    iscm - iterative = v_k * l1 * (delta + v_k * e) / (1 - v_k * l1)

which vanishes only when `l1` does. That corrected form is exact against the
data: slope 1.0, intercept zero, and a largest deviation of 7.2e-15 over the
570 replications with a non-zero cross-weight. The 150 replications where the
simplex sets `l1 = 0` on its own still agree to 5.3e-15, so the study's
headline identity is intact where it was claimed.

The practical reading is that the inclusive system is not free. It removes a
term proportional to `l1 * tau` and adds one proportional to `l1 * delta`, so
it pays when the treatment effect is large against the design's own fit error
and costs when it is not.

That shows up directly in the cost. At `tau = 0` the inclusive system is
marginally behind in five of six DGPs, by 0.001 to 0.017. At three times the
panel scale it is ahead in all six, by 0.09 to 0.89. The determinant stays
benign throughout, averaging 0.965 and never falling below 0.867, so the
inflation is not what drives any of this.

Admitting the treated markets also buys a better rebuild, and the pre-period
rebuild error falls in all six DGPs when they are allowed in. Taking the two
together, the inclusive pool with the inclusive correction beats the clean pool
in four of six, ties one, and loses one:

| DGP | clean pool, iterative | inclusive pool, iscm |
| --- | --- | --- |
| `pangeo_seasonal` | 0.190 | 0.121 |
| `hsc_shared_trend` | 0.717 | 0.686 |
| `rank_shift_ok` | 0.050 | 0.045 |
| `rank_shift_dormant` | 0.049 | 0.045 |
| `fdid_ar1_rho.7` | 0.032 | 0.031 |
| `marex_native` | 0.372 | 0.381 |

The seasonal panel gains most, which is the panel whose clean rebuild was worst
and which defeated the threshold estimator earlier. Borrowing from the treated
markets is how a poorly reconstructable market gets rebuilt, and the inclusive
system is what makes the borrowing safe. Pairing the inclusive pool with the
iterative correction is the combination to avoid: it carries the full leak, and
at three times the panel scale it runs to 1.275 against 0.381 on
`marex_native`.

This arm imposes a homogeneous treatment effect on every DGP, including the one
shipping its own treated potential outcomes, so that `l1 * tau` is exact. It
therefore says nothing about effects that vary across treated markets, where
the leak becomes a weighted average that `l1` alone no longer summarises.

## Scope

What is measured assumes the contaminated market is known, and known from
outside the outcome data. Identifying it from the panel is a different problem,
not addressed in this study but addressed in the library: control markets
looking unusual after the intervention is often the estimator working, and
dropping the ones that look unusual is a specification search that biases
toward finding effects, so the identification needs a method and not a glance.

:class:`~mlsynth.SPOTSYNTH` is that method (O'Riordan and Gilligan-Lee 2025).
It forecasts each donor's post-intervention values from pre-intervention donor
data alone and flags the ones the forecast misses, returning the flagged set on
``excluded_idx``, with sensitivity analysis bounding the bias from flagging a
valid donor or missing an invalid one. Composing it with the repairs measured
here is the obvious next arm, and it would measure what detection error costs
the repair.

The contamination is a level shift on one market over the post-period. Two or
more contaminated markets, and dynamic contamination, are not measured.
Melnychuk (2024) reports the methods converging and degrading as the affected
share of the donor pool rises.

What is measured is contamination, not spillover. The distinction is where the
donor pool comes in. These panels contaminate one market and leave the rest
clean, so there is always clean material to rebuild from. Spillover decays with
proximity, so the donors that best reconstruct a contaminated market are the
ones most likely contaminated by the same leakage, and the repair's raw material
is what spillover takes away. `spillover_pool.py` relaxes the cross-weight but
keeps the clean pool, so it measures the machinery of the correction and not
that harder problem.

The library carries a method built for that harder case.
:class:`~mlsynth.RRSC` (He, Li, Shi and Miao 2026) writes the direct and
interference effects as a sparse-outlier component of a robust regression
against pre-period factor loadings, so it needs neither a nominated set of
contaminated markets nor a pool known to be clean, only that a majority of
controls are unaffected without knowing which. It returns an average effect, an
interval and a p-value for every unit.

RRSC and the repairs measured here answer different questions, and the
difference matters for an experimental design. RRSC replaces the weighting: it
is an estimator, and the weights it implies come from the loadings it fits
after the fact. The repairs here leave the design's committed ``w`` and ``v``
untouched and change only the data the control aggregate reads. Where the
weights were fixed before the experiment ran, that is the whole point, and RRSC
cannot stand in for the repair however well it estimates. Where there is no
such commitment, RRSC asks less of the analyst and should be the comparison
this study is measured against.

MAREX's own design spreads control weight thinly, which limits the exposure: a
mean largest control weight of 0.08 to 0.19 across these panels, against 0.51
for a plain single-treated-unit synthetic control on comparable draws. The
exposure grows as the pool shrinks or as a penalised design concentrates the
weights.

## What the two thresholds buy, measured end to end

The blank-window threshold is better calibrated at the selected `k*`, and it
places the crossover where a calibrated threshold should (40 replications per
DGP, contamination swept as a multiple of `thr_oos`):

| DGP | in-sample ratio | blank-window ratio | crossover |
| --- | --- | --- | --- |
| `fdid_ar1_rho.7` | 0.778 | 1.055 | 1.0 |
| `hsc_shared_trend` | 0.637 | 1.061 | 1.0 |
| `marex_native` | 1.008 | 1.077 | 1.0 |
| `rank_shift_dormant` | 0.835 | 1.090 | 1.0 |
| `rank_shift_ok` | 0.917 | 1.193 | 1.0 |
| `pangeo_seasonal` | 0.367 | 0.577 | 2.0 |

Five of six land on 1.0, against 1.25 to 3.0 under the in-sample threshold.
Seasonality stays unfixed at 0.577.

The calibration does not carry through to the decision. On the `panel` grid,
where the contamination size depends on neither threshold, both rules are
accurate and their costs are indistinguishable:

| DGP | accuracy, in-sample | accuracy, blank-window | cost, in-sample | cost, blank-window | cost, best possible | cost, never repair |
| --- | --- | --- | --- | --- | --- | --- |
| `fdid_ar1_rho.7` | 1.000 | 1.000 | 0.024 | 0.024 | 0.024 | 0.310 |
| `marex_native` | 0.993 | 0.989 | 0.227 | 0.228 | 0.227 | 1.421 |
| `rank_shift_ok` | 0.961 | 0.946 | 0.032 | 0.032 | 0.030 | 0.196 |
| `rank_shift_dormant` | 0.957 | 0.918 | 0.028 | 0.030 | 0.027 | 0.144 |
| `pangeo_seasonal` | 0.914 | 0.904 | 0.079 | 0.081 | 0.067 | 0.291 |
| `hsc_shared_trend` | 0.900 | 0.896 | 0.306 | 0.309 | 0.279 | 1.048 |

The in-sample rule is marginally ahead in five of six and tied in the sixth,
and both sit within a few percent of the best choice available with hindsight.
The two grids disagree because they ask different questions: the `thr_oos` grid
concentrates the contamination around the threshold, which is where a decision
is hard and a better threshold helps, and the `panel` grid spreads it over
realistic magnitudes, where most calls are not close and the threshold's
calibration stops mattering.

So the blank window is what to use when reporting a threshold as a number, and
the choice between the two does not change the repair-or-not call on contamination
of a realistic size. Never repairing is the expensive policy everywhere, by four
to six times.

Repairing the data continues to beat re-weighting across every DGP, on the same
grid: `iterative` against `renorm` runs 0.032/0.070, 0.717/1.763, 0.372/0.926,
0.190/0.327, 0.049/0.094 and 0.050/0.082.

## Status

Exploratory. Nothing here is wired into the library, and the repair is not a
new estimator: it is a post-hoc adjustment that applies to any design estimator
exposing weights, so it belongs with weight caps and concentration diagnostics
on the design estimators.
