# Is Bayesian synthetic control viable for geo design?

The question this study answers is narrow and practical: can a Bayesian
synthetic control stand in for augsynth inside a GeoLift-style design loop, and
should a team be told yes or no. It is not an attempt to replicate Meta's
GeoLift numerically. The replication already exists
(`benchmarks/cases/geox_augsynth_geolift.py`), and one of the findings here is
that numerical agreement with it is not available to any other estimator.

## The verdict

Yes, with one requirement.

A Bayesian synthetic control drops into GEOX's engine seam with no change to
the harness, recovers the known effect on Meta's own test panel, and produces
credible intervals that cover. The requirement is that the interval be a
posterior predictive carrying the pre-period autocorrelation. Without that step
the intervals under-cover badly, and the MDEs built on them are optimistic by
roughly a factor of two.

On Meta's GeoLift test panel, treating chicago + portland from period 91, MVBBSC
returns ATT +5.8% of treated volume with a 90% credible interval of
[+3.3%, +8.3%] and P(effect > 0) = 1.000, at max r-hat 1.002 with no
divergences. GeoLift's own walkthrough puts the lift on this panel at about 5%.
The counterfactual and its bands are in `results/bayes_sc_geolift.png`.

What a team should not expect is the published market ranking. See arm 2.

## What is here

| file | what it measures |
| --- | --- |
| `engines.py` | BSCM and MVBBSC wired into GEOX's five-function engine seam, with the posterior-predictive ATT and the AR(1) shock |
| `calibration.py` | arm 1: placebo coverage of every interval, on Meta's pre-test panel |
| `market_selection.py` | arm 2: each engine against GeoLift's published BestMarkets top five |
| `criterion.py` | arm 3: what the MDE ranks on, and what to rank on instead |
| `plot.py` | the counterfactual and credible bands on the test panel |
| `results/` | the runs behind the tables below |

Every arm runs from the repository root against data already in `basedata/`:

```bash
python -m benchmarks.studies.bayesian_geo_design.calibration      results/calibration.json
python -m benchmarks.studies.bayesian_geo_design.market_selection results/market_selection.json
python -m benchmarks.studies.bayesian_geo_design.criterion        results/criterion.json
python -m benchmarks.studies.bayesian_geo_design.plot
```

## Arm 1 — do the credible intervals cover?

Each of the 40 locations in `geolift_market_data.csv` takes a turn as a placebo
treated unit, pre-period 1..75, placebo post 76..90. Nothing happened to any of
them, so a nominal 90% interval should contain zero about 90% of the time.

| interval | coverage | mean width |
| --- | --- | --- |
| augsynth, conformal | 89.7% | 716 |
| sdid, placebo | 92.5% | 1323 |
| BSCM, horseshoe, AR shock | 75.0% | 315 |
| MVBBSC as shipped, iid shock | 65.0% | 299 |
| MVBBSC, AR shock | 87.5% | 418 |

The AR(1) step is what separates 65% from 87.5%, and both intervals come off the
same MCMC run, so the comparison isolates that one choice. The mechanism is
specific to an averaged estimand: under an iid shock the variance of a
15-period mean falls as `sigma^2 / h`, under positive autocorrelation it falls
more slowly. A per-period band looks acceptable either way, so averaging is what
exposes the defect.

MVBBSC starts ahead of BFSC here because it already adds the sigma shock before
the back transform. `mlsynth/estimators/bfsc.py:106` takes percentiles of `cf`,
which `bfsc_helpers/model.py:111` registers as the latent mean, so that band is
a credible interval on the mean and is about 3.5x too narrow to compare against
an observed series. Measured on six untreated Walmart stores, the shipped band
covers 70% of pre-period weeks and 66% of post-period weeks against a nominal
95%; the predictive version covers 98% and 96%. The pre-period figure is the
informative one, since BFSC fits the pre-period and a calibrated band has to
cover there.

## Arm 2 — the market ranking is not transferable

`benchmarks/cases/geox_augsynth_geolift.py` pins GEOX(engine="augsynth") to the
BestMarkets top five that GeoLift publishes. Same data, same config, same
harness, engine swapped:

| design (duration) | published | augsynth | bscm | mvbbsc |
| --- | --- | --- | --- | --- |
| chicago+portland (15) | 1 | 1 | 21 | 1 |
| chicago+cincinnati+houston+portland (15) | 1 | 1 | 8 | 2 |
| chicago+portland (10) | 3 | 3 | 36 | 14 |
| chicago+cincinnati+houston+portland (10) | 3 | 3 | 16 | 20 |
| chicago+houston+portland (10) | 5 | 5 | 2 | 29 |
| mean absolute rank error | — | 0.00 | 15.20 | 10.60 |

Calibration buys the top of the table and not the rest. MVBBSC picks GeoLift's
own first design first; BSCM, whose intervals under-cover, puts it 21st. Mean
rank error improves from 15.20 to 10.60 and stops there.

The investments give the reason. Both Bayesian engines report exactly half the
published figure for three of the five designs, and match to the cent on the
fourth. Investment is `cpic * effect_size * treated volume`, so a halved
investment is an MDE of 0.05 where GeoLift reports 0.10, and the design that
matches is the one whose published MDE is already 0.05. MVBBSC's calibrated
interval is 418 wide against augsynth's 716 at 87.5% against 89.7% coverage, so
on a 0.05-step effect grid a 1.7x tighter interval drops the MDE one grid step.
Both intervals land near nominal; augsynth spends more width to get there.

The conclusion is that the published ranking is a property of augsynth's
interval geometry and not of the designs. augsynth scores 0.00 because it is the
reference implementation. Arm 3 measures the same fragility directly.

## Arm 3 — the MDE ranks on bias

This arm is independent of the engine. Scoring the same 990-design pool
(Walmart, cardinality-2 MAREX designs) against the realised out-of-sample
contrast error, over three origins:

| criterion | mean rho | selected design's realised error |
| --- | --- | --- |
| `mde` | −0.018 | 11.94% |
| power at 5% | +0.468 | 4.31% |
| power at 10% | +0.524 | 6.89% |
| centred `mde` | +0.414 | 44.21% |
| `att_error_rmse` | +0.243 | 11.40% |
| in-sample contrast | +0.862 | 2.93% |
| best in pool | — | 1.40% |

`placebo_detection_boundary` returns `up = (z sigma - tau0)/baseline` and
`down = (-z sigma - tau0)/baseline`, an interval displaced by `tau0`, the
backtest's estimation error with nothing injected. `compute_mde` keeps the
smaller magnitude of the two, which is by construction the side `tau0` pushes
toward, so a biased backtest reports a smaller MDE and reads as more sensitive.

The ladder behind that claim:

- rho(|tau0|/sigma, |MDE|) = −0.79 (sdid) and −0.93 (tasc). The MDE is
  determined by the bias term, not by the noise term: rho(sigma, |MDE|) is
  +0.57 and −0.31, inconsistent in sign.
- Detection-interval asymmetry |up|/|down| = 0.73 and 6.64, against 1.00 for a
  centred interval. sign(MDE) equals sign(tau0) in 96% of sdid designs.
- `tau0` is a persistent property of the candidate, not backtest noise: sign
  consistency across eight backtests is 0.77 (sdid) and 0.99 (tasc) against 0.35
  for a coin flip, with between-candidate spread 2.6x and 4.2x the
  within-candidate spread. It does not average out.
- Removing the bias channel flips the sign: ranking on `z sigma / baseline`
  alone gives rho +0.414, positive at every origin.

`compute_accuracy` already computes the un-conflated quantities and says so in
its own docstring: "a ratio above one says the design's error exceeds what its
null admits, so the p-values, and the MDE built from them, are
anti-conservative". `att_error_over_sigma` is merged into the shortlist at
`orchestration.py:329` and nothing ranks or gates on it. Median on the Walmart
panel is 0.65, so this is a guard and not a live fire.

## Negative results

Recorded because each cost a day and each would otherwise be attempted again.

- Shrinking the design's matching target does not help. Matching MAREX designs
  on a rank-r denoised panel or on latent loadings instead of raw pre-period
  outcomes changes which markets are selected at every rank, and improves
  nothing: five wins and four losses across a fit-window sweep. The Walmart
  panel is 98.1% rank-1, so there is no noise to shrink and truncation discards
  matching signal.
- A rolling-refit cross-validated selection criterion is worse than the
  in-sample contrast it was meant to improve on: 3.06% against 2.48%, winning 4
  of 12 origins. The sub-window refits are too unstable to score with.
- Residual autocorrelation does not predict which designs degrade out of
  sample: rho(lag-1 autocorrelation, out-of-sample degradation) = 0.031, sign
  flipping across origins. It matters for the interval and not for the ranking.
- Centring the MDE confirms the diagnosis and makes the decision worse (44.21%
  against 11.94%), because `z sigma / baseline` ranks on noise relative to
  treated volume and so favours large markets. `tau0` was partly offsetting
  that.
- Recast's multiplier `M` is inert for a design that targets the population
  mean. `M* = 1.0000` at 26 of 26 origins when the target is the grand mean,
  which lies in the donor hull by construction; `M* = 1.049` with a 46% fit
  improvement when the treated set is the three largest stores. The multiplier
  is a correction for a treated set that was not designed.
- TASC as a GEOX engine is 6.2x faster than sdid single-core (0.273s against
  1.69s a candidate, because the state-space fit is candidate-independent and
  amortises) and makes the MDE worse: median |tau0|/sigma 1.84 against 0.74, and
  48% of its MDEs at the effect-grid floor. The fault is above the engine seam,
  so no engine swap reaches it.

## Limitations

- Arm 1 is one panel and one post-window length. Coverage is a function of the
  horizon and the residual process, and 15 periods of 90 is one point.
- Arm 2 is one configuration of one walkthrough. The rank comparison is against
  five published designs.
- Arm 3 runs three origins of one panel with 150 sampled designs each, scored
  with the sdid engine. The `tau0` mechanism is engine-independent; the
  magnitudes are not.
- `market_selection.py` runs MVBBSC at 600 warmup and 600 samples over two
  chains. The arm-1 calibration run used 1000 and 1000, at max r-hat 1.003.
  Per-fit r-hat is recorded but not asserted in arm 2.
- The realised error arm 3 scores against is the MAREX weighted contrast. Every
  criterion correlates far worse with the realised sdid gap (0.12 to 0.25,
  including the in-sample contrast at 0.196), because sdid refits donor and time
  weights per candidate and so depends less on which markets were chosen. The
  in-sample contrast leads the table partly because it shares a functional form
  with the yardstick.
