# TBR's design search, batched the way FDID's selection is

References:

- Tim C. Au (2018), *A Time-Based Regression Matched Markets Approach for
  Designing Geo Experiments*, Google LLC. Algorithm 1, the climb.
- Jouni Kerman, Peng Wang and Jon Vaver (2017), *Estimating Ad Effectiveness
  using Geo Experiments in a Time-Based Regression Framework*, Google. The model
  the climb scores designs with.
- `mlsynth.utils.fdid_helpers.estimation.forward_did_select`, which scores every
  candidate donor of a forward step from one matrix-vector product.

This is a study. It produces a recommendation, not estimator code.

```
python -m benchmarks.studies.tbr_batched_climb.run              # GeoLift and synthetic panels
python -m benchmarks.studies.tbr_batched_climb.run --seattle    # the Seattle DMI block groups
```

Results land in `results/`. The tests are `benchmarks/tests/test_tbr_batched_climb.py`
(42, every line of the study covered); the mutants are the `tbr-batched-climb`
target in `tools/mutation/targets.toml` (7 killed, 1 recorded as equivalent).

## The step, and why it batches

`greedy_search` in `mlsynth/utils/tbr_helpers/design/search.py` alternates two
step functions. `_augment` adds a treatment geo and removes it from the control
group if it was there; `_match` toggles one control geo in or out. Either way a
candidate is the current pair of group aggregates plus or minus one geo's
column, the structure FDID's forward step has with one aggregate where TBR has
two.

Every term of the reference objective's key is a column operation on those
candidates. The centred sums give the slope, the residual scale, the correlation
and the smallest detectable impact; the residual columns give the Durbin-Watson
statistic and the Brownian-bridge CUSUM; the same sums over the first
`T - n_test` periods give the A/A test. `batched_keys` forms the candidates as
the columns of a `T x M` matrix, centres each column directly as
`groupfit.group_sums` does for one candidate, and returns every key at once.
`match_step` and `augment_step` replace the two step functions inside the
module's own `greedy_search`, so the orchestration under test is the shipped one.

## Exactness

The claim is equality with the exact climb, not closeness. The key compares
four gates, a correlation rounded to two decimals and an inverse impact, and a
batched key can differ from `score_split`'s in its last bits. That matters only
at a decision, so the batched step sends to `score_split`:

- every candidate within `1e-9` of a decision: the 0.8 correlation gate, a
  `round(corr, 2)` boundary, the correlation clip, the Durbin-Watson band, the
  Brownian-bridge envelope, the A/A interval's zero crossing and its
  false-positive cutoff;
- every candidate within `1e-9` of the best key;
- every candidate whose fit is degenerate, including any sum of squares within
  `64 eps` of the rounding of its uncentred terms (`agents_solvers.md` rule 4).

The last rule came from a measurement. Forming a candidate's aggregate as
`x - Y_j` at the data's level leaves an error of `eps |x|` that centring does
not remove, so a fit that is exact on paper read `1e-13`, and the first version
kept climbing on a key made of rounding noise where the exact climb stops.

The near-tie rule has a pinned witness: on seed 1, with geo 9 equal to geo 8
times `1 + 5 eps`, the two candidates' batched keys tie exactly while their
exact keys differ by one unit in the last place, and the exact climb takes geo
9. A first-index rule on the batched keys would take geo 8.

## Validation

| panel | geos | treatment sizes | `control_start` | identical to the exact climb |
| --- | --- | --- | --- | --- |
| GeoLift | 40 | up to 3, 4, 10 | carried, pool, best | 9 of 9 |
| synthetic | 100 | up to 3, 4, 25 | carried, pool, best | 9 of 9 |
| synthetic | 200 | up to 3, 4, 50 | carried | 3 of 3 |
| synthetic | 500 | up to 3, 4 | carried | 2 of 2 |
| Seattle block groups | 480 | up to 3 | carried, pool, best | 3 of 3 |

Identical means the same treatment group, control group and unassigned geos at
every size, the same exact key, the same trace and the same convergence flag.
`benchmarks/cases/tbrmm.py` pins `google/matched_markets`' designs on GeoLift
geo for geo; the batched climb reaches every one of them.

## Speed

Search time, the default `control_start`, one core:

| geos | treatment sizes | exact climb | batched | speedup | exact scores, exact climb against batched |
| --- | --- | --- | --- | --- | --- |
| 40 | up to 3 | 0.17 s | 0.027 s | 6.3 | 1,391 against 37 |
| 100 | up to 4 | 1.70 s | 0.17 s | 10.3 | 12,525 against 128 |
| 200 | up to 50 | 26.6 s | 2.0 s | 13.3 | 164,125 against 895 |
| 480 (Seattle) | up to 3 | 20.6 s | 0.90 s | 23.0 | 78,697 against 165 |
| 500 | up to 3 | 72.2 s | 3.5 s | 20.4 | 250,447 against 502 |

The user-facing `TBR(...).fit()` in searched mode goes from 0.39 s to 0.05 s on
GeoLift, from 24.9 s to 2.1 s on 200 geos at up to 50 treated, and from 15.7 s to
0.92 s on Seattle, with the same designs. The speedup grows with the number of
geos, since a step's batched cost grows with `T x M` while the exact one pays a
Python call per candidate.

For scale, Google's `meridian-geox` (commit `3424ae0`, JAX 0.12 on CPU, measured
on the same machine with its default 100,000 stratified candidates) took 1.35 s
on GeoLift, 2.28 s on 100 geos and 4.54 s on 200 in steady state, and 10.6 s on
its first call. Its search samples and scores candidates instead of climbing,
so the two answer different search problems; the times are from panel to
recommended design.

## Seattle

`basedata/seattledmi.parquet` is the R `microsynth` package's Drug Market
Intervention panel: 9,642 census blocks over 16 quarters, the intervention
starting in quarter 13 on 39 blocks. At block level it cannot run: 308 blocks
record no crime before the intervention, and a constant treatment series stops
the search (its A/A refit has no residual scale; see below). Dropping them is
not enough, since 423 more are constant over the A/A window, 1,948 repeat
another block's series, and 74 percent of block-quarters hold 0 or 1 crime.
Summed to the 480 block groups no series is constant or repeated, the median
block group records 18 crimes a quarter, and the intervention's 39 blocks fall
in exactly three of them, which matches a design of up to three treated units.

Both climbs agree on all three `control_start` settings (above). The designs
they agree on are not believable, and the reason is the panel's shape, not the
batching:

| treatment size | controls | in-sample residual scale | RMSE on the four quarters the search never saw | ratio |
| --- | --- | --- | --- | --- |
| 1 | searched (418) | 0.01 | 5.16 | 710 |
| 1 | every other block group (479) | 1.37 | 5.09 | 3.7 |
| 2 | searched (405) | 0.01 | 5.25 | 642 |
| 2 | every other block group (478) | 1.66 | 4.92 | 3.0 |
| 3 | searched (391) | 0.01 | 6.09 | 760 |
| 3 | every other block group (477) | 1.96 | 5.27 | 2.7 |

(`seattle_holdout`: the search runs on quarters 1 to 8, each design's fit
predicts quarters 9 to 12.) With 8 quarters to fit and some 400 control geos to
toggle, the climb assembles a control aggregate that matches the treated series
to 0.01 crimes in-sample, and out of sample it predicts no better than every
other block group does. The detectable impacts the full-sample search reports,
1.3 to 1.4 crimes, are that fit, and the A/A gate cannot see it because the A/A
window is among the periods the control group was chosen on. This is Au's
search, not the batching; the reference implementation climbs the same
objective.

The intervention's own three block groups, scored as a design against every
other block group, correlate with the rest of Seattle at 0.10, fail the
correlation gate, and could detect 171 crimes over four quarters.

## Two defects in shipped code

1. A candidate whose A/A refit has a residual scale of exactly zero makes
   `false_positive_probability` divide by zero, and the whole search raises
   `ZeroDivisionError`. Two geos with identical series in the treatment and
   control groups do it, and so does any constant treatment series, which is
   why the Seattle blocks cannot be searched. The tests pin that both climbs
   fail alike; the fix is a gate that fails, or reports, a zero scale.
2. The search overfits when periods are few against geos, as above, and none of
   its four gates is computed on periods the control group was not chosen with.

## Recommendation

Build the batched steps into `mlsynth/utils/tbr_helpers/design/search.py` as
the reference objective's step functions, with this study's tests and mutants,
and `score_split` kept as the authority on every decision. They are exact by
construction and by test, and 6 to 23 times faster on panels of 40 to 500 geos.
The `"paper"` objective (CUSUM and Breusch-Godfrey) keeps the exact steps until
its Breusch-Godfrey regression is batched.

Before that, on their own branches: the zero-scale defect, which the batched
steps reproduce faithfully, and a guard against the overfitting, for instance
choosing the control group on the earlier pretest periods and reporting each
design's error on the later ones, the check `seattle_holdout` runs here.
