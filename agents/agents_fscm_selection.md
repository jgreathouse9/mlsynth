# FSCM donor selection: is the greedy path leaving anything on the table?

A spike, prompted by a question: could FSCM be seeded with FDID's selected
donors and polish from there, instead of starting its forward search from
nothing?

The answer is no, and the measurements that settle it turn up a different
defect, and that one needs a ticket. This document is the evidence, not a change.

## What FSCM actually optimizes

`_forward_select` uses two criteria, and conflating them is the first way to
get this wrong.

| step | criterion | scope |
| --- | --- | --- |
| which donor to add | `_outcome_rmspe` over `slice(0, T0)` | in-sample, simplex-weighted |
| how many to keep | `argmin` of `_rolling_origin_rmspe` over sizes | out-of-sample, simplex-weighted |

The greedy ordering is an in-sample fit criterion. The rolling-origin
criterion never chooses a donor; it only picks where to cut the path.

FDID selects by `argmax` of pre-period R-squared on an equal-weighted DID fit
(`fdid_helpers/estimation.py`, `np.argmax(R2_path)`). R-squared and RMSPE are
monotone in each other for a fixed target, so FDID and FSCM's greedy step
differ in exactly one thing: the weight restriction. FDID asks which donors fit
best when averaged equally; FSCM asks which fit best when weighted on the
simplex. That is closer than it first appears, which is what makes the seeding
idea reasonable enough to test instead of dismissing.

## Is greedy suboptimal at all?

Exhaustive search over every subset at each size, scored on FSCM's own greedy
criterion. Proposition 99 has 38 donors over 19 pre-periods; Basque 16 over 20.

| panel | size | greedy | exhaustive | gap | same set |
| --- | ---: | ---: | ---: | ---: | --- |
| Proposition 99 | 1 | 4.47543000 | 4.47543000 | 0 | yes |
| Proposition 99 | 2 | 3.98278491 | 2.58156240 | +1.401 | no |
| Proposition 99 | 3 | 1.97276826 | 1.97276826 | 0 | yes |
| Basque | 1 | 0.15968246 | 0.15968246 | 0 | yes |
| Basque | 2 | 0.08426660 | 0.08426660 | 0 | yes |

At `k = 2` greedy is 54 percent worse than the best pair. It takes Montana at
`k = 1` and is then locked into it, while the best pair is Nevada with New
Mexico and contains neither of greedy's picks in the same combination. By
`k = 3` greedy has caught up and lands on the exhaustive optimum exactly.

`k = 3` is the size FSCM chose on this panel, and `k = 2` is the size it chose
on Basque. So at the size that produces the answer, greedy is optimal on both
canonical panels, and there is nothing for a better search to recover.

Cost: 8436 subsets at `k = 3` took 161 seconds. `k = 4` is 73815 subsets, so
exhaustive stops being viable one step later on a pool this size.

## Would FDID's set have helped?

Scored on FSCM's greedy criterion, lower is better.

| set | size | in-sample RMSPE |
| --- | ---: | ---: |
| greedy `k = 2`, Montana and Nevada | 2 | 3.98278491 |
| exhaustive `k = 2`, Nevada and New Mexico | 2 | 2.58156240 |
| greedy `k = 3`, Montana, Nevada, Utah | 3 | 1.97276826 |
| FDID's set: Colorado, Connecticut, Montana, Nevada | 4 | 3.70609498 |
| FDID's first two by weight | 2 | 6.25804885 |
| FDID's first three by weight | 3 | 3.99375520 |

FDID's four-donor set scores 3.71 where greedy's three-donor set scores 1.97.
It is worse at a larger size, and its subsets are worse again. Seeding the
search there starts it further from the answer than starting from nothing.

The reason is the weight restriction, and it is not a small difference. Equal
weighting rewards a set that brackets the treated unit symmetrically, so that
the average lands near it. Simplex weighting rewards a set whose convex hull
contains the treated unit, and then picks the point in that hull. A set chosen
for the first property has no reason to have the second.

So the seeding proposal is rejected on measurement, not on the combinatorial
argument. The combinatorial objection would have been the right one if FDID's
set had been a good start; it is not.

## The defect this turned up instead

`optimal_size = argmin(test_rmspe) + 1` compares greedy's `k = 1` model,
greedy's `k = 2` model, greedy's `k = 3` model, and so on. It does not compare
the best model at each size. On Proposition 99 the `k = 2` entry of that curve
comes from a pair 54 percent worse than the best pair, so the size decision is
made on a curve that is not the curve it is read as.

Proposition 99 chose 3 anyway, so nothing reported is wrong there. Whether the
curve's shape can invert a size decision on some panel is open, and it is the
question to answer before anything is built. It is also cheap to answer now:
the migration in #622 took a Proposition 99 fit from 27.93s to 0.57s, so
exhaustive `k <= 3` curves across the benchmark panels are affordable.

A second observation from the same source. The root-cause ladder on #622 found
that correcting the weight solver by 1e-5 reordered the greedy path from step 6
onward. Candidates that deep are near-tied, so the tail of the path is not a
well-determined object whatever solver computes it. That is consistent with
what this spike measures: the path is unstable where it does not matter and
correct where it does.

## Recommendation

Do not seed FSCM from FDID. The evidence is the table above: the set is worse
by FSCM's own criterion at a larger size.

Do not build multi-start or branch and bound on this evidence either. Greedy
attains the exhaustive optimum at the chosen size on both canonical panels, so
there is no measured gap for a better search to close, and building a search to
close a gap nobody has observed is speculative.

Do open a ticket on the size-selection curve. The finding that stands on its
own is that `test_rmspe[k]` is greedy's `k`-donor model and is read as the best
`k`-donor model, and those differ by 54 percent at one point on Proposition 99.
Establishing whether that can move `optimal_size` on any panel is a bounded
piece of work and does not require a new search algorithm to answer.

## Reproducing

Both scripts are one-off measurement, not library code:

- exhaustive against greedy, per size, on both panels;
- FDID's set scored on FSCM's greedy criterion.

Each calls `_forward_select`, `_fit_weights` and `_outcome_rmspe` directly on
inputs from `prepare_fscm_inputs`, so nothing here depends on a private copy of
the criterion.
