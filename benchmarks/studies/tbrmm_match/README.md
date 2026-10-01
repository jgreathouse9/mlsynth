# TBRMM: where the two hill climbs agree

Compares mlsynth's `TBRMM` against `google/matched_markets`' `greedy_search` on
the GeoLift panel the repository ships, and separates the two ways they could
differ.

* Au, T. C. (2018). *A Time-Based Regression Matched Markets Approach for
  Designing Geo Experiments.* Technical report, Google LLC.

The reference is Apache 2.0 and is not vendored. Point
`MLSYNTH_MATCHED_MARKETS` at a checkout:

```
MLSYNTH_MATCHED_MARKETS=/path/to/matched_markets \
    python benchmarks/studies/tbrmm_match/run.py
```

## Why the comparison is split into sections

A hill climb over partitions can disagree with another hill climb for two
unrelated reasons, and one number confounds them:

1. the two engines score the same split differently, which is an arithmetic
   fault in one of them;
2. the two engines score identically and still walk different paths, which is
   two heuristics differing and is nobody's fault.

`run.py` answers them in that order. Section 1 scores every design either engine
recommends with both engines. Section 2 takes one augmentation step from a split
both agree on, enumerates the candidates once, scores them with each engine, and
reports every candidate they disagree about alongside the step each would take.
Section 3 reports the recommended designs and, where they differ, which one the
reference's own score prefers.

`reference.py` and `engine.py` expose the same two functions -- a scorer and a
`designs` mapping -- so `candidates.py` never knows which implementation it holds.
The reference indexes geos by position in a volume-sorted list while its designs
report labels; that translation lives in `reference.py` and nothing outside it
handles an index.

## Result

Complete agreement at `n_test=14`, `K=4`, `n_pretest_max=90`: the same treatment
group and the same control group at every treatment size, the same four
assumption gates, and the inverse detectable impact agreeing to 9.3e-15 at its
worst. Of the 39 candidates at the first augmentation step, zero are scored
differently and both engines take the same one.

Agreeing on the path and not only the answer is the stronger claim. A search
returns a list of geo names, so two implementations can land on similar scores
from different partitions; here they make the same move.

## The defect this found

The A/A test was implemented as a scan over every `n_test`-long window of
cumulative residuals, asking whether any of them would already read as an
effect. The reference does something narrower: hold out the last `n_test`
pretest periods, refit eqn 1 on what remains, estimate that one window, and pass
when its interval covers zero or when the probability of a false positive stays
under 0.2.

The scan is much stricter and failed nearly every split, so the gate read 0
where the reference read 1 on all four designs. At `k = 2` that cost the search
denver, which the reference takes, and the two walks separated from there. The
54 unit tests in place did not catch it: the gate returned a bool of the right
type, and the search returned a plausible list of cities either way.

That is the argument for cross-validating a design search at all. An estimator
that is miscoded usually shows itself in a residual, a counterfactual or a
placebo. A design search has none of those to show.

The corrected test runs through TBR's own `cumulative_posterior`, so the A/A
window is estimated by the estimator the design is being chosen for.

## Durable form

`benchmarks/cases/tbrmm.py` pins this comparison as a CI gate. The reference's
output is captured there as a literal, so the case needs no checkout and no
network: 33 metrics covering membership, the gates, the correlation and the
detectable impact at all four treatment sizes.
