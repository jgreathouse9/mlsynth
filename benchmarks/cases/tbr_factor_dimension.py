r"""Where TBR's posterior is calibrated: the one-factor condition, measured.

Li and Van den Bulte (2022) develop ADID under a linear factor model, and Li
(2024) web appendix A states the identifying assumption under a one-factor
model, under which the correlation between the treated series and the control
average does not depend on time, so parallel trends reduces to parallel
pretrends. TBR fits that relation with a free intercept and a free slope on one
control aggregate and prices the interval with equation 6.

This case measures what that condition buys and what its absence costs, on
panels carrying no treatment at all, where the cumulative effect is zero in
every period and coverage is the share of intervals containing zero.

Why one factor is the condition, and not merely a convenience
------------------------------------------------------------

Aggregating gives each group the mean loading of its members. The fitted
relation holds at every period exactly when the two mean loading vectors are
proportional. At one factor they are scalars and proportionality is automatic.
Past one factor two independently drawn mean vectors are not proportional, and
one regressor cannot absorb the difference, so a gap persists with no treatment
anywhere.

That is asserted directly and not only through coverage: with the noise
switched off, the best affine fit leaves a relative gap at machine precision at
one factor and more than ten orders of magnitude larger past it. The sine of
the angle between the two mean loading vectors is exactly zero at one factor.

Averaging more geos does not help. The sweep runs treated groups of 2, 5 and 25
of 50 geos, and multi-factor coverage stays far below nominal at every size, so
the failure is not a small-group artefact.

What is asserted, and what is not
---------------------------------

The one-factor cell is asserted against the nominal level, with a tolerance of
four sampling standard deviations of the pooled rate at the replication count
in this module, so it tightens when ``MLSYNTH_TBR_FACTOR_REPS`` is raised and
is not a number read off a previous run.

The multi-factor cells are asserted as an upper bound on coverage. Coverage is
bounded below by zero, so a two-sided window whose lower end is vacuous states
the claim: the best multi-factor cell sits no higher than 0.70 against a
nominal 0.90. The size of the shortfall is reported and not pinned, because it
depends on the loading distribution, which the papers do not specify.
"""
from __future__ import annotations

import os

import numpy as np

from benchmarks.studies.tbr_factor import coverage as cv
from benchmarks.studies.tbr_factor import dgp

#: Replications per (factor count, treated-group size) cell.
REPS = int(os.environ.get("MLSYNTH_TBR_FACTOR_REPS", "25"))

#: Nominal interval level the sweep is read at.
LEVEL = 0.90

#: Replications behind each factor count's pooled rate.
_N = REPS * len(cv.KS)


def _sd(p: float) -> float:
    return (p * (1.0 - p) / _N) ** 0.5


def run() -> dict:
    rows = cv.run_sweep(REPS, level=LEVEL, seed=0)
    s = cv.summarise(rows)

    one = [dgp.noiseless_gap(seed, 1, k) for seed in range(8) for k in cv.KS]
    many = [dgp.noiseless_gap(seed, r, k)
            for seed in range(8) for k in cv.KS for r in cv.RS if r > 1]
    decades = float(np.log10(min(many) / max(one)))

    print(f"  coverage at nominal {LEVEL}, {int(s['n_per_r'])} reps per factor count")
    for r in cv.RS:
        print(f"    r = {r}: {s[f'cov_r{r}']:.3f} "
              f"[{s[f'cov_r{r}_lo']:.3f}, {s[f'cov_r{r}_hi']:.3f}]   "
              f"mean width {s[f'width_r{r}']:8.3f}   "
              f"collinearity {s[f'collinearity_r{r}']:.4f}")
    print(f"  noiseless relative gap: one factor max {max(one):.2e}, "
          f"multi-factor min {min(many):.2e}  ({decades:.1f} decades apart)")
    print(f"  shortfall from nominal at the best multi-factor cell: "
          f"{LEVEL - s['multifactor_cov_max_over_k']:+.3f}   (reported, not pinned)")

    return {
        "cov_r1": s["cov_r1"],
        "collinearity_r1": s["collinearity_r1"],
        "noiseless_gap_r1": max(one),
        "noiseless_gap_decades": decades,
        "multifactor_cov_max": s["multifactor_cov_max_over_k"],
        "n_factor_counts": float(len(cv.RS)),
    }


EXPECTED = {
    # The method's own condition: under one factor the relation is exact and
    # the posterior should attain its nominal level.
    "cov_r1": (LEVEL, round(4 * _sd(LEVEL), 4)),
    # Proportionality at one factor is not approximate.
    "collinearity_r1": (0.0, 1e-12),
    "noiseless_gap_r1": (0.0, 1e-9),
    # And past one factor the gap is orders of magnitude larger, not merely
    # bigger. The lower end of this window carries the claim.
    "noiseless_gap_decades": (13.0, 3.0),
    # Coverage is bounded below by zero, so the upper end carries the claim:
    # the best multi-factor cell stays well under nominal.
    "multifactor_cov_max": (0.35, 0.35),
    "n_factor_counts": (4.0, 0.5),
}
