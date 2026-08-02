"""Path A: Wiltshire (2023), Walmart Supercenters and local employment, STACKEDSC.

Reproduces the event-study geometry of Section 4.2 on the author's own panel:
566 treated counties in six adoption cohorts against 39 never-treated donors,
1990-2005, effects in percent of each county's own pre-treatment level and
averaged under 1990 population weights.

Geometry, not a cell match -- and why
-------------------------------------

This case does not pin the paper's Table 4 magnitudes, and the reason is a
component that cannot be recovered from the published materials rather than a
tolerance that could be widened. ``allsynth`` calls Stata's ``synth``, whose
default predictor-weight rule is a regression on the pre-treatment outcomes, not
the Abadie-Diamond-Hainmueller nested search. That rule ships as a compiled Mata
library with no source distributed. Its structure is recoverable from the
library's symbol table, but two choices are not -- whether the regression runs
once over all unit-period observations or once per pre-treatment period, and
whether an intercept is fitted -- and across eight of the paper's cells the two
readings tie while moving the five-year estimate by more than the estimate
itself. See ``docs/replications/stackedsc.rst``.

What the paper states in prose is matched, and that is what is pinned here: the
design's counts, an excellent pre-treatment fit, no effect in the year of entry,
a decline beginning the year after, and a large negative effect by five years.

The paper's specification, and the part of it that is not expressible
--------------------------------------------------------------------

Appendix A.2 matches on ten covariate averages plus four outcome lags. The
covariate averages are over an absolute window -- the five years before any unit
is treated -- which ``covariate_windows`` expresses. The outcome lags are
relative to each cohort's own base period, which it does not: a covariate is one
value per unit, and a donor serves six cohorts with six different base periods,
so no single column can carry "the outcome two years before treatment" for a
donor. Issue #345 tracks the base-period-relative window option that would
close this.

Matching on the outcome path is used instead. It is not a workaround chosen for
convenience: the pre-treatment path is a superset of four lags of it, and it is
the specification that reproduces the paper's stated pre-treatment fit. The
covariate specification as expressible today is measured too, and reported --
its pre-treatment RMSE is 27x worse, which is the quantitative form of the
missing lags and is pinned so that a future relative-window option can be seen
to fix it.

Structural invariants
---------------------

Three quantities here are not comparisons with the paper but properties the
estimator must have on this design, and each would fail loudly if broken.

``base_period_gap`` is the indexing identity. Rescaling each unit and its donors
to their own value at ``T0i`` while requiring the weights to sum to one forces
the synthetic control to reproduce the treated level exactly there, so the gap
at ``e = -1`` is zero by construction rather than by fit. It comes out at 6e-16;
a wrong base year would break it immediately and break nothing else visibly.

``batched`` records that each cohort was solved as one multiple-right-hand-side
program. It is what makes 566 counties six solves rather than 566, and it is
false exactly when a donor predicate binds -- also measured here.

``county_dispersion_at_five`` is the caution against reading a single county.
The per-county five-year gaps have a standard deviation of 14.1 percent around a
weighted mean of -0.90. Part of that is genuine heterogeneity; part is that with
5 to 10 pre-treatment periods against 39 donors the individual weight vectors
are not identified at all, so two solvers reaching the same fit assign different
weights and different post-treatment paths. The average is the estimand.

Inference is deliberately absent. The placebo layer costs 22,074 simplex solves
in its default pool on this panel, far past the runtime budget for a case, and
the paper's p-values rest on the same unrecoverable predictor-weight rule as its
magnitudes.

Reference values: Wiltshire, J. C. (2023), Section 4.2 and Figure 6, for the
prose claims; the remaining rows are this estimator's own deterministic output
on ``basedata/allsynth_walmart.parquet``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# Appendix A.2's covariate block, less the two the shipped panel does not carry.
COVARIATES = ["pop_t", "unr", "lfp", "emr", "emps_shr_n44", "prop_white",
              "pe_salary_n10", "pe_salary_n44"]
# "the five years before any unit is treated" -- the first cohort adopts in 1995.
COVARIATE_WINDOW = {c: (1990, 1994) for c in COVARIATES}


def _panel() -> pd.DataFrame:
    df = pd.read_parquet(_BASE / "allsynth_walmart.parquet")
    df["treat"] = ((df.supercenter == 1)
                   & (df.year >= df.super_year)).astype(int)
    return df


def _fit(df: pd.DataFrame, **over):
    from mlsynth import STACKEDSC

    cfg = {"df": df, "outcome": "emps_n10", "treat": "treat",
           "unitid": "cty_fips", "time": "year",
           "agg_weights": "pop90",          # fn 28: 1990 population
           "n_lags": 5, "n_leads": 6,       # the balanced e = [-5, 5] window
           "display_graphs": False}
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return STACKEDSC(cfg).fit()


def _path(res):
    return (np.asarray(res.event_study.horizons, int),
            np.asarray(res.event_study.tau, float))


def run() -> dict:
    df = _panel()
    res = _fit(df)
    e, tau = _path(res)
    pre = e < 0
    out: dict = {}

    # --- the design is the paper's ------------------------------------
    out["n_treated"] = float(res.design.n_treated)
    out["n_donors"] = float(res.design.n_donors)
    out["n_cohorts"] = float(len(res.design.cohorts))
    out["n_horizons"] = float(len(e))

    # --- structural invariants ----------------------------------------
    out["base_period_gap"] = float(abs(tau[e == -1][0]))
    out["batched"] = float(res.design.batched)
    out["weights_sum_to_one"] = float(
        max(abs(sum(u.donor_weights.values()) - 1.0)
            for u in res.per_unit.values()))

    # --- the paper's prose claims, in order ---------------------------
    # "an excellent pre-treatment fit"
    out["pre_rmse"] = float(res.fit_diagnostics.rmse_pre)
    out["max_pre_gap"] = float(np.abs(tau[pre]).max())
    # "no effect in the year of Supercenter entry"
    out["gap_at_entry"] = float(tau[e == 0][0])
    # "a downward trend begins in the year following"
    out["declines_monotonically_after_entry"] = float(
        np.all(np.diff(tau[e >= 1]) < 0))
    # large and negative by five years
    out["gap_at_five"] = float(tau[e == 5][0])
    out["att"] = float(res.effects.att)

    # --- read the average, not a county -------------------------------
    five = np.array([float(np.asarray(u.tau)[np.asarray(u.horizons, int) == 5][0])
                     for u in res.per_unit.values()])
    gamma = np.array([u.agg_weight for u in res.per_unit.values()])
    out["county_dispersion_at_five"] = float(five.std(ddof=1))
    # The aggregate is exactly the weighted mean of the parts, which is what
    # makes the dispersion above a statement about the parts and not about a
    # discrepancy in the aggregation.
    out["aggregate_is_the_weighted_mean"] = float(
        abs(float(five @ gamma) - float(tau[e == 5][0])))

    # --- the specification choices change the estimand -----------------
    # fn 28 reports that equal weighting gives the same pattern with different
    # magnitudes, so the pattern is asserted and the magnitude is only reported.
    eq = _fit(df, agg_weights=None)
    _, tau_eq = _path(eq)
    out["equal_weighted_gap_at_five"] = float(tau_eq[e == 5][0])
    out["equal_weighting_keeps_the_sign"] = float(
        np.all(np.sign(tau_eq[e >= 2]) == np.sign(tau[e >= 2])))

    # The covariate specification as expressible today: the missing outcome
    # lags show up as a pre-treatment fit an order of magnitude worse, which is
    # why the outcome path is what this case matches on.
    cov = _fit(df, covariates=COVARIATES, covariate_windows=COVARIATE_WINDOW)
    out["covariate_spec_pre_rmse"] = float(cov.fit_diagnostics.rmse_pre)
    out["covariate_spec_pre_rmse_ratio"] = float(
        cov.fit_diagnostics.rmse_pre / res.fit_diagnostics.rmse_pre)

    # A binding donor restriction forfeits the per-cohort batching, and the
    # result says so rather than reporting a speed it did not achieve.
    czone = df.groupby("cty_fips").czone.first().to_dict()
    cz = _fit(df, donor_predicate=lambda i, d: czone[i] != czone[d])
    out["restricted_pool_is_not_batched"] = float(not cz.design.batched)
    out["restricted_pool_gap_at_five"] = float(_path(cz)[1][e == 5][0])
    return out


#: (expected, absolute tolerance).
#:
#: Everything here is deterministic -- no resampling, no simulation, and the
#: simplex solve is seeded only in the trivial sense that FISTA starts from
#: uniform weights. So tolerances cover solver drift across BLAS builds, not
#: Monte-Carlo noise, and are correspondingly tight.
#:
#: The exception is `county_dispersion_at_five`. Individual county weights are
#: not identified on this design, so a different solver reaching the same fit
#: reports a different spread; #341 measured up to 6.7pp of per-county movement
#: against 0.03 on the aggregate. Its band is wide on purpose, and it is a
#: floor-not-a-value row: what would be a real regression is the dispersion
#: collapsing, which would mean the per-county fits had stopped varying.
EXPECTED = {
    # the design
    "n_treated": (566.0, 0.5),
    "n_donors": (39.0, 0.5),
    "n_cohorts": (6.0, 0.5),
    "n_horizons": (11.0, 0.5),

    # invariants: identities, so pinned at machine precision
    "base_period_gap": (0.0, 1e-12),
    "batched": (1.0, 0.5),
    "weights_sum_to_one": (0.0, 1e-6),
    "aggregate_is_the_weighted_mean": (0.0, 1e-9),

    # the paper's prose claims. "An excellent pre-treatment fit" is read as a
    # pre-treatment RMSE below a tenth of a percent, against post-treatment
    # effects an order of magnitude larger.
    "pre_rmse": (0.049, 0.01),
    "max_pre_gap": (0.073, 0.02),
    "gap_at_entry": (0.158, 0.05),
    "declines_monotonically_after_entry": (1.0, 0.5),
    "gap_at_five": (-0.904, 0.02),
    "att": (-0.352, 0.02),

    # read the average, not a county
    "county_dispersion_at_five": (14.1, 6.0),

    # specification choices
    "equal_weighted_gap_at_five": (-1.100, 0.05),
    "equal_weighting_keeps_the_sign": (1.0, 0.5),
    # 27x worse without the paper's outcome lags. The band is wide because the
    # point is the order of magnitude, not the figure.
    "covariate_spec_pre_rmse": (1.354, 0.10),
    "covariate_spec_pre_rmse_ratio": (27.5, 8.0),
    "restricted_pool_is_not_batched": (1.0, 0.5),
    "restricted_pool_gap_at_five": (-1.010, 0.05),
}
