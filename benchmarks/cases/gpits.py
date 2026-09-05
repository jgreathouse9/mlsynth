"""GPITS Path-A: Gaussian-process ITS on the Heller decision.

Path A (empirical, scenario: the paper's own data and application). Cho (2026)
estimates the effect of the 2008 Supreme Court decision in *District of Columbia
v. Heller* on legal handgun purchases. The ruling bound every U.S. jurisdiction
at once, so no donor pool exists and the counterfactual is extrapolated from
D.C.'s own pre-treatment history by a Gaussian process.

The paper reports a cumulative four-month effect of 15.1 background checks per
100,000 population with a 95% interval of [13.0, 17.3] (Section 6). This case
pins that, the two hyperparameters the fit turns on, and the placebo checks.

The fit is deterministic -- the hyperparameters come from a bounded scalar search
and a marginal-likelihood fit, with no RNG anywhere -- so the cells below are
exact re-runs. They were additionally cross-validated cell-for-cell against the
author's R package ``gpss`` (agreement to ~1e-11 on every quantity); that harness
lives in ``benchmarks/reference/gpits_heller/``.

Provenance: Cho (2026), arXiv:2608.20610, Section 6. Panel:
``basedata/dc_handgun_heller.csv``, the D.C. rows of the FBI NICS monthly series
divided by Census population and scaled per 100,000, 2002-07 to 2008-10.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "dc_handgun_heller.csv")


def run() -> dict:
    from mlsynth import GPITS

    d = pd.read_csv(os.path.abspath(_DATA), parse_dates=["date"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GPITS({
            "df": d, "outcome": "handgun_rate", "treat": "treated",
            "unitid": "unit", "time": "date",
            "covariates": ["month"], "categorical_covariates": ["month"],
            "kernel": "gaussian_periodic_linear", "period": 12,
            "placebo_periods": 4, "display_graphs": False,
        }).fit()

    lo, hi = res.cumulative_ci[-1]
    return {
        "cumulative_4m": float(res.cumulative_effect[-1]),
        "cumulative_ci_lower": float(lo),
        "cumulative_ci_upper": float(hi),
        "length_scale": float(res.design.length_scale),
        "noise_variance": float(res.design.noise_variance),
        "placebo_all_cover": float(res.placebo.all_cover),
        "max_abs_placebo_tau": float(np.max(np.abs(res.placebo.tau))),
    }


# Deterministic (bounded scalar search + marginal likelihood, no RNG) => exact
# re-runs. The paper reports 15.1 [13.0, 17.3]; the tolerances below are the
# rounding the paper's own precision implies, not slack. The hyperparameters are
# pinned tightly because they are what the cross-validation against gpss
# established, and a drift in either would move the headline.
EXPECTED = {
    "cumulative_4m": (15.1323, 0.001),
    "cumulative_ci_lower": (12.9687, 0.001),
    "cumulative_ci_upper": (17.2960, 0.001),
    "length_scale": (2.7367, 0.001),
    "noise_variance": (0.8488, 0.001),
    "placebo_all_cover": (1.0, 0.0),
    "max_abs_placebo_tau": (0.4950, 0.01),
}
