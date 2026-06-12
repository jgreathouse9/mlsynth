"""GEOLIFT cross-validation vs GeoLift/augsynth: the GeoLift_Walkthrough.

Cross-validation (scenario: match an authoritative reference implementation).
Meta's GeoLift package walkthrough runs Ben-Michael, Feller & Rothstein's
Augmented SCM through augsynth with ``fixed_effects=TRUE`` (the package default)
and Chernozhukov-Wuthrich-Zhu conformal inference. The vignette's public call is

    GeoLift_Test <- GeoLift(Y_id = "Y", data = GeoTestData_Test,
                            locations = c("chicago", "portland"),
                            treatment_start_time = 91, treatment_end_time = 105)
    summary(GeoLift_Test)

and ``summary`` reports, for the ``chicago`` + ``portland`` test markets over the
last 15 of 105 days (40 markets, the other 38 as donors):

* Average ATT (per treated unit, per period): ``155.556``
* Percent Lift: ``5.4%``
* Incremental Y (summed over both units, 15 periods): ``4667``
* Conformal p-value: ``0.01``

This case drives mlsynth's **public** estimator -- ``GEOLIFT(...).fit()`` -- to the
same scenario (the two markets forced via ``to_be_treated`` + ``treatment_size``,
the post window marked by ``post_col``), exactly as a user would, and checks the
realized report (``res.report``, the analogue of ``summary(GeoLift_Test)``)
against those published numbers. See ``docs/replications/geolift.rst``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from mlsynth import GEOLIFT

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_test_data.csv")
_TREATED = ["chicago", "portland"]
_PRE = 90
_NS = 2000


def run() -> dict:
    df = pd.read_csv(os.path.abspath(_DATA))
    # Mark the 15 post-treatment periods (days 91-105), as the walkthrough does
    # with treatment_start_time = 91.
    dates = sorted(df["date"].unique())
    df["post"] = df["date"].isin(set(dates[_PRE:])).astype(int)

    config = {
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": len(_TREATED), "to_be_treated": _TREATED,
        "durations": [len(dates) - _PRE], "effect_sizes": [0.0, 0.10],
        "lookback_window": 1, "post_col": "post",
        "how": "mean", "augment": "ridge", "fixed_effects": True,
        "power_threshold": 0.8, "alpha": 0.1, "ns": _NS, "seed": 0,
        "conformal_type": "iid", "display_graphs": False,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GEOLIFT(config).fit()

    rep = res.report
    gap_post = np.asarray(rep.time_series.estimated_gap, dtype=float)[_PRE:]
    cf_post = float(np.mean(rep.time_series.counterfactual_outcome[_PRE:]))
    att_per_unit = float(rep.effects.att)                 # per-unit (how="mean")
    return {
        # the design forces the walkthrough's two markets
        "selected_chicago_portland": float(
            set(res.selected_units) == set(_TREATED)),
        "att_per_unit": att_per_unit,                     # GeoLift 155.556
        "pct_lift": 100.0 * att_per_unit / cf_post,       # GeoLift 5.4
        "incremental": float(np.sum(gap_post)) * len(_TREATED),  # GeoLift 4667
        "conformal_p": float(rep.inference.p_value),      # GeoLift 0.01
        "significant": float(rep.inference.p_value < 0.05),
    }


# Deterministic (fixed CV lambda, fixed seed/ns). The augsynth/GeoLift target is
# in the comment; tolerances accept the small numerical gap (mlsynth hits
# 156.8 / 5.47% / 4704 / 0.011) while pinning the value-for-value match.
EXPECTED = {
    "selected_chicago_portland": (1.0, 0.5),
    "att_per_unit": (155.556, 5.0),     # GeoLift 155.556
    "pct_lift": (5.4, 0.5),             # GeoLift 5.4%
    "incremental": (4667.0, 150.0),     # GeoLift 4667
    "conformal_p": (0.01, 0.02),        # GeoLift 0.01 (deterministic at seed=0)
    "significant": (1.0, 0.5),
}
