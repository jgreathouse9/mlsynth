"""MULTICELLGEOLIFT cross-validation vs augsynth (per-cell, donor-excluded).

Cross-validation against the engine GeoLiftMultiCell wraps. GeoLiftMultiCell
measures each cell against the shared control pool, **excluding the other cells'
markets from the donor pool** (``filter(!location %in% other_cells)``). This pins
mlsynth's per-cell ATT against augsynth run the same way on the GeoLift_Test
panel: cell A = {chicago, portland} (the real effect), cell B = {atlanta, boston}
(a placebo cell), the rest shared controls, treatment over days 91-105.

The ATT is the deterministic fixed-effect ASCM estimate, so it matches augsynth
to the decimal; the donor-exclusion invariant (cell A's synthetic control never
uses cell B's markets, and vice versa) is asserted directly. (The live
GeoLiftMultiCell summary path is blocked by augsynth 0.2.0's multi-unit-cohort
``treated_table`` bug, so we cross-check against augsynth's own per-cell fit.)
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from mlsynth import MULTICELLGEOLIFT

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_test_data.csv")
_CELLS = {"chicago": "A", "portland": "A", "atlanta": "B", "boston": "B"}


def run() -> dict:
    df = pd.read_csv(os.path.abspath(_DATA))
    df["cell"] = df["location"].map(_CELLS).fillna("")        # blank = control
    dates = sorted(df["date"].unique())
    df["post"] = df["date"].isin(set(dates[90:])).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MULTICELLGEOLIFT({
            "df": df, "outcome": "Y", "unitid": "location", "time": "date",
            "cell_column_name": "cell", "post_col": "post", "how": "mean",
            "fixed_effects": True, "ns": 1000, "seed": 0, "display_graphs": False,
        }).fit()

    a_donors = set(res.cells["A"].weights.donor_weights)
    b_donors = set(res.cells["B"].weights.donor_weights)
    return {
        "att_A": float(res.cells["A"].effects.att),           # augsynth 156.837
        "att_B": float(res.cells["B"].effects.att),           # augsynth 119.383
        # donor-exclusion: A never uses B's markets, B never uses A's
        "donor_exclusion": float(
            a_donors.isdisjoint({"atlanta", "boston"})
            and b_donors.isdisjoint({"chicago", "portland"})),
    }


# ATT is the deterministic fixed-effect ASCM estimate -> matches augsynth exactly.
EXPECTED = {
    "att_A": (156.837, 0.05),     # augsynth {chicago, portland} excl. {atlanta, boston}
    "att_B": (119.383, 0.05),     # augsynth {atlanta, boston} excl. {chicago, portland}
    "donor_exclusion": (1.0, 0.5),
}
