"""TBRMM cross-validation: the hill climb against google/matched_markets.

Cross-validation against the reference implementation, on the GeoLift panel the
repository already ships. The reference is Apache 2.0 and is not vendored: its
output was captured once with ``benchmarks/studies/tbrmm_match`` and is pinned
below, so this case runs with no external dependency.

What is pinned is stronger than a matching score. A design search returns a list
of geo names, and two searches can agree on a number while having walked
different paths, so the case pins membership: for each treatment size, whether
the treatment group and the control group are the sets the reference chose, geo
for geo. All four sizes match, together with the four assumption gates, the
rounded correlation and the inverse detectable impact.

Why membership and not the score alone. TBR has no weight vector, so a geo's
weight is its membership and the design is the estimate; a search that reached a
different partition with a similar score is a different experiment. The score
agreeing is necessary and not sufficient.

The reference's parameters here are ``n_test=14``, ``treatment_geos_range=(1,4)``
and ``n_pretest_max=90``, the last of which truncates the 105-period panel inside
its constructor, so the case scores the same final 90 periods.

One defect surfaced only here. The A/A test was implemented as a scan over every
window of cumulative residuals, where the reference holds out the last ``n_test``
pretest periods, refits, and asks whether that one window's interval excludes
zero. The scan is far stricter and failed almost every split, which cost the
search the geo the reference takes at k = 2 and carried the two walks apart from
there. No unit test caught it: the gate returned a bool, and a search's output is
a plausible list of cities either way.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from mlsynth import TBR
from mlsynth.config_models import TBRConfig

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_test_data.csv")

N_TEST, K_MAX, N_PRETEST = 14, 4, 90

#: google/matched_markets' ``greedy_search`` output, captured once. Each entry is
#: the treatment group, the control group and the score
#: ``(corr_test, aa_test, bb_test, dw_test, corr, inv_required_impact)``.
REFERENCE_DESIGNS: Dict[int, Tuple[List[str], List[str], Tuple[float, ...]]] = {
    1: (["dallas"],
        ["baltimore", "boston", "chicago", "cleveland", "denver", "houston",
         "kansas city", "los angeles", "memphis", "orlando", "philadelphia",
         "reno", "salt lake city", "san francisco", "washington"],
        (1.0, 1.0, 1.0, 1.0, 0.99, 0.0009987278043297272)),
    2: (["dallas", "denver"],
        ["baltimore", "baton rouge", "boston", "chicago", "cleveland", "houston",
         "kansas city", "los angeles", "memphis", "orlando", "reno",
         "san francisco", "washington"],
        (1.0, 1.0, 1.0, 1.0, 0.99, 0.0004502559979805353)),
    3: (["cincinnati", "dallas", "denver"],
        ["atlanta", "baltimore", "baton rouge", "chicago", "cleveland", "houston",
         "jacksonville", "kansas city", "los angeles", "memphis", "minneapolis",
         "nashville", "new york", "orlando", "philadelphia", "reno", "san diego",
         "san francisco", "washington"],
        (1.0, 1.0, 1.0, 1.0, 0.99, 0.0004328635232407686)),
    4: (["chicago", "cincinnati", "dallas", "denver"],
        ["atlanta", "baltimore", "baton rouge", "cleveland", "houston",
         "kansas city", "memphis", "minneapolis", "nashville", "new york",
         "portland", "reno", "san diego", "washington"],
        (1.0, 1.0, 1.0, 1.0, 0.99, 0.00036863374362752256)),
}


def _scoring_window() -> pd.DataFrame:
    """The panel truncated the way the reference truncates it."""
    panel = pd.read_csv(os.path.abspath(_DATA)).rename(columns={"location": "geo"})
    dates = sorted(panel["date"].unique())[-N_PRETEST:]
    return panel[panel["date"].isin(dates)]


def run() -> dict:
    result = TBR(TBRConfig(
        df=_scoring_window(), unitid="geo", time="date", outcome="Y",
        max_treatment_size=K_MAX, n_test=N_TEST)).fit()
    by_size = {d.k: d for d in result.designs}

    out: dict = {"n_designs": float(len(result.designs))}
    for k, (treatment, control, key) in REFERENCE_DESIGNS.items():
        design = by_size[k]
        detail = design.detail
        out[f"k{k}_treatment_matches"] = float(sorted(design.treatment_units) == treatment)
        out[f"k{k}_control_matches"] = float(sorted(design.control_units) == control)
        out[f"k{k}_corr_test"] = float(detail["corr_test"])
        out[f"k{k}_aa_test"] = float(detail["aa_test"])
        out[f"k{k}_bb_test"] = float(detail["bb_test"])
        out[f"k{k}_dw_test"] = float(detail["dw_test"])
        out[f"k{k}_corr"] = round(float(detail["corr"]), 2)
        out[f"k{k}_inv_impact"] = float(design.objective_value)
    return out


def _expected() -> dict:
    """The pins, built from the captured reference output.

    The power term is pinned to a relative 1e-9 of the reference's value, which
    the measured agreement (9.3e-15 at its worst) clears by six orders of
    magnitude. Membership, the gates and the rounded correlation are exact.
    """
    pins = {"n_designs": (float(len(REFERENCE_DESIGNS)), 0.0)}
    for k, (_, _, key) in REFERENCE_DESIGNS.items():
        pins[f"k{k}_treatment_matches"] = (1.0, 0.0)
        pins[f"k{k}_control_matches"] = (1.0, 0.0)
        pins[f"k{k}_corr_test"] = (key[0], 0.0)
        pins[f"k{k}_aa_test"] = (key[1], 0.0)
        pins[f"k{k}_bb_test"] = (key[2], 0.0)
        pins[f"k{k}_dw_test"] = (key[3], 0.0)
        pins[f"k{k}_corr"] = (key[4], 0.0)
        pins[f"k{k}_inv_impact"] = (key[5], abs(key[5]) * 1e-9)
    return pins


EXPECTED = _expected()
