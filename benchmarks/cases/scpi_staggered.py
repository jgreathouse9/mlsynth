"""Cross-validation benchmark: staggered VanillaSC vs the ``scpi`` package.

Path A (empirical, runnable Python reference). Reproduces Cattaneo, Feng, Palomba
and Titiunik's (2025) multiple-treated-unit synthetic control on the canonical
Germany reunification panel (the public ``scpi_germany.csv`` shipped with the
``scpi`` distribution), driven entirely through the public ``VanillaSC.fit()``.

Two treated units adopt at different times -- West Germany in 1991 and Italy in
1992 (the placebo unit from the package's own illustration) -- with the 15
never-treated countries as donors. Run outcome-only with a simplex constraint,
``scpi``'s ``scest`` produces the per-cell synthetic-control predictions; the
per-unit, event-time, and overall *causal predictand* point estimates derived
from them are reproduced by ``mlsynth`` to a few thousandths.

Reference: ``scpi_pkg`` (PyPI, pinned at 4.0.0), ``scest`` /  ``scdataMulti``.
The point estimates are deterministic SC quantities, so they match to the
published digits. NOTE: this case validates the *point-estimate* family
(``effect="unit"`` / ``"time"`` / ``"unit-time"``). The cross-unit prediction
*intervals* (TSUA / TAUA) are not yet reproduced and are out of scope here.

Skips itself (``BenchmarkSkipped``) when ``scpi_pkg`` is absent, so it runs under
``--all`` without a hard dependency.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "scpi_germany.csv")
_ADOPT = {"West Germany": 1991, "Italy": 1992}


def _panel() -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(_DATA))
    df["status"] = 0
    for unit, yr in _ADOPT.items():
        df.loc[(df["country"] == unit) & (df["year"] >= yr), "status"] = 1
    return df


def _scpi_gaps(df: pd.DataFrame) -> pd.Series:
    """Per-cell post-treatment gaps from scpi's scest (effect='unit-time')."""
    from scpi_pkg.scdataMulti import scdataMulti
    from scpi_pkg.scest import scest
    aux = scdataMulti(df=df, id_var="country", time_var="year",
                      outcome_var="gdp", treatment_var="status", features=None,
                      cov_adj=None, constant=False, cointegrated_data=False,
                      effect="unit-time")
    res = scest(aux, w_constr={"name": "simplex"})
    gap = np.asarray(res.Y_post).ravel() - np.asarray(res.Y_post_fit).ravel()
    return pd.Series(gap, index=res.Y_post.index)   # index: (unit, year)


def _event_study(gaps_by_unit: dict, min_post: int) -> np.ndarray:
    """Balanced event-study series: mean gap across units at each event time."""
    return np.array([
        np.mean([gaps_by_unit[u][ell] for u in gaps_by_unit])
        for ell in range(min_post)
    ])


def run() -> dict:
    try:
        import scpi_pkg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional reference dep
        raise BenchmarkSkipped("scpi_pkg not installed "
                               "(`pip install scpi_pkg==4.0.0`)") from exc

    from mlsynth import VanillaSC

    df = _panel()

    # --- mlsynth, through the public fit() ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": df, "outcome": "gdp", "treat": "status",
                         "unitid": "country", "time": "year",
                         "display_graphs": False}).fit()
    ml_unit = {f.treated_unit_name: float(f.att)
               for f in res.sub_method_results.values()}
    ml_event = res.additional_outputs["event_study"]            # {ell: effect}
    ml_overall = float(res.effects.att)

    # --- scpi reference ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaps = _scpi_gaps(df)
    sc_by_unit = {}
    sc_unit = {}
    for unit, yr in _ADOPT.items():
        g = gaps.xs(unit, level=0).sort_index()
        sc_by_unit[unit] = g.to_numpy(float)
        sc_unit[unit] = float(g.mean())
    sc_overall = float(gaps.mean())
    min_post = min(len(v) for v in sc_by_unit.values())
    sc_event = _event_study(sc_by_unit, min_post)
    ml_event_arr = np.array([ml_event[k] for k in sorted(ml_event)])[:min_post]

    return {
        # mlsynth's own values (pinned to the published SC numbers)
        "att_west_germany": ml_unit["West Germany"],
        "att_italy": ml_unit["Italy"],
        "att_overall": ml_overall,
        # cell-by-cell agreement with scpi (driven through fit())
        "per_unit_max_abs_diff_vs_scpi": max(
            abs(ml_unit[u] - sc_unit[u]) for u in _ADOPT),
        "event_study_max_abs_diff_vs_scpi": float(
            np.max(np.abs(ml_event_arr - sc_event))),
        "overall_abs_diff_vs_scpi": abs(ml_overall - sc_overall),
    }


# Deterministic SC point estimates: mlsynth's fit() reproduces scpi's scest to a
# few thousandths across the per-unit, event-time, and overall predictands.
EXPECTED = {
    "att_west_germany": (-1.8476, 0.01),     # scpi scest, effect="unit"
    "att_italy": (-1.1211, 0.01),
    "att_overall": (-1.4989, 0.01),          # effect="unit-time"
    "per_unit_max_abs_diff_vs_scpi": (0.0, 0.01),
    "event_study_max_abs_diff_vs_scpi": (0.0, 0.01),   # effect="time" series
    "overall_abs_diff_vs_scpi": (0.0, 0.01),
}
