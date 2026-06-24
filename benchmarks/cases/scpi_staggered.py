"""Cross-validation benchmark: staggered VanillaSC vs the ``scpi`` package.

Path A (empirical). Reproduces Cattaneo, Feng, Palomba and Titiunik's (2025)
multiple-treated-unit synthetic control on the canonical Germany reunification
panel (``basedata/scpi_germany.csv``), driven entirely through the public
``VanillaSC.fit()``.

Two treated units adopt at different times -- West Germany in 1991 and Italy in
1992 (the placebo unit from the package's own illustration) -- with the 15
never-treated countries as donors. Run outcome-only with a simplex constraint,
``scpi``'s ``scest`` produces the per-cell synthetic-control predictions; the
per-unit, event-time, and overall *causal predictand* point estimates derived
from them are reproduced by ``mlsynth`` to a few thousandths.

The ``scpi`` reference is hard-coded below rather than computed live: ``scpi`` is
GPL-licensed and ``mlsynth`` is MIT, so the benchmark records ``scpi``'s ``scest``
numbers once and checks ``fit()`` against them, with no run-time dependency on
``scpi_pkg``. The point estimates are deterministic SC quantities, so they match
to the published digits.

Reference: ``scpi_pkg`` (PyPI), ``scest`` / ``scdataMulti``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "scpi_germany.csv")
_ADOPT = {"West Germany": 1991, "Italy": 1992}

# scpi reference (scest, outcome-only simplex): per-unit average effects, the
# overall unit-time ATT, and the event-time (effect="time") series.
_SCPI_ATT = {"West Germany": -1.8476, "Italy": -1.1211, "overall": -1.4989}
_SCPI_EVENT_TIME = np.array([
    0.131, 0.0316, -0.4255, -0.67, -0.9013, -1.2304,
    -1.2529, -1.6015, -2.2758, -2.6734, -2.8457, -3.0832])


def _panel() -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(_DATA))
    df["status"] = 0
    for unit, yr in _ADOPT.items():
        df.loc[(df["country"] == unit) & (df["year"] >= yr), "status"] = 1
    return df


def run() -> dict:
    from mlsynth import VanillaSC

    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": df, "outcome": "gdp", "treat": "status",
                         "unitid": "country", "time": "year",
                         "display_graphs": False}).fit()
    ml_unit = {f.treated_unit_name: float(f.att)
               for f in res.sub_method_results.values()}
    ml_event = res.additional_outputs["event_study"]            # {ell: effect}
    ml_overall = float(res.effects.att)

    ev = np.array([ml_event[k] for k in sorted(ml_event)])
    n = min(len(ev), len(_SCPI_EVENT_TIME))

    return {
        "att_west_germany": ml_unit["West Germany"],
        "att_italy": ml_unit["Italy"],
        "att_overall": ml_overall,
        # agreement with the recorded scpi reference (driven through fit())
        "per_unit_max_abs_diff_vs_scpi": max(
            abs(ml_unit[u] - _SCPI_ATT[u]) for u in _ADOPT),
        "event_study_max_abs_diff_vs_scpi": float(
            np.max(np.abs(ev[:n] - _SCPI_EVENT_TIME[:n]))),
        "overall_abs_diff_vs_scpi": abs(ml_overall - _SCPI_ATT["overall"]),
    }


def comparison() -> dict:
    """mlsynth vs the recorded ``scpi`` scest reference, quantity by quantity.

    The mlsynth side comes from a fresh ``VanillaSC.fit()``; the reference side is
    the ``scpi`` scest numbers captured in ``_SCPI_ATT`` / ``_SCPI_EVENT_TIME``
    (recorded once, since ``scpi`` is GPL and ``mlsynth`` MIT). Pairs the per-unit
    ATTs, the overall unit-time ATT, and the event-time series. Returns the
    ``{"rows": [...], "mlsynth_call": ..., "reference": ...}`` exporter contract.
    """
    from mlsynth import VanillaSC

    df = _panel()
    cfg = {"outcome": "gdp", "treat": "status", "unitid": "country",
           "time": "year"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({**cfg, "df": df, "display_graphs": False}).fit()
    ml_unit = {f.treated_unit_name: float(f.att)
               for f in res.sub_method_results.values()}
    ml_event = res.additional_outputs["event_study"]            # {ell: effect}
    ml_overall = float(res.effects.att)
    ev = np.array([ml_event[k] for k in sorted(ml_event)])
    n = min(len(ev), len(_SCPI_EVENT_TIME))

    rows = [
        {"quantity": "ATT[West Germany]",
         "mlsynth": round(ml_unit["West Germany"], 6),
         "reference": round(_SCPI_ATT["West Germany"], 6)},
        {"quantity": "ATT[Italy]", "mlsynth": round(ml_unit["Italy"], 6),
         "reference": round(_SCPI_ATT["Italy"], 6)},
        {"quantity": "ATT[overall]", "mlsynth": round(ml_overall, 6),
         "reference": round(_SCPI_ATT["overall"], 6)},
    ]
    rows += [{"quantity": f"event_time[ell={i}]",
              "mlsynth": round(float(ev[i]), 6),
              "reference": round(float(_SCPI_EVENT_TIME[i]), 6)}
             for i in range(n)]

    try:
        from scpi_pkg import __version__ as _scpi_ver
        version = f"scpi_pkg {_scpi_ver}"
    except Exception:                                           # noqa: BLE001
        version = "scpi_pkg (pip)"
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
        "reference": {"impl": "Python package scpi_pkg", "version": version},
    }


# Deterministic SC point estimates: mlsynth's fit() reproduces scpi's recorded
# scest numbers across the per-unit, event-time, and overall predictands.
EXPECTED = {
    "att_west_germany": (-1.8476, 0.01),     # scpi scest, effect="unit"
    "att_italy": (-1.1211, 0.01),
    "att_overall": (-1.4989, 0.01),          # effect="unit-time"
    "per_unit_max_abs_diff_vs_scpi": (0.0, 0.01),
    "event_study_max_abs_diff_vs_scpi": (0.0, 0.01),   # effect="time" series
    "overall_abs_diff_vs_scpi": (0.0, 0.01),
}
