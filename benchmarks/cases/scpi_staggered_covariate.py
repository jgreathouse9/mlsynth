"""Cross-validation benchmark: covariate staggered VanillaSC vs the ``scpi``
multiple-treated-unit illustration.

Path A (empirical). Reproduces the covariate (multi-feature) staggered synthetic
control of Cattaneo, Feng, Palomba & Titiunik (2025) -- their ``scpi``
multiple-treated illustration (``scpi_illustration-multi.py``) -- on the Germany
reunification panel, driven through the public ``VanillaSC.fit()`` with a
``staggered_spec``. The shared specification matches on GDP and trade with a
constant-and-trend covariate adjustment and cointegrated differencing; the
treated units (West Germany 1991, Italy 1992) come from the treatment indicator,
never named in the spec.

The ``EXPECTED`` values are the ``scpi`` numbers, pinned as constants rather than
computed live: ``scpi``'s multi-feature design produces duplicate donor-column
names that current ``scikit-learn`` / ``narwhals`` reject, so ``scpi_pkg`` cannot
compute these prediction intervals in a modern environment. The per-unit average
effects are deterministic ``scest`` quantities reproduced exactly; the
event-time prediction-interval widths were generated against ``scpi`` (with a
one-line column-name coercion) at ``seed=8894``, ``sims=500``, and the engine
reproduces ``scpi``'s draw sequence, so they are stable. This case therefore
needs no ``scpi_pkg`` at run time -- it pins the published reference and checks
``fit()`` against it.

Reference: ``scpi_pkg`` (PyPI), ``scpi_illustration-multi.py``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "scpi_germany.csv")
_ADOPT = {"West Germany": 1991, "Italy": 1992}
_SPEC = {"features": ["gdp", "trade"], "cov_adj": ["constant", "trend"],
         "constant": True, "cointegrated": True}


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
        res = VanillaSC({
            "df": df, "outcome": "gdp", "treat": "status", "unitid": "country",
            "time": "year", "display_graphs": False, "inference": "scpi",
            "scpi_sims": 500, "seed": 8894, "scpi_compat": True,
            "staggered_spec": _SPEC,
        }).fit()

    pu = res.additional_outputs["per_unit_att"]
    es = res.additional_outputs["event_study"]
    esi = res.additional_outputs["event_study_intervals"]
    w1 = esi[1]["synthetic_ci"][1] - esi[1]["synthetic_ci"][0]

    return {
        "per_unit_att_italy": float(pu["Italy"]),
        "per_unit_att_west_germany": float(pu["West Germany"]),
        "overall_att": float(res.effects.att),
        "event_study_ell1": float(es[1]),
        "event1_synthetic_ci_width": float(w1),
    }


# scpi reference: per-unit average effects are the deterministic scest values;
# the overall ATT, event-time effect and prediction-interval width are the
# scpi-reproduced numbers at seed=8894, sims=500 (see module docstring).
EXPECTED = {
    "per_unit_att_italy": (-0.8902, 0.005),          # scpi scest
    "per_unit_att_west_germany": (-1.7467, 0.005),   # scpi scest
    "overall_att": (-1.3356, 0.01),
    "event_study_ell1": (0.2831, 0.01),
    "event1_synthetic_ci_width": (2.8624, 0.1),      # scpi CI_all_gaussian width
}
