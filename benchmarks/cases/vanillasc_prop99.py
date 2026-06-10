"""VanillaSC Path-A: the canonical Abadie-Diamond-Hainmueller Proposition 99 fit.

Path A (empirical, scenario: the authors' canonical study and predictor set).
Abadie, Diamond & Hainmueller (2010) introduced the modern synthetic control on
California's Proposition 99 tobacco-control program, matching the treated unit
on a specific predictor set and three lagged outcomes. This reproduces their
headline result -- Table 2's donor weights and the estimated effect path -- with
mlsynth's standard synthetic control (the MSCMT backend doing the nested
predictor-weight ``V`` optimisation, as in Abadie's ``Synth``).

The ADH predictor specification, reproduced exactly:

  * ln(personal income), retail cigarette price and percent aged 15-24, each
    averaged over 1980-1988;
  * beer consumption per capita, averaged over 1984-1988;
  * cigarette sales in 1975, 1980 and 1988 (three lagged outcomes).

All predictors ship in ``basedata/augmented_cali_long.csv``. The fit is
deterministic given the canonicalised ``V``, so the cells below are exact
re-runs.

Provenance: Abadie, Diamond & Hainmueller (2010), *"Synthetic Control Methods
for Comparative Case Studies,"* JASA 105(490), Table 2 and Figure 2.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "augmented_cali_long.csv")

# ADH 2010 Table 2 synthetic-California donor weights.
_REF_WEIGHTS = {
    "Utah": 0.334, "Nevada": 0.234, "Montana": 0.199,
    "Colorado": 0.164, "Connecticut": 0.069,
}
_LAGS = (1975, 1980, 1988)


def run() -> dict:
    from mlsynth import VanillaSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    for L in _LAGS:                                   # lagged-outcome predictors
        m = d[d.year == L].set_index("state")["cigsale"]
        d[f"cig{L}"] = d["state"].map(m)

    covs = ["loginc", "p_cig", "pct15-24", "pc_beer",
            "cig1975", "cig1980", "cig1988"]
    windows = {
        "loginc": (1980, 1988), "p_cig": (1980, 1988),
        "pct15-24": (1980, 1988), "pc_beer": (1984, 1988),
        "cig1975": (1975, 1975), "cig1980": (1980, 1980),
        "cig1988": (1988, 1988),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treated",
            "unitid": "state", "time": "year",
            "covariates": covs, "covariate_windows": windows,
            "backend": "mscmt", "canonical_v": "min.loss.w",
            "seed": 0, "display_graphs": False,
        }).fit()

    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    mv = np.array([w.get(s, 0.0) for s in _REF_WEIGHTS])
    rv = np.array(list(_REF_WEIGHTS.values()))

    years = sorted(d["year"].unique())
    gap = np.asarray(res.time_series.estimated_gap)

    return {
        "att_1989_2000": float(res.effects.att),
        "weight_max_abs_dev": float(np.max(np.abs(mv - rv))),
        "weight_utah": w.get("Utah", 0.0),
        "weight_nevada": w.get("Nevada", 0.0),
        "n_positive_donors": float(sum(1 for v in w.values() if v > 1e-3)),
        "gap_2000": float(gap[years.index(2000)]),
        "pre_rmspe": float(res.fit_diagnostics.rmse_pre),
    }


# Deterministic (canonicalised V, fixed seed) => exact re-runs. The cells
# reproduce ADH 2010 value-for-value: the five-state donor pool (Utah, Nevada,
# Montana, Colorado, Connecticut) matches Table 2 to ~0.003, the average effect
# is ~-19 packs, the gap reaches ~-26 by 2000, and the pre-period RMSPE is ~1.75.
EXPECTED = {
    "att_1989_2000": (-18.98, 0.6),            # ADH ~ -19
    "weight_max_abs_dev": (0.004, 0.015),      # vs Table 2; fails above ~0.019
    "weight_utah": (0.335, 0.03),              # ADH 0.334
    "weight_nevada": (0.236, 0.03),            # ADH 0.234
    "n_positive_donors": (5.0, 1.0),           # ADH's five-state pool
    "gap_2000": (-25.73, 2.0),                 # ADH ~ -26
    "pre_rmspe": (1.75, 0.3),
}
