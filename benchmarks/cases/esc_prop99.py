"""ESC empirical validation -- California (Proposition 99).

Path A (empirical). Ensemble Synthetic Control (Lian & Pu 2026) on the canonical
Prop 99 cigarette panel (California + 38 donors, treatment 1989, T0=19). ESC's
ensemble counterfactual should reproduce the classic convex synthetic-control
path while its prediction interval stays informative under the panel.

The authors' ``esc`` package is not yet released, so this pins ESC's own
reproducible (seeded) outputs against the established Prop 99 result: the
counterfactual correlates ~0.998 with the convex synthetic California, the ATT
matches Abadie et al.'s ~-19 packs, and the 90% band places observed California
below its lower bound throughout the post-period. Data ship as
``basedata/california_panel.csv``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "california_panel.csv")
_SEED = 20260201


def run() -> dict:
    from mlsynth import ESC, VanillaSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    years = np.array(sorted(d.year.unique()))
    post = years >= 1989

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ESC({"df": d, "outcome": "cigsale", "treat": "treat", "unitid": "state",
                 "time": "year", "n_learners": 200, "block_length": 3,
                 "subspace_ratio": 0.5, "alpha": 0.10, "seed": _SEED,
                 "display_graphs": False}).fit()
        rc = VanillaSC({"df": d, "outcome": "cigsale", "treat": "treat",
                        "unitid": "state", "time": "year", "backend": "outcome-only",
                        "inference": False, "display_graphs": False}).fit()

    ts = r.time_series
    obs = np.asarray(ts.observed_outcome, float)
    cf = np.asarray(ts.counterfactual_outcome, float)
    cf_sc = np.asarray(rc.time_series.counterfactual_outcome, float)
    lo = np.asarray(ts.counterfactual_lower, float)
    gap = obs - cf
    return {
        "avg_post_gap": float(np.mean(gap[post])),
        "gap_2000": float(gap[years == 2000][0]),
        "pre_rmspe": float(np.sqrt(np.mean(gap[~post] ** 2))),
        "corr_with_classic_sc": float(np.corrcoef(cf, cf_sc)[0, 1]),
        "obs_below_lower_frac": float(np.mean(obs[post] < lo[post])),
    }


# Deterministic (the perturbation draws are seeded). ESC reproduces the classic
# Prop 99 result and its band separates the effect throughout the post-period.
EXPECTED = {
    "avg_post_gap": (-22.73, 4.0),
    "gap_2000": (-33.55, 6.0),
    "pre_rmspe": (1.64, 0.5),
    "corr_with_classic_sc": (0.997, 0.02),
    "obs_below_lower_frac": (1.0, 0.2),
}


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:24s} {v:.4f}")
