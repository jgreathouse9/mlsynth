"""ESC empirical validation -- Sao Paulo homicides (short pre-window).

Path A (empirical). The Sao Paulo homicide study (Freire 2018; treated 1999,
T0=9 annual pre-periods, 24 donor states) is the short-panel regime ESC targets.
The deliverable is the honesty of the interval: ESC reports a moderate central
effect with a wide prediction interval on lives saved that reflects the genuine
model uncertainty a single specification hides. Data ship as
``basedata/saopaulo_homicides.csv`` (code, state, year, homicide rate,
population projection).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "saopaulo_homicides.csv")
_FREIRE_DONORS = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 31, 32,
                  33, 41, 42, 43, 50, 51, 52, 53]
_SEED = 19990101


def run() -> dict:
    from mlsynth import ESC

    raw = pd.read_csv(os.path.abspath(_DATA))
    d = raw[raw.code.isin([35] + _FREIRE_DONORS)].copy()
    d["treat"] = ((d.code == 35) & (d.year >= 1999)).astype(int)
    years = np.array(sorted(d.year.unique()))
    post = years >= 1999
    pop = raw[raw.code == 35].set_index("year")["population.projection"].reindex(years).to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ESC({"df": d, "outcome": "homicide.rates", "treat": "treat",
                 "unitid": "code", "time": "year", "n_learners": 300,
                 "block_length": 2, "subspace_ratio": 0.5, "alpha": 0.10,
                 "seed": _SEED, "display_graphs": False}).fit()

    ts = r.time_series
    obs = np.asarray(ts.observed_outcome, float)
    cf = np.asarray(ts.counterfactual_outcome, float)
    lo = np.asarray(ts.counterfactual_lower, float)
    gap = obs - cf
    lives = float(np.sum((cf[post] - obs[post]) / 1e5 * pop[post]))
    # lives-saved interval from the ensemble of base-learner counterfactuals
    ens = np.asarray(r.fit.ensemble, float)                       # (B, T)
    mem_lives = np.array([np.sum((ens[b][post] - obs[post]) / 1e5 * pop[post])
                          for b in range(ens.shape[0])])
    return {
        "att": float(r.effects.att),
        "gap_2009": float(gap[years == 2009][0]),
        "pre_rmspe": float(r.fit_diagnostics.rmse_pre),
        "lives_saved": lives,
        "lives_pi_lo": float(np.nanquantile(mem_lives, 0.05)),
        "lives_pi_hi": float(np.nanquantile(mem_lives, 0.95)),
        "n_below_lower_pi": float(np.sum(obs[post] < lo[post])),
    }


# Deterministic (seeded). Central effect is moderate with a wide, honest interval;
# the observed series clears the lower band only in the later post-treatment years.
EXPECTED = {
    "att": (-6.61, 3.0),
    "gap_2009": (-19.27, 5.0),
    "pre_rmspe": (1.28, 0.5),
    "lives_saved": (30664.0, 12000.0),
    "n_below_lower_pi": (6.0, 4.0),
}


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:20s} {v:,.4f}")
