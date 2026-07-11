"""BFSC Path-A: West Germany reunification (Pinkney 2021).

Cross-validation + Path A. Pinkney (2021) ships the model as an appendix Stan
program; mlsynth's BFSC ports it to NumPyro. On the German reunification panel
the NumPyro posterior matches the author's Stan cell-for-cell (counterfactual
corr 0.999999, sigma to 4 decimals -- see docs/replications/bfsc). This case
pins the reproducible headline quantities of a fresh NumPyro fit: the classical
negative reunification effect (~-1600 PPP-USD per capita), a positive-then-
negative gap path, a close pre-period fit, and converged NUTS.

Bayesian (NUTS) => the cells carry MCMC tolerances; the seed is fixed so a run
is reproducible. Requires the ``[bayes]`` optional dependency (NumPyro).

Provenance: Pinkney (2021), arXiv:2103.16244, Section 2.1 + appendix Stan;
Abadie & Gardeazabal (2003) / Abadie, Diamond & Hainmueller (2015) for the data
and the ~-1600 benchmark.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "german_reunification.csv")


def run() -> dict:
    try:
        import numpyro  # noqa: F401
    except ImportError as exc:  # optional dependency -> graceful skip
        raise BenchmarkSkipped(
            "BFSC needs NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc

    from mlsynth import BFSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = d["Reunification"].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = BFSC({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "n_factors": 8, "n_warmup": 500, "n_samples": 500, "n_chains": 4,
            "seed": 0, "display_graphs": False,
        }).fit()

    ts = res.time_series
    yr = np.asarray(ts.time_periods)
    gap = np.asarray(ts.estimated_gap)
    g = lambda y: float(gap[yr == y][0])
    return {
        "mean_att": float(res.att),
        "att_negative": float(res.att < 0.0),
        "gap_1990": g(1990),
        "gap_2003": g(2003),
        "pre_rmse": float(res.fit_diagnostics.rmse_pre),
        "ci_excludes_zero": float(res.inference.ci_upper < 0.0),
        "max_rhat": float(res.weights.summary_stats["max_rhat"]),
    }


# NUTS with a fixed seed => reproducible up to MCMC error; tolerances absorb it.
# Validated cell-for-cell against the author's appendix Stan (corr 0.999999).
EXPECTED = {
    "mean_att": (-1600.0, 600.0),      # Abadie ~-1600/yr reunification effect
    "att_negative": (1.0, 0.0),        # negative effect (classical finding)
    "gap_1990": (350.0, 500.0),        # ~0 at the intervention year
    "gap_2003": (-3670.0, 900.0),      # deepening loss by 2003
    "pre_rmse": (66.0, 130.0),         # close pre-period fit (GDP units)
    "ci_excludes_zero": (1.0, 0.0),    # 95% ATT band excludes zero
    "max_rhat": (1.0, 0.1),            # converged on the identified quantities
}
