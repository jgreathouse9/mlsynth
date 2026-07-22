"""MVBBSC Path-A: West Germany reunification (Martinez & Vives-i-Bastida 2024).

Cross-validation + Path A. Martinez and Vives-i-Bastida (2024) ship their method
as the ``bsynth`` R package (Stan via rstan); mlsynth's MVBBSC is a clean-room
NumPyro port of the same generative model (uniform-Dirichlet simplex weights,
HalfNormal scale, Gaussian likelihood on the pre-period-standardized series). On
the German reunification panel a fresh NumPyro fit matches the bsynth posterior
to MCMC error -- pre-RMSE 62.2 vs 62.2, mean post-ATT -2078 vs -2075, and the
95% credible-band widths agree year-by-year to within a few percent (see
docs/replications/mvbbsc; reference produced by benchmarks/R/mvbbsc_bsynth_ref.R).

This case pins the reproducible headline quantities of a fresh NumPyro fit: the
negative reunification effect near the paper's own ~7.5% magnitude, a positive-
then-negative gap path, a close pre-period fit, an ATT band that excludes zero,
and converged NUTS.

Bayesian (NUTS) => the cells carry MCMC tolerances; the seed is fixed so a run
is reproducible. Requires the ``[bayes]`` optional dependency (NumPyro).

Provenance: Martinez & Vives-i-Bastida (2024), arXiv:2206.01779, Sections 2--3
and the bsynth package; Abadie, Diamond & Hainmueller (2015) for the data and
the reunification benchmark.
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
            "MVBBSC needs NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc

    from mlsynth import MVBBSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = d["Reunification"].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MVBBSC({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "n_warmup": 500, "n_samples": 500, "n_chains": 4,
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
# Cross-validated against the authors' bsynth package (pre-RMSE 62.2, ATT -2075).
EXPECTED = {
    "mean_att": (-2080.0, 500.0),      # ~7.5% decline (the paper's own magnitude)
    "att_negative": (1.0, 0.0),        # negative effect (classical finding)
    "gap_1990": (426.0, 500.0),        # near the intervention year
    "gap_2003": (-4989.0, 1200.0),     # deepening loss by 2003
    "pre_rmse": (62.2, 20.0),          # close pre-period fit (GDP units)
    "ci_excludes_zero": (1.0, 0.0),    # 95% ATT band excludes zero
    "max_rhat": (1.0, 0.1),            # converged simplex weights + sigma
}
