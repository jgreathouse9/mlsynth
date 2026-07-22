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
from benchmarks.reference import reference_value

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "german_reunification.csv")

# The one MVBBSC call shared by run() and comparison(): the authors' outcome-only
# model, four NUTS chains, fixed seed so a run is reproducible up to MCMC error.
_MLSYNTH_KW = {
    "outcome": "gdp", "treat": "treat", "unitid": "country", "time": "year",
    "n_warmup": 500, "n_samples": 500, "n_chains": 4,
    "seed": 0, "display_graphs": False,
}


def _fit():
    """Fit MVBBSC on the German reunification panel. Skips gracefully when
    NumPyro (the ``[bayes]`` optional dependency) is absent."""
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
        return MVBBSC({"df": d, **_MLSYNTH_KW}).fit()


def run() -> dict:
    res = _fit()
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


def _mlsynth_summaries(res) -> dict:
    """The four posterior summaries cross-validated against bsynth: the pre-1990
    in-sample RMSE, the mean post-1990 ATT, and the mean 95% credible-band width
    in the pre- and post-treatment windows (the band is ``counterfactual_upper``
    minus ``counterfactual_lower`` period by period)."""
    inf = res.inference_detail
    width = np.asarray(inf.counterfactual_upper) - np.asarray(inf.counterfactual_lower)
    T0 = res.inputs.T0
    return {
        "pre_rmse": float(res.fit_diagnostics.rmse_pre),
        "mean_att": float(res.att),
        "band_width_pre": float(np.mean(width[:T0])),
        "band_width_post": float(np.mean(width[T0:])),
    }


def comparison() -> dict:
    """mlsynth MVBBSC vs the authors' ``bsynth`` package on the German
    reunification panel, posterior summary by posterior summary. Both fit the
    identical outcome-only model (``predictor_match = FALSE``, ``ci_width =
    0.95``); the reference numbers are the live captured bsynth run under
    ``benchmarks/reference/mvbbsc_germany/``. Agreement is to Monte-Carlo error
    between two independent NUTS runs."""
    m = _mlsynth_summaries(_fit())
    pairs = [
        ("pre_1990_RMSE", "pre_rmse"),
        ("mean_post_ATT", "mean_att"),
        ("band_width_pre_1990", "band_width_pre"),
        ("band_width_post_1990", "band_width_post"),
    ]
    rows = [{"quantity": q, "mlsynth": round(float(m[k]), 4),
             "reference": round(reference_value("mvbbsc_germany", k), 4)}
            for q, k in pairs]
    cfg = {k: v for k, v in _MLSYNTH_KW.items() if k != "display_graphs"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MVBBSC", "config": cfg},
        "reference": {
            "impl": "R package bsynth (bayesianSynth, predictor_match=FALSE, live run)",
            "version": "Martinez & Vives-i-Bastida (2024), bsynth @ 22d960f (arXiv:2206.01779)",
        },
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
