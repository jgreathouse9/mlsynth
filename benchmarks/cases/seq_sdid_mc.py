"""Sequential SDiD Path-B: the calibrated-panel coverage/RMSE Monte Carlo.

Path B (Monte Carlo, scenario 1 -- paper only). Reproduces the *geometry* of
Arkhangelsky & Samkov (2025) Table 1 (Section 5.2.2, "Experiment 2: Calibrated
State-Level Panel"): under an interactive-fixed-effects violation of parallel
trends with adoption correlated to the leading factor loading,

  * the **standard DiD** estimator (the eta -> infinity / Remark-2.2 limit,
    ``mode="sdid_imputation"``) is severely biased and its 95% CIs
    **under-cover** -- the paper reports coverage collapsing to ~0.70; and
  * **Sequential SDiD** (``mode="ssdid"``) is approximately unbiased with
    coverage near the nominal 0.95 and lower RMSE.

We have the paper only -- no replication code, and the authors' CPS women's
log-wage panel is not public -- so the DGP is re-implemented from the paper's
description in the reusable :mod:`mlsynth.utils.seq_sdid_helpers.simulate`
helper (a rank-one differential-trend IFE; structural truth fixed, only the
AR(2) shocks redrawn per draw; units replicated x4 as in Section 5.2.1). The
*standard-DiD comparator is the same estimator at eta -> infinity*, exactly the
paper's "Original Results" line (Figure 1), so both arms share the Bayesian
bootstrap and the comparison is apples-to-apples.

Donor balance: a treated cohort needs >= 2 later-adopting / never-treated donor
cohorts to balance its loading, so the design caps ``a_max`` at the
``donor_tail``-th-latest cohort; the paper's many-cohort regime avoids
starvation by construction.

Determinism: structural seed 2024; per-draw shock seeds ``8000 + m`` for
``M = 40`` draws; bootstrap seed 7 with ``B = 50`` -- fully reproducible.

Provenance: arXiv:2404.00164v2, Section 5.2.2, Table 1 and Figures 4-5.
"""
from __future__ import annotations

import warnings

import numpy as np

# Locked parameters (calibrated to the paper's qualitative regime).
_TAU = 1.0          # planted constant effect
_K = 4              # horizons 0..4
_M = 40             # Monte-Carlo draws (paper: 1000)
_B = 50             # bootstrap reps per fit (paper: 100)
_ETA = 0.05         # finite SSDiD penalty (eta -> inf gives the DiD comparator)
_STRUCT_SEED = 2024
_BOOT_SEED = 7


def _fit(df, mode, a_max):
    from mlsynth import SequentialSDID
    res = SequentialSDID({
        "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
        "time": "year", "mode": mode, "eta": _ETA, "K": _K, "a_max": a_max,
        "n_bootstrap": _B, "seed": _BOOT_SEED, "display_graphs": False,
    }).fit()
    return res.event_study.tau, res.event_study.ci


def run() -> dict:
    from mlsynth.utils.seq_sdid_helpers.simulate import (
        calibrate_staggered_ife,
        simulate_replication,
    )

    design = calibrate_staggered_ife(seed=_STRUCT_SEED)

    arms = ("ssdid", "sdid_imputation")
    sq = {m: [] for m in arms}      # squared error per draw, per horizon
    cov = {m: [] for m in arms}     # coverage indicator per draw, per horizon
    bias = {m: [] for m in arms}    # signed error per draw, per horizon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in range(_M):
            df = simulate_replication(
                design, np.random.default_rng(8000 + m), tau=_TAU)
            for mode in arms:
                tau_hat, ci = _fit(df, mode, design.a_max)
                sq[mode].append((tau_hat - _TAU) ** 2)
                cov[mode].append(
                    ((ci[:, 0] <= _TAU) & (_TAU <= ci[:, 1])).astype(float))
                bias[mode].append(tau_hat - _TAU)

    out = {"n_draws": float(_M), "n_cohorts": float(design.n_cohorts),
           "a_max": float(design.a_max)}
    for mode in arms:
        rmse = float(np.sqrt(np.mean(sq[mode])))
        coverage = float(np.mean(cov[mode]))
        abs_bias = float(np.mean(np.abs(np.mean(bias[mode], axis=0))))
        tag = "ssdid" if mode == "ssdid" else "did"
        out[f"{tag}_mean_coverage"] = coverage
        out[f"{tag}_mean_rmse"] = rmse
        out[f"{tag}_mean_abs_bias"] = abs_bias
    # Headline contrasts (the paper's "DiD severely biased, SSDiD reliable").
    out["coverage_gap"] = out["ssdid_mean_coverage"] - out["did_mean_coverage"]
    return out


# Deterministic (fixed structural + shock + bootstrap seeds) => exact re-runs.
# The cells reproduce Table 1's geometry at a small M: Sequential SDiD coverage
# is near the nominal 0.95 while standard DiD collapses well below it (paper
# ~0.70; this reconstruction lands ~0.4-0.5 because its differential-trend
# violation is sharper than the CPS calibration), and SSDiD's bias is several
# times smaller. Tolerances are wide enough to absorb the gap to the paper's
# M = 1000 / B = 100 (coverage SE ~ sqrt(p(1-p)/(M*5)) ~ 0.03) and tight enough
# to catch a regression that flips the geometry (e.g. SSDiD under-covering or
# DiD recovering nominal coverage).
EXPECTED = {
    "n_draws": (40.0, 0.0),
    "ssdid_mean_coverage": (0.945, 0.085),  # near nominal; fails below ~0.86
    "did_mean_coverage": (0.45, 0.16),      # collapsed; fails above ~0.61
    "ssdid_mean_abs_bias": (0.062, 0.055),  # ~unbiased
    "did_mean_abs_bias": (0.305, 0.13),     # severely biased
    "ssdid_mean_rmse": (0.252, 0.09),
    "did_mean_rmse": (0.346, 0.11),
    "coverage_gap": (0.495, 0.18),          # SSDiD covers, DiD does not
}
