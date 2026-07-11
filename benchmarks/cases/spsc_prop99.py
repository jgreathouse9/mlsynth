"""SPSC Path-A / cross-validation: the California (Proposition 99) example.

Reproduces the authors' California illustration from the ``qkrcks0218/SPSC`` R
package value-for-value. SPSC views the 38 donor states' cigarette sales as a
single proxy for treatment-free California, detrends against a linear time
trend, and fits a *linear-in-time* treatment-effect path (the package's
``att.ft = (1, t)``). The effect grows from about -4.8 packs in the first
post-period (1988) to -35.3 in the last (2000).

This case cross-checks mlsynth's SPSC -- configured to that exact
parameterisation (``spsc_att_degree=1``, ``spsc_detrend_basis="poly"``, ridge
``lambda`` fixed at ``10**0``) -- against the reference R implementation run
live on the same panel (``basedata/smoking_data.csv``, ``T0 = 18``). The
per-period effect path and its per-period standard errors match the reference
to solver tolerance. Path A / cross-validation (scenario 3): the data and the
reference are the authors'. Skips gracefully when ``Rscript`` or the SPSC clone
is unavailable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "smoking_data.csv")
_T0 = 18
_LAMBDA = 0.0


def _panel():
    df = pd.read_csv(os.path.abspath(_DATA))
    states = list(dict.fromkeys(df["state"]))
    donors = [s for s in states if s != "California"]
    y = df["cigsale"][df["state"] == "California"].to_numpy(float)
    W = np.column_stack([df["cigsale"][df["state"] == s].to_numpy(float)
                         for s in donors])
    return y, W, donors


# T0 = 18 cannot support a 95% conformal level (the finest achievable is
# 2/19 ~ 0.105), so the conformal cross-check runs at ~89%.
_CONFORMAL_ALPHA = 0.11


def run() -> dict:
    from benchmarks.reference.clone_spsc import run_reference
    from mlsynth.utils.proximal_helpers.spsc.conformal import conformal_intervals
    from mlsynth.utils.proximal_helpers.spsc.estimation import estimate_spsc

    y, W, donors = _panel()
    T1 = len(y) - _T0
    ref = run_reference(y, W, _T0, detrend=True, att_degree=1,
                        detrend_linear=True, ridge_lambda=_LAMBDA,
                        conformal_periods=list(range(1, T1 + 1)),
                        conformal_alpha=_CONFORMAL_ALPHA)             # skips if no R
    out = estimate_spsc(y, W, _T0, detrend=True, ridge_lambda=_LAMBDA,
                        att_degree=1, detrend_basis="poly", detrend_degree=1)
    path, path_se = out[6], out[7]
    band = conformal_intervals(
        y, W, _T0, gamma=out[1], ridge_lambda=_LAMBDA, detrend=True,
        spline_df=5, att_se=out[3], period_se=path_se, alpha=_CONFORMAL_ALPHA,
        att_degree=1, detrend_basis="poly", detrend_degree=1)
    return {
        "att": float(out[2]),                       # mean of the fitted path
        "path_first": float(path[0]),
        "path_last": float(path[-1]),
        "n_donors": float(len(donors)),
        "path_vs_ref": float(np.max(np.abs(path - ref["effect_path"]))),
        "se_vs_ref": float(np.max(np.abs(path_se - ref["path_se"]))),
        "conformal_lb_vs_ref": float(np.max(np.abs(band["lower"] - ref["conformal_lb"]))),
        "conformal_ub_vs_ref": float(np.max(np.abs(band["upper"] - ref["conformal_ub"]))),
    }


def comparison() -> dict:
    """mlsynth SPSC vs the authors' ``qkrcks0218/SPSC`` R: the California
    (Proposition 99) linear effect path, side by side."""
    from benchmarks.reference.clone_spsc import run_reference
    from mlsynth.utils.proximal_helpers.spsc.estimation import estimate_spsc

    y, W, donors = _panel()
    ref = run_reference(y, W, _T0, detrend=True, att_degree=1,
                        detrend_linear=True, ridge_lambda=_LAMBDA)     # skips if no R
    out = estimate_spsc(y, W, _T0, detrend=True, ridge_lambda=_LAMBDA,
                        att_degree=1, detrend_basis="poly", detrend_degree=1)
    path, r_path = out[6], ref["effect_path"]
    yrs = list(range(1989, 1989 + len(path)))
    rows = [{"quantity": "ATT (mean path)", "mlsynth": round(float(out[2]), 4),
             "reference": round(float(np.mean(r_path)), 4)}]
    for yr, m, r in zip(yrs, path, r_path):
        rows.append({"quantity": f"effect[{yr}]", "mlsynth": round(float(m), 4),
                     "reference": round(float(r), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SPSC",
                         "config": {"detrend": True, "att_degree": 1,
                                    "ridge_lambda": _LAMBDA}},
        "reference": {"impl": "qkrcks0218/SPSC R (single-proxy synthetic control)",
                      "version": "@054f1fbb"},
    }


# Validated value-for-value against qkrcks0218/SPSC @ 054f1fbb (lambda = 10**0):
# the linear effect path runs -4.845 ... -35.284 (mean -20.06), per-period SE
# 0.0020 ... 0.0235; mlsynth reproduces both to solver tolerance. The pointwise
# conformal prediction intervals (level ~0.11, the finest T0=18 supports) run
# from a width of 0.139 (1988) to 1.645 (2000) and match the reference to ~1e-3.
EXPECTED = {
    "att": (-20.064, 0.05),
    "path_first": (-4.845, 0.02),
    "path_last": (-35.284, 0.05),
    "n_donors": (38.0, 0.0),
    "path_vs_ref": (0.0, 5e-3),       # bit-for-bit vs the R package
    "se_vs_ref": (0.0, 5e-4),
    "conformal_lb_vs_ref": (0.0, 3e-3),
    "conformal_ub_vs_ref": (0.0, 3e-3),
}
