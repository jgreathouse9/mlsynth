"""Delete-one jackknife inference for PPSCM (Ben-Michael et al. 2022).

The paper's jackknife drops each unit ``i`` (treated *or* control), refits the
full staggered estimator on the remaining ``n - 1`` units (holding ``nu``
fixed), and forms

    se^2 = (n - 1) / n * sum_i (theta_i - mean_i theta_i)^2

separately for the overall ATT and each relative-time horizon. Wald intervals
are built from these SEs around the full-sample point estimates.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.stats import norm

from .engine import run_multisynth


def jackknife_inference(
    Xy: np.ndarray, trt: np.ndarray, d: int, n_leads: int, n_lags: int,
    *, fixedeff: bool, time_cohort: bool, nu_used: float, lam: float,
    solver: Any, alpha: float, per_time_full: np.ndarray, att_full: float,
) -> Tuple[float, float, Tuple[float, float], np.ndarray, np.ndarray]:
    """Return ``(att, se, ci, per_time_se, per_time_ci)``."""
    n = Xy.shape[0]
    H = n_leads
    att_loo = np.full(n, np.nan)
    pt_loo = np.full((n, H), np.nan)

    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        trt_i = trt[keep]
        if not np.isfinite(trt_i).any() or np.isfinite(trt_i).all():
            continue                                  # need >=1 treated and >=1 control
        try:
            fit_i = run_multisynth(
                Xy[keep], trt_i, d, n_leads, n_lags,
                fixedeff=fixedeff, time_cohort=time_cohort,
                nu=nu_used, lam=lam, solver=solver,
            )
        except Exception:
            continue
        att_loo[i] = fit_i["att"]
        pt = fit_i["per_time"]
        pt_loo[i, : len(pt)] = pt

    def _se(col: np.ndarray) -> float:
        x = col[~np.isnan(col)]
        m = x.size
        if m < 2:
            return float("nan")
        return float(np.sqrt((m - 1) / m * np.sum((x - x.mean()) ** 2)))

    se = _se(att_loo)
    per_time_se = np.array([_se(pt_loo[:, h]) for h in range(H)])
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci = (att_full - z * se, att_full + z * se)
    per_time_ci = np.column_stack([per_time_full - z * per_time_se,
                                   per_time_full + z * per_time_se])
    return float(att_full), se, ci, per_time_se, per_time_ci
