"""Leave-one-treated-unit-out jackknife for PPSCM.

For each treated unit ``j``, drop unit ``j`` from the panel, refit
PPSCM on the remaining ``J - 1`` treated units and the same donor pool,
and record the leave-one-out ATT. The standard error follows the
standard jackknife formula

    se_jack = sqrt( (J - 1) / J * sum_j (att_loo_j - att_loo_mean)^2 ).

A Wald-style confidence interval is built from this SE.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.stats import norm

from .optimization import solve_ppscm


def event_study_taus(Y_treated_post: np.ndarray, Y_donors_post: np.ndarray,
                     Gamma: np.ndarray) -> np.ndarray:
    """Return ``(K+1,)`` array of per-horizon ATTs ``(1/J) sum_j tau_{j, k}``."""
    fitted = np.einsum("kij,ij->kj", Y_donors_post, Gamma)
    return (Y_treated_post - fitted).mean(axis=1)


def jackknife_inference(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    Y_treated_post: np.ndarray,
    Y_donors_post: np.ndarray,
    nu: float | str,
    lam: float,
    solver: Any,
    nu_grid_size: int,
    alpha: float = 0.05,
) -> Tuple[float, float, Tuple[float, float], np.ndarray, np.ndarray]:
    """Return ``(att, se, ci, taus_per_horizon, se_per_horizon)``.

    The ``att`` is the average ATT across horizons; the per-horizon
    arrays come back too so the caller can plot the trajectory.
    """
    L, J = Y_treated_pre.shape
    if J < 2:
        # Jackknife is undefined for a single treated unit.
        return float("nan"), float("nan"), (float("nan"), float("nan")), \
               np.full(Y_treated_post.shape[0], float("nan")), \
               np.full(Y_treated_post.shape[0], float("nan"))

    # Refit on each leave-one-out panel.
    horizon_count = Y_treated_post.shape[0]
    loo_taus = np.zeros((J, horizon_count))
    for j in range(J):
        keep = np.array([k for k in range(J) if k != j])
        Yt_pre = Y_treated_pre[:, keep]
        Yd_pre = Y_donors_pre[:, :, keep]
        Yt_post = Y_treated_post[:, keep]
        Yd_post = Y_donors_post[:, :, keep]
        try:
            Gamma_loo, *_ = solve_ppscm(
                Yt_pre, Yd_pre, nu=nu, lam=lam, solver=solver,
                nu_grid_size=nu_grid_size,
            )
        except Exception:
            loo_taus[j, :] = np.nan
            continue
        loo_taus[j, :] = event_study_taus(Yt_post, Yd_post, Gamma_loo)

    # Per-horizon ATTs across the full panel and their jackknife SEs.
    # The "full ATT at horizon k" computed via solve_ppscm is the
    # comparison object; we recompute the leave-one-out average for the
    # SE.
    valid = np.isfinite(loo_taus).all(axis=1)
    if valid.sum() < 2:
        return float("nan"), float("nan"), (float("nan"), float("nan")), \
               np.full(horizon_count, float("nan")), \
               np.full(horizon_count, float("nan"))
    loo_taus = loo_taus[valid, :]
    loo_means = loo_taus.mean(axis=0)
    j_valid = loo_taus.shape[0]
    se_per_horizon = np.sqrt(
        (j_valid - 1.0) / j_valid * ((loo_taus - loo_means) ** 2).sum(axis=0)
    )
    overall_atts = loo_taus.mean(axis=1)
    overall_att_mean = overall_atts.mean()
    se_overall = float(np.sqrt(
        (j_valid - 1.0) / j_valid * ((overall_atts - overall_att_mean) ** 2).sum()
    ))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci = (float(overall_att_mean - z * se_overall),
          float(overall_att_mean + z * se_overall))
    return (float(overall_att_mean), se_overall, ci,
            loo_means, se_per_horizon)
