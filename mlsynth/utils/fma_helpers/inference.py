"""Inference procedures for FMA (Li & Sonnier 2023).

Three procedures live here and can run in any combination:

* :func:`asymptotic_inference` -- Theorem 3.1 (stationary) /
  Theorem 3.3 (non-stationary) normal CI for the ATT.
* :func:`bootstrap_inference` -- Web Appendix F residual bootstrap
  for per-period ``ATT_t`` CIs. Uses the pre-period residuals as the
  bootstrap distribution and refits the loading on each draw, so the
  CIs reflect the joint variability of the loading estimate and the
  idiosyncratic shock at time ``t``.
* :func:`placebo_inference` -- Web Appendix G placebo test where
  every control is treated as a pseudo-treated unit in turn. Returns
  the pointwise quantile band across the placebo ATT curves.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from .factors import extract_factors
from .fit import estimate_loading_and_counterfactual


# ---------------------------------------------------------------------------
# Asymptotic (Theorem 3.1)
# ---------------------------------------------------------------------------

def asymptotic_inference(
    treated_outcome: np.ndarray,
    counterfactual: np.ndarray,
    factors_with_const: np.ndarray,
    residual_variance: float,
    T0: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Theorem 3.1 normal CI for the ATT.

    Returns
    -------
    se_att, lower, upper, p_value
    """

    T = treated_outcome.shape[0]
    T2 = T - T0
    if T2 <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    gap = treated_outcome - counterfactual
    att = float(np.mean(gap[T0:]))

    F_pre = factors_with_const[:T0]
    F_post = factors_with_const[T0:]
    F_post_mean = F_post.mean(axis=0).reshape(-1, 1)
    XtX_pre = F_pre.T @ F_pre
    try:
        psi_hat = np.linalg.inv(XtX_pre)
    except np.linalg.LinAlgError:
        psi_hat = np.linalg.pinv(XtX_pre)

    # Omega_hat = Omega1 + Omega2
    omega1 = (T2 / max(T0, 1)) * float(
        (F_post_mean.T @ psi_hat @ F_post_mean).item()
    )
    omega2 = float(residual_variance)
    omega_total = omega1 + omega2

    se_att = float(np.sqrt(max(omega_total, 0.0)) / np.sqrt(T2))
    if not np.isfinite(se_att) or se_att <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = att - z * se_att
    upper = att + z * se_att
    p_value = 2.0 * (1.0 - float(norm.cdf(abs(att) / se_att)))
    return se_att, lower, upper, p_value


# ---------------------------------------------------------------------------
# Bootstrap (Web Appendix F)
# ---------------------------------------------------------------------------

def bootstrap_inference(
    treated_outcome: np.ndarray,
    factors: np.ndarray,
    counterfactual: np.ndarray,
    T0: int,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    seed: int = 0,
) -> dict:
    """Web Appendix F residual bootstrap for per-period ATT_t CIs.

    Procedure
    ---------
    1. Compute pre-period residuals ``e_hat_1t = y_1t - F_aug_t' lambda_hat``
       for ``t = 1, ..., T0``.
    2. For each bootstrap draw b = 1, ..., B:
       a. Sample ``e*_1t`` from {e_hat} with replacement for every t.
       b. Form ``y*_1t = F_aug_t' lambda_hat + e*_1t``.
       c. Re-fit the loading on the bootstrap pre-period.
       d. Compute ``Delta*_1t = y*_1t - F_aug_t' lambda*_hat`` for t > T0.
    3. The (1 - alpha) CI for Delta_1t is
       ``[Delta_hat_1t - q_{1 - alpha/2}, Delta_hat_1t - q_{alpha/2}]``,
       with quantiles taken across bootstrap replicates of Delta*.
    """
    T = treated_outcome.shape[0]
    T2 = T - T0
    if T2 <= 0 or T0 < 2:
        return {
            "lower": np.asarray([], dtype=float),
            "upper": np.asarray([], dtype=float),
            "replicates": np.asarray([], dtype=float),
            "n_replicates": 0,
        }

    ones = np.ones((T, 1))
    F_aug = np.concatenate([ones, factors], axis=1)
    F_aug_pre = F_aug[:T0]
    F_aug_post = F_aug[T0:]

    # Loading from the observed treated pre-period
    XtX_pre = F_aug_pre.T @ F_aug_pre
    diag_mean = float(np.trace(XtX_pre) / max(F_aug_pre.shape[1], 1))
    XtX_reg = XtX_pre + max(diag_mean, 1.0) * 1e-10 * np.eye(F_aug_pre.shape[1])
    XtX_inv = np.linalg.inv(XtX_reg)

    lambda_hat = XtX_inv @ (F_aug_pre.T @ treated_outcome[:T0])
    counterfactual_post = F_aug_post @ lambda_hat
    delta_hat = treated_outcome[T0:] - counterfactual_post

    # Pre-period residuals -- bootstrap pool.
    residuals = treated_outcome[:T0] - F_aug_pre @ lambda_hat
    if residuals.size == 0:
        return {
            "lower": np.asarray([], dtype=float),
            "upper": np.asarray([], dtype=float),
            "replicates": np.asarray([], dtype=float),
            "n_replicates": 0,
        }

    rng = np.random.default_rng(seed)
    replicates = np.empty((n_replicates, T2), dtype=float)

    Pmat = XtX_inv @ F_aug_pre.T       # (r+1, T0) -- avoids re-solving each draw
    Hmat = F_aug_post @ Pmat            # (T2, T0) -- maps pre-resids -> post-CF
    Imat = np.eye(T2)
    for b in range(n_replicates):
        # Bootstrap residuals for the full T-period sequence; only the
        # pre-period subset affects lambda*_hat (Step 2c) while the
        # post-period subset enters as the y* component (Step 2a, b).
        e_star_pre = rng.choice(residuals, size=T0, replace=True)
        e_star_post = rng.choice(residuals, size=T2, replace=True)
        # lambda*_hat = (XtX)^{-1} X'_pre (F_pre lambda_hat + e_star_pre)
        #             = lambda_hat + (XtX)^{-1} X'_pre e_star_pre
        # so the post-period CF perturbation is H_mat @ e_star_pre.
        # Delta* = (F_post lambda_hat + e_star_post) - F_post lambda*_hat
        #        = e_star_post - H_mat @ e_star_pre
        replicates[b] = e_star_post - Hmat @ e_star_pre

    # CI for ATT_t = delta_hat[t] - quantile(replicates[:, t])
    q_lower = np.quantile(replicates, 1.0 - alpha / 2.0, axis=0)
    q_upper = np.quantile(replicates, alpha / 2.0, axis=0)
    lower = delta_hat - q_lower
    upper = delta_hat - q_upper

    return {
        "lower": lower,
        "upper": upper,
        "replicates": replicates,
        "n_replicates": int(n_replicates),
    }


# ---------------------------------------------------------------------------
# Placebo (Web Appendix G)
# ---------------------------------------------------------------------------

def placebo_inference(
    control_outcomes: np.ndarray,
    treated_outcome: np.ndarray,
    T0: int,
    n_factors: Optional[int],
    stationarity: str,
    preprocessing: str,
    alpha: float = 0.05,
    max_factors: int = 10,
) -> dict:
    """Web Appendix G: pseudo-ATT curves with each control as the treated unit.

    For each control ``k``, swap it into the treated slot, refit the
    factor model on the remaining ``N_co - 1`` controls, project the
    pseudo-treated pre-period onto those factors, compute the pseudo-
    ATT curve. The output band is the pointwise alpha/2 / (1 - alpha/2)
    quantile across the placebo curves at each period.

    Parameters
    ----------
    control_outcomes : np.ndarray
        ``(T, N_co)`` control panel.
    treated_outcome : np.ndarray
        Real treated outcome series, shape ``(T,)``. Used for the
        leading row of the curves matrix (so the caller can compare).
    T0 : int
        Pre-treatment periods.
    n_factors : int or None
        Number of factors to fit on each placebo iteration. ``None``
        means re-select per iteration via the criterion in
        ``stationarity``.
    stationarity : {"stationary", "nonstationary"}
    preprocessing : {"demean", "standardize"}
    alpha : float
        Significance level for the quantile band.
    max_factors : int
        Upper bound on the factor-selection routine.

    Returns
    -------
    dict
        Keys ``curves`` (``(N_co + 1, T)``; first row is the real
        treated unit), ``q_lower``, ``q_upper`` (``(T,)`` bands).
    """
    T, N_co = control_outcomes.shape

    # Real treated unit
    n_real, _, F_real, _ = extract_factors(
        control_outcomes,
        stationarity=stationarity,
        preprocessing=preprocessing,
        n_factors=n_factors,
        max_factors=max_factors,
    )
    _, cf_real, _, _ = estimate_loading_and_counterfactual(
        treated_outcome, F_real, T0
    )
    real_gap = treated_outcome - cf_real

    curves = np.empty((N_co + 1, T), dtype=float)
    curves[0] = real_gap

    for k in range(N_co):
        mask = np.ones(N_co, dtype=bool)
        mask[k] = False
        loo_controls = control_outcomes[:, mask]
        if loo_controls.shape[1] < 1:
            curves[k + 1] = np.nan
            continue
        try:
            _, _, F_k, _ = extract_factors(
                loo_controls,
                stationarity=stationarity,
                preprocessing=preprocessing,
                n_factors=n_factors,
                max_factors=min(max_factors, loo_controls.shape[1]),
            )
            _, cf_k, _, _ = estimate_loading_and_counterfactual(
                control_outcomes[:, k], F_k, T0
            )
        except Exception:
            curves[k + 1] = np.nan
            continue
        curves[k + 1] = control_outcomes[:, k] - cf_k

    placebo_band = curves[1:]
    # Skip rows with any NaNs from failed placebos.
    valid_mask = np.all(np.isfinite(placebo_band), axis=1)
    if valid_mask.any():
        q_lower = np.quantile(placebo_band[valid_mask], alpha / 2.0, axis=0)
        q_upper = np.quantile(placebo_band[valid_mask], 1.0 - alpha / 2.0, axis=0)
    else:
        q_lower = np.full(T, np.nan)
        q_upper = np.full(T, np.nan)

    return {
        "curves": curves,
        "q_lower": q_lower,
        "q_upper": q_upper,
    }
