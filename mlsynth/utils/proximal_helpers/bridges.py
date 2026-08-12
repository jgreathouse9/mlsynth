"""Shared building blocks for the doubly-robust proximal estimators.

Implements the two confounding-bridge fits and the just-identified GMM
sandwich used by both the doubly-robust (``dr``) and treatment-bridge
weighting (``pipw``) estimators of

    Qiu, H., Shi, X., Miao, W., Dobriban, E., & Tchetgen Tchetgen, E. (2024).
    "Doubly robust proximal synthetic controls." Biometrics 80(2), ujae055.

Bridges (with an intercept column appended to ``W`` and ``Z``):

* **outcome bridge** ``h_alpha(W) = (1, W) alpha`` -- a just-identified IV
  fit of the treated outcome on the donors ``W`` instrumented by the
  proxies ``Z`` on the pre-period.
* **treatment bridge** ``q_beta(Z) = exp((1, Z) beta)`` -- a covariate-shift
  / likelihood-ratio weight solving the pre-period moment
  ``E_pre[q(Z)(1, W)] = E_post[(1, W)]``.

Both estimators are just-identified, so the parameters solve the empirical
moment equations exactly and the asymptotic variance is the GMM sandwich
``G^{-1} Omega G^{-T} / T`` with a Bartlett-HAC ``Omega``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import root

from ...exceptions import MlsynthEstimationError


def augment(matrix: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones."""
    return np.column_stack([np.ones(len(matrix)), matrix])


def fit_outcome_bridge(Y_pre: np.ndarray, Wc_pre: np.ndarray, Zc_pre: np.ndarray) -> np.ndarray:
    """Just-identified IV for ``alpha``: ``E_pre[(1,Z)(Y - (1,W) alpha)] = 0``."""
    return np.linalg.solve(Zc_pre.T @ Wc_pre, Zc_pre.T @ Y_pre)


def fit_treatment_bridge(
    Zc_pre: np.ndarray,
    Wc_pre: np.ndarray,
    psi: np.ndarray,
    beta_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve ``E_pre[exp((1,Z) beta) (1,W)] = psi`` for ``beta``.

    ``psi = E_post[(1, W)]`` is the post-period donor mean. The system is
    square (``dim(beta) = dim(W)+1``); a Newton/hybr solve from ``beta=0``
    (with a logistic-regression fallback init) recovers it.
    """
    p = Zc_pre.shape[1]

    def moment(beta: np.ndarray) -> np.ndarray:
        q = np.exp(Zc_pre @ beta)
        return (q[:, None] * Wc_pre).mean(0) - psi

    inits = [np.zeros(p)] if beta_init is None else [beta_init, np.zeros(p)]
    last = None
    for b0 in inits:
        sol = root(moment, b0, method="hybr")
        last = sol.x
        if sol.success:
            return sol.x
    return last  # best effort; downstream SE will reflect any residual error


def _bartlett_hac(U: np.ndarray, bandwidth: int) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance of the moment rows ``U``."""
    T = U.shape[0]
    omega = U.T @ U / T
    for lag in range(1, max(bandwidth, 0) + 1):
        if lag >= T:
            break
        weight = 1.0 - lag / (bandwidth + 1)
        auto = U[:-lag].T @ U[lag:] / T
        omega += weight * (auto + auto.T)
    return omega


def gmm_sandwich_se(
    theta: np.ndarray,
    moments: Callable[[np.ndarray], np.ndarray],
    param_index: int,
    total_periods: int,
    bandwidth: int,
    eps: float = 1e-6,
    expected_value: float | None = None,
) -> float:
    """Sandwich SE for one parameter of a just-identified GMM.

    Parameters
    ----------
    theta : np.ndarray
        Solved parameter vector.
    moments : callable
        ``theta -> U`` returning the ``(T, p)`` per-period moment matrix.
    param_index : int
        Index into ``theta`` of the parameter whose SE is wanted.
    total_periods : int
        ``T`` (sandwich normalization).
    bandwidth : int
        Bartlett-HAC bandwidth.
    expected_value : float, optional
        The estimate this standard error is meant to belong to. When supplied it
        is checked against ``theta[param_index]``, which is the only thing tying
        the index to the estimand.

        The caller states the layout twice -- once in the offsets its moment
        function unpacks ``theta`` with, once in this index -- and nothing else
        makes the two agree. When they disagree the sandwich still returns a
        finite, positive, plausibly-sized number, because every entry on that
        diagonal is some parameter's standard error. Passing the estimate turns a
        silent mislabel into a raised error.

    Returns
    -------
    float
        ``sqrt(Cov[param_index, param_index])``; ``np.nan`` if the Jacobian
        is singular or the variance is negative.

    Raises
    ------
    MlsynthEstimationError
        If ``expected_value`` is given and does not match ``theta[param_index]``.
    """
    if expected_value is not None:
        got = float(theta[param_index])
        if not np.isclose(got, float(expected_value), rtol=1e-8, atol=1e-10):
            raise MlsynthEstimationError(
                f"the GMM sandwich was asked for index {param_index}, which holds "
                f"{got:.10g}, but the estimate it is meant to describe is "
                f"{float(expected_value):.10g}. The moment function's parameter "
                f"layout and the index disagree, so this standard error would "
                f"belong to a different parameter."
            )
    base = moments(theta).mean(0)
    p = len(theta)
    G = np.zeros((len(base), p))
    for j in range(p):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        G[:, j] = (moments(tp).mean(0) - moments(tm).mean(0)) / (2 * eps)
    omega = _bartlett_hac(moments(theta), bandwidth)
    try:
        G_inv = np.linalg.inv(G)
        cov = G_inv @ omega @ G_inv.T / total_periods
        var = cov[param_index, param_index]
        return float(np.sqrt(var)) if var >= 0 else np.nan
    except np.linalg.LinAlgError:
        return np.nan
