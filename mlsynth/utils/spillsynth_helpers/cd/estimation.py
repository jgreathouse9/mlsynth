"""Cao-Dowd spillover-adjusted treatment-effect estimator (eq. 5).

Given the leave-one-out SCM artifacts ``(a, B)`` from
:mod:`.scm_core` and a spillover-structure matrix ``A``, the
per-post-period parameter estimate is

    gamma_hat(T+1) = (A' M A)^{-1} A' (I - B)' [(I - B) Y_{T+1} - a]

with ``M = (I - B)' (I - B)``. The full effect vector is then
``alpha_hat = A @ gamma_hat``; its first entry is the
spillover-adjusted treatment effect on the treated unit, and entries
2..p+1 are the per-affected-unit spillover effects.

This module also provides a thin :func:`vanilla_scm_path` helper that
returns the standard SCM counterfactual using the treated unit's own
leave-one-out fit, so the orchestrator can produce the SCM-vs-SP
comparison without re-fitting.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError

_M_RIDGE = 1e-8

# Above this condition number, A' M A is numerically degenerate and the
# Cao-Dowd identification requirement (Assumption 1(d)) is effectively
# violated -- the inverse blows up small residual noise into the effect
# estimates. At cond > 1e8 roughly half of float64's ~16 significant digits
# are already lost, so we warn (opt-in) rather than hard-fail because some
# LAPACK builds still return a (numerically dubious) inverse instead of
# raising. Whether the solve raises or merely warns is platform-dependent;
# both outcomes flag the same near-non-identification.
_COND_WARN = 1e8


def _invert_AMA(AMA: np.ndarray, *, label: str, warn: bool) -> Tuple[np.ndarray, float]:
    """Invert ``A' M A`` with a clear error on singularity and an optional
    ill-conditioning warning. Returns ``(inverse, condition_number)``.
    """
    cond = float(np.linalg.cond(AMA))
    try:
        inv = np.linalg.inv(AMA)
    except np.linalg.LinAlgError as exc:
        raise MlsynthEstimationError(
            f"SPILLSYNTH/cd: {label} is singular (cond={cond:.3e}); the "
            "spillover structure A is not identified (Cao-Dowd Assumption "
            "1(d) fails). Declare fewer affected units or a lower-rank A."
        ) from exc
    if warn and cond > _COND_WARN:
        warnings.warn(
            f"SPILLSYNTH/cd: {label} is ill-conditioned (cond={cond:.3e} > "
            f"{_COND_WARN:.0e}); the spillover estimates may be numerically "
            "unstable (Cao-Dowd Assumption 1(d) near-violation).",
            RuntimeWarning,
            stacklevel=3,
        )
    return inv, cond


def build_M(B: np.ndarray) -> np.ndarray:
    """Gram matrix ``M = (I - B)' (I - B) + 1e-8 * I``."""
    N = B.shape[0]
    I_B = np.eye(N) - B
    return I_B.T @ I_B + np.eye(N) * _M_RIDGE


def sp_estimate(
    Y_post: np.ndarray,
    *,
    a: np.ndarray,
    B: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    warn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute per-period spillover-adjusted effects.

    Parameters
    ----------
    Y_post : np.ndarray
        Shape ``(N, T1)`` post-treatment outcomes.
    a : np.ndarray
        Length-``N`` leave-one-out intercepts.
    B : np.ndarray
        Shape ``(N, N)`` leave-one-out SCM weight matrix.
    M : np.ndarray
        Shape ``(N, N)`` Gram matrix from :func:`build_M`.
    A : np.ndarray
        Shape ``(N, k)`` spillover-structure matrix.

    Returns
    -------
    gamma : np.ndarray
        Shape ``(k, T1)`` parameter estimates per post-period.
    alpha : np.ndarray
        Shape ``(N, T1)`` effect estimates per post-period.
    cond_AMA : float
        Condition number of ``A' M A`` (Assumption 1(d) diagnostic).
    """
    N, T1 = Y_post.shape
    I_B = np.eye(N) - B
    AMA = A.T @ M @ A
    AMA_inv, cond_AMA = _invert_AMA(AMA, label="A' M A", warn=warn)
    # Each column of Y_post is one Y_{T+1}; vectorise across periods.
    residual = I_B @ Y_post - a[:, None]                # (N, T1)
    gamma = AMA_inv @ (A.T @ I_B.T @ residual)           # (k, T1)
    alpha = A @ gamma                                    # (N, T1)
    return gamma, alpha, cond_AMA


def vanilla_scm_path(
    Y_post: np.ndarray, *, a: np.ndarray, B: np.ndarray,
) -> np.ndarray:
    """Vanilla (no-spillover) SCM counterfactual for the treated unit.

    Returns a length-``T1`` vector ``a[0] + B[0, :] @ Y_post`` —
    the treated unit's own leave-one-out fit applied to the post-period
    outcomes.
    """
    return a[0] + B[0] @ Y_post


# ---------------------------------------------------------------------------
# Cao-Dowd v3 Section S.1.1: efficient-weighted (GMM-style) variant
# ---------------------------------------------------------------------------


def estimate_omega_from_pre_residuals(
    Y_pre: np.ndarray, a: np.ndarray, B: np.ndarray, *, ridge: float = 1e-6,
) -> np.ndarray:
    """Sample covariance of the pre-period residuals plus a ridge.

    Implements :math:`\\widehat \\Omega = T_0^{-1} \\sum_{t=1}^{T_0}
    \\widehat u_t \\widehat u_t^\\prime + \\lambda I`, where
    :math:`\\widehat u_t = (I - \\widehat B) Y_t - \\widehat a` is the
    in-sample SCM residual vector at pre-period :math:`t`. The ridge
    ensures positive-definiteness when :math:`T_0 < N`.
    """
    N, T0 = Y_pre.shape
    U = (np.eye(N) - B) @ Y_pre - a[:, None]            # (N, T0)
    Omega = (U @ U.T) / T0
    Omega = Omega + ridge * np.eye(N)
    return Omega


def sp_estimate_weighted(
    Y_post: np.ndarray,
    *,
    a: np.ndarray,
    B: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    warn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted Cao-Dowd estimator with a user-supplied weighting matrix.

    Cao-Dowd v3 Section S.1.1 generalisation: minimise
    :math:`\\|W^{1/2} \\widehat u_{T+1}\\|` rather than the unweighted
    Euclidean norm, where :math:`\\widehat u_{T+1} = (I - \\widehat B)
    (Y_{T+1} - A \\gamma) - \\widehat a`. The closed form is

    .. math::

       \\widehat \\gamma_W = (A^\\prime \\widehat M_W A)^{-1}
       A^\\prime (I - \\widehat B)^\\prime W
       \\left[(I - \\widehat B) Y_{T+1} - \\widehat a\\right],

    with :math:`\\widehat M_W = (I - \\widehat B)^\\prime W (I - \\widehat B)`.

    Pass ``W`` = identity to recover :func:`sp_estimate`. Pass ``W``
    = an estimator of :math:`\\Omega^{-1}` (e.g.\\
    ``np.linalg.inv(estimate_omega_from_pre_residuals(...))``) to obtain
    the efficient variant :math:`\\widehat \\alpha^e` of Proposition S.1
    -- which has asymptotic variance no larger than the unweighted
    estimator.

    Parameters
    ----------
    Y_post : np.ndarray
        Shape ``(N, T1)`` post-treatment outcomes.
    a, B : np.ndarray
        Leave-one-out SCM artefacts.
    A : np.ndarray
        Shape ``(N, k)`` spillover-structure matrix.
    W : np.ndarray
        Shape ``(N, N)`` weighting matrix. Should be positive-definite;
        the caller is responsible for any regularisation.

    Returns
    -------
    gamma : np.ndarray
        Shape ``(k, T1)``.
    alpha : np.ndarray
        Shape ``(N, T1)``.
    cond_AMA_W : float
        Condition number of :math:`A^\\prime \\widehat M_W A`
        (Assumption 1(d) diagnostic under the weighted estimator).
    """
    N, T1 = Y_post.shape
    I_B = np.eye(N) - B
    M_W = I_B.T @ W @ I_B
    AMA_W = A.T @ M_W @ A
    AMA_W_inv, cond_AMA_W = _invert_AMA(AMA_W, label="A' M_W A", warn=warn)
    residual = I_B @ Y_post - a[:, None]                # (N, T1)
    gamma = AMA_W_inv @ (A.T @ I_B.T @ W @ residual)     # (k, T1)
    alpha = A @ gamma                                    # (N, T1)
    return gamma, alpha, cond_AMA_W
