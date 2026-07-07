"""Systematic causal-effect estimation for CFM (Bai & Wang 2026, Sec 7.1).

Given the common factors ``F_hat`` estimated from the control units, the
treated unit is regressed on ``(1, f_t)`` separately over the pre- and
post-treatment periods:

.. math::

   (\\hat a_0, \\hat\\lambda_0) = \\arg\\min \\sum_{t \\le T_0}
        (y_t - a_0 - \\lambda_0' f_t)^2, \\quad
   (\\hat a_1, \\hat\\lambda_1) = \\arg\\min \\sum_{t > T_0}
        (y_t - a_1 - \\lambda_1' f_t)^2,

and the systematic causal effect is
``tau*_t = (lambda1 - lambda0)' f_t + (a1 - a0)`` for ``t > T0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SystematicFit:
    """Regression output of :func:`fit_systematic_effect`."""

    a0: float
    a1: float
    lambda0: np.ndarray
    lambda1: np.ndarray
    kappa: float
    tau: np.ndarray
    tau_full: np.ndarray
    counterfactual: np.ndarray
    att: float


def _design(factors: np.ndarray) -> np.ndarray:
    """Prepend a constant column: ``[1, f_t]``."""
    return np.column_stack([np.ones(factors.shape[0]), factors])


def fit_systematic_effect(
    treated_outcome: np.ndarray, factors: np.ndarray, T0: int
) -> SystematicFit:
    """Estimate the systematic causal effect for a single treated unit.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    factors : np.ndarray
        Estimated factor matrix, shape ``(T, r)``.
    T0 : int
        Number of pre-treatment periods.

    Returns
    -------
    SystematicFit
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    T = y.shape[0]
    if factors.shape[0] != T:
        raise ValueError(
            f"factors has {factors.shape[0]} rows; expected T={T}."
        )
    Z = _design(factors)
    b0, *_ = np.linalg.lstsq(Z[:T0], y[:T0], rcond=None)
    b1, *_ = np.linalg.lstsq(Z[T0:], y[T0:], rcond=None)
    delta = b1 - b0
    tau_full = Z @ delta
    counterfactual = Z @ b0  # systematic untreated path a0 + lambda0' f_t
    tau = tau_full[T0:]
    return SystematicFit(
        a0=float(b0[0]),
        a1=float(b1[0]),
        lambda0=np.asarray(b0[1:], dtype=float),
        lambda1=np.asarray(b1[1:], dtype=float),
        kappa=float(b1[0] - b0[0]),
        tau=tau,
        tau_full=tau_full,
        counterfactual=counterfactual,
        att=float(np.mean(tau)) if tau.size else float("nan"),
    )


def _rss(Z: np.ndarray, y: np.ndarray) -> float:
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    return float(e @ e)


def chow_break_statistic(
    treated_outcome: np.ndarray, factors: np.ndarray, T0: int
) -> float:
    """Chow F-statistic for a structural break at the treatment date.

    Tests parameter stability of the treated-unit regression on ``(1, f_t)``
    at ``T0``. A diagnostic supporting the relevance of allowing treated
    factor loadings to change (Bai & Wang Sec 7.1).
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    T = y.shape[0]
    Z = _design(factors)
    k = Z.shape[1]
    denom_df = T - 2 * k
    if denom_df <= 0:
        return float("nan")
    rss_pooled = _rss(Z, y)
    rss_split = _rss(Z[:T0], y[:T0]) + _rss(Z[T0:], y[T0:])
    if rss_split <= 0:  # pragma: no cover - a perfect split fit yields tiny positive RSS, not exactly 0
        return float("nan")
    return float(((rss_pooled - rss_split) / k) / (rss_split / denom_df))
