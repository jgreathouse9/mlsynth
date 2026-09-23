"""DROSC point estimand and its simplex sub-solver (Koo & Guo 2026, helpers.R
``sc`` and ``DRoSC`` with ``Inference = FALSE``). Ported value-for-value against
the authors' ``limSolve::lsei``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np


def sc_weights(Y0: np.ndarray, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard simplex synthetic control on pre-treatment outcomes:
    ``min ||Y0 - X0 w||^2`` s.t. ``w >= 0``, ``sum w = 1`` (helpers.R ``sc``).

    Returns ``(weights, residuals)``.
    """
    Y0 = np.asarray(Y0, float).ravel()
    X0 = np.asarray(X0, float)
    J = X0.shape[1]
    w = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(X0 @ w - Y0)), [cp.sum(w) == 1]).solve(
        solver=cp.CLARABEL)
    if w.value is None:  # pragma: no cover - CLARABEL failure on a degenerate panel
        raise RuntimeError("simplex SC solve failed")
    weights = np.clip(np.asarray(w.value, float).ravel(), 0.0, None)
    return weights, Y0 - X0 @ weights


def _band_ls(a: np.ndarray, ey: float, Sig: np.ndarray,
             lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """min (a'b - ey)^2 s.t. b on the simplex, lo <= Sig b <= hi."""
    N = a.shape[0]
    b = cp.Variable(N, nonneg=True)
    cons = [cp.sum(b) == 1, Sig @ b >= lo, Sig @ b <= hi]
    cp.Problem(cp.Minimize(cp.square(a @ b - ey)), cons).solve(solver=cp.CLARABEL)
    if b.value is None:
        raise RuntimeError("moment-band solve infeasible")
    return np.clip(np.asarray(b.value, float).ravel(), 0.0, None)


def drosc_point(Y0: np.ndarray, Y1: np.ndarray, X0: np.ndarray, X1: np.ndarray,
                robustness_lambda: float = 0.0, nu: float = 0.01, eta: float = 0.01,
                slack_rate: float = 0.5) -> Tuple[float, Optional[np.ndarray]]:
    """DROSC worst-case point estimand (helpers.R ``DRoSC``, ``Inference=FALSE``).

    Chooses donor weights ``b`` minimising the post-treatment fit
    ``(mean(X1) b - mean(Y1))^2`` subject to the simplex and the pre-treatment
    moment-compatibility band ``|Sigmahat b - gammahat|_inf <= lambda + slack``,
    where the data-driven slack shrinks at rate ``log(max(T0,N))^c / sqrt(T0)``.
    The ``nu`` term is inflated until the band is feasible (the R feasibility
    retry).

    Minimising the squared post-treatment gap is minimising the squared effect,
    so ``b`` is the adversarial weighting driving the effect closest to zero and
    ``tau`` is the projection of the origin onto the effects the band admits.
    This is the computational form of the paper's max-min problem, not its
    definition; Theorem 1 is what makes the two the same.

    Returns ``(tau, beta)``. ``tau = mean(Y1) - mean(X1)'beta`` is unique even
    when ``beta`` is not (the post objective is rank one), which is the
    non-uniqueness the method exists to tolerate: it declines to report a
    weighting and still reports an effect.
    """
    Y0 = np.asarray(Y0, float).ravel()
    Y1 = np.asarray(Y1, float).ravel()
    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)
    T0, N = X0.shape
    _, resid = sc_weights(Y0, X0)
    sig = float(np.sqrt(np.var(resid, ddof=1)))
    Sig = X0.T @ X0 / T0                       # Sigmahat
    gam = (Y0[:, None] * X0).mean(0)           # gammahat
    EY1 = Y1.mean()
    EX1 = X1.mean(0)
    maxnorm = float(np.max(np.sqrt((X0 ** 2).mean(0))))
    beta: Optional[np.ndarray] = None
    while beta is None:
        thres = (robustness_lambda
                 + np.log(max(T0, N)) ** slack_rate / np.sqrt(T0)
                 * (nu * sig * maxnorm + eta * robustness_lambda))
        try:
            beta = _band_ls(EX1, EY1, Sig, gam - thres, gam + thres)
        except Exception:
            beta = None
        if nu <= 0:
            break
        nu *= 1.25
    tau = float(EY1 - EX1 @ beta) if beta is not None else float("nan")
    return tau, beta
