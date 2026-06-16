"""Regularized nuisance estimation for OSC: the control weights (delta) and the
orthogonalization weights (eta).

Both solve a two-stage penalized program -- an LP that finds the smallest moment
slack ``lambda`` achievable, then a min-norm solve among the values whose sample
moments sit within that (log-inflated) slack. This drives the partially
identified nuisance to a unique element of the identified set. ``delta`` is
simplex-constrained (a synthetic control); ``eta`` is normalized so its last
(post-moment) entry is one, making the orthogonalized moments identify the ATT.

Faithful to the reference ``RegularizedEstimate.R`` (EstimateDelta /
EstimateNormalizedEta + their lambda tuners), modulo clean array orientation:
``pre_y0`` is ``(T0,)``, ``pre_yj`` ``(J, T0)``, ``Z`` ``(Q, T0)``.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from mlsynth.exceptions import MlsynthEstimationError


def _check(y0, YJ, Z):
    y0 = np.asarray(y0, float).ravel()
    YJ = np.atleast_2d(np.asarray(YJ, float))
    Z = np.atleast_2d(np.asarray(Z, float))
    J, T0 = YJ.shape
    if y0.shape[0] != T0:
        raise MlsynthEstimationError(
            f"pre_y0 has {y0.shape[0]} periods but pre_yj has {T0}.")
    if Z.size and Z.shape[1] != T0:
        raise MlsynthEstimationError(
            f"Z has {Z.shape[1]} periods but the pre-period is {T0}.")
    if Z.shape[0] == 0 or Z.size == 0:
        raise MlsynthEstimationError(
            "at least one instrument unit is required for OSC.")
    return y0, YJ, Z, J, T0


def _time_scale(stack: np.ndarray) -> np.ndarray:
    """Divide every series at each period by that period's cross-series sd
    (R's ``apply(cbind(...), 1, var)`` rescaling). Zero-variance periods pass
    through unscaled."""
    sd = stack.std(axis=0, ddof=1)
    sd = np.where(sd == 0.0, 1.0, sd)
    return stack / sd


def _with_constant(Z: np.ndarray, T0: int, include_constant: bool) -> np.ndarray:
    return np.vstack([Z, np.ones(T0)]) if include_constant else Z


def estimate_delta(pre_y0, pre_yj, Z, scaled: bool = True,
                   include_constant: bool = True, T1: int | None = None):
    """Regularized IV control weights on the simplex.

    Returns ``dict`` with ``delta`` (J,) and ``lambda_`` (the inflated slack).
    """
    y0, YJ, Z, J, T0 = _check(pre_y0, pre_yj, Z)
    Z = _with_constant(Z, T0, include_constant)
    if scaled:
        sd = np.vstack([y0[None, :], YJ, Z]).std(axis=0, ddof=1)
        sd = np.where(sd == 0.0, 1.0, sd)
        y0s, YJs, Zs = y0 / sd, YJ / sd, Z / sd
    else:
        y0s, YJs, Zs = y0, YJ, Z
    K = Zs.shape[0]
    M = Zs @ YJs.T / np.sqrt(T0)        # (K, J)
    m0 = Zs @ y0s / np.sqrt(T0)         # (K,)

    # Stage 1: smallest achievable max-moment slack on the simplex.
    d = cp.Variable(J)
    lam = cp.Variable(nonneg=True)
    simplex = [cp.sum(d) == 1, d >= 0, d <= 1]
    cp.Problem(cp.Minimize(lam),
               simplex + [M @ d - m0 <= lam, m0 - M @ d <= lam]).solve(solver=cp.CLARABEL)
    n = min(T0, T1) if T1 is not None else T0
    log_factor = (np.log(n) * np.log(J) / np.log(K)) if (J > 1 and K > 1) else 1.0
    lam_val = float(lam.value) * log_factor

    # Stage 2: min-norm simplex weights within the inflated slack.
    d2 = cp.Variable(J)
    cp.Problem(cp.Minimize(cp.sum_squares(d2)),
               [cp.sum(d2) == 1, d2 >= 0, d2 <= 1,
                M @ d2 - m0 <= lam_val, m0 - M @ d2 <= lam_val]).solve(solver=cp.CLARABEL)
    delta = np.asarray(d2.value, float).ravel()
    return {"delta": delta, "lambda_": lam_val}


def estimate_eta(pre_y0, pre_yj, post_y0, post_yj, Z, scaled: bool = True,
                 include_constant: bool = True):
    """Regularized, normalized orthogonalization weights ``eta`` (last entry 1).

    Returns ``dict`` with ``eta`` (Q+1,) and ``lambda_``.
    """
    y0, YJ, Z, J, T0 = _check(pre_y0, pre_yj, Z)
    py0 = np.asarray(post_y0, float).ravel()
    PYJ = np.atleast_2d(np.asarray(post_yj, float))
    T1 = PYJ.shape[1]
    if PYJ.shape[0] != J or py0.shape[0] != T1:
        raise MlsynthEstimationError("post-period control/treated shapes are inconsistent.")
    Z = _with_constant(Z, T0, include_constant)
    if scaled:
        sd = np.vstack([y0[None, :], YJ, Z]).std(axis=0, ddof=1)
        sd = np.where(sd == 0.0, 1.0, sd)
        YJs, Zs = YJ / sd, Z / sd
        sdp = np.vstack([py0[None, :], PYJ]).std(axis=0, ddof=1)
        sdp = np.where(sdp == 0.0, 1.0, sdp)
        PYJs = PYJ / sdp
    else:
        YJs, Zs, PYJs = YJ, Z, PYJ
    gdelta = np.vstack([Zs @ YJs.T / T0, PYJs.mean(axis=1)[None, :]])  # (Q+1, J)
    K = gdelta.shape[0]
    GT = gdelta.T                          # (J, Q+1)

    eta = cp.Variable(K)
    lam = cp.Variable(nonneg=True)
    cp.Problem(cp.Minimize(lam),
               [eta[K - 1] == 1, GT @ eta <= lam, -(GT @ eta) <= lam]).solve(solver=cp.CLARABEL)
    lam_val = float(lam.value) * np.log(min(T0, T1))

    eta2 = cp.Variable(K)
    cp.Problem(cp.Minimize(cp.sum_squares(eta2)),
               [eta2[K - 1] == 1, GT @ eta2 <= lam_val,
                -(GT @ eta2) <= lam_val]).solve(solver=cp.CLARABEL)
    return {"eta": np.asarray(eta2.value, float).ravel(), "lambda_": lam_val}
