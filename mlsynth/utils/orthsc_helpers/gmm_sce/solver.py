"""One-step GMM-SCE control-weight solver (Fry 2024, *J. Econometrics* 244).

Ports the ``GMMSC()`` routine of the author's ``GMM-SCE.R``: synthetic-control
weights on the unit simplex are estimated by a General Method of Moments that
uses the *instrument* units (units excluded from the control pool) as
instruments for the control outcomes, fixing the attenuation bias that the naive
OLS regression of the treated unit on the controls suffers (Ferman and Pinto
2021). The pre-treatment data are normalized to unit variance in each time period
before the solve, exactly as in the reference.

All arrays here are time-major (``T0`` rows) to mirror the reference; the
pipeline transposes from mlsynth's unit-major convention.
"""
from __future__ import annotations

from typing import Dict

import cvxpy as cp
import numpy as np

from ....exceptions import MlsynthEstimationError


def _row_normalize(stacked: np.ndarray) -> np.ndarray:
    """Scale each row (time period) of ``stacked`` to unit sample variance.

    Mirrors ``divisor <- sqrt(apply(big.dataframe, 1, var))`` in the reference,
    where ``var`` is R's sample (``ddof=1``) variance across the columns of a row.
    """
    divisor = np.sqrt(np.var(stacked, axis=1, ddof=1))
    if not np.all(np.isfinite(divisor)) or np.any(divisor <= 0):
        raise MlsynthEstimationError(
            "GMM-SCE normalization failed: a pre-treatment period has zero or "
            "non-finite cross-unit variance (constant across all units).")
    return stacked / divisor[:, None]


def gmm_sc_weights(
    pre_y0: np.ndarray,
    pre_yj: np.ndarray,
    pre_yk: np.ndarray,
    *,
    include_constant: bool = True,
    solver: str = "CLARABEL",
) -> Dict[str, object]:
    """One-step GMM-SCE control weights and the over-identification J-statistic.

    Parameters
    ----------
    pre_y0
        Treated unit's pre-treatment outcomes, shape ``(T0,)``.
    pre_yj
        Control units' pre-treatment outcomes, shape ``(T0, J)`` (time x control).
    pre_yk
        Instrument units' pre-treatment outcomes, shape ``(T0, K)`` (time x
        instrument), *without* the constant instrument.
    include_constant
        If true, prepend a constant column to the instruments (the mean-matching
        moment ``meanfit`` of the reference).
    solver
        cvxpy conic solver for the simplex-constrained quadratic program.

    Returns
    -------
    dict
        ``weights`` (J,), ``jstatistic`` (float), ``n_instruments`` (the K used,
        counting the constant), and ``status`` (the cvxpy solve status).
    """
    y0 = np.asarray(pre_y0, dtype=float).ravel()
    YJ = np.atleast_2d(np.asarray(pre_yj, dtype=float))
    YK = np.atleast_2d(np.asarray(pre_yk, dtype=float))
    if YJ.shape[0] != y0.shape[0]:
        YJ = YJ.T
    if YK.shape[0] != y0.shape[0]:
        YK = YK.T
    T0, J = YJ.shape
    if T0 < 2:
        raise MlsynthEstimationError(
            "GMM-SCE needs at least two pre-treatment periods to normalize.")

    if include_constant:
        YK = np.hstack([np.ones((T0, 1)), YK])

    # Normalize each time period to unit variance across all units (the constant
    # column is included in the variance, exactly as in the reference).
    stacked = np.hstack([y0[:, None], YJ, YK])
    scaled = _row_normalize(stacked)
    y0s = scaled[:, 0]
    YJs = scaled[:, 1:1 + J]
    YKs = scaled[:, 1 + J:]

    w = cp.Variable(J)
    residual = y0s - YJs @ w
    moments = YKs.T @ residual                          # g, the (K,) moment vector
    objective = cp.Minimize(cp.sum_squares(moments))    # one-step GMM, A = I_K
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError as exc:  # pragma: no cover - solver fallback
        prob.solve(solver="SCS")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise MlsynthEstimationError(
                f"GMM-SCE weight solve failed: {exc}") from exc
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):  # pragma: no cover - defensive: non-optimal solver status
        raise MlsynthEstimationError(
            f"GMM-SCE weight solve did not converge (status={prob.status}).")

    weights = np.clip(np.asarray(w.value, dtype=float).ravel(), 0.0, None)
    s = weights.sum()
    if s > 0:
        weights = weights / s                            # numerical simplex tidy-up
    g = YKs.T @ (y0s - YJs @ weights)
    jstatistic = float(g @ g) / T0
    return {
        "weights": weights,
        "jstatistic": jstatistic,
        "n_instruments": int(YKs.shape[1]),
        "status": prob.status,
    }
