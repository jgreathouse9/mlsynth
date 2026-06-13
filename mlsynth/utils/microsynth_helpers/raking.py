"""Raking / GREG calibration weights for the MicroSynth panel method.

Robbins, Saunders & Kilmer (2017, JASA) and the ``microsynth`` R package
(Robbins & Davenport 2021, JSS v97i02) choose control weights by **raking
calibration** (``survey::calibrate(calfun = "raking")``): the weights are an
exponential tilt of base weights that *exactly* matches the treated group's
column totals (the calibration targets) on the matching variables -- time-
invariant covariates **and each pre-intervention outcome**, plus group dummies.

This is the Deville-Sarndal (1992) raking-ratio estimator, equivalent to
entropy balancing (Hainmueller 2012) on totals: with base weights :math:`d_i`,

    w_i = d_i * exp(g_i^T lambda),

where ``g_i`` is unit ``i``'s matching-variable row and ``lambda`` solves the
calibration equations ``sum_i w_i g_i = targets``. ``lambda`` is the minimizer
of the smooth convex potential

    F(lambda) = sum_i d_i * exp(g_i^T lambda) - targets^T lambda,

whose gradient ``X_C^T w - targets`` is exactly the calibration residual, so the
unconstrained minimizer enforces exact balance. When the matching matrix
contains an intercept (a column of ones), the weights automatically sum to the
treated count ``targets[intercept]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

_CLIP = 50.0  # exp overflow guard on the linear predictor


@dataclass(frozen=True)
class RakingSolution:
    """Raking calibration result: primal weights + dual multipliers."""

    w: np.ndarray
    dual_lambda: np.ndarray
    balance_residual: float
    converged: bool
    n_iter: int


def solve_raking_weights(
    X_C: np.ndarray,
    targets: np.ndarray,
    base_weight: Optional[np.ndarray] = None,
    *,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> RakingSolution:
    """Raking-calibration control weights that exactly balance ``targets``.

    Parameters
    ----------
    X_C : np.ndarray
        Control matching matrix, shape ``(n_C, d)`` -- one row per control unit,
        one column per matching variable (typically an intercept, the covariates,
        and each pre-intervention outcome).
    targets : np.ndarray
        Calibration targets, shape ``(d,)`` -- the treated group's column totals
        of the same matching variables.
    base_weight : np.ndarray, optional
        Base weights ``d_i`` (length ``n_C``); defaults to all ones (the raking
        ratio tilts these to hit the targets).
    max_iter, tol : int, float
        L-BFGS-B controls on the dual potential.

    Returns
    -------
    RakingSolution
        ``w`` (length ``n_C``, strictly positive), the dual ``lambda``, the max
        absolute balance residual, convergence flag and iteration count.
    """
    X_C = np.asarray(X_C, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n_C, d = X_C.shape
    base = np.ones(n_C) if base_weight is None else np.asarray(base_weight, dtype=float)

    def potential_and_grad(lmbda: np.ndarray) -> Tuple[float, np.ndarray]:
        z = np.clip(X_C @ lmbda, -_CLIP, _CLIP)
        w = base * np.exp(z)
        f = float(w.sum() - targets @ lmbda)
        grad = X_C.T @ w - targets
        return f, grad

    res = minimize(
        potential_and_grad, np.zeros(d), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol, "ftol": tol},
    )
    w = base * np.exp(np.clip(X_C @ res.x, -_CLIP, _CLIP))
    residual = float(np.max(np.abs(X_C.T @ w - targets)))
    return RakingSolution(
        w=w, dual_lambda=res.x, balance_residual=residual,
        converged=bool(res.success), n_iter=int(res.nit),
    )
