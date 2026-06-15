"""Convex building blocks for the two-way relaxed SYNDES solver.

This module contains the small, side-effect-free numerical primitives the
annealing pipeline composes:

- :func:`solve_weights_global` — the convex QP w-step (given ``D``)
- :func:`compute_energy` — the relaxed objective
- :func:`compute_rmse_gap` — RMSE of the synthetic gap
- :func:`synthetic_paths` — synthetic treated/control series
- :func:`extract_weights` — normalized treated/control/contrast weights

These functions are kept independent of the annealing schedule so they can
be tested in isolation and reused outside of the simulated-annealing loop.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError

_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def solve_weights_global(
    Y: np.ndarray,
    D: np.ndarray,
    lam: float = 0.0,
) -> np.ndarray:
    """Solve the convex w-step QP for a fixed assignment ``D``.

    The problem minimizes the squared distance between the synthetic
    treated mean and the synthetic control mean subject to simplex
    constraints on each side and an optional ridge penalty.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix of shape ``(T, N)``.
    D : np.ndarray
        Binary assignment vector of shape ``(N,)`` with ``1`` for treated
        and ``0`` for control units.
    lam : float, optional
        Non-negative ridge penalty on the weight magnitudes.

    Returns
    -------
    np.ndarray
        Combined weight vector ``w`` of shape ``(N,)`` with treated entries
        on the simplex over ``D == 1`` and control entries on the simplex
        over ``D == 0``.

    Raises
    ------
    MlsynthEstimationError
        If the convex solve does not return an optimal status.
    """

    _, N = Y.shape
    treated = np.where(D == 1)[0]
    control = np.where(D == 0)[0]

    Y_T = Y[:, treated]
    Y_C = Y[:, control]

    w_T = cp.Variable(len(treated))
    w_C = cp.Variable(len(control))

    objective = cp.Minimize(
        cp.sum_squares(Y_T @ w_T - Y_C @ w_C)
        + lam * (cp.sum_squares(w_T) + cp.sum_squares(w_C))
    )
    constraints = [
        w_T >= 0,
        w_C >= 0,
        cp.sum(w_T) == 1,
        cp.sum(w_C) == 1,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    # This simplex-constrained LSQ is always feasible (uniform weights satisfy
    # it), but OSQP (first-order ADMM) can spuriously report infeasible/inaccurate
    # on large-magnitude, ill-scaled panels -- which would otherwise crash the
    # annealed solver mid-search. Fall back to the robust interior-point CLARABEL
    # solver before giving up.
    if problem.status not in _OPTIMAL_STATUSES or w_T.value is None:
        problem.solve(solver=cp.CLARABEL, verbose=False)

    if problem.status not in _OPTIMAL_STATUSES or w_T.value is None:
        raise MlsynthEstimationError(
            f"Relaxed weight QP failed with status: {problem.status}"
        )

    w = np.zeros(N)
    w[treated] = w_T.value
    w[control] = w_C.value
    return w


def compute_energy(
    Y: np.ndarray,
    D: np.ndarray,
    w: np.ndarray,
    lam: float,
) -> float:
    """Evaluate the relaxed two-way SYNDES energy for a given state.

    The energy is the mean-squared distance between the synthetic treated
    and control trajectories plus a ridge penalty on the weights.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    D : np.ndarray
        Binary assignment vector of shape ``(N,)``.
    w : np.ndarray
        Combined weight vector of shape ``(N,)``.
    lam : float
        Ridge penalty coefficient.

    Returns
    -------
    float
        Value of the relaxed objective at ``(D, w)``.
    """

    treated = D == 1
    control = D == 0

    mu_T = Y[:, treated] @ w[treated]
    mu_C = Y[:, control] @ w[control]

    return float(
        np.mean((mu_T - mu_C) ** 2)
        + lam * (np.sum(w[treated] ** 2) + np.sum(w[control] ** 2))
    )


def compute_rmse_gap(
    Y: np.ndarray,
    D: np.ndarray,
    w: np.ndarray,
) -> float:
    """RMSE of the synthetic gap implied by ``(D, w)``.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    D : np.ndarray
        Binary assignment vector of shape ``(N,)``.
    w : np.ndarray
        Combined weight vector of shape ``(N,)``.

    Returns
    -------
    float
        Root-mean-square of ``Y[:, treated] @ w[treated] - Y[:, control] @ w[control]``.
    """

    treated = D == 1
    control = D == 0

    mu_T = Y[:, treated] @ w[treated]
    mu_C = Y[:, control] @ w[control]

    return float(np.sqrt(np.mean((mu_T - mu_C) ** 2)))


def synthetic_paths(
    Y: np.ndarray,
    D: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the synthetic treated, synthetic control, and gap series.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    D : np.ndarray
        Binary assignment vector of shape ``(N,)``.
    w : np.ndarray
        Combined weight vector of shape ``(N,)``.

    Returns
    -------
    tuple of np.ndarray
        ``(mu_T, mu_C, mu_T - mu_C)`` with each array of shape ``(T,)``.
    """

    treated = D == 1
    control = D == 0

    mu_T = Y[:, treated] @ w[treated]
    mu_C = Y[:, control] @ w[control]
    return mu_T, mu_C, mu_T - mu_C


def extract_weights(D: np.ndarray, w: np.ndarray) -> Dict[str, np.ndarray]:
    """Decompose a combined weight vector into treated/control/contrast.

    Parameters
    ----------
    D : np.ndarray
        Binary assignment vector of shape ``(N,)``.
    w : np.ndarray
        Combined weight vector of shape ``(N,)``.

    Returns
    -------
    dict of str -> np.ndarray
        Mapping with keys ``"treated_weights"``, ``"control_weights"``, and
        ``"contrast_weights"``. Treated and control weights are normalized
        to sum to one over their respective groups.
    """

    treated_raw = D * w
    control_raw = (1 - D) * w

    treated_weights = treated_raw / (treated_raw.sum() + 1e-12)
    control_weights = control_raw / (control_raw.sum() + 1e-12)
    contrast_weights = (2 * D - 1) * w

    return {
        "treated_weights": treated_weights,
        "control_weights": control_weights,
        "contrast_weights": contrast_weights,
    }
