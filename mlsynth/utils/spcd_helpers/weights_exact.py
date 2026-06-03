"""Exact convex-QP weights (Algorithm 1 of the paper).

Implements Eq. (6) of the paper (page 7), the convex quadratic program
used by "Algorithm 1: Synthetic principal component Design" once the
sign vector ``gamma`` is fixed by the iteration::

    Eq. (6):
        min_{w_i >= 0}    (1/T) sum_t (
                              sum_{i: gamma(i)=1}  w_i Y_{it}
                            - sum_{i: gamma(i)=-1} w_i Y_{it}
                          )^2 + sigma sum_i w_i^2

        s.t.   w_i >= 0  for all i in [N],
               sum_{gamma(i)=1}  w_i = 1,
               sum_{gamma(i)=-1} w_i = 1.

This is a convex QP and is solved exactly via cvxpy. ``sigma`` plays
the role of the ridge coefficient ``alpha`` from Eq. (2).

The paper notes that in their experiments they prefer the closed-form
approximation in Eq. (9) (see ``weights_empirical.py``), but Eq. (6)
is the original optimization problem and is provided here for users
who want the exact Algorithm 1 weights.

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError


def exact_weights(
    Y_pre: np.ndarray,
    y_star: np.ndarray,
    sigma: float,
    solver: Optional[Any] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Solve Eq. (6) via cvxpy and return the signed weight vector.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(T_pre, N)`` (mlsynth
        convention; the paper's ``Y in R^{N x T}`` corresponds to
        ``Y_pre.T``).
    y_star : np.ndarray
        Length-N sign vector ``gamma in {-1, +1}^N`` from the iteration.
        Used to identify the treated and control groups in the
        constraints of Eq. (6).
    sigma : float
        Ridge coefficient ``sigma`` from Eq. (6). Plays the same role
        as ``alpha`` in Eq. (2).
    solver : optional
        Optional cvxpy solver name or object passed through to
        ``cp.Problem.solve``.
    verbose : bool
        Whether to print solver progress.

    Returns
    -------
    w : np.ndarray
        Length-N signed weight vector ``(2 * indicator(gamma=1) - 1) *
        w_raw`` so that downstream consumers see one signed vector per
        unit. The unsigned positive weights ``w_raw[i]`` sum to 1 within
        each group.

    Raises
    ------
    MlsynthEstimationError
        If cvxpy is not installed, or if the solver fails to find an
        optimal solution.
    """

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise MlsynthEstimationError(
            "weights='exact' requires cvxpy. Install with `pip install cvxpy`."
        ) from exc

    T_pre, N = Y_pre.shape
    treated = (y_star > 0)
    control = (y_star < 0)

    if not treated.any() or not control.any():
        raise MlsynthEstimationError(
            "Exact-weight QP requires both treated and control groups to be non-empty."
        )

    w = cp.Variable(N, nonneg=True)

    treated_series = Y_pre @ cp.multiply(w, treated.astype(float))
    control_series = Y_pre @ cp.multiply(w, control.astype(float))
    contrast = treated_series - control_series

    objective = cp.Minimize(
        (1.0 / T_pre) * cp.sum_squares(contrast) + sigma * cp.sum_squares(w)
    )
    constraints = [
        cp.sum(cp.multiply(w, treated.astype(float))) == 1,
        cp.sum(cp.multiply(w, control.astype(float))) == 1,
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, verbose=verbose)
    except Exception as exc:
        raise MlsynthEstimationError(f"Eq. (6) QP failed to solve: {exc}") from exc

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise MlsynthEstimationError(
            f"Eq. (6) QP did not reach an optimal status (status={problem.status})."
        )

    w_pos = np.asarray(w.value, dtype=float)
    sign_vec = np.where(treated, 1.0, np.where(control, -1.0, 0.0))
    return sign_vec * w_pos
