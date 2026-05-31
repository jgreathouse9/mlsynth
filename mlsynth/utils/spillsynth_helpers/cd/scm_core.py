"""Per-unit demeaned simplex SCM for Cao-Dowd 2023.

Implements equation (2) of Cao & Dowd (2023): for each unit ``i``, fit
SCM weights ``b_i`` on the simplex (non-negative, sum to 1, with
``b_{ii} = 0``) plus a free intercept ``a_i``, against the demeaned
pre-treatment outcomes. Stacking these for all ``i`` gives the
``(N, N)`` weight matrix ``B`` and the length-``N`` intercept vector
``a`` that drive the spillover estimator.

The demeaning is the Ferman-Pinto (2021) modification: it ensures the
estimator is asymptotically unbiased when the pre-treatment fit is
imperfect, which is the regime Cao-Dowd assume throughout the paper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from ....exceptions import MlsynthEstimationError


def fit_demeaned_sc(
    Y_block: np.ndarray, *, solver: Optional[str] = None,
) -> Tuple[float, np.ndarray]:
    """Fit a single demeaned simplex SCM (eq. 2 of Cao-Dowd 2023).

    Parameters
    ----------
    Y_block : np.ndarray
        Shape ``(N, T)``. Row 0 is the treated unit for this fit; the
        remaining ``N - 1`` rows are donors.
    solver : str, optional
        cvxpy solver to use. Defaults to ``"CLARABEL"``.

    Returns
    -------
    a_hat : float
        Estimated intercept.
    b_full : np.ndarray
        Length-``N`` weight vector with ``b_full[0] = 0`` (the treated
        unit's own weight) and the remaining entries on the simplex.
    """
    if Y_block.ndim != 2 or Y_block.shape[0] < 2:
        raise MlsynthEstimationError(
            "fit_demeaned_sc requires Y_block with at least 2 rows."
        )
    N, T = Y_block.shape
    y_t = Y_block[0]
    Y_u = Y_block[1:]
    y_t_mean = float(y_t.mean())
    Y_u_mean = Y_u.mean(axis=1)                              # (N - 1,)
    y_d = y_t - y_t_mean                                     # (T,)
    X_d = (Y_u.T - Y_u_mean[None, :])                        # (T, N - 1)

    w = cp.Variable(N - 1, nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(y_d - X_d @ w)),
        [cp.sum(w) == 1],
    )
    try:
        problem.solve(solver=solver or "CLARABEL")
    except Exception as exc:                                # pragma: no cover
        raise MlsynthEstimationError(
            f"SPILLSYNTH/cd: simplex QP failed ({type(exc).__name__})."
        ) from exc
    if w.value is None:
        raise MlsynthEstimationError(
            f"SPILLSYNTH/cd: simplex QP returned no solution "
            f"(status={problem.status!r})."
        )
    b_hat = np.clip(np.asarray(w.value).flatten(), 0.0, None)
    s = b_hat.sum()
    if s <= 0:                                              # pragma: no cover
        raise MlsynthEstimationError(
            "SPILLSYNTH/cd: degenerate weights (all zero)."
        )
    b_hat /= s
    a_hat = y_t_mean - float((Y_u_mean * b_hat).sum())
    return a_hat, np.r_[0.0, b_hat]


def fit_leave_one_out_sc(
    Y_pre: np.ndarray, *, solver: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out demeaned SCM for every unit.

    For each ``i``, swap unit ``i`` into row 0 of ``Y_pre`` and call
    :func:`fit_demeaned_sc`. Re-permute the returned weight vector so
    that ``B[i, j]`` always refers to the original column ``j``.

    Parameters
    ----------
    Y_pre : np.ndarray
        Shape ``(N, T0)``.
    solver : str, optional
        cvxpy solver to use.

    Returns
    -------
    a : np.ndarray
        Length-``N`` intercepts.
    B : np.ndarray
        Shape ``(N, N)`` with ``B[i, i] == 0`` and ``B[i, :].sum() == 1``
        for all ``i``.
    """
    N = Y_pre.shape[0]
    a = np.zeros(N)
    B = np.zeros((N, N))
    for i in range(N):
        if i == 0:
            Y_temp = Y_pre
        else:
            Y_temp = Y_pre.copy()
            Y_temp[[0, i]] = Y_temp[[i, 0]]
        a_i, b_temp = fit_demeaned_sc(Y_temp, solver=solver)
        a[i] = a_i
        if i == 0:
            B[0] = b_temp
        else:
            # b_temp is indexed in the swapped ordering: position 0 is
            # the "treated" (which is original i, weight 0), position i
            # is the original 0. Other positions match.
            b_rear = np.zeros(N)
            b_rear[i] = b_temp[0]                            # = 0 by construction
            b_rear[0] = b_temp[i]
            others = [k for k in range(N) if k not in (0, i)]
            b_rear[others] = b_temp[others]
            B[i] = b_rear
    return a, B
