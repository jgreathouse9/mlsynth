"""HiGHS, through SciPy, as the independent reference for the study's own solvers.

The estimator path (``oracle``, ``lp``, ``solver_pairfree``) solves its linear
programmes in NumPy and imports nothing from ``scipy.optimize``. This module is
what the tests and the solver comparisons check those solutions against.

* ``wlad`` is the weighted least-absolute-deviation fit Remark 1 of Lv and Li
  (2026) sends to ``quantreg::rq``: ``argmin_beta sum_k c_k |y_k - D_k beta|``,
  solved as the dual programme ``max y'd s.t. D'd = 0, -c <= d <= c`` with the
  coefficients read off the equality multipliers.
* ``simplex_wlad`` is the same loss over the simplex, Section 2.4's programme.

HiGHS returned status 15 (model status Unknown, primal feasible) on one
programme of the first 500-replication run, so both retry with the dual simplex
without presolve and then the interior-point method; ``LP_FALLBACKS["retry"]``
counts the retries.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

_HIGHS_ATTEMPTS = (
    dict(method="highs"),
    dict(method="highs-ds", options=dict(presolve=False)),
    dict(method="highs-ipm"),
)
LP_FALLBACKS = {"retry": 0}


def _linprog_retrying(*args, **kwargs):
    """``linprog`` over ``_HIGHS_ATTEMPTS``; the first result with status 0, else the last."""
    for k, attempt in enumerate(_HIGHS_ATTEMPTS):
        res = linprog(*args, **kwargs, **attempt)
        if res.status == 0:
            LP_FALLBACKS["retry"] += k > 0
            return res
    return res


def wlad(y: np.ndarray, D: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``argmin_beta sum_k c_k |y_k - D_k beta|``; raises when no HiGHS setting solves it."""
    res = _linprog_retrying(-y, A_eq=D.T, b_eq=np.zeros(D.shape[1]),
                            bounds=np.column_stack([-c, c]))
    if res.status != 0:
        raise RuntimeError(f"weighted LAD failed: {res.message}")
    return -np.asarray(res.eqlin.marginals, dtype=float)


def simplex_wlad(y: np.ndarray, D: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``argmin_w sum_k c_k |y_k - D_k w|`` over the simplex.

    Dual: ``max y'd + mu s.t. D'd + mu 1 <= 0, -c <= d <= c``; the weights are
    the negated multipliers of the inequality rows.
    """
    S = D.shape[1]
    if S == 1:
        return np.ones(1)
    cost = -np.r_[y, 1.0]
    A_ub = np.column_stack([D.T, np.ones(S)])
    bounds = np.vstack([np.column_stack([-c, c]), [-np.inf, np.inf]])
    res = _linprog_retrying(cost, A_ub=A_ub, b_ub=np.zeros(S), bounds=bounds)
    if res.status != 0:
        raise RuntimeError(f"simplex LAD failed: {res.message}")
    w = np.clip(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0, None)
    return w / w.sum()
