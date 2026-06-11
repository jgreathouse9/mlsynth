"""Exact simplex-constrained least squares via a primal active-set method.

Pure-NumPy, warm-startable, and PSD-safe: the free-set subproblem is solved with
``lstsq``, so a rank-deficient donor Gram (``J > T0`` or collinear donors) is
handled without an epsilon-I fudge. Intended as a drop-in for the cvxpy
``simplex_qp`` that avoids the per-call canonicalisation overhead in the hot
conformal / market-selection loops.

Status: TDD scaffold (PR #58). The correctness contract -- cvxpy parity, a
solver-independent KKT certificate, and a fuzzed differential test -- is pinned
in ``tests/test_simplex_active_set.py``. The solver is implemented *against*
that red harness; until then this raises ``NotImplementedError`` so the tests
fail for the right reason.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def solve_simplex_qp(
    B: np.ndarray,
    A: np.ndarray,
    *,
    warm_start: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: Optional[int] = None,
    return_info: bool = False,
):
    """Minimise ``||A - B w||^2`` over ``w >= 0, sum(w) == 1``.

    Parameters
    ----------
    B : np.ndarray, shape (m, J)
        Donor / design matrix (e.g. pre-period donor outcomes).
    A : np.ndarray, shape (m,)
        Target vector (e.g. the treated unit's pre-period outcomes).
    warm_start : np.ndarray, shape (J,), optional
        A feasible initial weight vector (e.g. the solution of a neighbouring
        problem in a conformal / market-selection sweep) used to seed the
        active set. Must lie on the simplex; ignored if infeasible.
    tol : float
        KKT / feasibility tolerance.
    max_iter : int, optional
        Cap on active-set pivots; defaults to a small multiple of ``J``.
    return_info : bool
        If ``True`` also return a diagnostics dict (``iterations``, ``pivots``,
        ``converged``) so the performance tests can assert bounded work.

    Returns
    -------
    np.ndarray, shape (J,)
        The optimal weights, or ``(w, info)`` when ``return_info=True``.
    """
    raise NotImplementedError(
        "active-set simplex QP is implemented against the red test harness in "
        "tests/test_simplex_active_set.py"
    )
