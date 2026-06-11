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
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    if B.ndim != 2:
        raise ValueError("B must be a 2-D (m, J) matrix.")
    m, J = B.shape
    if A.shape[0] != m:
        raise ValueError(f"len(A)={A.shape[0]} must equal B's row count {m}.")
    if J == 0:
        raise ValueError("B has no columns: at least one donor is required.")

    def _finish(w, pivots, converged):
        w = np.maximum(np.asarray(w, dtype=float), 0.0)
        total = w.sum()
        if total > 0:
            w = w / total
        if return_info:
            return w, {"pivots": int(pivots), "converged": bool(converged)}
        return w

    if J == 1:
        return _finish(np.array([1.0]), 0, True)

    if max_iter is None:
        max_iter = 50 * J
    G = B.T @ B                                   # (J, J) Gram
    c = B.T @ A

    # Feasible start: a valid warm start (on the simplex) seeds the active set;
    # otherwise the uniform point.
    w = None
    if warm_start is not None:
        ws = np.asarray(warm_start, dtype=float).ravel()
        if (ws.shape == (J,) and np.all(np.isfinite(ws))
                and ws.min() >= -tol and abs(ws.sum() - 1.0) <= 1e-6):
            w = np.clip(ws, 0.0, None)
            w = w / w.sum()
    if w is None:
        w = np.full(J, 1.0 / J)
    active = w <= tol                             # variables pinned at the 0 bound

    pivots = 0
    converged = False
    for _ in range(max_iter):
        free = np.where(~active)[0]
        if free.size == 0:                        # never pin every variable
            active[int(np.argmax(w))] = False
            free = np.where(~active)[0]
        BF = B[:, free]
        nF = free.size
        # Equality-constrained LSQ on the free set: min ||BF wF - A||^2 s.t.
        # 1' wF = 1, solved on the null space of 1' so we factor BF *directly*
        # rather than the normal equations BF'BF (which would square the
        # condition number). lstsq tolerates a rank-deficient BF (|free| > T0,
        # collinear donors), so no epsilon-I is needed.
        if nF == 1:
            wF = np.array([1.0])
        else:
            wF0 = np.full(nF, 1.0 / nF)           # particular solution, 1'wF0 = 1
            Z = np.zeros((nF, nF - 1))            # null-space basis of 1'
            Z[:nF - 1] = np.eye(nF - 1)
            Z[nF - 1] = -1.0
            v, *_ = np.linalg.lstsq(BF @ Z, A - BF @ wF0, rcond=None)
            wF = wF0 + Z @ v

        if wF.min() >= -tol:
            # Full step to the free-set optimum.
            w = np.zeros(J)
            w[free] = np.maximum(wF, 0.0)
            g = G @ w - c
            nu = float(g[free].mean())            # sum-to-one multiplier (g_i == nu on free)
            if active.any():
                # Dual feasibility: a pinned variable is optimal iff its reduced
                # gradient g_i >= nu. Release the most-violating one if any.
                ai = np.where(active)[0]
                reduced = g[ai] - nu
                scale = 1.0 + float(np.max(np.abs(g)))
                k = int(np.argmin(reduced))
                if reduced[k] < -tol * scale:
                    active[ai[k]] = False
                    pivots += 1
                    continue
            converged = True
            break

        # Blocking constraint: line-search toward wF until a free variable hits 0.
        cur = w[free]
        direction = wF - cur
        blocking = direction < -tol
        ratios = np.where(blocking, cur / np.maximum(-direction, tol), np.inf)
        step = min(1.0, float(ratios.min()))
        cur = cur + step * direction
        w = np.zeros(J)
        w[free] = np.maximum(cur, 0.0)
        hit = free[cur <= tol]
        if hit.size == 0:                         # numerical safety
            hit = free[[int(np.argmin(cur))]]
        active[hit] = True
        pivots += 1

    return _finish(w, pivots, converged)
