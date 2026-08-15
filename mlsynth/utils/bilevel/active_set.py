"""Exact simplex-constrained least squares via a primal active-set method.

Pure-NumPy, warm-startable, and PSD-safe: the free-set subproblem is a
rank-revealing QR least squares (LAPACK ``gelsy``), so a rank-deficient donor
Gram (``J > T0`` or collinear donors) is handled without an epsilon-I fudge.
Intended as a drop-in for the cvxpy ``simplex_qp`` that avoids the per-call
canonicalisation overhead in the hot conformal / market-selection loops.

``gelsy`` is called directly (see :func:`_gelsy_lstsq`) instead of through
``scipy.linalg.lstsq``. The free sets here are small enough that the wrapper's
fixed per-call cost exceeded the LAPACK work it wrapped.

A cold solve seeds itself. Starting from the uniform point, the active set has
to shed one donor per pivot until only the support is left, so its work scales
with the *pool* and not with the support it ends on: on factor panels the pivot
count runs 0.6 to 0.9 times ``J`` from ``J = 20`` to ``J = 320``, while the
support grows 7 to 43. A Gram-collapsed FISTA warm start
(:func:`mlsynth.utils.bilevel.accelerate.fista_warm_start`) names that support
up front and the same pivot counts drop to 0 or 1. So for a pool of at least
``ACCEL_MIN_DONORS``, with no warm start from the caller, the seed is computed
here -- once, for every caller -- instead of at a call site. It is speed only:
the exact active set still determines the weights, and a seed it cannot use is
discarded. Pass ``accelerate=False`` to force the cold path.

The correctness contract -- cvxpy parity, a solver-independent KKT certificate,
and a fuzzed differential test -- is pinned in
``tests/test_simplex_active_set.py``; the LAPACK call's bit-identity to the
scipy path is pinned in ``tests/test_active_set_lapack.py``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import get_lapack_funcs

from .accelerate import ACCEL_MIN_DONORS, fista_warm_start

# Workspace size and LAPACK handle per free-set shape. The active set solves a
# sequence of small systems whose shapes repeat across pivots, units and
# placebo draws, so the query is done once per shape and reused.
_GELSY_CACHE: Dict[Tuple[int, int], Tuple[object, int]] = {}


def _gelsy_lstsq(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``lstsq(M, b, lapack_driver="gelsy")[0]``, without the wrapper overhead.

    Bit-identical to :func:`scipy.linalg.lstsq` under the same driver -- it is
    the same LAPACK routine on the same bytes. What is skipped is per-call
    Python work that dominates at these sizes: ``asarray_chkfinite`` over both
    arguments, the generic argument validation, and a fresh workspace query
    every call. On the Prop 99 free sets that is 1.7x at the first pivot and
    4.3x by the time the support has formed, since the wrapper's cost is fixed
    while LAPACK's shrinks with the system.

    ``b`` is copied into a ``max(m, n)``-tall buffer because gelsy returns the
    solution in that argument's storage and needs room for the longer of the
    two; ``M`` is passed with ``overwrite_a=0`` so the caller's array survives.
    """
    m, n = M.shape
    key = (m, n)
    entry = _GELSY_CACHE.get(key)
    if entry is None:
        gelsy, = get_lapack_funcs(("gelsy",), (M, b))
        minmn = min(m, n)
        # LAPACK's documented floor for nrhs = 1, times four so the blocked
        # path has room; the array is tiny at synthetic-control sizes.
        lwork = max(minmn + 3 * n + 1, 2 * minmn + 1) * 4
        entry = (gelsy, lwork)
        _GELSY_CACHE[key] = entry
    gelsy, lwork = entry
    rhs = np.zeros((max(m, n), 1))
    rhs[:m, 0] = b
    x = gelsy(M, rhs, np.zeros(n, dtype=np.int32), 1e-12, lwork,
              overwrite_a=0, overwrite_b=1)[1]
    return x[:n, 0]


def solve_simplex_qp(
    B: np.ndarray,
    A: np.ndarray,
    *,
    warm_start: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: Optional[int] = None,
    return_info: bool = False,
    accelerate: bool = True,
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
    accelerate : bool
        Whether a cold solve on a pool of at least ``ACCEL_MIN_DONORS`` may seed
        itself with a FISTA warm start (default ``True``). Set ``False`` for the
        cold path -- it changes the work, not the weights.

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
    # otherwise the uniform point. A wide pool with nothing from the caller
    # seeds itself, since the uniform point costs a pivot per donor.
    if warm_start is None and accelerate and J >= ACCEL_MIN_DONORS:
        warm_start = fista_warm_start(B, A)
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
        # instead of the normal equations BF'BF (which would square the
        # condition number). lstsq tolerates a rank-deficient BF (|free| > T0,
        # collinear donors), so no epsilon-I is needed.
        if nF == 1:
            wF = np.array([1.0])
        else:
            # Null space of 1' via the difference basis Z[:, j] = e_j - e_{nF-1}.
            # Then BF @ Z = BF[:, :-1] - BF[:, -1:] and the uniform particular
            # solution gives BF @ wF0 = mean(BF) -- both without forming Z or a
            # matmul. Rank-revealing QR (LAPACK gelsy) is ~3x faster than SVD
            # lstsq and robust to a rank-deficient system (collinear free donors).
            M = BF[:, :nF - 1] - BF[:, nF - 1:nF]
            v = _gelsy_lstsq(M, A - BF.mean(axis=1))
            wF = np.empty(nF)
            wF[:nF - 1] = 1.0 / nF + v
            wF[nF - 1] = 1.0 / nF - v.sum()

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
