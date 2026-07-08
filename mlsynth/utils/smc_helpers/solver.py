"""Exact box-constrained QP for the SMC Cp criterion.

Minimise ``f(w) = 1/2 w' D w - d' w`` subject to ``lo <= w <= hi`` via a primal
active-set method with a ratio test. For a strictly-convex ``D`` (any
``ridge > 0``) this terminates in finitely many pivots at the exact KKT point --
the box analogue of :func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`.

Why not a first-order / interior-point QP: SMC re-solves this problem in the hot
placebo and bootstrap loops, and the reported weights must pin the box bounds
*exactly* (a donor at zero is exactly zero -- that is the sparsity the estimator
reports). An active-set method delivers both: it matches the reference R
``solve.QP`` to machine precision, is markedly faster than OSQP/CLARABEL at the
donor-pool sizes synthetic control uses, and is warm-startable for the sweeps.
The free-set subproblem is solved with a Cholesky ``solve`` (PSD via the ridge),
falling back to ``lstsq`` if a rank-deficient free block ever arises.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

FREE, LO, HI = 0, -1, 1


def solve_box_qp(
    D: np.ndarray,
    d: np.ndarray,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    warm_start: Optional[np.ndarray] = None,
    tol: float = 1e-11,
    max_iter: Optional[int] = None,
    return_info: bool = False,
):
    """Minimise ``1/2 w' D w - d' w`` over the box ``[lo, hi]``.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Symmetric positive-definite (with ``ridge > 0``) quadratic term.
    d : np.ndarray, shape (n,)
        Linear term.
    lo, hi : float
        Box bounds applied to every coordinate.
    warm_start : np.ndarray, shape (n,), optional
        Feasible starting point seeding the active set (e.g. the neighbouring
        solution in a placebo / bootstrap sweep). Clipped to the box if outside.
    tol : float
        KKT / feasibility tolerance.
    max_iter : int, optional
        Cap on active-set pivots; defaults to ``50 n + 50``.
    return_info : bool
        If ``True`` also return a diagnostics dict (``pivots``, ``converged``,
        ``kkt``) so tests can assert exactness and bounded work.

    Returns
    -------
    np.ndarray, shape (n,)
        The optimal ``w``, or ``(w, info)`` when ``return_info=True``.
    """
    D = np.asarray(D, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    n = d.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must be ({n}, {n}) to match d of length {n}.")
    if hi <= lo:
        raise ValueError(f"Need lo < hi; got lo={lo}, hi={hi}.")
    if max_iter is None:
        max_iter = 50 * n + 50

    if warm_start is not None:
        w = np.clip(np.asarray(warm_start, dtype=float).ravel(), lo, hi)
    else:
        w = np.full(n, lo)
    state = np.where(w >= hi - tol, HI, np.where(w <= lo + tol, LO, FREE))
    w = np.where(state == LO, lo, np.where(state == HI, hi, w))

    scale = 1.0 + float(np.max(np.abs(d))) if n else 1.0
    pivots = 0
    converged = False
    for _ in range(max_iter):
        free = np.where(state == FREE)[0]
        bnd = np.where(state != FREE)[0]
        if free.size:
            rhs = d[free] - (D[np.ix_(free, bnd)] @ w[bnd] if bnd.size else 0.0)
            Dff = D[np.ix_(free, free)]
            try:
                wF = np.linalg.solve(Dff, rhs)
            except np.linalg.LinAlgError:            # pragma: no cover - ridge keeps PD
                wF = np.linalg.lstsq(Dff, rhs, rcond=None)[0]

            cur = w[free]
            direction = wF - cur
            with np.errstate(divide="ignore", invalid="ignore"):
                r_lo = np.where(direction < -tol, (cur - lo) / (-direction), np.inf)
                r_hi = np.where(direction > tol, (hi - cur) / direction, np.inf)
            ratios = np.minimum(r_lo, r_hi)
            kk = int(np.argmin(ratios))
            t_step = min(1.0, float(ratios[kk]))
            if t_step < 1.0 - 1e-12:
                # A free variable reaches a face before the interior optimum: pin it.
                w[free] = cur + t_step * direction
                j = int(free[kk])
                to_hi = r_hi[kk] <= r_lo[kk]
                w[j] = hi if to_hi else lo
                state[j] = HI if to_hi else LO
                pivots += 1
                continue
            w[free] = wF

        # Free set at its interior optimum: check dual feasibility on the bounds.
        g = D @ w - d
        if bnd.size:
            viol = np.where(state[bnd] == LO, -g[bnd], g[bnd])
            m = int(np.argmax(viol))
            if viol[m] > tol * scale:
                state[bnd[m]] = FREE     # release the most-violating bound variable
                pivots += 1
                continue
        converged = True
        break

    w = np.clip(w, lo, hi)
    if return_info:
        g = D @ w - d
        res = _kkt_residual(w, g, lo, hi, tol)
        return w, {"pivots": pivots, "converged": converged, "kkt": res}
    return w


def _kkt_residual(w: np.ndarray, g: np.ndarray, lo: float, hi: float,
                  tol: float) -> float:
    """Max KKT violation: stationarity on free vars, sign on active vars."""
    res = np.where(w <= lo + tol, -g, np.where(w >= hi - tol, g, np.abs(g)))
    return float(np.max(res)) if res.size else 0.0
