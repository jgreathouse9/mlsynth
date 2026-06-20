"""In-house non-negative least squares (Lawson-Hanson active set).

Pure-NumPy solver for

    min_w  ||A w - b||^2   s.t.   w >= 0,

a drop-in for :func:`scipy.optimize.nnls` used by the MSCMT backend's inner
solve. Implementing it here removes a dependency on scipy's solver in a hot
loop that runs tens of thousands of times, and -- crucially -- makes the result
independent of the installed scipy version: scipy 1.13's ``nnls`` *raises*
``RuntimeError`` when it reaches ``maxiter`` (which the ill-conditioned big-M
formulation can trigger on Python 3.9's older scipy), whereas newer scipy
returns a best-effort iterate. This solver follows the latter contract -- it
always returns ``(w, residual_norm)`` and never raises in the inner loop.

Algorithm: the classical Lawson & Hanson (1974) active-set method, which is
finite (it terminates in a bounded number of steps) and exact for a
full-column-rank ``A``, so it reproduces scipy's optimum to numerical
tolerance. The ``maxiter`` guard only protects against cycling on degenerate
(rank-deficient) inputs, where it returns the current non-negative iterate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def nnls(
    A: np.ndarray,
    b: np.ndarray,
    maxiter: Optional[int] = None,
    *,
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Solve ``min ||A w - b||^2`` subject to ``w >= 0``.

    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape ``(m, n)``.
    b : np.ndarray
        Target vector, shape ``(m,)``.
    maxiter : int, optional
        Maximum number of active-set iterations. Defaults to ``3 * n``. On
        exhaustion the current non-negative iterate is returned (no exception),
        matching modern scipy's best-effort behaviour.
    tol : float, optional
        Tolerance for the optimality (gradient) test and for detecting weights
        driven to the boundary. Defaults to a problem-scaled machine epsilon.

    Returns
    -------
    w : np.ndarray
        Non-negative minimiser, shape ``(n,)``.
    rnorm : float
        Residual two-norm ``||A w - b||``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if maxiter is None:
        maxiter = 3 * n
    if tol is None:
        # Scale with the problem like scipy's default heuristic.
        tol = 10.0 * np.finfo(float).eps * float(np.linalg.norm(A, 1)) * max(m, n)
        tol = max(tol, 1e-12)

    P = np.zeros(n, dtype=bool)        # passive (potentially positive) set
    w = np.zeros(n, dtype=float)
    Atb = A.T @ b
    grad = Atb.copy()                  # A'(b - A w); w = 0 initially
    it = 0

    # Outer loop: bring the most violating coordinate into the passive set.
    while True:
        free = ~P
        if not free.any() or float(np.max(grad[free])) <= tol:
            break                      # KKT satisfied: no improving direction
        # Index (among free coords) with the largest gradient toward feasibility.
        cand = np.where(free)[0]
        j = cand[int(np.argmax(grad[cand]))]
        P[j] = True

        # Inner loop: solve the unconstrained LS on the passive set, retreating
        # to the boundary whenever a passive weight goes non-positive.
        while True:
            it += 1
            if it > maxiter:           # pragma: no cover - degenerate guard
                w = np.maximum(w, 0.0)
                return w, float(np.linalg.norm(A @ w - b))
            Pidx = np.where(P)[0]
            Ap = A[:, Pidx]
            sp, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if float(np.min(sp)) > tol:
                w = np.zeros(n)
                w[Pidx] = sp
                break
            # Some passive weight is non-positive: move w toward s maximally
            # while keeping all passive weights >= 0.
            s = np.zeros(n)
            s[Pidx] = sp
            bad = P & (s <= tol)
            denom = w - s
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(bad & (denom > 0), w / denom, np.inf)
            alpha = float(np.min(ratios))
            if not np.isfinite(alpha):  # pragma: no cover - numerical safety
                alpha = 0.0
            w = w + alpha * (s - w)
            # Drop coordinates that hit the boundary out of the passive set.
            P[(w <= tol) & P] = False
            w[~P] = 0.0

        grad = Atb - A.T @ (A @ w)

    w = np.maximum(w, 0.0)
    return w, float(np.linalg.norm(A @ w - b))
