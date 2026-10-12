"""The exposure-fit frontier on mlsynth's active-set solver, for the solver choice.

``solve_simplex_qp`` minimises ``||a - A w||^2 + linear' w`` over the simplex
and takes no further constraints. With a scalar exposure index only one side of
``|EI(w)| <= d`` binds below the synthetic-control imbalance, so the frontier
point is ``w(mu) = argmin ||a - A w||^2 - mu sigma s0' w`` at the ``mu >= 0``
where ``sigma EI(w(mu)) = d``, ``sigma`` the sign of the synthetic-control
imbalance. ``sigma EI(w(mu))`` is non-increasing in ``mu``, so ``mu`` is found
by bisection. At ``d = d_min`` with the treated unit outside the hull the
multiplier diverges; the point is then the simplex fit on the donors at the
nearest end of the exposure range.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from mlsynth.utils.solvers.active_set import solve_simplex_qp

from . import oracle as O


def frontier_point(a: np.ndarray, A: np.ndarray, s1: float, s0: np.ndarray, d: float, w_sc: np.ndarray,
                   rel_tol: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Weights at tolerance ``d`` and the multiplier that produced them."""
    ei_sc = s1 - s0 @ w_sc
    if abs(ei_sc) <= d:
        return w_sc, 0.0
    sig = float(np.sign(ei_sc))
    dmin = O.min_imbalance(s1, s0)
    if dmin > 0 and d <= dmin + 1e-9:
        face = np.flatnonzero(np.isclose(s0, s0.max() if sig > 0 else s0.min()))
        w = np.zeros_like(w_sc)
        w[face] = solve_simplex_qp(A[:, face], a)
        return w, np.inf

    def excess(mu: float) -> float:
        return sig * (s1 - s0 @ solve_simplex_qp(A, a, linear=-mu * sig * s0)) - d

    lo, hi = 0.0, 1e-8
    while excess(hi) > 0:
        lo, hi = hi, 2 * hi
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return solve_simplex_qp(A, a, linear=-hi * sig * s0), hi
