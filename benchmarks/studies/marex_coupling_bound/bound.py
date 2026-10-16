"""A lower bound on the weakly-targeted objective that sees the control pool.

A prototype for issue #770. It is not wired into
:mod:`mlsynth.utils.marex_helpers.exact`; ``search.py`` puts it in front of
``coupled_fit`` from the outside to measure what it would do.

For a treated set ``S`` with control pool ``C``, write ``P = conv(B_S)`` and
``Q = conv(B_C)``. The weakly-targeted objective is

    f(S) = min over t in P, c in Q of ||A - t||^2 + beta ||t - c||^2.

A minimum of a sum is at least the sum of the minimums, so

    f(S) >= g(S) + beta dist(P, Q)^2,

with ``g(S) = min over t in P of ||A - t||^2`` the treated fit, which the search
computes before any coupled fit. ``P - Q`` is the convex hull of the differences
``b_i - b_j``, ``i`` in ``S`` and ``j`` in ``C``, so ``dist(P, Q)`` is the norm
of the point of that polytope nearest the origin. Wolfe (1976) finds that point,
and for any ``x`` in the polytope his Theorem 2.2 bounds the distance from
below,

    dist(P, Q) >= (min over i in S of x'b_i - max over j in C of x'b_j) / |x|,

the separating-hyperplane dual of the distance (Boyd and Vandenberghe 2004,
sec. 8.2.3). The right side reads the data only through the products ``x'b_j``,
so every ``x`` gives a valid bound: an iterate stopped early, or computed
inaccurately, cannot make it invalid.

:func:`hull_distance_bound` runs Wolfe's method on ``S`` and ``C`` directly, as
his sec. 10 does for two point sets. A step needs the extreme point of
``P - Q`` along ``-x``, which is ``b_i - b_j`` with ``i`` minimising ``x'b_i``
over ``S`` and ``j`` maximising ``x'b_j`` over ``C``: the support mapping of a
Minkowski difference (Gilbert, Johnson and Keerthi 1988). Each step costs one
product ``B'x`` and never forms the ``m (n - m)`` differences. It stops as soon
as the bound clears what the caller needs.

Wolfe's affine step, the point of the corral's affine hull nearest the origin,
is solved as least squares on the corral's edges, ``min ||z_0 + E alpha||``. The
bordered Gram system ``[[Z'Z, 1], [1', 0]]`` is unusable here: on GeoLift its
Gram entries are of order 1e7 against a border of 1, and a least-squares solve
of it returns ``w = 0``, which stalls the method at its first step.

:func:`fenchel_bound` is lead 1 of the issue completed in closed form, the
construction of Hazimeh, Mazumder and Saab (2022, sec. 3.2) applied to this
objective. For any ``lam`` and ``mu``, with no feasibility condition, weak
duality gives

    f(S) >= lam'A - |lam|^2 / 4 - |mu|^2 / (4 beta)
            + min over i in S of (mu - lam)'b_i - max over j in C of mu'b_j,

and the best non-negative rescaling of ``(lam, mu)`` is free. At a fitted
candidate, ``lam = 2 (A - t*)`` and ``mu = 2 beta (t* - c*)`` make it an
equality.

References
----------
Wolfe, P. (1976). Finding the nearest point in a polytope. Mathematical
Programming 11, 128-149.
Gilbert, E. G., Johnson, D. W. and Keerthi, S. S. (1988). A fast procedure for
computing the distance between complex objects in three-dimensional space. IEEE
Journal of Robotics and Automation 4(2), 193-203.
Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge University
Press.
Hazimeh, H., Mazumder, R. and Saab, A. (2022). Sparse regression at scale:
branch-and-bound rooted in first-order optimization. Mathematical Programming
196, 347-388.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: Wolfe's major cycles before the bound gives up and the caller fits.
MAX_MAJOR = 300

#: A corral weight at or below this is treated as zero.
WEIGHT_TOL = 1e-12


def hull_distance_bound(B: np.ndarray, S: Sequence[int], C: Sequence[int],
                        need: float = np.inf, *,
                        pair_d2: Optional[np.ndarray] = None,
                        max_major: int = MAX_MAJOR) -> Tuple[float, int, str]:
    """A lower bound on ``dist(conv B_S, conv B_C)``, by Wolfe's method.

    Stops once the bound's square reaches ``need`` (``"cleared"``), once the
    iterate is the nearest point (``"converged"``), or after ``max_major``
    cycles (``"cap"``). Returns the best bound seen, the cycles taken and which
    of the three stopped it. The bound may be negative, which certifies nothing.

    ``pair_d2``, the squared distances between all pairs of columns of ``B``,
    picks the starting pair; it is computed when not given.
    """
    S = np.asarray(S, dtype=int)
    C = np.asarray(C, dtype=int)
    if pair_d2 is None:
        pair_d2 = squared_pair_distances(B)
    sub = pair_d2[np.ix_(S, C)]
    i0, j0 = np.unravel_index(np.argmin(sub), sub.shape)
    pairs = [(int(S[i0]), int(C[j0]))]
    Z = (B[:, S[i0]] - B[:, C[j0]])[:, None]
    lam = np.ones(1)
    x = Z[:, 0].copy()
    best = -np.inf
    for k in range(max_major):
        pr = B.T @ x
        i = int(S[np.argmin(pr[S])])
        j = int(C[np.argmax(pr[C])])
        xx = float(x @ x)
        best = max(best, (pr[i] - pr[j]) / np.sqrt(xx))
        if best > 0 and best * best >= need:
            return best, k, "cleared"
        if xx - (pr[i] - pr[j]) <= 1e-12 * xx or (i, j) in pairs:
            return best, k, "converged"
        pairs.append((i, j))
        Z = np.column_stack([Z, B[:, i] - B[:, j]])
        lam = np.append(lam, 0.0)
        for _ in range(4 * len(pairs) + 10):
            E = Z[:, 1:] - Z[:, :1]
            alpha = np.linalg.lstsq(E, -Z[:, 0], rcond=None)[0]
            w = np.concatenate([[1.0 - alpha.sum()], alpha])
            if np.all(w > WEIGHT_TOL):
                lam = w
                break
            out = w <= WEIGHT_TOL
            theta = np.min(lam[out] / (lam[out] - w[out]))
            lam = lam + theta * (w - lam)
            keep = lam > WEIGHT_TOL
            Z, lam = Z[:, keep], lam[keep] / lam[keep].sum()
            pairs = [p for p, kept in zip(pairs, keep) if kept]
        x = Z @ lam
    return best, max_major, "cap"


def squared_pair_distances(B: np.ndarray) -> np.ndarray:
    """``|b_i - b_j|^2`` for every pair of columns."""
    sq = (B * B).sum(axis=0)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * (B.T @ B), 0.0)


def fenchel_bound(B: np.ndarray, A: np.ndarray, S: Sequence[int],
                  C: Sequence[int], lam: np.ndarray, mu: np.ndarray,
                  beta: float) -> float:
    """The weak-duality bound at ``(a lam, a mu)``, maximised over ``a >= 0``."""
    S, C = list(S), list(C)
    lin = (lam @ A + (B[:, S].T @ (mu - lam)).min()
           - (B[:, C].T @ mu).max())
    quad = lam @ lam / 4.0 + mu @ mu / (4.0 * beta)
    return max(lin, 0.0) ** 2 / (4.0 * quad) if quad > 0 else 0.0
