"""Pentagonal inequalities past the triangles (#837, candidate 2).

For any five indices of the lifted ``X`` (index 0 the homogenising one) and
signs ``b`` in ``{-1, +1}^5``, a sign vector has ``(b'x)^2 >= 1`` since
``b'x`` is odd, so

    sum_{i<j} b_i b_j X_ij >= -2,

the pentagonal case of the hypermetric inequalities (BiqBin, Gusmeroli et al.
2022, p. 6, valid on the convex hull of rank-one ``X``). ``b`` and ``-b`` give
the same row, so each five indices carry 16 rows. They are linear rows on
``X``, like the triangles, and would enter the lift's dual as more multipliers.

:func:`pentagons` separates them by enumeration, which a 24-unit node (25
lifted indices, 53,130 five-subsets) affords; BiqBin separates them
heuristically. :func:`closure` solves the lift's relaxation by interior point
with triangles, and with pentagonal rows when asked, adding violated rows
round by round, so the gain of the family is measured against the closure it
would extend.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from mlsynth.utils.syndes_helpers import branch

#: The sign patterns, one of each pair ``b, -b``: those with ``b_0 = +1``.
_PATTERNS = np.array([(1,) + rest for rest in
                      __import__("itertools").product((1, -1), repeat=4)], dtype=float)
_PAIRS = np.array(list(combinations(range(5), 2)))


def value(X: np.ndarray, subset, pattern) -> float:
    """``sum_{i<j} b_i b_j X_ij`` over ``subset`` with signs ``pattern``."""
    s = np.asarray(subset)
    b = np.asarray(pattern, dtype=float)
    return float(sum(b[p] * b[q] * X[s[p], s[q]] for p, q in _PAIRS))


def pentagons(X: np.ndarray, *, gap: float = 1e-6, limit: int = 500) -> list:
    """The pentagonal rows ``X`` violates by more than ``gap``, most violated
    first, at most ``limit``, as ``(subset, pattern)`` pairs."""
    size = X.shape[0]
    if size < 5:
        return []
    subsets = np.array(list(combinations(range(size), 5)))
    # the ten pairwise entries of each subset, and each pattern's ten products
    entries = X[subsets[:, _PAIRS[:, 0]], subsets[:, _PAIRS[:, 1]]]       # (S, 10)
    coeff = _PATTERNS[:, _PAIRS[:, 0]] * _PATTERNS[:, _PAIRS[:, 1]]          # (16, 10)
    vals = entries @ coeff.T + 2.0                                           # (S, 16)
    s, p = np.nonzero(vals < -gap)
    order = np.argsort(vals[s, p], kind="stable")[:limit]
    return [(subsets[s[k]], _PATTERNS[p[k]]) for k in order]


def closure(c: float, b: np.ndarray, A: np.ndarray, s: int, *, pentagons: bool = True,
            gap: float = 1e-6, per_round: int = 1_000, rounds: int = 8,
            solver: str = "CLARABEL") -> dict:
    """The lift's relaxation by interior point at a node's parts, with the
    triangles and, with ``pentagons``, the pentagonal rows, each added where
    violated by more than ``gap``, at most ``per_round`` of each per round.
    Returns the value ``U`` on ``c + 2 b'x + x'A x``, the rounds, the rows of
    each family added and how many the last solve still violated. The free
    part sets the scale and ``c`` is added back (#864)."""
    import cvxpy as cp

    from .bounds import _lift_problem

    unit = max(float(np.abs(b).max(initial=0.0)), float(np.abs(A).max()))
    prob, _, _ = _lift_problem(0.0, b / unit, A / unit, s)
    X = prob.variables()[0]
    signs = branch._SIGNS
    rows = []
    counts = {"triangles": 0, "pentagons": 0}
    violated = 0
    value_ = np.nan
    done = 0
    for done in range(1, rounds + 1):
        value_ = cp.Problem(prob.objective, prob.constraints + rows).solve(solver=solver)
        Xv = np.asarray(X.value)
        tri = branch.separate(Xv, gap=gap, limit=per_round)
        pent = _separate(Xv, gap, per_round) if pentagons else []
        violated = len(tri) + len(pent)
        if not violated:
            break
        if len(tri):
            sg = signs[tri.t - 1]
            rows.append(1 + cp.multiply(sg[:, 0], X[tri.i, tri.j])
                        + cp.multiply(sg[:, 1], X[tri.i, tri.k])
                        + cp.multiply(sg[:, 2], X[tri.j, tri.k]) >= 0)
            counts["triangles"] += len(tri)
        for subset, pattern in pent:
            rows.append(sum(pattern[p] * pattern[q] * X[subset[p], subset[q]]
                            for p, q in _PAIRS) >= -2)
        counts["pentagons"] += len(pent)
    return {"U": c + unit * float(value_), "rounds": done, "violated": violated, **counts}


def _separate(X, gap, limit):
    """:func:`pentagons`, by a name :func:`closure`'s flag does not shadow."""
    return pentagons(X, gap=gap, limit=limit)


def closures(Y: np.ndarray, K: int, *, depths=(0, 2, 4, 6, 3, 5, 7, 1), seed: int = 0) -> list:
    """At ``duals.ratios``'s nodes of a panel small enough to price every set:
    the triangle closure and the closure with pentagonal rows, each as its
    node bound over the subtree's minimum price, beside ``need``, the ratio a
    bound must pass to prune with the optimum as the incumbent."""
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    from .duals import _node
    from .quality import price_all

    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    N = Y.shape[1]
    sets, price, _ = price_all(eng, N, K)
    best = float(price.min())
    member = np.zeros((len(sets), N), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    rng = np.random.default_rng(seed)
    rows = []
    for d in depths:
        I, O, F = _node(rng, N, K, d)
        mask = np.ones(len(sets), dtype=bool)
        for u in I:
            mask &= member[:, u]
        for u in O:
            mask &= ~member[:, u]
        sub = float(price[mask].min())
        c, b, A = branch.node_parts(eng.R, I, O, F)
        s = 2 * (K - len(I)) - len(F)
        tri = closure(c, b, A, s, pentagons=False)
        pent = closure(c, b, A, s, pentagons=True)
        ratio = lambda U: 4.0 * eng.alpha / U / sub
        rows.append({"N": N, "K": K, "fixed": len(I) + len(O), "need": best / sub,
                     "triangles": ratio(tri["U"]), "pentagons": ratio(pent["U"]),
                     "pentagon_rows": pent["pentagons"], "left": pent["violated"]})
    return rows
