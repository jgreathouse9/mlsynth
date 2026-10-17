"""A branch and bound for MAREX's standard design, small enough to read whole.

Twelve markets, three treated: 220 treated sets, so every node can be checked
against brute force.

The design. ``f(S) = g(S) + h(C)``, where ``g(S)`` is the simplex fit of the
target ``A`` on the treated markets ``S`` and ``h(C)`` the same fit on the
controls ``C``, the other nine. Each is solved exactly by the library's
``side_fit``. Measured from the target (columns ``b_j - A``), ``g(S)`` is the
squared distance from the origin to the convex hull of the markets in ``S``.

The tree. The markets get an order. A node fixes its first ``d`` treated
markets in that order, ``I``; the markets later than the last of them are
free, ``F``; the earlier ones not in ``I`` are controls, ``O``. Its children
add one free market each. A node at depth 3 is a leaf, a complete treated set.
Every set appears exactly once. ``r = 3 - d`` markets remain to be chosen.

The bounds. Each is a valid lower bound on ``f`` over a node's subtree. The
control side is always the monotone ``h(O | F)``: a simplex fit over a larger
support can only fit better, and the control fit is a small share of ``f``.
They differ on the treated side, ``U = I | F``:

- ``monotone``: ``g(U)``. It ignores that exactly three markets are treated.
- ``sparse``: ``g(U) + lam * delta^2``. With ``w`` the fit over ``U`` and
  ``q(v) = v'Gv`` on the centred columns, optimality of ``w`` over the simplex
  gives ``q(v) >= q(w) + (v - w)'G(v - w) >= q(w) + lam ||v - w||^2`` for every
  ``v`` on that simplex, ``lam`` the smallest eigenvalue of ``G_U`` on the
  zero-sum directions. A completion puts weight on ``I`` and on at most ``r``
  markets of ``F``, so its distance from ``w`` is at least ``delta``:
  ``delta^2 = sum over the rest of w_i^2 + (sum over the rest of w_i)^2 / 3``,
  the rest being ``F`` less its ``r`` largest weights. Bienstock's (2010)
  idea, in its simplest valid form.
- ``cover``: direction covering. For a unit direction ``u``, if every chosen
  market projects at least ``theta`` onto ``u``, the hull is at least
  ``theta`` from the origin, so ``g >= theta^2``. The directions ``I`` leaves
  uncovered at ``theta`` (no market of ``I`` below it) must be covered by the
  ``r`` markets still to choose. When even the ``r`` largest coverage counts
  in ``F`` fall short of that number, no completion covers them all, and
  ``theta^2`` is a bound. The largest such ``theta`` is found by bisection.
- ``relaxation``: the MIQP's continuous relaxation at the node, the textbook
  branch-and-bound bound. Selections ``z`` are 1 on ``I``, 0 on ``O`` and in
  ``[0, 1]`` on ``F`` with ``sum over F of z = r``; ``0 <= w <= z`` and
  ``0 <= v <= 1 - z``, both on the simplex. Unlike the bounds above it sees
  that a free market split between the sides serves each only in part. Solved
  by Clarabel; valid up to the solver's tolerance, which the check below
  allows for.
- ``best``: the largest of ``monotone``, ``sparse``, ``cover`` and
  ``relaxation``.
- ``closed_form``: the least of the library's closed-form treated bound over
  every completion. A reference, not a node bound: it costs as much as
  enumerating the subtree, and shows what a near-exact treated side captures.

The search is depth first. A node whose bound is at least the incumbent is
pruned with its whole subtree.

From the repository root::

    python -m benchmarks.studies.marex_bnb.bnb12
"""
from __future__ import annotations

import itertools
import os
from math import comb

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.linalg import null_space

from benchmarks.studies.marex_bnb.panels import county, factor, geolift, noise
from mlsynth.utils.marex_helpers.exact import side_fit, treated_fit_bound

N, M = 12, 3


def panels():
    rng = np.random.default_rng(0)
    B, _ = geolift()
    B = B[:, np.sort(rng.choice(B.shape[1], N, replace=False))]
    return {
        "factor": factor(N, 40, seed=1),
        "noise": noise(N, 40, seed=3),
        "geolift12": (B, B.mean(axis=1)),
        "county12": county(N, seed=0),
    }


def fit(B, A, cols):
    return side_fit(B, A, sorted(cols))[0] if cols else np.inf


def objective(B, A, S):
    S = set(S)
    return fit(B, A, S) + fit(B, A, [j for j in range(B.shape[1]) if j not in S])


class Bounds:
    """The node bounds of one panel; ``I``, ``F``, ``O`` are market lists."""

    def __init__(self, B, A, n_dir=200, seed=0):
        self.B, self.A = B, A
        self.Bc = B - A[:, None]
        n = B.shape[1]
        U, _, Vt = np.linalg.svd(self.Bc, full_matrices=False)
        dirs = [U[:, :min(10, U.shape[1])].T, -U[:, :min(10, U.shape[1])].T]
        norms = np.linalg.norm(self.Bc, axis=0)
        dirs += [(self.Bc / norms).T, -(self.Bc / norms).T]
        R = np.random.default_rng(seed).normal(size=(n_dir, B.shape[0]))
        dirs.append(R / np.linalg.norm(R, axis=1, keepdims=True))
        self.P = np.vstack(dirs) @ self.Bc            # (directions, markets)

    def controls(self, O, F):
        return fit(self.B, self.A, O + F)

    def monotone(self, I, F, O, r):
        return fit(self.B, self.A, I + F) + self.controls(O, F)

    def sparse(self, I, F, O, r):
        U = I + F
        g, w = side_fit(self.B, self.A, U)
        k = len(U)
        if k <= M:
            return g + self.controls(O, F)
        Bu = self.Bc[:, U]
        Z = null_space(np.ones((1, k)))
        lam = max(float(np.linalg.eigvalsh(Z.T @ (Bu.T @ Bu) @ Z).min()), 0.0)
        wF = np.asarray(w[len(I):], float)
        rest = np.sort(wF)[: max(len(F) - r, 0)]
        delta2 = float(rest @ rest) + float(rest.sum()) ** 2 / M
        return g + lam * delta2 + self.controls(O, F)

    def cover(self, I, F, O, r):
        P = self.P
        pI = P[:, I].min(axis=1) if I else np.full(P.shape[0], np.inf)
        PF = P[:, F]

        def impossible(theta):
            open_ = pI >= theta                    # directions I leaves open
            need = int(open_.sum())
            if need == 0:
                return False
            counts = (PF[open_] < theta).sum(axis=0)
            return int(np.sort(counts)[::-1][:r].sum()) < need

        grid = np.unique(P[P > 0])
        lo, hi = -1, len(grid) - 1
        if hi < 0 or not impossible(grid[0]):
            theta = 0.0
        else:
            lo = 0
            while lo < hi:                         # largest index still impossible
                mid = (lo + hi + 1) // 2
                if impossible(grid[mid]):
                    lo = mid
                else:
                    hi = mid - 1
            theta = float(grid[lo])
        return theta * theta + self.controls(O, F)

    def relaxation(self, I, F, O, r):
        import cvxpy as cp
        n = self.B.shape[1]
        w = cp.Variable(n, nonneg=True)
        v = cp.Variable(n, nonneg=True)
        z = cp.Variable(n)
        cons = [cp.sum(w) == 1, cp.sum(v) == 1, w <= z, v <= 1 - z]
        if I:
            cons.append(z[I] == 1)
        if O:
            cons.append(z[O] == 0)
        cons += [z[F] >= 0, z[F] <= 1, cp.sum(z[F]) == r]
        obj = cp.sum_squares(self.Bc @ w) + cp.sum_squares(self.Bc @ v)
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.CLARABEL)
        return max(float(prob.value), 0.0)

    def best(self, I, F, O, r):
        return max(self.monotone(I, F, O, r), self.sparse(I, F, O, r),
                   self.cover(I, F, O, r), self.relaxation(I, F, O, r))

    def closed_form(self, I, F, O, r):
        subs = np.array([sorted(I + list(R))
                         for R in itertools.combinations(F, r)])
        return float(treated_fit_bound(self.B, self.A, subs).min()) \
            + self.controls(O, F)


BOUNDS = ("monotone", "sparse", "cover", "relaxation", "best", "closed_form")


def bnb(B, A, order, bound, incumbent=np.inf):
    """Depth-first branch and bound; returns the design and what it did."""
    n = B.shape[1]
    best = [incumbent, None]
    stats = {"bounded": 0, "pruned": 0, "pruned_sets": 0, "leaves": 0}
    nodes = []

    def visit(pos):
        d = len(pos)
        if d == M:
            S = tuple(sorted(int(order[p]) for p in pos))
            stats["leaves"] += 1
            value = objective(B, A, S)
            if value < best[0]:
                best[0], best[1] = value, S
            return
        start = pos[-1] + 1 if pos else 0
        for q in range(start, n - (M - d) + 1):
            child = pos + (q,)
            if len(child) < M:
                I = [int(order[p]) for p in child]
                F = [int(order[p]) for p in range(q + 1, n)]
                O = [int(order[p]) for p in range(q + 1) if p not in child]
                lb = bound(I, F, O, M - len(child))
                stats["bounded"] += 1
                cut = lb >= best[0]
                nodes.append((tuple(I), lb, cut))
                if cut:
                    stats["pruned"] += 1
                    stats["pruned_sets"] += comb(n - 1 - q, M - len(child))
                    continue
            visit(child)

    visit(())
    return best[0], best[1], stats, nodes


def _in_subtree(T, I, rank):
    return sorted(T, key=rank.get)[:len(I)] == list(I)


def main():
    for name, (B, A) in panels().items():
        n = B.shape[1]
        brute = {S: objective(B, A, S)
                 for S in itertools.combinations(range(n), M)}
        S_star = min(brute, key=brute.get)
        f_star = brute[S_star]
        single = np.array([fit(B, A, [j]) for j in range(n)])
        bd = Bounds(B, A)
        print(f"\n=== {name}: optimum {S_star}, f* = {f_star:.6g}; existing "
              f"search fits {_existing_fits(B, A)} sets")
        orders = {"index": np.arange(n),
                  "best_first": np.argsort(single, kind="stable"),
                  "worst_first": np.argsort(-single, kind="stable")}
        print(f"  {'bound':12s} {'order':12s} {'bounded':>7s} {'pruned':>6s} "
              f"{'sets cut':>8s} {'priced':>6s}   depth-1 bound / subtree min")
        for bname in BOUNDS:
            fn = getattr(bd, bname)
            for oname, order in orders.items():
                rank = {int(m): p for p, m in enumerate(order)}
                value, S, stats, nodes = bnb(B, A, order, fn)
                assert S == S_star and abs(value - f_star) <= 1e-12 * f_star
                ratios = []
                for I, lb, _ in nodes:
                    sub = min(v for T, v in brute.items()
                              if _in_subtree(T, I, rank))
                    tol = 1e-6 if bname in ("relaxation", "best") else 1e-9
                    assert lb <= sub * (1 + tol) + 1e-12, (bname, I, lb, sub)
                    if len(I) == 1:
                        ratios.append(lb / sub)
                early = ratios[: len(ratios) // 2]
                print(f"  {bname:12s} {oname:12s} {stats['bounded']:7d} "
                      f"{stats['pruned']:6d} {stats['pruned_sets']:8d} "
                      f"{stats['leaves']:6d}   first half median "
                      f"{np.median(early):.3f}, all median "
                      f"{np.median(ratios):.3f}")


def _existing_fits(B, A):
    from mlsynth.utils.marex_helpers import exact as ex
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    calls = [0]
    real = ex.side_fit

    def spy(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    ex.side_fit = spy
    try:
        ex.search_cluster(B, A, [M], filt=SubsetFilter())
    finally:
        ex.side_fit = real
    return calls[0] // 2


if __name__ == "__main__":
    main()
