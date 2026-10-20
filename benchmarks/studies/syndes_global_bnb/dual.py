"""The lift's bound by its regularised dual, without an interior-point solve (#803).

The node's quadratic ``c + 2 b'x + x'A x`` over sign vectors with ``1'x = s``
is lifted to ``X = [1; x][1; x]'``: semidefinite, unit diagonal, trace
``n + 1``. With multipliers ``y`` on the equalities (the diagonal, ``1'x = s``
and the products ``X 1 = s x``) and ``v >= 0`` on triangle inequalities, the
Lagrangian is ``const + <C, X>`` for a matrix ``C`` affine in the multipliers,
and at least the objective at every design. Two upper bounds follow, for any
multipliers:

    const + (n + 1) lambda_max(C)                                  (exact)
    const + |C_+|_F^2 / (2 alpha) + alpha (n + 1)^2 / 2            (regularised)

the second because a design's ``X`` has Frobenius norm ``n + 1``, so
``<C, X> <= (n + 1) |C_+|_F``, and by Young's inequality for any ``alpha > 0``.
The regularised bound is differentiable, with gradient the constraints'
residuals at ``X_alpha = C_+ / alpha``. This is the bounding procedure of
Krislock, Malick and Roupin (BiqCrunch), read from its source for the procedure
and its defaults, nothing copied (it is GPL): L-BFGS-B on
the regularised bound, ``alpha`` halved from 0.1 to 5e-5, triangle
inequalities violated at ``X_alpha`` by more than 0.05 added up to 500 a round
and dropped once their multiplier is zero and they are slack. Every evaluation
reports the exact bound, never above the regularised one, and the smallest is
kept; so the method stops as soon as the node prunes, and gives up, as
BiqCrunch does, on a node whose bound has stalled at a small ``alpha``. A second
rule, off by default (``PATIENCE = inf``), abandons a node whose bound would take
more than ``PATIENCE`` rounds at its last round's rate to reach the target. With
``PATIENCE = 10`` it abandoned nodes that would have pruned: the bound falls
fastest late, as ``alpha`` shrinks, so an early round's rate is pessimistic. On
GeoLift with eight treated 0 of 14,367 lifts pruned, and with ten treated the
search ran past 900 s against 77.6 s without it. A child can start from
its parent's multipliers (``restrict``), at ``alpha = 1e-2``: they fit a small
``alpha``, and restarted at 0.1 they saved 4% of the evaluations over eight
children, against 12% at 1e-2 (and a loss at 1e-4). Read on a child, the
parent's multipliers bound it at the price of one eigendecomposition, which is
the dual fixing of semidefinite branch and bound.

The triangle inequalities through the homogenising index 0 are the lift's
products ``(1 +- x_i)(1 +- x_j) >= 0`` of ``bounds.lift_upper``; with
``cuts="products"`` only those are separated, and the bound is the lift's.
``cuts="triangles"`` separates those among the free units as well, every one
valid at a design.

The schedule's constants are BiqCrunch's defaults: the problem's parts are
divided by their largest entry, so ``X``'s entries and the constraints'
residuals are of order one, as in BiqCrunch's max-cut instances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .bounds import split, trs_parts

ALPHA0, ALPHA_SCALE, ALPHA_MIN = 1e-1, 0.5, 5e-5
ALPHA_WARM = 1e-2           # a warm start begins here: its multipliers fit a small alpha
TOL0, TOL_SCALE, TOL_MIN = 1e-1, 0.95, 1e-2
CUT_GAP = 5e-2              # a triangle enters when violated by more than this
CUT_LIMIT = 500             # triangles added per round
MIN_CUTS = 50               # fewer added than this: halve alpha
ROUNDS_AT_ALPHA = 50        # or this many rounds at one alpha
MIN_ROUNDS, MAX_ROUNDS = 12, 100
EVALUATIONS = 2000          # per round
MAXCOR = 10
PATIENCE = math.inf         # off; at 10 it abandoned nodes that would have pruned
TARGET_ROUNDS = 3           # rounds before a target can be abandoned
KAPPA = 1.0 / (1.0 + math.sqrt(3.0))   # scale of the triangles' multipliers

# signs of (X_ij, X_ik, X_jk) in each type; type 1 has no negative edge, the
# others one positive edge: (i, j), (i, k), (j, k)
_SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


@dataclass
class Cuts:
    """Triangle inequalities ``1 + s1 X_ij + s2 X_ik + s3 X_jk >= 0`` on
    ``i < j < k`` of the node's lifted indices (0 the homogenising one), with
    signs by type ``t`` in 1..4."""

    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    t: np.ndarray

    @staticmethod
    def empty() -> "Cuts":
        z = np.zeros(0, dtype=np.int64)
        return Cuts(z, z.copy(), z.copy(), z.copy())

    def __len__(self) -> int:
        return len(self.i)

    def take(self, idx) -> "Cuts":
        return Cuts(self.i[idx], self.j[idx], self.k[idx], self.t[idx])

    def keys(self, size: int) -> np.ndarray:
        return ((self.i * size + self.j) * size + self.k) * 4 + (self.t - 1)

    def values(self, X: np.ndarray) -> np.ndarray:
        s = _SIGNS[self.t - 1]
        return (1.0 + s[:, 0] * X[self.i, self.j] + s[:, 1] * X[self.i, self.k]
                + s[:, 2] * X[self.j, self.k])

    @staticmethod
    def concat(a: "Cuts", b: "Cuts") -> "Cuts":
        return Cuts(*(np.concatenate([x, y]) for x, y in
                      ((a.i, b.i), (a.j, b.j), (a.k, b.k), (a.t, b.t))))


@lru_cache(maxsize=None)
def _triples(size: int, products: bool):
    idx = np.array([(i, j, k) for i in range(size) for j in range(i + 1, size)
                    for k in range(j + 1, size) if not products or i == 0],
                   dtype=np.int64).reshape(-1, 3)
    idx.flags.writeable = False
    return idx


def separate(X: np.ndarray, *, gap: float = CUT_GAP, limit: int = CUT_LIMIT,
             products: bool = False) -> Cuts:
    """The triangle inequalities ``X`` violates by more than ``gap``, most
    violated first, at most ``limit``."""
    T = _triples(X.shape[0], products)
    if not len(T):
        return Cuts.empty()
    xij, xik, xjk = X[T[:, 0], T[:, 1]], X[T[:, 0], T[:, 2]], X[T[:, 1], T[:, 2]]
    vals = 1.0 + (np.stack([xij, xik, xjk], axis=1) @ _SIGNS.T)      # (triples, 4)
    p, t = np.nonzero(vals < -gap)
    order = np.argsort(vals[p, t], kind="stable")[:limit]
    p, t = p[order], t[order]
    return Cuts(T[p, 0].copy(), T[p, 1].copy(), T[p, 2].copy(), (t + 1).astype(np.int64))


def theta(c: float, b: np.ndarray, A: np.ndarray, s: int, y: np.ndarray,
          cuts: Cuts, v: np.ndarray, alpha: float):
    """The exact bound, the regularised bound, their gradients in ``y`` and in
    ``v`` (of the regularised one) and ``X_alpha``.

    ``y``: the diagonal's ``n + 1`` multipliers (index 0 first), then
    ``1'x = s``'s, then the ``n`` products ``sum_j X_ij - s X_0i = 0``. Both
    bounds hold for any ``y`` and any ``v >= 0``."""
    n = len(b)
    size = n + 1
    C = np.empty((size, size))
    C[0, 0] = 0.0
    C[0, 1:] = C[1:, 0] = b
    C[1:, 1:] = A
    yd, ys, yr = y[:size], y[size], y[size + 1:]
    C[np.diag_indices(size)] -= yd
    C[0, 1:] -= 0.5 * ys
    C[1:, 0] -= 0.5 * ys
    u = np.concatenate([[-float(s)], np.ones(n)])
    rows = np.concatenate([[0.0], yr])
    C -= 0.5 * (np.outer(rows, u) + np.outer(u, rows))
    w = KAPPA * v
    if len(cuts):
        sg = _SIGNS[cuts.t - 1] * (0.5 * w)[:, None]
        for (p, q), col in (((cuts.i, cuts.j), 0), ((cuts.i, cuts.k), 1), ((cuts.j, cuts.k), 2)):
            np.add.at(C, (p, q), sg[:, col])
            np.add.at(C, (q, p), sg[:, col])
    const = c + yd.sum() + ys * s + w.sum()
    lam, U = np.linalg.eigh(C)
    pos = np.maximum(lam, 0.0)
    X = (U * (pos / alpha)) @ U.T
    exact = const + size * lam[-1]
    smooth = const + float(pos @ pos) / (2.0 * alpha) + 0.5 * alpha * size * size
    gy = np.empty(len(y))
    gy[:size] = 1.0 - np.diag(X)
    gy[size] = s - X[0, 1:].sum()
    gy[size + 1:] = -(X[1:, :] @ u)
    gv = KAPPA * cuts.values(X) if len(cuts) else np.zeros(0)
    return exact, smooth, gy, gv, X


def restrict(state, parent: Sequence[int], child: Sequence[int]):
    """A parent's ``(y, cuts, v)`` read on a child's free units, a subset of
    the parent's in any order. Cuts through a unit the child does not leave free
    are dropped; the others keep their units and their positive edge."""
    y, cuts, v = state
    n, k = len(parent), len(child)
    where = {u: p + 1 for p, u in enumerate(parent)}
    idx = np.array([0] + [where[u] for u in child], dtype=np.int64)
    cy = np.empty(2 * k + 2)
    cy[:k + 1] = y[:n + 1][idx]
    cy[k + 1] = y[n + 1]
    cy[k + 2:] = y[n + 2:][idx[1:] - 1]
    inv = np.full(n + 1, -1, dtype=np.int64)
    inv[idx] = np.arange(k + 1)
    ci, cj, ck = inv[cuts.i], inv[cuts.j], inv[cuts.k]
    keep = (ci >= 0) & (cj >= 0) & (ck >= 0)
    ci, cj, ck, t = ci[keep], cj[keep], ck[keep], cuts.t[keep]
    ea = np.where(t == 4, cj, ci)                    # the positive edge
    eb = np.where(t == 2, cj, ck)
    tri = np.sort(np.stack([ci, cj, ck], axis=1), axis=1)
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    nt = np.where(t == 1, 1, np.where((lo == tri[:, 0]) & (hi == tri[:, 1]), 2,
                                      np.where((lo == tri[:, 0]) & (hi == tri[:, 2]), 3, 4)))
    return cy, Cuts(tri[:, 0], tri[:, 1], tri[:, 2], nt.astype(np.int64)), np.asarray(v)[keep].copy()


@dataclass
class DualInfo:
    state: Tuple[np.ndarray, Cuts, np.ndarray]   # in the node's unit-scaled parts
    evaluations: int
    rounds: int


class _Pruned(Exception):
    pass


class _Converged(Exception):
    pass


def dual_upper(R, I, O, F, K, *, target: Optional[float] = None, warm=None,
               info: bool = False, cuts: str = "triangles"):
    """The lift's bound by the regularised dual; never above the trust-region
    bound. ``target``: stop once the bound is at or under it (the node prunes).
    ``warm``: a starting ``(y, cuts, v)``, e.g. ``restrict`` of the parent's.
    ``cuts``: ``"products"`` or ``"triangles"``."""
    c, b, A = split(R, I, O, F)
    n = len(F)
    s = 2 * (K - len(I)) - n
    products = cuts == "products"

    def out(value, state=None, evaluations=0, rounds=0):
        if not info:
            return value
        if state is None:
            state = (np.zeros(2 * n + 2), Cuts.empty(), np.zeros(0))
        return value, DualInfo(state, evaluations, rounds)

    if n == 0:
        return out(c)
    if n == 1:
        return out(trs_parts(c, b, A, s)[0])
    unit = max(abs(c), float(np.abs(b).max(initial=0.0)), float(np.abs(A).max()))
    if unit == 0.0:
        return out(c)
    c, b, A = c / unit, b / unit, A / unit
    fallback = trs_parts(c, b, A, s)[0]
    goal = -np.inf if target is None else target / unit
    size = n + 1
    if warm is None:
        y, cut, v = np.zeros(2 * n + 2), Cuts.empty(), np.zeros(0)
    else:
        y, cut, v = np.array(warm[0], dtype=float), warm[1], np.maximum(np.asarray(warm[2], dtype=float), 0.0)
    alpha, tol = (ALPHA0 if warm is None else ALPHA_WARM), TOL0
    best = [np.inf]
    evaluations = [0]
    rounds = at_alpha = 0
    previous = None
    reached = np.inf
    ny = 2 * n + 2

    while True:
        rounds += 1
        at_alpha += 1
        last = {}

        def fun(x, cut=cut, alpha=alpha):
            exact, smooth, gy, gv, _ = theta(c, b, A, s, x[:ny], cut, x[ny:], alpha)
            evaluations[0] += 1
            best[0] = min(best[0], exact)
            last["x"], last["g"] = x.copy(), (gy, gv)
            if min(best[0], fallback) <= goal:
                raise _Pruned
            return smooth, np.concatenate([gy, gv])

        def stop(xk, tol=tol):
            if "x" not in last or not np.array_equal(last["x"], xk):
                return  # pragma: no cover - L-BFGS-B calls back at the point it evaluated last; this keeps a stale gradient from deciding convergence
            gy, gv = last["g"]
            if (np.abs(gy).max() < tol
                    and (not len(gv) or -min(gv.min(), 0.0) / KAPPA < tol)):
                raise _Converged

        x0 = np.concatenate([y, v])
        bounds = [(None, None)] * ny + [(0.0, None)] * len(v)
        pruned = False
        try:
            res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds, callback=stop,
                           options={"maxcor": MAXCOR, "maxfun": EVALUATIONS,
                                    "maxiter": EVALUATIONS, "ftol": 0.0, "gtol": 0.0})
            x = res.x
        except _Pruned:
            pruned = True
            x = last["x"]
        except _Converged:
            x = last["x"]
        y, v = x[:ny], x[ny:]
        if pruned:
            break
        # with PATIENCE finite, a node that will not prune at this rate is
        # abandoned: its bound is still a bound, and the search splits it
        now = min(best[0], fallback)
        if (target is not None and math.isfinite(PATIENCE) and rounds >= TARGET_ROUNDS
                and now - goal > PATIENCE * max(reached - now, 0.0)):
            break
        reached = now
        _, smooth, gy, gv, X = theta(c, b, A, s, y, cut, v, alpha)
        stalled = (previous is not None and abs(previous - smooth) <= 1e-6 * abs(smooth)
                   and alpha < 1e-4)
        previous = smooth
        giveup = rounds >= MIN_ROUNDS and (rounds >= MAX_ROUNDS or stalled)
        # drop slack cuts whose multiplier is zero, add the violated ones
        violation = -min(gv.min(), 0.0) / KAPPA if len(gv) else 0.0
        if len(cut):
            keep = ~((v < 1e-8) & (cut.values(X) > violation / 10.0))
            cut, v = cut.take(keep), v[keep]
        found = separate(X, products=products)
        fresh = ~np.isin(found.keys(size), cut.keys(size))
        added = int(fresh.sum())
        if added:
            cut = Cuts.concat(cut, found.take(fresh))
            v = np.concatenate([v, np.zeros(added)])
        if giveup or (alpha == ALPHA_MIN and added == 0):
            break
        if added < MIN_CUTS or at_alpha > ROUNDS_AT_ALPHA:
            at_alpha = 0
            alpha = max(alpha * ALPHA_SCALE, ALPHA_MIN)
            tol = max(tol * TOL_SCALE, TOL_MIN)
    return out(unit * min(best[0], fallback), (y, cut, v), evaluations[0], rounds)
