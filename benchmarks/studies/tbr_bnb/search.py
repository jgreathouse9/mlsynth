"""Branch and bound on geos with smoothed L-BFGS-B as the node solver (#965).

A node prunes when the exact certificate of its multipliers (certificate.joint_bound)
clears the incumbent, branches on its most fractional geo when it stalls, and
stops at a depth limit. The fractional values are read from the eigenvector
behind the bound, the primal point Z* = T v v' of the dual function
(oracle.py). Children warm-start from the parent's multipliers, the fixing
multiplier at zero. Inside a branch and bound an approximate dual solution is
enough, since every multiplier vector gives a valid bound (Fischer, Gruber,
Rendl and Sotirov 2006, Sect. 4.1; Palagi et al. in Anjos and Lasserre 2012,
Ch. 28).

The stall rule is read from measured trajectories (#965): at the end of stage 4
of the schedule 60 * 2^k <= 800 evaluations (900 evaluations in all), every node
that never pruned sat below -0.9% of the incumbent and every node that pruned
later above -1.6%; a node below -2% there, or not pruned by 1700, branches.
The solver aims HEADROOM past the incumbent: stopping exactly at it left margins
near 1e-8, which the exact certificate can lose, and a child warm-started from
such a parent stops at the same margin again, which once grew one node into 511
open leaves.

Fischer, I., Gruber, G., Rendl, F. and Sotirov, R. (2006). Computational
    experience with a bundle approach for semidefinite cutting plane relaxations
    of Max-Cut and Equipartition. Math. Program. 105, 451-469.
Anjos, M. F. and Lasserre, J. B., eds. (2012). Handbook on Semidefinite, Conic
    and Polynomial Optimization. Springer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Set, Tuple

import cvxpy as cp
import numpy as np

from .certificate import ExactData, hull_bound, joint_bound, shor_bound
from .dual import smoothed_lbfgs
from .hull import Band, ControlBound, rss_of
from .lifts import Lift
from .oracle import Oracle

#: Stage k of the smoothing schedule takes up to this many evaluations
#: (measured on the 40-geo nodes, #965: shorter early stages and longer late ones
#: halve an easy prune's cost against a flat 400).
BUDGET = lambda k: min(60 * 2 ** k, 800)  # noqa: E731
#: The solver aims this far past the incumbent, relative, so the exact certificate
#: (which loses up to about 3e-8 relative to the double bound) still clears it.
HEADROOM = 1e-6
#: The stall rule: at ``STALL_AT`` evaluations a node below ``STALL_BELOW`` of the
#: incumbent branches, and every node not pruned by ``STALL_LAST`` does.
STALL_AT, STALL_BELOW, STALL_LAST = 900, -0.02, 1700


@dataclass
class SearchStats:
    """What the joint-lift search did."""

    nodes: int = 0
    pruned: int = 0
    branched: int = 0
    infeasible: int = 0
    cert_failed: int = 0
    evals: int = 0
    lbfgs_seconds: float = 0.0
    cert_seconds: float = 0.0
    open: List[tuple] = field(default_factory=list)


@dataclass
class Certificate:
    """The outcome of :func:`certify` for one treatment group."""

    incumbent: Fraction                 # the incumbent's exact RSS at its double slope
    controls: List[int]                 # the incumbent's control group, indices into X
    slope: float
    slope_cap: Fraction
    pieces: int
    pairs: int
    after_hull: int
    after_shor: int
    search: SearchStats
    certified: bool                     # every other design's RSS is at least the incumbent's
    seconds: float


def unpack(O: Oracle, yv: np.ndarray, p: int):
    """The oracle's multiplier vector in the form :func:`joint_bound` reads."""
    g = lambda k: yv[O.sl[k]]  # noqa: E731
    duals = ([g("z00")[0], g("dU"), g("dV"), g("su")[0], g("rU"), g("cV"), g("sym").reshape(p, p)]
             + [f for f in g("fam").reshape(16, p, p)])
    fixes = [(j, v, float(d)) for (j, v), d in zip(O.fixes, g("fix"))]
    return duals, float(g("ng")[0]), fixes


def stalled(evals: int, best: float, target: float) -> bool:
    margin = (best - target) / abs(target)
    return (evals >= STALL_AT and margin < STALL_BELOW) or evals >= STALL_LAST


def run_node(y, X, lift, C, UB, scale2, lo, hi, m, fixed, y_warm, stats: SearchStats,
             depth: int = 0, max_depth: int = 10) -> None:
    """Prune the node, or branch on its most fractional geo; record it open past ``max_depth``."""
    p = X.shape[1]
    ones = sum(v for _, v in fixed)
    if ones > m or len(fixed) - ones > p - m:
        stats.infeasible += 1
        return
    O = Oracle(y, X, lo, hi, m, C=C, fixes=fixed)
    y0 = np.zeros(O.dim)
    if y_warm is not None:
        y0[:len(y_warm)] = y_warm[:O.dim]          # same layout; the new fixing starts at 0
    stats.nodes += 1
    yb = y0
    for head in (HEADROOM, 100 * HEADROOM):
        target = float(UB) / scale2 * (1 + head)
        t0 = time.perf_counter()
        th, yb, inf = smoothed_lbfgs(O, y0, target=target, stage_budget=BUDGET, handoff=stalled)
        stats.lbfgs_seconds += time.perf_counter() - t0
        stats.evals += inf["evals"]
        if th < target:
            break
        duals, ngd, fixes = unpack(O, yb, p)
        t1 = time.perf_counter()
        cb, _, _ = joint_bound(lift, y, X, lo, hi, m, duals=duals, nogood=(C, ngd), fixes=fixes)
        stats.cert_seconds += time.perf_counter() - t1
        if cb is not None and cb >= UB:
            stats.pruned += 1
            return
        stats.cert_failed += 1
        y0 = yb
    if depth >= max_depth or len(fixed) == p:
        stats.open.append((lo, hi, m, tuple(fixed)))
        return
    # branch on the most fractional geo of the eigen-primal behind the bound
    Sm, _ = O.S_k(yb)
    v = np.linalg.eigh(Sm)[1][:, 0]
    u = np.clip(v[1:p + 1] / v[0], 0, 1) if abs(v[0]) > 1e-12 else np.full(p, 0.5)
    free = [j for j in range(p) if j not in dict(fixed)]
    j = min(free, key=lambda j: abs(u[j] - 0.5))
    stats.branched += 1
    for val in (0, 1):
        run_node(y, X, lift, C, UB, scale2, lo, hi, m, fixed + [(j, val)], yb, stats, depth + 1, max_depth)


def incumbent(y: np.ndarray, X: np.ndarray, *, restarts: int = 200, seed: int = 0) -> Tuple[float, Set[int]]:
    """The best control group an RSS-only add/drop/swap descent finds from random starts."""
    p = X.shape[1]
    rng = np.random.default_rng(seed)

    def descend(C):
        cur = rss_of(y, X, C)
        while True:
            moves = [mv for mv in ([C ^ {j} for j in range(p)]
                                   + [(C - {a}) | {b} for a in C for b in range(p) if b not in C]) if mv]
            if not moves:                       # one candidate control: nothing to move
                return cur, C
            vals = [rss_of(y, X, mv) for mv in moves]
            i = int(np.argmin(vals))
            if vals[i] < cur * (1 - 1e-12):
                C, cur = moves[i], vals[i]
            else:
                return cur, C

    starts = [set(rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist())
              for _ in range(restarts)]
    return min((descend(C) for C in starts), key=lambda r: r[0])


def pieces_to(b_inc: float, cap: Fraction, ratio: float = 1.25) -> List[Tuple[float, float]]:
    """Geometric slope pieces from ``b_inc / 1000`` past ``cap``, after ``(0, edge_0]``."""
    edges = [b_inc / 1e3]
    while Fraction(edges[-1]) < cap:
        edges.append(edges[-1] * ratio)
    return [(0.0, edges[0])] + list(zip(edges[:-1], edges[1:]))


def hull_survivors(y, X, D: ExactData, pieces, UB, *, two_tier: bool = True):
    """The (piece, size) pairs the hull stage cannot rule out, and what it did.

    The first tier is the band without the per-geo cap (hull.Band): one NNLS and
    simplex QPs at the band's ends, certified by the hull's first-order bound
    with the cap at m hi. Only the pairs it cannot rule out reach the second
    tier, the capped hull by Clarabel, so the survivors are the capped hull's.
    """
    p = X.shape[1]
    band = Band(y, X) if two_tier else None
    hull = ControlBound(y, X)
    stats = {"band_pruned": 0, "capped_solved": 0}
    out = []
    for lo, hi in pieces:
        for m in range(1, p + 1):
            if band is not None:
                v, w = band.value(lo, hi, m)
                if v >= float(UB) * (1 - 1e-6) and hull_bound(D, w, lo, hi, m, cap=m * hi) >= UB:
                    stats["band_pruned"] += 1
                    continue
            stats["capped_solved"] += 1
            v = hull.piece(lo, hi, m)
            if v < float(UB) * (1 - 1e-6) or hull_bound(D, hull.w.value, lo, hi, m) < UB:
                out.append((lo, hi, m))
    stats["band_solves"] = 0 if band is None else band.solves
    return out, stats


def certify(y: np.ndarray, X: np.ndarray, *, ratio: float = 1.25, restarts: int = 200,
            seed: int = 0, max_depth: int = 10) -> Certificate:
    """Certify that no control group with a positive slope has a lower RSS than the incumbent.

    Every prune compares an exact lower bound with the incumbent's exact RSS: the
    hull's first-order bound, the Shor certificate, and the joint lift's
    certificate with the no-good cut, each in :mod:`certificate`.
    """
    t_all = time.perf_counter()
    p = X.shape[1]
    _, C = incumbent(y, X, restarts=restarts, seed=seed)
    x = X[:, sorted(C)].sum(axis=1)
    b_inc = float((y @ x) / (x @ x))
    if not b_inc > 0:
        raise ValueError("no control group has a positive slope: every design fails the "
                         "correlation gate, so there is no incumbent to certify")
    D = ExactData(y, X)
    UB = D.rss(C, b_inc)
    cap = D.slope_cap()
    if cap is None:
        raise ValueError("the controls' Gram matrix is singular: no slope cap, no certificate")
    pieces = pieces_to(b_inc, cap, ratio)
    s1, _ = hull_survivors(y, X, D, pieces, UB)
    shor = Lift(y, X, "shor")
    s2 = []
    for pc in s1:
        v = shor.piece(*pc)
        if v < float(UB) * (1 - 1e-4):
            s2.append(pc)
            continue
        b, _, _ = shor_bound(shor, D, *pc, shor.scale)
        if b is None or b < UB:
            s2.append(pc)
    full = Lift(y, X, "full")
    stats = SearchStats()
    scale2 = float(np.linalg.norm(y)) ** 2
    for pc in s2:
        run_node(y, X, full, set(C), UB, scale2, *pc, [], None, stats, max_depth=max_depth)
    return Certificate(incumbent=UB, controls=sorted(int(j) for j in C), slope=b_inc, slope_cap=cap,
                       pieces=len(pieces), pairs=len(pieces) * p, after_hull=len(s1), after_shor=len(s2),
                       search=stats, certified=not stats.open, seconds=time.perf_counter() - t_all)
