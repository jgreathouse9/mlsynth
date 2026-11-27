"""POPMUSIC over a PANGEO design: the incumbent improved by exact re-covers of clusters of its groups.

After the exact path's proof stops at its cap (:func:`.columns.solve_exact`), the best design it holds can sit far
above the optimum: at 120 geos and Q = 2 the start design was 11% above the best design known, which is what most of
the certified gap measured. :func:`improve` lowers it before the certify rounds. Each pass splits the design's groups
into disjoint clusters (:func:`clusters`): a seed group drawn at random and the groups nearest it by the centroid of
their markets' centred paths, while the cluster holds at most ``dim`` markets. Each cluster's markets are re-covered
exactly: the volume-bound tree (:func:`.vtree.search`) over every group inside them (:func:`subproblem`), from the
arm root's duals on those markets, searching below the cluster's own rigorous upper bound, with as many groups as the
rest of the design leaves short of the floor k. A cover found there replaces the cluster's groups, and its upper bound
is below theirs, so the design's bound never rises. ``dim`` starts at ``alpha`` and grows by ``delta`` after a pass
that improves nothing or after ``repeat`` passes at one size, while it is below the arm's size, and every tree draws
on one node budget, so the result is the same on every machine.

This is Algorithm 1 of Bijiga, Warwicker and Rebennack ("Branch-and-Cut POPMUSIC for the Electric Vehicle Routing
Problem With Time Windows", Networks 2026, doi:10.1002/net.70072, p. 12), with a design's groups in place of routes,
the centroid of a group's paths in place of a route's centre of gravity (their eq. 53, p. 11), and the volume tree in
place of their branch and cut. Their dimension grows after every pass; here it stays while passes keep improving, up to
``repeat``. The method is POPMUSIC (partial optimisation metaheuristic under special intensification conditions),
which they cite to Essays and Surveys in Metaheuristics (Springer, 2002), pp. 613-629.

Measured on a 120-geo Q = 2 seasonal arm from the start design (10.033, single-threaded): 9.0416 after 11.0 million
nodes over 53 subproblems, the last improvement in a cluster of 112 markets; the best design known is 9.0416, and
moves of 2 to 6 groups found nothing. No cluster is the whole arm: that would be the capped proof again.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence

import numpy as np

from ...exceptions import MlsynthConfigError
from . import columns, cover, vtree

try:  # numba is the optional ``fast`` extra, as for :mod:`.cover`
    from numba import njit
except ImportError:  # pragma: no cover - the suite runs with numba installed
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name (see :mod:`.columns`).
CANONICAL_NAME = "mlsynth.utils.pangeo_helpers.popmusic"
_CACHE = __name__ == CANONICAL_NAME

ALPHA = 16              # markets in a cluster at the first pass
DELTA = 8               # markets a cluster grows by
REPEAT = 3              # passes at one size while they keep improving
SUB_LIMIT = 2_000_000   # nodes one subproblem's tree may place
SEED = 0                # the generator of the clusters' seeds, fixed so the design is reproducible
WAVE = 8                # clusters solved together, on threads; fixed, so the design does not depend on the threads


@njit(cache=_CACHE, nogil=True)
def _enumerate(U, n, Q, B):  # pragma: no cover - compiled; checked through rows_within
    """Every group of 2 to min(2Q, |U|) of the ascending markets U: its row of a complete pool of n units (the groups
    of fewer units first, then the lexicographic rank, :func:`.columns.column_index`) and its members as positions
    into U, padded with -1. ``B[a, b]`` is the binomial coefficient a choose b."""
    nu = U.size
    top = min(2 * Q, nu)
    total = 0
    for m in range(2, top + 1):
        total += B[nu, m]
    rows = np.empty(total, dtype=np.int64)
    mem = -np.ones((total, max(top, 0)), dtype=np.int64)
    c = np.empty(max(top, 1), dtype=np.int64)
    out = 0
    base = 0
    for m in range(2, top + 1):
        for i in range(m):
            c[i] = i
        while True:
            idx = base
            prev = -1
            for i in range(m):
                g = U[c[i]]
                for j in range(prev + 1, g):
                    idx += B[n - 1 - j, m - 1 - i]
                prev = g
                mem[out, i] = c[i]
            rows[out] = idx
            out += 1
            i = m - 1
            while i >= 0 and c[i] == nu - m + i:
                i -= 1
            if i < 0:
                break
            c[i] += 1
            for j in range(i + 1, m):
                c[j] = c[j - 1] + 1
        base += B[n, m]
    return rows, mem


def _binomials(n: int, top: int) -> np.ndarray:
    """``(n + 1, top + 1)`` binomial coefficients in int64."""
    B = np.zeros((n + 1, top + 1), dtype=np.int64)
    for a in range(n + 1):
        for b in range(min(a, top) + 1):
            B[a, b] = math.comb(a, b)
    return B


def rows_within(units, n: int, Q: int):
    """The rows of a complete pool of ``n`` units (:func:`.columns.column_index`) of every group of 2 to
    min(2Q, |units|) of the ascending positions ``units``, ascending by size and then lexicographically, and their
    members as positions into ``units`` padded with -1: ``(rows, members)``."""
    U = np.ascontiguousarray(np.asarray(units, dtype=np.int64))
    top = min(2 * int(Q), U.size)
    if top < 2:
        return np.zeros(0, dtype=np.int64), np.zeros((0, max(top, 0)), dtype=np.int64)
    return _enumerate(U, int(n), int(Q), _binomials(int(n), top))


def subproblem(pool, units, Q: int):
    """The groups of ``pool`` (complete) inside the ascending positions ``units``, as a pool over positions into
    ``units`` with the arm pool's costs, bounds and splits: ``(sub, rows)``, ``rows`` their rows of ``pool``."""
    U = np.asarray(units, dtype=np.int64)
    rows, mem = rows_within(U, int(pool.unit_indices.size), Q)
    sub = columns.ColumnPool(np.arange(U.size), mem, pool.cost[rows], pool.side_a[rows], True,
                             pool.cost_lo[rows], pool.cost_up[rows])
    return sub, rows


def clusters(centroids: np.ndarray, sizes: Sequence[int], dim: int, rng: np.random.Generator) -> List[List[int]]:
    """A pass's clusters of a design's groups (their indices): while groups remain, a seed drawn by ``rng`` from
    them, then the remaining groups in order of squared distance from its centroid (ties by index), each joining
    when the cluster's markets stay within ``dim`` and skipped otherwise, until the cluster holds ``dim`` markets or
    none remain. A seed larger than ``dim`` stands alone. The clusters are disjoint and hold every group once."""
    C = np.asarray(centroids, dtype=float)
    left = list(range(len(sizes)))
    parts = []
    while left:
        seed = left[int(rng.integers(len(left)))]
        d = np.sum((C[left] - C[seed]) ** 2, axis=1)
        order = [left[i] for i in np.argsort(d, kind="mergesort")]
        pick, used = [seed], int(sizes[seed])
        for g in order:
            if used >= dim:
                break
            if g == seed or used + int(sizes[g]) > dim:
                continue
            pick.append(g)
            used += int(sizes[g])
        for g in pick:
            left.remove(g)
        parts.append(pick)
    return parts


@dataclass(frozen=True)
class _Duals:
    """The arm root's cover duals on a subproblem's markets, with no group-count dual: the form vtree.search reads."""

    pi: np.ndarray
    sigma: float = 0.0


def _count(name: str, value, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < least:
        raise MlsynthConfigError(f"POPMUSIC {name} must be an integer of at least {least}; got {value!r}.")
    return int(value)


def _upper(pool, rows) -> Fraction:
    """A design's rigorous upper bound: the sum of its groups' cost_up, in rationals."""
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _commit(old: List[List[int]], new: List[List[int]], cheaper: List[bool], total: int, k: int):
    """A wave's covers taken in cluster order: cluster i's ``new`` groups replace its ``old`` ones when ``cheaper[i]``
    (its rigorous upper bound is below theirs) and the design, ``total`` groups before the wave, keeps at least ``k``
    groups after it. Returns the groups each cluster ends with and the covers taken."""
    out, taken = [], 0
    for o, nw, ch in zip(old, new, cheaper):
        if ch and total - len(o) + len(nw) >= k:
            out.append(list(nw))
            total += len(nw) - len(o)
            taken += 1
        else:
            out.append(list(o))
    return out, taken


def _jobs(n_jobs) -> int:
    if n_jobs is None:
        return os.cpu_count() or 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise MlsynthConfigError(f"POPMUSIC n_jobs must be a positive integer, or None for every core; got {n_jobs!r}.")
    return int(n_jobs)


def improve(pool, root, k: int, Q: int, rows: Sequence[int], paths: np.ndarray, node_limit: int, *,
            alpha: Optional[int] = None, delta: Optional[int] = None, repeat: Optional[int] = None,
            sub_limit: Optional[int] = None, seed: Optional[int] = None,
            progress: Optional[np.ndarray] = None, n_jobs: Optional[int] = None) -> dict:
    """POPMUSIC from the design ``rows`` (rows of the complete ``pool``, an exact cover with at least ``k`` groups),
    as in the module docstring, its clusters by the centroids of ``paths`` (the arm's paths, one row a position of
    ``pool.unit_indices``, centred), its trees placing at most ``node_limit`` groups in all and ``sub_limit`` each.
    ``alpha``, ``delta``, ``repeat``, ``sub_limit`` and ``seed`` are the module's constants when None.

    Returns ``rows`` (the design, int64), ``upper`` (its rigorous upper bound, a Fraction, at most the start's),
    ``start_upper``, ``passes``, ``subproblems`` (trees run), ``improved`` (covers taken), ``nodes``, ``seconds`` and
    ``trace``, one ``(dim, subproblems, improved, nodes)`` a pass. ``progress`` (:func:`.cover.new_progress`)
    receives every tree's nodes and searches after each wave.

    A pass's clusters of two or more groups run in waves of ``WAVE``, in cluster order. Each tree of a wave of m
    trees may place the lesser of ``sub_limit`` and the budget left at the wave's start divided by m (when that is
    less than one node, only as many trees run as nodes are left, one node each), and every tree's group floor is k
    less the groups outside its cluster at the wave's start. The trees run on ``n_jobs`` threads (every core when
    None; 1 runs them in this thread): the compiled tree releases Python's lock, so they run in parallel. Their covers
    are then taken in cluster order (:func:`_commit`). None of this depends on ``n_jobs``, so neither does the
    design: ``n_jobs`` changes the time taken and nothing else."""
    t0 = time.perf_counter()
    if isinstance(node_limit, bool) or not isinstance(node_limit, (int, np.integer)) or node_limit < 0:
        raise MlsynthConfigError(f"POPMUSIC node_limit must be a non-negative integer; got {node_limit!r}.")
    alpha = _count("alpha", ALPHA if alpha is None else alpha, 2)
    delta = _count("delta", DELTA if delta is None else delta, 1)
    repeat = _count("repeat", REPEAT if repeat is None else repeat, 1)
    sub_limit = _count("sub_limit", SUB_LIMIT if sub_limit is None else sub_limit, 1)
    jobs = _jobs(n_jobs)
    n = int(pool.unit_indices.size)
    P = np.asarray(paths, dtype=float)
    pi = np.asarray(root.pi, dtype=float)
    design = [int(j) for j in rows]
    start = _upper(pool, design)
    rng = np.random.default_rng(SEED if seed is None else seed)
    units = lambda j: [int(u) for u in pool.members[j] if u >= 0]  # noqa: E731
    left = int(node_limit)
    passes = subs = taken = nodes = 0
    dim, at = alpha, 0
    trace = []
    while dim < n and left > 0:
        passes += 1
        cent = np.array([P[units(j)].mean(axis=0) for j in design])
        parts = clusters(cent, [len(units(j)) for j in design], dim, rng)
        groups_of = [[design[i] for i in part] for part in parts]
        active = [c for c, g in enumerate(groups_of) if len(g) >= 2]
        changed, subs0, nodes0 = 0, subs, nodes

        def solve(c, floor, limit):
            """Cluster c's markets re-covered exactly below its own upper bound: (old bound, new groups, nodes)."""
            groups = groups_of[c]
            U = sorted(u for j in groups for u in units(j))
            sub, srows = subproblem(pool, U, Q)
            old = _upper(pool, groups)
            res = vtree.search(sub, _Duals(pi[U]), np.arange(len(sub)), floor, columns._up(old), limit)
            return old, [int(j) for j in srows[res.rows]], int(res.nodes)
        for w0 in range(0, len(active), WAVE):
            if left <= 0:
                break
            wave = active[w0:w0 + WAVE]
            limit = min(sub_limit, left // len(wave))
            if limit == 0:
                wave, limit = wave[:left], 1
            total = sum(len(g) for g in groups_of)
            floors = [max(int(k) - (total - len(groups_of[c])), 1) for c in wave]
            if jobs > 1 and len(wave) > 1:
                with ThreadPoolExecutor(max_workers=jobs) as ex:
                    results = list(ex.map(solve, wave, floors, [limit] * len(wave)))
            else:
                results = [solve(c, f, limit) for c, f in zip(wave, floors)]
            cheaper = [bool(nw) and _upper(pool, nw) < old for old, nw, _ in results]
            kept, took = _commit([groups_of[c] for c in wave], [nw for _, nw, _ in results], cheaper, total, int(k))
            for c, g in zip(wave, kept):
                groups_of[c] = g
            spent = sum(u for _, _, u in results)
            subs += len(wave)
            nodes += spent
            left -= spent
            changed += took
            if progress is not None:
                progress[cover.P_SEARCHES] += float(len(wave))
                progress[cover.P_TOTAL] += float(spent)
        out = [j for g in groups_of for j in g]
        design = out
        taken += changed
        trace.append((dim, subs - subs0, changed, nodes - nodes0))
        at += 1
        if changed == 0 or at >= repeat:
            dim, at = dim + delta, 0
    return dict(rows=np.asarray(design, dtype=np.int64), upper=_upper(pool, design), start_upper=start,
                passes=passes, subproblems=subs, improved=taken, nodes=nodes, seconds=time.perf_counter() - t0,
                trace=trace)
