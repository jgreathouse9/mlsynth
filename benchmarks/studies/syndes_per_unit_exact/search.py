"""Per-unit SYNDES by a bound-ordered search over treated sets (#806).

Doudchenko et al. (2021, Appendix A.3) observe that the per-unit problem is ``K``
one-way problems, one per treated unit, each with every control unit as a donor.
In ``mlsynth``'s formulation the treated units' weights on other treated units
are forced to zero (``q_ij <= 1 - D_j``) and the penalty is on the weights, so
for a treated set ``S`` of size ``K``

    f(S) = (1/K) sum_{i in S} phi_i(S^c),
    phi_i(P) = min over v on the simplex over P of (1/T)||y_i - Y_P v||^2 + lam ||v||^2.

Removing donors can only raise ``phi_i``, so ``phi_i(N \\ {i}) <= phi_i(S^c)`` for
every ``S`` that contains ``i``, and

    lb(S) = (1/K) sum_{i in S} phi_i(N \\ {i})

bounds ``f(S)`` from below and is separable: ``N`` fits bound every set.
:func:`ordered_subsets` visits the sets in nondecreasing order of that bound,
:class:`Fits` prices each visited set exactly, and :func:`search` stops when the
next bound reaches the incumbent, which proves the incumbent optimal.

A second bound sits between the two. Removing every other treated unit raises
``phi_i`` at least as much as removing any one of them, so

    lb2(S) = (1/K) sum_{i in S} max_{j in S \ {i}} phi_i(N \ {i, j})

bounds ``f(S)`` too, and only ``j`` in the support of ``i``'s root fit can raise
a term above its root value. Those pair fits are few, and every set that
contains the pair reuses them; with ``pair=True`` the search prunes a set on
``lb2`` before pricing it.

Two reuses keep the pricing cheap, and both are exact. A treated unit's root fit
stays optimal on a smaller pool whenever the removed donors carried no weight in
it. And a fit with only the root's support removed stays optimal for any larger
removal that misses its own support.

This is a study script, not library code. :func:`scip` runs the MIP the library
hands SCIP, through ``miqp_accel.solve_warm_cut``, so SCIP's own status is
visible.
"""
from __future__ import annotations

import heapq
import math
from itertools import combinations
from time import perf_counter
from typing import Dict, FrozenSet, Iterator, Optional, Sequence, Tuple

import numpy as np

from mlsynth.utils.solvers.active_set import solve_simplex_qp


class Fits:
    """``phi_i`` on any pool, with exact reuse of solved fits.

    Every fit is checked: the active-set solver must report convergence, and
    the Frank-Wolfe gap of the returned weights, relative to the value, is
    recorded in ``max_rel_gap``.
    """

    def __init__(self, Y: np.ndarray, lam: float):
        self.Y = np.asarray(Y, dtype=float)
        self.T, self.N = self.Y.shape
        self.lam = float(lam)
        self.n_qp = 0
        self.n_reused = 0
        self.max_rel_gap = 0.0
        self._cache: Dict[Tuple[int, FrozenSet[int]], Tuple[float, FrozenSet[int]]] = {}
        self.root_value = np.empty(self.N)
        self.root_support = []
        for i in range(self.N):
            value, support = self._solve(i, frozenset())
            self.root_value[i] = value
            self.root_support.append(support)

    def _solve(self, i: int, removed: FrozenSet[int]) -> Tuple[float, FrozenSet[int]]:
        key = (i, removed)
        if key in self._cache:
            return self._cache[key]
        pool = [j for j in range(self.N) if j != i and j not in removed]
        root_t = math.sqrt(self.T)
        B = np.vstack([self.Y[:, pool] / root_t, math.sqrt(self.lam) * np.eye(len(pool))])
        A = np.concatenate([self.Y[:, i] / root_t, np.zeros(len(pool))])
        w, info = solve_simplex_qp(B, A, return_info=True)
        if not info["converged"]:
            raise RuntimeError(f"phi_{i} did not converge on a pool of {len(pool)}")
        r = A - B @ w
        value = float(r @ r)
        grad = -2.0 * (B.T @ r)
        fw = float(grad @ w - grad.min())
        if value > 0:
            self.max_rel_gap = max(self.max_rel_gap, abs(fw) / value)
        self.n_qp += 1
        support = frozenset(pool[k] for k in np.flatnonzero(w > 0))
        self._cache[key] = (value, support)
        return value, support

    def phi(self, i: int, removed: FrozenSet[int]) -> float:
        """``phi_i`` on the pool ``N`` minus ``{i}`` and ``removed``."""
        removed = removed - {i}
        hit = removed & self.root_support[i]
        if not hit:
            self.n_reused += 1
            return float(self.root_value[i])
        value, support = self._solve(i, hit)
        if not (support & removed):
            return value
        return self._solve(i, removed)[0]

    def pair_term(self, i: int, removed: FrozenSet[int]) -> float:
        """``max_j phi_i(N \\ {i, j})`` over ``j`` in ``removed``."""
        best = float(self.root_value[i])
        for j in (removed - {i}) & self.root_support[i]:
            best = max(best, self._solve(i, frozenset({j}))[0])
        return best

    def price(self, S: Sequence[int]) -> float:
        """``f(S)``, exactly."""
        S = frozenset(int(i) for i in S)
        return sum(self.phi(i, S) for i in S) / len(S)


def ordered_subsets(a: np.ndarray, K: int) -> Iterator[Tuple[float, Tuple[int, ...]]]:
    """``K``-subsets of positions ``0..n-1`` in nondecreasing order of their sum.

    ``a`` must be sorted ascending. Each subset is generated once, from the
    subset that lowers by one its first position not at its minimum, so a child
    never sums below its parent and a heap pops them in order. A node has at
    most two children.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    start = tuple(range(K))
    heap = [(float(a[:K].sum()), start)]
    while heap:
        total, p = heapq.heappop(heap)
        yield total, p
        m = next((j for j in range(K) if p[j] != j), K)
        for j in (m - 1, m):
            if 0 <= j < K:
                ceiling = p[j + 1] if j + 1 < K else n
                if p[j] + 1 < ceiling:
                    child = p[:j] + (p[j] + 1,) + p[j + 1:]
                    heapq.heappush(heap, (total - a[p[j]] + a[p[j] + 1], child))


def search(Y: np.ndarray, K: int, lam: float, *, pair: bool = True,
           max_sets: Optional[int] = None,
           time_limit: Optional[float] = None) -> dict:
    """The per-unit optimum by bound-ordered search, and what it cost."""
    t0 = perf_counter()
    fits = Fits(Y, lam)
    t_root = perf_counter() - t0
    order = np.argsort(fits.root_value, kind="stable")
    a = fits.root_value[order]
    best, best_set = math.inf, None
    popped = priced = cut_short = pair_pruned = 0
    closed = False
    root_bound = float(a[:K].sum()) / K
    for _, pos in ordered_subsets(a, K):
        S = tuple(int(order[k]) for k in pos)
        bound = math.fsum(fits.root_value[list(S)]) / K
        if bound >= best:
            closed = True
            break
        if (max_sets is not None and popped >= max_sets) or (
                time_limit is not None and perf_counter() - t0 > time_limit):
            break
        popped += 1
        removed = frozenset(S)
        if pair:
            floor = {i: fits.pair_term(i, removed) for i in S}
            running = math.fsum(floor.values())
            if running / K >= best:
                pair_pruned += 1
                continue
        else:
            floor = {i: float(fits.root_value[i]) for i in S}
            running = bound * K
        for i in sorted(S, key=lambda u: -floor[u]):
            running += fits.phi(i, removed) - floor[i]
            if running / K >= best:
                cut_short += 1
                break
        else:
            priced += 1
            value = running / K
            if value < best:
                best, best_set = value, tuple(sorted(S))
    else:
        closed = True
    return {
        "design": best_set, "value": best, "closed": closed,
        "root_bound": root_bound,
        "root_gap": (best - root_bound) / best if best_set else math.nan,
        "popped": popped, "priced": priced, "cut_short": cut_short,
        "pair_pruned": pair_pruned,
        "qp": fits.n_qp, "reused": fits.n_reused,
        "max_rel_gap": fits.max_rel_gap,
        "t_root": t_root, "t_total": perf_counter() - t0,
    }


def brute_force(Y: np.ndarray, K: int, lam: float) -> Tuple[Tuple[int, ...], float]:
    """Every ``K``-subset priced from fresh fits. For checking small panels."""
    best, best_set = math.inf, None
    for S in combinations(range(Y.shape[1]), K):
        value = _fresh_price(Y, S, lam)
        if value < best:
            best, best_set = value, S
    return best_set, best


def _fresh_price(Y: np.ndarray, S: Sequence[int], lam: float) -> float:
    T, N = Y.shape
    total = 0.0
    for i in S:
        pool = [j for j in range(N) if j not in S]
        B = np.vstack([Y[:, pool] / math.sqrt(T), math.sqrt(lam) * np.eye(len(pool))])
        A = np.concatenate([Y[:, i] / math.sqrt(T), np.zeros(len(pool))])
        w = solve_simplex_qp(B, A)
        r = A - B @ w
        total += float(r @ r)
    return total / len(S)


def scip(Y: np.ndarray, K: int, lam: float, *, time_limit: Optional[float] = 60.0,
         gap_limit: Optional[float] = 0.05,
         wall_limit: Optional[float] = None) -> dict:
    """:func:`scip_here` in a fresh process, killed at ``wall_limit``.

    In this study's first runs SCIP corrupted the heap (``malloc(): invalid
    size``) on two 40-unit panels, which killed the run; on one of them, solved
    alone, it ran past its 60 s limit until killed at 400 s; and on Walmart it
    returned ``solver_error`` at its limit, which cvxpy cannot unpack. In a child
    process with a wall clock these are outcomes: ``crashed``, ``hung`` and
    ``error``. ``wall_limit`` defaults to 600 s plus twice the time limit, which
    covers cvxpy's compile (70 s at ``N = 100``).
    """
    import multiprocessing as mp
    import queue

    if wall_limit is None:
        wall_limit = 600.0 + 2.0 * (time_limit if time_limit is not None else 300.0)
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    child = ctx.Process(target=_scip_child,
                        args=(results, Y, K, lam, time_limit, gap_limit))
    t0 = perf_counter()
    child.start()
    out = None
    while out is None and perf_counter() - t0 < wall_limit:
        try:
            out = results.get(timeout=1.0)
        except queue.Empty:
            if not child.is_alive():
                break
    if out is None:
        status = ("hung" if child.is_alive()
                  else f"crashed (exit {child.exitcode})")
        out = {"status": status, "wall": perf_counter() - t0,
               "scip_time": math.nan, "design": None, "value": math.nan,
               "scip_gap": math.nan}
    child.join(timeout=5)
    if child.is_alive():
        child.kill()
        child.join()
    return out


def _scip_child(results, Y, K, lam, time_limit, gap_limit) -> None:
    results.put(scip_here(Y, K, lam, time_limit=time_limit, gap_limit=gap_limit))


def scip_here(Y: np.ndarray, K: int, lam: float, *, time_limit: Optional[float] = 60.0,
              gap_limit: Optional[float] = 0.05) -> dict:
    """The library's per-unit MIP on SCIP, timed from building to the answer."""
    import cvxpy as cp

    from mlsynth.utils.miqp_accel import solve_warm_cut
    from mlsynth.utils.syndes_helpers.formulation import build_syndes_problem_components

    N = Y.shape[1]
    t0 = perf_counter()
    D = cp.Variable(N, boolean=True)
    comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam, mode="per_unit")
    prob = cp.Problem(cp.Minimize(comp.objective), list(comp.constraints))
    try:
        _, info = solve_warm_cut(prob, D, gap_limit=gap_limit, time_limit=time_limit)
    except Exception as exc:
        return {"status": f"error: {str(exc)[:60]}", "wall": perf_counter() - t0,
                "scip_time": math.nan, "design": None, "value": math.nan,
                "scip_gap": math.nan}
    wall = perf_counter() - t0
    if D.value is None:
        return {"status": info.status, "wall": wall, "scip_time": info.solve_time,
                "design": None, "value": math.nan, "scip_gap": info.gap}
    S = tuple(int(i) for i in np.flatnonzero(np.asarray(D.value) > 0.5))
    return {"status": info.status, "wall": wall, "scip_time": info.solve_time,
            "design": S, "value": _fresh_price(Y, S, lam), "scip_gap": info.gap}


def branch_and_bound(Y: np.ndarray, K: int, lam: float, *,
                     time_limit: Optional[float] = None) -> dict:
    """The per-unit optimum by branching on units, with a bound that tightens.

    A node fixes a set ``F`` of treated units and a set ``E`` of units kept as
    controls; the rest, ``U``, are undecided. Every completion ``S`` contains
    ``F``, so each pool in it lacks ``F``, and

        f(S) >= (1/K) [ sum_{i in F} phi_i(N \\ F)
                        + the K - |F| smallest phi_u(N \\ (F + u)), u in U ].

    At a leaf, ``F = S``, the bound is ``f(S)``. Units in ``E`` stay donors. The
    search is depth first, including the undecided unit with the smallest term
    before excluding it, from the incumbent of the ``K`` smallest root fits.
    """
    t0 = perf_counter()
    fits = Fits(Y, lam)
    N = fits.N
    order = np.argsort(fits.root_value, kind="stable")
    root_bound = float(fits.root_value[order[:K]].sum()) / K
    best_set = tuple(sorted(int(u) for u in order[:K]))
    best = fits.price(best_set)
    nodes = pruned = 0
    closed = True
    stack = [(frozenset(), frozenset())]
    while stack:
        if time_limit is not None and perf_counter() - t0 > time_limit:
            closed = False
            break
        F, E = stack.pop()
        nodes += 1
        U = [u for u in range(N) if u not in F and u not in E]
        need = K - len(F)
        if need > len(U):
            continue
        fixed = math.fsum(fits.phi(i, F) for i in F)
        terms = sorted((fits.phi(u, F), u) for u in U)
        bound = (fixed + math.fsum(v for v, _ in terms[:need])) / K
        if bound >= best:
            pruned += 1
            continue
        if need == 0:
            best, best_set = bound, tuple(sorted(F))
            continue
        if need == len(U):
            S = tuple(sorted(F | set(U)))
            value = fits.price(S)
            if value < best:
                best, best_set = value, S
            continue
        u = terms[0][1]
        stack.append((F, E | {u}))
        stack.append((F | {u}, E))
    return {
        "design": best_set, "value": best, "closed": closed,
        "root_bound": root_bound, "root_gap": (best - root_bound) / best,
        "nodes": nodes, "pruned": pruned, "qp": fits.n_qp,
        "reused": fits.n_reused, "max_rel_gap": fits.max_rel_gap,
        "t_total": perf_counter() - t0,
    }
