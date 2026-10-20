"""A branch and bound over the two-way global SYNDES design (#803).

The tree fixes one unit at a time into the treated set (``I``) or out of it
(``O``), in the order of the root relaxation's maximiser, treated side first. A
node is priced set by set when that costs less than bounding it, and otherwise
bounded by ``node_lower(alpha, node_upper(...))`` from ``bounds.py`` and pruned
when the bound reaches the incumbent. The costs decide where the time goes,
never the answer.

Pricing is the library's: the closed form of ``gram.py`` for every set in a
batch, exact where its minimiser keeps the sign conditions; the sets where it
does not, and whose closed form is under the incumbent, are solved at the end by
projected gradient (``solve_partition_batch``), as the library's enumeration
does. The incumbent starts from the library's swap-search candidates.

Every discarded set lies under a node whose bound reached an incumbent no
better than the final one, or was priced, so the search is exact. Its bound is
the final value when it closes; in floating point, to the rounding of the
closed forms and of the trust-region bound, which this study does not certify.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations, islice
from math import comb as sets_count
from typing import Optional, Sequence

import numpy as np

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers.exact import _settle
from mlsynth.utils.syndes_helpers.gram import (BoundEngine, TwoWayProblem,
                                               maxcut_candidates, signs_from_subsets)
from mlsynth.utils.syndes_helpers.partition import solve_partition_batch, split_indices

from .bounds import lift_upper, node_lower, node_upper, trs_upper
from .dual import dual_upper

# The lifted bound: the interior-point solve of ``bounds.lift_upper``, or its
# regularised dual (``dual.py``) with the triangle inequalities, or with the
# lift's products alone; the dual stops as soon as the node prunes.
LIFTS = {
    "ipm": lambda R, I, O, F, K, target=None: lift_upper(R, I, O, F, K),
    "dual": lambda R, I, O, F, K, target=None: dual_upper(R, I, O, F, K, target=target),
    "products": lambda R, I, O, F, K, target=None: dual_upper(R, I, O, F, K, target=target,
                                                            cuts="products"),
}

PRUNE_RTOL = 1e-12
CHUNK = 20_000
TABLE_LIMIT = 1_000_000     # entries of a cached combination table
_TABLES = {}


def _combos(n: int, r: int) -> np.ndarray:
    """``combinations(range(n), r)`` as an array, cached when small."""
    table = _TABLES.get((n, r))
    if table is None:
        table = (np.zeros((1, 0), dtype=np.int64) if r == 0 else
                 np.array(list(combinations(range(n), r)), dtype=np.int64))
        if table.size <= TABLE_LIMIT:
            _TABLES[(n, r)] = table
    return table


@dataclass
class Result:
    treated: np.ndarray
    value: float
    lower: float
    certified: bool
    nodes_bounded: int
    nodes_lifted: int
    nodes_pruned: int
    sets_priced: int
    sets_solved: int
    seconds: float


def _price_seconds(engine: BoundEngine, N: int, K: int) -> float:
    """Seconds per set on the enumeration's own path: build the sets, score them."""
    t = time.perf_counter()
    block = np.array(list(_first(combinations(range(N), K), CHUNK)), dtype=np.int64)
    engine.bound(signs_from_subsets(N, block))
    return (time.perf_counter() - t) / max(len(block), 1)


def _first(it, k):
    """The next ``k`` items of ``it``. ``islice`` stops at ``k`` without pulling
    a ``k + 1``-th item; a loop that tests its index after pulling drops one item
    per call."""
    return islice(it, k)


def _bound_seconds(R: np.ndarray, K: int):
    """Seconds per bound as a function of the free units, from two timings."""
    N = R.shape[0]
    sizes = sorted({max(4, N // 4), N})
    times = []
    for n in sizes:
        F = list(range(n)); O = list(range(n, N))
        r = min(K, n - 1)
        t = time.perf_counter()
        for _ in range(3):
            node_upper(R, [], O, F, r)
        times.append((time.perf_counter() - t) / 3)
    if len(sizes) == 1:
        return lambda n: times[0]
    (n0, n1), (t0, t1) = sizes, times
    slope = max((t1 - t0) / (n1 ** 3 - n0 ** 3), 0.0)
    return lambda n: max(t0, t0 + slope * (n ** 3 - n0 ** 3))


def _lift_seconds(R: np.ndarray, K: int, lift=lift_upper):
    """Seconds per lift as a power law in the free units, fitted at two sizes no
    larger than 20 (a lift at the full size can take longer than the search)."""
    N = R.shape[0]
    sizes = sorted({min(10, N), min(20, N)})
    times = []
    for n in sizes:
        F = list(range(n)); O = list(range(n, N))
        t = time.perf_counter()
        lift(R, [], O, F, min(K, n - 1))
        times.append(time.perf_counter() - t)
    if len(sizes) == 1:
        return lambda n: times[0]
    (n0, n1), (t0, t1) = sizes, times
    p = max(np.log(t1 / t0) / np.log(n1 / n0), 1.0)
    return lambda n: t0 * (max(n, n0) / n0) ** p


def search(
    Y: np.ndarray,
    K: int,
    *,
    lam: Optional[float] = None,
    forced: Sequence[int] = (),
    forbidden: Sequence[int] = (),
    incumbent: Optional[np.ndarray] = None,
    restarts: int = 32,
    price_cost: Optional[float] = None,
    bound_cost: Optional[float] = None,
    lift_cost: Optional[float] = None,
    adaptive: bool = True,
    seed: int = 0,
    lift: str = "ipm",
) -> Result:
    """The two-way design's optimum by branch and bound. ``price_cost`` (seconds
    per set), ``bound_cost`` (per trust-region bound) and ``lift_cost`` (per
    lift; ``inf`` never lifts) override the calibration. ``lift``: ``"ipm"``,
    ``"dual"`` or ``"products"`` (``LIFTS``)."""
    if lift not in LIFTS:
        raise ValueError(f"lift must be one of {sorted(LIFTS)}, not {lift!r}.")
    lifted = LIFTS[lift]
    start = time.perf_counter()
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    engine = BoundEngine.from_problem(problem)
    N, R = problem.N, engine.R
    I0 = sorted(int(u) for u in forced)
    O0 = sorted(int(u) for u in forbidden)
    free0 = [u for u in range(N) if u not in set(I0) | set(O0)]
    if not 0 <= K - len(I0) <= len(free0):
        raise MlsynthEstimationError(
            "SYNDES found no admissible design: the forced and forbidden units "
            f"leave no treated set of size {K}.")

    price = _price_seconds(engine, N, K) if price_cost is None else price_cost
    bound = _bound_seconds(R, K) if bound_cost is None else (lambda n: bound_cost)
    lift_time = _lift_seconds(R, K, lifted) if lift_cost is None else (lambda n: lift_cost)

    def admissible(sets: np.ndarray) -> np.ndarray:
        ok = np.ones(len(sets), dtype=bool)
        for u in I0:
            ok &= (sets == u).any(axis=1)
        for u in O0:
            ok &= ~(sets == u).any(axis=1)
        return ok

    best = [np.inf, None]
    pending = []
    counts = {"bounded": 0, "lifted": 0, "pruned": 0, "priced": 0}
    buffer = []
    buffered = [0]
    # Prune rate by subtree size (log2 bucket), with a Laplace prior: a bound
    # that does not prune is paid for nothing, so it pays only on a subtree
    # whose pricing costs more than the bound over its chance of pruning.
    tried = np.zeros(80)
    hits = np.zeros(80)

    def price_block(sets: np.ndarray):
        out = engine.bound(signs_from_subsets(N, sets))
        counts["priced"] += len(sets)
        if out.exact.any():
            j = int(np.argmin(np.where(out.exact, out.lower, np.inf)))
            if out.exact[j] and out.lower[j] < best[0]:
                best[0], best[1] = float(out.lower[j]), sets[j].copy()
        open_ = (~out.exact) & (out.lower < best[0])
        if open_.any():
            pending.append((sets[open_], out.lower[open_]))

    def flush():
        """Price the buffered subtrees in one call: a subtree a bound did not
        pay for is small, and one call per subtree is mostly call overhead."""
        if buffer:
            price_block(np.concatenate(buffer))
            buffer.clear()
            buffered[0] = 0

    def enumerate_subtree(I, F, r):
        if sets_count(len(F), r) * max(r, 1) <= TABLE_LIMIT:
            idx = _combos(len(F), r)
            block = np.empty((len(idx), K), dtype=np.int64)
            block[:, :len(I)] = I
            block[:, len(I):] = np.asarray(F, dtype=np.int64)[idx]
            buffer.append(block)
            buffered[0] += len(block)
            if buffered[0] >= CHUNK:
                flush()
            return
        flush()
        gen = (tuple(I) + c for c in combinations(F, r))
        while True:
            block = np.array(list(_first(gen, CHUNK)), dtype=np.int64).reshape(-1, K)
            if not len(block):
                break
            price_block(block)

    seeds = []
    if incumbent is not None:
        seeds.append(np.atleast_2d(np.asarray(incumbent, dtype=np.int64)))
    if restarts:
        seeds.append(maxcut_candidates(engine, K, n_restarts=restarts, seed=seed))
    for block in seeds:
        block = np.sort(block, axis=1)
        block = block[admissible(block)]
        for s in block:
            value = _settle(problem, engine, s)[0]
            if value < best[0]:
                best[0], best[1] = value, s.copy()

    _, x = trs_upper(R, I0, O0, free0, K)
    order = [free0[i] for i in np.argsort(-x, kind="stable")]

    stack = [(I0, O0, 0)]
    while stack:
        I, O, pos = stack.pop()
        F = order[pos:]
        r, n = K - len(I), len(F)
        if r < 0 or r > n:
            continue
        sets_here = sets_count(n, r)
        bucket = min(int(np.log2(max(sets_here, 1))), 79)
        p_prune = (hits[bucket] + 1.0) / (tried[bucket] + 2.0) if adaptive else 1.0
        if r == 0 or r == n or sets_here * price * p_prune <= bound(n):
            enumerate_subtree(I, F, r)
            continue
        counts["bounded"] += 1
        tried[bucket] += 1
        lb = node_lower(engine.alpha, node_upper(R, I, O, F, K))
        if lb >= best[0] * (1 - PRUNE_RTOL):
            counts["pruned"] += 1
            hits[bucket] += 1
            continue
        if sets_here * price > lift_time(n):
            counts["lifted"] += 1
            goal = 4.0 * engine.alpha / (best[0] * (1 - PRUNE_RTOL))
            lb = node_lower(engine.alpha, lifted(R, I, O, F, K, target=goal))
            if lb >= best[0] * (1 - PRUNE_RTOL):
                counts["pruned"] += 1
                continue
        u = F[0]
        stack.append((I, O + [u], pos + 1))
        stack.append((I + [u], O, pos + 1))

    flush()
    solved = 0
    if pending:
        sets = np.concatenate([s for s, _ in pending])
        lows = np.concatenate([b for _, b in pending])
        sets = sets[lows < best[0]]
        solved = len(sets)
        if solved:
            t_idx, c_idx = split_indices(N, sets)
            out = solve_partition_batch(problem, t_idx, c_idx, max_iter=4000, tol=1e-14)
            j = int(np.argmin(out.value))
            if out.value[j] < best[0]:
                best[0], best[1] = float(out.value[j]), sets[j].copy()
    if best[1] is None:
        raise MlsynthEstimationError("SYNDES found no admissible design.")  # pragma: no cover - the size check above
    value, _, _, gap = _settle(problem, engine, best[1])
    return Result(treated=np.sort(best[1]), value=value, lower=value - gap,
                  certified=gap <= 1e-8 * max(abs(value), 1e-300),
                  nodes_bounded=counts["bounded"], nodes_lifted=counts["lifted"],
                  nodes_pruned=counts["pruned"],
                  sets_priced=counts["priced"],
                  sets_solved=solved, seconds=time.perf_counter() - start)
