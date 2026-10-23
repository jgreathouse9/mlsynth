"""SYNDES's two-way branch and bound under other branching rules and incumbents (#839).

:func:`search` is the library's ``branch.branch_and_bound`` loop with two
choices opened up, and everything else, the bounds, their certificates, the
cost model, the node order and the work limit, taken from the library as it
stands. A test holds it to the library count for count under the library's
choices (``rule="root"``, ``seeds="swap"``).

The branching unit (``rule``):

- ``root``, the library's: the free units in the order of the root
  trust region's maximiser, largest first, and every node splits on its first;
- ``easy``: at each bounded node, the free unit whose entry in that node's
  trust-region maximiser is largest in magnitude, the one the relaxation is
  surest of;
- ``hard``: the one whose entry is smallest in magnitude;
- ``pseudo``: reliability branching (Achterberg, Koch and Martin 2005). Each
  unit's pseudo-cost is the mean gain in bound per split of it, up and down. A
  unit with fewer than ``RELIABLE`` gains on either side is tried, at most
  ``LOOKAHEAD`` such trials in a row without a better score: both children's
  trust-region bounds, each charged as a bound. The score is
  ``(1 - MU) min + MU max`` of the two gains.

The incumbent (``seeds``):

- ``swap``, the library's: the swap search's local optima from ``RESTARTS``
  restarts;
- ``none``: no seed, so the search dives until it prices a design;
- ``ascent``: one seed, the ``K`` largest entries of the root trust region's
  maximiser, then the concave ascent of Qian, Shen and Xu: put the ``K``
  units with the largest entries of ``R sigma`` in the treated set until it
  stops changing.

Given the optimum, :func:`search` records the work at which the incumbent first
came within ``0.1%`` of it.
"""
from __future__ import annotations

import heapq
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from mlsynth.utils.marex_helpers.certify import round_down
from mlsynth.utils.syndes_helpers import branch as lib
from mlsynth.utils.syndes_helpers.enumeration import SearchSpace
from mlsynth.utils.syndes_helpers.exact import (CERTIFY_RTOL, CHUNK, WORK_LIMIT,
                                                SubsetFilter, _resolve_open, _settle,
                                                _stream_from_space, with_complements)
from mlsynth.utils.syndes_helpers.gram import maxcut_candidates, signs_from_subsets

RULES = ("root", "easy", "hard", "pseudo")
SEEDS = ("swap", "none", "ascent")

#: Reliability branching's defaults in Achterberg, Koch and Martin (2005):
#: eight gains a side before a pseudo-cost is trusted, four trials without a
#: better score, and a sixth of the larger gain in the score.
RELIABLE, LOOKAHEAD, MU = 8, 4, 1.0 / 6.0

#: The incumbent counts as found once within this fraction of the optimum.
NEAR = 1e-3


def ascent(R: np.ndarray, K: int, start: np.ndarray, max_steps: int = 100) -> np.ndarray:
    """The ``K`` units with the largest entries of ``R sigma``, repeated from
    ``start`` until the treated set stops changing: each step raises
    ``sigma'R sigma``, since the objective is convex in ``sigma`` and the step
    maximises its linearisation over the designs of size ``K``."""
    S = np.sort(np.asarray(start, dtype=np.int64))
    N = R.shape[0]
    for _ in range(max_steps):
        sigma = -np.ones(N)
        sigma[S] = 1.0
        nxt = np.sort(np.argsort(-(R @ sigma), kind="stable")[:K])
        if np.array_equal(nxt, S):
            break
        S = nxt
    return S


class PseudoCosts:
    """Mean bound gain per split of each unit, by side (include, exclude)."""

    def __init__(self):
        self.total: Dict[Tuple[int, bool], float] = {}
        self.count: Dict[Tuple[int, bool], int] = {}

    def record(self, u: int, include: bool, gain: float) -> None:
        key = (u, include)
        self.total[key] = self.total.get(key, 0.0) + max(gain, 0.0)
        self.count[key] = self.count.get(key, 0) + 1

    def reliable(self, u: int) -> bool:
        return min(self.count.get((u, True), 0), self.count.get((u, False), 0)) >= RELIABLE

    def mean(self, u: int, include: bool, default: float) -> float:
        key = (u, include)
        return self.total[key] / self.count[key] if self.count.get(key) else default

    def overall(self) -> float:
        n = sum(self.count.values())
        return sum(self.total.values()) / n if n else 1.0


def score(down: float, up: float) -> float:
    return (1.0 - MU) * min(down, up) + MU * max(down, up)


def search(problem, engine, K: int, filt: Optional[SubsetFilter] = None, *,
           rule: str = "root", seeds: str = "swap", optimum: Optional[float] = None,
           restarts: int = lib.RESTARTS, seed: int = 0,
           work_limit: float = WORK_LIMIT) -> dict:
    """One size's search under ``rule`` and ``seeds``; the library's counts,
    design and bound, the trials made, and the work at which the incumbent first
    came within :data:`NEAR` of ``optimum`` (``None`` if it never did)."""
    if rule not in RULES or seeds not in SEEDS:
        raise ValueError(f"rule must be one of {RULES} and seeds one of {SEEDS}")
    filt = SubsetFilter() if filt is None else filt
    N, R, alpha = problem.N, engine.R, engine.alpha
    xr = lib.ExactR(engine)
    four_alpha = Fraction(4.0 * alpha)
    keep = Fraction(1) - Fraction(lib.PRUNE_RTOL)
    price = lib.PRICE_SECONDS_PER_UNIT * N
    learned = lib.LiftCost()

    def bound_cost(n, p):
        return lib.trust_region_seconds(n) + p * lib.certificate_seconds(n)

    def lift_cost(n, p):
        return learned(n) + p * lib.certificate_seconds(n)

    bound_rates, lift_rates = lib.PruneRates(), lib.PruneRates()
    costs = PseudoCosts()

    I0 = tuple(sorted(filt.forced_in))
    O0 = tuple(sorted(filt.forbidden))
    free0 = [u for u in range(N) if u not in filt.forced_in and u not in filt.forbidden]
    partners: Dict[int, set] = {}
    for i, j in filt.conflict_pairs:
        partners.setdefault(int(i), set()).add(int(j))
        partners.setdefault(int(j), set()).add(int(i))

    best: List = [np.inf, None]
    floor = [np.inf]
    pending: List[Tuple[np.ndarray, np.ndarray]] = []
    buffer: List[np.ndarray] = []
    counts = {"priced": 0, "buffered": 0, "nodes": 0, "lifts": 0, "pruned": 0,
              "work": 0.0, "trials": 0}
    near: List = [None]

    def offer(value: float, S: np.ndarray) -> None:
        if value < best[0]:
            best[0], best[1] = float(value), S.copy()
            if (near[0] is None and optimum is not None
                    and best[0] <= optimum * (1.0 + NEAR)):
                near[0] = counts["work"]

    def price_block(sets: np.ndarray) -> None:
        out = engine.bound(signs_from_subsets(N, sets))
        counts["priced"] += len(sets)
        if out.exact.any():
            j = int(np.argmin(np.where(out.exact, out.lower, np.inf)))
            offer(float(out.lower[j]), sets[j])
        open_ = (~out.exact) & (out.lower < best[0])
        if open_.any():
            pending.append((sets[open_], out.lower[open_]))

    def flush() -> None:
        if buffer:
            price_block(np.concatenate(buffer))
            buffer.clear()
            counts["buffered"] = 0

    def enumerate_subtree(space: SearchSpace) -> None:
        if space.size > CHUNK:
            flush()
            for block in _stream_from_space(space, filt, CHUNK):
                price_block(block)
            return
        for block in _stream_from_space(space, filt, CHUNK):
            buffer.append(block)
            counts["buffered"] += len(block)
        if counts["buffered"] >= CHUNK:
            flush()

    def discards(I, O, F, U, m) -> bool:
        if not (np.isfinite(best[0]) and m is not None and np.isfinite(U) and U > 0.0
                and 4.0 * alpha / U >= best[0] * (1.0 - lib.PRUNE_RTOL)):
            return False
        counts["work"] += lib.certificate_seconds(len(F))
        exact = xr.certify(I, O, F, K, m)
        if exact is None or exact <= 0 or not four_alpha >= Fraction(best[0]) * keep * exact:
            return False
        floor[0] = min(floor[0], round_down(four_alpha / exact))
        return True

    def space_of(I, O) -> SearchSpace:
        return SearchSpace.build(N, K, forced=I, forbidden=O, strata=filt.strata)

    def lower_of(I, O, F) -> float:
        """A child's trust-region bound on ``f``, as a trial reads it."""
        r, n = K - len(I), len(F)
        if r < 0 or r > n:
            return np.inf
        if r == 0 or r == n:
            sigma = -np.ones(N)
            sigma[list(I) + (list(F) if r == n else [])] = 1.0
            return 4.0 * alpha / float(sigma @ R @ sigma)
        counts["work"] += lib.trust_region_seconds(n)
        counts["trials"] += 1
        U, _, _ = lib.trust_region(*lib.node_parts(R, I, O, F), 2 * r - n)
        return 4.0 * alpha / U if np.isfinite(U) and U > 0.0 else 0.0

    def children(I, O, F, u):
        out = partners.get(u, set())
        rest = tuple(k for k in F if k != u)
        return ((I, O + (u,), rest),
                (I + (u,), O + tuple(k for k in rest if k in out),
                 tuple(k for k in rest if k not in out)))

    def choose(I, O, F, x, here: float):
        """The unit to split on, under ``rule``."""
        if rule == "root":
            return F[0]
        mag = np.abs(np.asarray(x, dtype=float))
        if rule == "easy":
            return F[int(np.argmax(mag))]
        if rule == "hard":
            return F[int(np.argmin(mag))]
        # pseudo: candidates by their pseudo-cost score; unreliable ones tried
        default = costs.overall()
        order = sorted(F, key=lambda k: -score(costs.mean(k, False, default),
                                               costs.mean(k, True, default)))
        top, top_score, stale = order[0], -np.inf, 0
        for k in order:
            if costs.reliable(k):
                s = score(costs.mean(k, False, default), costs.mean(k, True, default))
            else:
                (Id, Od, Fd), (Iu, Ou, Fu) = children(I, O, F, k)
                down = lower_of(Id, Od, Fd) - here
                up = lower_of(Iu, Ou, Fu) - here
                costs.record(k, False, down)
                costs.record(k, True, up)
                s = score(max(down, 0.0), max(up, 0.0))
                stale += 1
            if s > top_score:
                top, top_score, stale = k, s, 0
            elif stale >= LOOKAHEAD:
                break
        return top

    # the incumbent
    blocks = []
    if seeds == "swap" and restarts:
        blocks.append(maxcut_candidates(engine, K, n_restarts=restarts, seed=seed))
    r0 = K - len(I0)
    x0 = None
    if free0 and 0 < r0 < len(free0):
        _, x0, _ = lib.trust_region(*lib.node_parts(R, I0, O0, free0), 2 * r0 - len(free0))
        order = tuple(free0[i] for i in np.argsort(-x0, kind="stable"))
    else:
        order = tuple(free0)
    if seeds == "ascent" and x0 is not None and not I0 and not O0:
        start = np.array(order[:K], dtype=np.int64)
        blocks.append(ascent(R, K, start)[None, :])
    for block in blocks:
        for S in with_complements(block, N):
            if len(S) == K and filt.accepts(tuple(int(i) for i in S)):
                offer(_settle(problem, engine, S)[0], S)

    heap: List = []
    dive: List = []
    pushed = [0]

    def found() -> bool:
        return bool(np.isfinite(best[0]) or pending or buffer)

    def push(key, I, O, F, space, inherited, origin) -> None:
        pushed[0] += 1
        entry = (key, -pushed[0], I, O, F, space, inherited, origin)
        if found():
            heapq.heappush(heap, entry)
        else:
            dive.append(entry)

    def pop():
        if dive and found():
            for waiting in dive:
                heapq.heappush(heap, waiting)
            dive.clear()
        return dive.pop() if dive else heapq.heappop(heap)

    root = space_of(I0, O0)
    if root.size and not any(partners.get(u, set()) & set(I0) for u in I0):
        push(-np.inf, I0, O0, order, root, -np.inf, None)
    closed = True
    while heap or dive:
        if counts["work"] >= work_limit:
            closed = False
            break
        entry = pop()
        key, _, I, O, F, space, inherited, origin = entry
        r, n = K - len(I), len(F)
        bucket = space.size.bit_length()
        p_bound = bound_rates.rate(bucket)
        if r == 0 or r == n or space.size * price * p_bound <= bound_cost(n, p_bound):
            pricing = space.size * price
            if counts["work"] + pricing > work_limit and not (r == 0 or r == n):
                dive.append(entry)
                closed = False
                break
            counts["work"] += pricing
            enumerate_subtree(space)
            continue
        counts["nodes"] += 1
        counts["work"] += lib.trust_region_seconds(n)
        parts = lib.node_parts(R, I, O, F)
        s = 2 * r - n
        U, x, m = lib.trust_region(*parts, s)
        here = 4.0 * alpha / U if np.isfinite(U) and U > 0.0 else 0.0
        if rule == "pseudo" and origin is not None:
            u_split, include, parent = origin
            costs.record(u_split, include, here - parent)
        hit = discards(I, O, F, U, m)
        bound_rates.record(bucket, hit)
        p_lift = lift_rates.rate(bucket)
        lifted = not hit and space.size * price * p_lift > lift_cost(n, p_lift)
        if lifted:
            counts["lifts"] += 1
            goal = (4.0 * alpha / (best[0] * (1.0 - lib.PRUNE_RTOL))
                    if np.isfinite(best[0]) else None)
            U, m, evaluations = lib.lift(*parts, s, target=goal)
            learned.observe(evaluations)
            counts["work"] += evaluations * lib.evaluation_seconds(n)
            hit = discards(I, O, F, U, m)
            lift_rates.record(bucket, hit)
        if hit:
            counts["pruned"] += 1
            continue
        if lifted and m is not None:
            counts["work"] += lib.certificate_seconds(n)
            exact = xr.certify(I, O, F, K, m)
            if exact is not None and exact > 0:
                inherited = max(inherited, round_down(four_alpha / exact))
        if np.isfinite(U) and U > 0.0:
            key = max(key, 4.0 * alpha / U)
        u = choose(I, O, F, x, here)
        for include, child in zip((False, True), children(I, O, F, u)):
            sub = space_of(child[0], child[1])
            if sub.size:
                push(key, *child, sub, inherited, (u, include, here))

    open_lower = np.inf
    rayleigh = engine.global_lower_bound(K)
    for _, _, I, O, F, space, inherited, _ in heap + dive:
        r, n = K - len(I), len(F)
        if r == 0 or r == n:
            enumerate_subtree(space)
            continue
        counts["work"] += lib.trust_region_seconds(n)
        U, _, m = lib.trust_region(*lib.node_parts(R, I, O, F), 2 * r - n)
        if np.isfinite(U) and U > 0.0 and 4.0 * alpha / U > inherited:
            counts["work"] += lib.certificate_seconds(n)
            exact = xr.certify(I, O, F, K, m)
            if exact is not None and exact > 0:
                inherited = max(inherited, round_down(four_alpha / exact))
        open_lower = min(open_lower, inherited)
    flush()

    if pending:
        candidates = np.concatenate([p for p, _ in pending])
        bounds = np.concatenate([q for _, q in pending])
        value, S, _ = _resolve_open(problem, candidates, bounds, best[0], best[1])
        if S is not None:
            offer(value, np.asarray(S))
    value, _, _, gap = _settle(problem, engine, best[1])
    lower = min(floor[0], open_lower, value - gap)
    if rayleigh is not None:
        lower = max(lower, rayleigh)
    scale = max(abs(value), 1e-300)
    return {"rule": rule, "seeds": seeds, "nodes": counts["nodes"],
            "lifts": counts["lifts"], "pruned": counts["pruned"],
            "priced": counts["priced"], "trials": counts["trials"],
            "work": counts["work"], "value": value, "lower": float(lower),
            "closed": closed,
            "certified": bool(gap / scale <= CERTIFY_RTOL
                              and value - lower <= CERTIFY_RTOL * scale),
            "near_work": near[0],
            "treated": [int(i) for i in np.sort(best[1])]}
