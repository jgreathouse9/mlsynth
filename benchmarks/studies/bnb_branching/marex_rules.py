"""MAREX's branch and bound under other branching rules, incumbents and node orders (#839).

:func:`search` is the library's ``marex_helpers.branch.branch_and_bound`` loop
with three choices opened up, and everything else, the persp+N bound and its
certificate, the cheap bound, pricing, the cost model and the work limit, taken
from the library as it stands. A test holds it to the library count for count
under the library's choices (``rule="largest"``, ``guess=True``,
``order="best"``).

The branching market (``rule``), read from the bounded node's persp+N indicator
``z`` (the treated weight where the bound returned none):

- ``largest``, the library's: the free market with the largest ``z``, the one
  the relaxation most wants treated;
- ``decided``: the largest ``|2 z - 1|``, the market the relaxation is surest
  of either way, BiqMac's "easy first";
- ``fractional``: the smallest ``|2 z - 1|``, the market it is least sure of,
  BiqMac's "difficult first";
- ``pseudo``: reliability branching (Achterberg, Koch and Martin 2005), as in
  :mod:`.syndes_rules`. A market's pseudo-cost is the mean gain in bound per
  split of it, read when a child is bounded. A market with fewer than
  ``RELIABLE`` gains a side is tried on the node's cheap bound (the treated fit
  over the markets it may treat plus the control floor), both children, at most
  ``LOOKAHEAD`` trials in a row without a better score, each trial charged as
  two fits. Markets are tried in the ``largest`` order.

The incumbent (``guess``): at each bounded node the library fits the ``r`` free
markets with the largest score as a candidate; ``guess=False`` leaves the
incumbent to the subtrees the search prices.

The node order (``order``): ``best``, the library's, pops the smallest bound;
``depth`` pops the last node pushed, the child that includes the market first.

Given the optimum, :func:`search` records the work at which the incumbent first
came within ``NEAR`` of it.
"""
from __future__ import annotations

import dataclasses
import heapq
import itertools
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mlsynth.utils.marex_helpers import branch as lib
from mlsynth.utils.marex_helpers.certify import ExactData, round_down, round_up
from mlsynth.utils.marex_helpers.exact import (ClusterDesign, _TopK, _chunks,
                                               count_candidates, side_fit)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

from .syndes_rules import LOOKAHEAD, NEAR, RELIABLE, PseudoCosts, score

RULES = ("largest", "decided", "fractional", "pseudo")
ORDERS = ("best", "depth")

#: Modelled seconds per fit, for the pseudo-cost trials. The library's cost
#: model has no fit term (it fits only under pricing and its guess), so a trial
#: is charged as pricing this many sets at the library's price.
FIT_SETS = 200
FIT_SECONDS = FIT_SETS * lib.PRICE_SECONDS


def search(B: np.ndarray, A: np.ndarray, m: int, *, rule: str = "largest",
           guess: bool = True, order: str = "best",
           optimum: Optional[float] = None,
           price_seconds: float = lib.PRICE_SECONDS,
           lift_seconds=lib.lift_seconds,
           work_limit: float = lib.WORK_LIMIT) -> Dict:
    """One cluster's standard design of size ``m`` by the branch and bound.

    Returns the design, its bracket and whether the search closed, the counts
    the library reports, the trials made, and the work at which the incumbent
    first came within ``NEAR`` of ``optimum``.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    filt = SubsetFilter()
    data = ExactData(B, A)
    B, A, n = data.B, data.A, data.n
    best = _TopK(1)
    brackets: Dict[Tuple[int, ...], Tuple[Fraction, Fraction]] = {}
    floors: Dict[Tuple[int, ...], Fraction] = {}
    state = {"lifts": 0, "priced": 0, "work": 0.0, "trials": 0}
    left: List[Fraction] = []
    near: List[Optional[float]] = [None]
    costs = PseudoCosts()

    def fit_lower(cols: Sequence[int], linear) -> Tuple[Fraction, np.ndarray]:
        cols = list(cols)
        _, w = side_fit(B, A, cols, None)
        return data.interval(cols, w, None)[0], w

    def control_floor(cols: Tuple[int, ...]) -> Fraction:
        if cols not in floors:
            floors[cols] = fit_lower(cols, None)[0]
        return floors[cols]

    def threshold() -> Optional[Fraction]:
        if len(best.items) < best.k:
            return None
        return max(brackets[d.treated][1] for d in best.items)

    def clears(bound: Fraction) -> bool:
        cut = threshold()
        return cut is not None and bound > cut

    def fit(S: Tuple[int, ...], h_floor: Fraction) -> None:
        if S in brackets:
            return
        cols = list(S)
        g, w = side_fit(B, A, cols, None)
        lo_g, hi_g = data.interval(cols, w, None)
        if clears(lo_g + h_floor):
            return
        control = [j for j in range(n) if j not in S]
        h, v = side_fit(B, A, control, None)
        lo_h, hi_h = data.interval(control, v, None)
        brackets[S] = (lo_g + lo_h, hi_g + hi_h)
        best.offer(ClusterDesign(value=g + h, treated=S, control=tuple(control),
                                 w=w, v=v))
        if (near[0] is None and optimum is not None
                and best.items[0].value <= optimum * (1.0 + NEAR)):
            near[0] = state["work"]

    def price(sub: SubsetFilter, O: Tuple[int, ...], F: Tuple[int, ...]) -> None:
        h_floor = control_floor(tuple(sorted(O + F)))
        h_down = round_down(h_floor)
        for block in _chunks(n, [m], sub):
            state["priced"] += len(block)
            priced = np.nextafter(data.treated_bounds(block, None) + h_down, -np.inf)
            for i in np.argsort(priced, kind="stable"):
                cut = threshold()
                if cut is not None and priced[i] > round_up(cut):
                    break
                fit(tuple(int(j) for j in block[i]), h_floor)

    def cheap(I: Tuple[int, ...], O: Tuple[int, ...], F: Tuple[int, ...]) -> float:
        """A child's cheap bound, as a trial reads it."""
        state["trials"] += 1
        state["work"] += 2 * FIT_SECONDS
        if m - len(I) < 0 or m - len(I) > len(F) or not O + F:
            return np.inf
        return float(fit_lower(sorted(I + F), None)[0]
                     + control_floor(tuple(sorted(O + F))))

    def choose(I, O, F, s: Dict[int, float], here: float) -> int:
        if rule == "largest":
            return max(F, key=lambda k: (s[k], -k))
        if rule == "decided":
            return max(F, key=lambda k: (abs(2.0 * s[k] - 1.0), -k))
        if rule == "fractional":
            return min(F, key=lambda k: (abs(2.0 * s[k] - 1.0), k))
        base = costs.overall()
        pick, top, stale = None, -np.inf, 0
        for j in sorted(F, key=lambda k: (-s[k], k)):
            if costs.reliable(j):
                val = score(costs.mean(j, False, base), costs.mean(j, True, base))
            elif stale >= LOOKAHEAD:
                continue
            else:
                up = cheap(tuple(sorted(I + (j,))), O, tuple(k for k in F if k != j))
                down = cheap(I, tuple(sorted(O + (j,))), tuple(k for k in F if k != j))
                val = score(max(min(down, 1e300) - here, 0.0), max(min(up, 1e300) - here, 0.0))
            if val > top:
                pick, top, stale = j, val, 0
            else:
                stale += 1
        return pick

    counter = itertools.count()
    queue: List = [(Fraction(0), next(counter), (), (), tuple(range(n)), None)]

    def pop():
        return heapq.heappop(queue) if order == "best" else queue.pop()

    def push(items) -> None:
        if order == "best":
            for item in items:
                heapq.heappush(queue, item)
        else:
            queue.extend(reversed(items))

    while queue:
        bound, _, I, O, F, origin = pop()
        if clears(bound):
            continue
        r = m - len(I)
        if r < 0 or r > len(F):
            continue
        sub = dataclasses.replace(filt, forced_in=frozenset(I), forbidden=frozenset(O))
        count = count_candidates(n, [m], sub)
        if not count:
            continue
        if len(O) + len(F) - r < 1:
            continue
        k = max(len(I) + len(F), len(O) + len(F))
        if count < 2 or not count * price_seconds > lift_seconds(k):
            charge = count * price_seconds
            if state["work"] + charge > work_limit:
                left.append(bound)
                break
            state["work"] += charge
            price(sub, O, F)
            continue
        h_floor = control_floor(tuple(sorted(O + F)))
        g_floor, w = fit_lower(sorted(I + F), None)
        bound = max(bound, g_floor + h_floor)
        if clears(bound):
            continue
        charge = lift_seconds(k)
        if state["work"] + charge > work_limit:
            left.append(bound)
            break
        state["work"] += charge
        nb = lib.node_bound(data, m, I, O, F, None, None)
        state["lifts"] += 1
        if nb.lower is not None:
            bound = max(bound, nb.lower)
        if origin is not None:
            j0, include, parent = origin
            costs.record(j0, include, max(float(bound) - parent, 0.0))
        U = sorted(I + F)
        s = nb.z if nb.z is not None else {j: float(w[U.index(j)]) for j in F}
        if guess:
            fit(tuple(sorted(I + tuple(sorted(F, key=lambda q: (-s[q], q))[:r]))),
                h_floor)
        if clears(bound):
            continue
        j = choose(I, O, F, s, float(bound))
        rest = tuple(q for q in F if q != j)
        push([(bound, next(counter), tuple(sorted(I + (j,))), O, rest,
               (j, True, float(bound))),
              (bound, next(counter), I, tuple(sorted(O + (j,))), rest,
               (j, False, float(bound)))])

    left += [item[0] for item in queue]
    top = best.items[0]
    lower, upper = brackets[top.treated]
    for S, (lo, _) in brackets.items():
        if S != top.treated and lo <= upper:
            lower = min(lower, lo)
    still_open = [b for b in left if not clears(b)]
    lower = min([lower] + still_open)
    return {"value": top.value, "treated": list(top.treated),
            "lower": float(round_down(lower)), "upper": float(round_up(upper)),
            "closed": not still_open, "lifts": state["lifts"],
            "priced": state["priced"], "work": state["work"],
            "trials": state["trials"], "fitted": len(brackets),
            "near_work": near[0]}
