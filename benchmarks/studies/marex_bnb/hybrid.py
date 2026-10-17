"""The certified search with a cost rule: bound a node or enumerate it.

``certified_bfs`` bounded every node whose subtree held more than 64 treated
sets. On GeoLift with 8 treated that made it 22 times slower than the
library's exact backend, which prices every set by a closed form in about a
microsecond and fits only the 22 of 77 million sets the closed form cannot
rule out. A lift costs as much as pricing tens of millions of sets, so most of
the 33 lifts were spent on subtrees cheaper to enumerate (``test_hybrid`` has
the decomposition).

Here a node is bounded only when ``should_bound``: when pricing its subtree
would cost more than ``kappa`` times its lift. Otherwise its subtree is
enumerated, with certified bounds:

``certified_enumerate`` prices a block of treated sets by the library's closed
form, ``min { w'Gw : 1'w = 1 }`` over each set, and certifies it. The closed
form's minimiser ``c = x / 1'x`` (``G_S x = 1``) defines ``y = Bc_S c``, and
for any ``y`` the Wolfe dual bounds the simplex fit:
``g(S) >= 2 min_j y'b_j - |y|^2 = 2 min_j (G_S c)_j - c'G_S c``. At this
``c`` the dual equals the closed form. ``c`` is whatever the floating-point
solve returned; the dual holds for it exactly. Each set reads its block of the
panel's floating-point Gram ``Gf`` and never the data, so a set costs O(m^3)
like the library's pricing. Two roundings separate the computed dual from the
exact one, both bounded whatever order the sums run in (Higham, Accuracy and
Stability, ch. 3): the Gram's own, ``|Gf - G| <= gamma_T |Bc|'|Bc|``
entrywise, and the products with ``c``, ``gamma_m |Gf||c|``, with
``gamma_k = k u / (1 - k u)``, ``u = 2^-53``. The bound is evaluated in
floating point, less those errors, inflated for the error of computing them,
and rounded down. A set whose solve fails gets ``0``.

A set of the subtree is then worth ``g_bound + h(O | F)``, the control side by
its Wolfe certificate once per subtree. Sets under the incumbent's certified
value are priced exactly (``certified_bnb.certified_fit``) in bound order.

Costs: ``price_cost`` is seconds per set, measured on the panel unless given;
``lift_cost(K)`` is seconds for a lift with a ``K``-row PSD block, by default
``210 (K / 116)^4.8``, fitted to Clarabel on GeoLift (``node_solvers.log``).
Both are machine-dependent estimates; they decide where the search spends
time, never what it proves.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from fractions import Fraction
from math import comb

import numpy as np

from benchmarks.studies.marex_bnb.certified_bnb import certified_fit
from benchmarks.studies.marex_bnb.certified_dnn import (indicator_lift,
                                                         lift_bound,
                                                         treated_lift)
from benchmarks.studies.marex_bnb.certify import Panel, down

U = 2.0 ** -53
CHUNK = 20_000


def should_bound(sets, price_cost, lift_cost, kappa=1.0):
    """Whether pricing ``sets`` costs more than ``kappa`` lifts.

    A subtree of one set is a leaf: there is nothing to bound it against but
    the set itself, so it is never bounded whatever the costs.
    """
    return sets >= 2 and sets * price_cost > kappa * lift_cost


def default_lift_cost(K):
    return 210.0 * (K / 116.0) ** 4.8


def certified_enumerate(panel, sets):
    """Certified lower bounds on the treated fit ``g`` of each row of ``sets``.

    Reads each set's block of the panel's floating-point Gram ``Gf`` (and of
    ``Gabs = |Bc|'|Bc|``), never the data, so a set costs O(m^3) like the
    library's pricing.
    """
    T = panel.Bc.shape[0]
    Gf, Gabs = panel.Gf, panel.Gabs
    gT = T * U / (1 - T * U)
    gT = gT / (1 - gT)                       # Gabs is itself rounded down by at most this
    sets = np.asarray(sets, dtype=np.int64)
    out = np.zeros(len(sets))
    if not len(sets):
        return out
    m = sets.shape[1]
    gm = m * U / (1 - m * U)
    for lo in range(0, len(sets), CHUNK):
        S = sets[lo:lo + CHUNK]
        b = len(S)
        G = Gf[S[:, :, None], S[:, None, :]]
        Ga = Gabs[S[:, :, None], S[:, None, :]]
        with np.errstate(all="ignore"):
            try:
                x = np.linalg.solve(G, np.ones((b, m, 1)))[:, :, 0]
            except np.linalg.LinAlgError:
                x = np.full((b, m), np.nan)
                for i in range(b):
                    try:
                        x[i] = np.linalg.solve(G[i], np.ones(m))
                    except np.linalg.LinAlgError:
                        pass
            s = x.sum(axis=1)
            ok = np.isfinite(s) & (s > 0) & np.all(np.isfinite(x), axis=1)
            c = np.where(ok[:, None], x / s[:, None], 0.0)     # y = Bc_S c, any c
            ac = np.abs(c)
            v = np.einsum("bij,bj->bi", G, c)                  # ~ G_S c = (b_j'y)_j
            q = np.einsum("bi,bi->b", c, v)                    # ~ c'G_S c = |y|^2
            av = np.einsum("bij,bj->bi", np.abs(G), ac)
            Ac = np.einsum("bij,bj->bi", Ga, ac)
            err_p = gm * av + gT * Ac
            err_q = (gm * np.einsum("bi,bi->b", ac, np.abs(v))
                     + np.einsum("bi,bi->b", ac, err_p))
            val = 2.0 * v.min(axis=1) - q
            slack = (2.0 * err_p.max(axis=1) + err_q) * (1 + 8 * U) \
                + 8 * U * (2.0 * np.abs(v).max(axis=1) + np.abs(q))
            lb = np.nextafter(val - slack, -np.inf)
        lb[~ok | ~np.isfinite(lb)] = 0.0
        out[lo:lo + b] = np.maximum(lb, 0.0)
    return out


def _measure_price_cost(panel, m, reps=200_000):
    """Seconds per set of enumerating, generation included, on a full batch.

    A small batch overstates the per-set cost several times over (fixed
    overheads), which once made the search bound a subtree it should have
    enumerated; so the measurement runs the enumeration's own path on blocks
    of ``CHUNK`` sets, twice, and keeps the faster.
    """
    n = panel.Bc.shape[1]
    best = np.inf
    for _ in range(2):
        walk = itertools.islice(itertools.combinations(range(n), m), reps)
        t0 = time.perf_counter()
        count = 0
        while True:
            flat = np.fromiter(itertools.chain.from_iterable(
                itertools.islice(walk, CHUNK)), dtype=np.int64)
            if not flat.size:
                break
            sets = flat.reshape(-1, m)
            certified_enumerate(panel, sets)
            count += len(sets)
        if count:
            best = min(best, (time.perf_counter() - t0) / count)
    return best


def search(B, A, m, price_cost=None, lift_cost=default_lift_cost, kappa=1.0,
           full_margin=0.10, time_limit=None, log=None):
    """Best-first certified search; returns the incumbent, the gap and a log."""
    if kappa < 1.0:
        raise ValueError(f"kappa must be at least 1: a lift pays only on a "
                         f"subtree that costs more to enumerate. Got {kappa}.")
    n = B.shape[1]
    panel = Panel(B - A[:, None])
    if price_cost is None:
        price_cost = _measure_price_cost(panel, m)
    INF = Fraction(10) ** 300
    inc = {"set": None, "lo": None, "hi": INF}
    kept, seen = [], set()
    st = {"nodes": 0, "lifts": 0, "lift_s": 0.0, "enumerated": 0,
          "enumerated_sets": 0, "enumerate_s": 0.0, "fitted": 0,
          "pruned": 0}
    t_start = time.perf_counter()

    def price(S):
        key = tuple(sorted(int(j) for j in S))
        if key in seen:
            return
        seen.add(key)
        st["fitted"] += 1
        C = [j for j in range(n) if j not in key]
        lo_g, hi_g, _ = certified_fit(panel, B, A, list(key))
        lo_h, hi_h, _ = certified_fit(panel, B, A, C)
        lo, hi = lo_g + lo_h, hi_g + hi_h
        if lo >= inc["hi"]:
            return
        if hi < inc["hi"]:
            if inc["set"] is not None:
                kept.append((inc["lo"], inc["set"]))
            inc.update(set=key, lo=lo, hi=hi)
        else:
            kept.append((lo, key))

    def enumerate_subtree(I, O, F, r, floor):
        """Price every set of the subtree, certified; return its lower bound."""
        t0 = time.perf_counter()
        st["enumerated"] += 1
        h_lo = certified_fit(panel, B, A, sorted(O + F))[0]
        lowest = INF
        walk = itertools.combinations(F, r)
        while True:
            flat = np.fromiter(itertools.chain.from_iterable(
                itertools.islice(walk, CHUNK)), dtype=np.int64)
            if not flat.size:
                break
            R = flat.reshape(-1, r) if r else np.zeros((1, 0), dtype=np.int64)
            sets = np.hstack([np.tile(np.array(I, dtype=np.int64), (len(R), 1)), R])
            st["enumerated_sets"] += len(sets)
            g_lo = certified_enumerate(panel, sets)
            lbs = [Fraction(v) + h_lo for v in g_lo] if len(sets) <= 64 else None
            order = np.argsort(g_lo, kind="stable")
            for i in order:
                lb = Fraction(g_lo[i]) + h_lo
                lb = max(lb, floor)
                lowest = min(lowest, lb)
                if lb >= inc["hi"]:
                    break
                price(list(sets[i]))
            del lbs
        st["enumerate_s"] += time.perf_counter() - t0
        return lowest

    def lift(level, I, O, F, h_lo):
        r = m - len(I)
        L = treated_lift(n, m, I, O, F) if level == 2 else \
            indicator_lift(n, m, I, O, F, cuts=False)
        t0 = time.perf_counter()
        lb, _, _, marg = lift_bound(panel, L, marginals=True)
        st["lift_s"] += time.perf_counter() - t0
        st["lifts"] += 1
        lb = Fraction(lb) + (h_lo if level == 2 else 0)
        score = {j: marg.get(("z", j), 0.0) for j in F} if marg else None
        if score:
            price(I + sorted(F, key=lambda k: (-score[k], k))[:r])
        return lb, score

    counter = itertools.count()
    queue = [(Fraction(0), next(counter), [], [], list(range(n)))]
    while queue:
        if time_limit and time.perf_counter() - t_start > time_limit:
            break
        prio, _, I, O, F = heapq.heappop(queue)
        if prio >= inc["hi"]:
            continue
        r = m - len(I)
        if r < 0 or r > len(F) or len(O) + len(F) - r < 1:  # pragma: no cover
            # Unreachable: only nodes with 1 <= r <= |F| - 1 are branched, which
            # keeps 0 <= r <= |F| in both children and the control count at n - m.
            continue
        size = comb(len(F), r)
        kw, kv, kf = len(I) + len(F), len(O) + len(F), len(F)
        K2, K3 = kw + kf - 1, kw + kv + kf - 2
        if not should_bound(size, price_cost, lift_cost(K2), kappa):
            enumerate_subtree(I, O, F, r, prio)
            continue
        st["nodes"] += 1
        lo_g, _, w = certified_fit(panel, B, A, I + F)
        lo_h, _, _ = certified_fit(panel, B, A, O + F)
        lb = max(lo_g + lo_h, prio)
        U_ = sorted(I + F)
        score = {j: w[U_.index(j)] for j in F}
        if lb < inc["hi"]:
            lb2, sc = lift(2, I, O, F, lo_h)
            lb, score = max(lb, lb2), sc or score
        if (lb < inc["hi"] and inc["hi"] < INF and lb >= (1 - full_margin) * inc["hi"]
                and should_bound(size, price_cost, lift_cost(K3), kappa)):
            lb3, sc = lift(3, I, O, F, lo_h)
            lb, score = max(lb, lb3), sc or score
        if lb >= inc["hi"]:
            st["pruned"] += 1
            continue
        j = max(F, key=lambda k: (score[k], -k))
        rest = [k for k in F if k != j]
        heapq.heappush(queue, (lb, next(counter), I + [j], O, rest))
        heapq.heappush(queue, (lb, next(counter), I, O + [j], rest))
        if log is not None:
            log(f"{time.perf_counter() - t_start:8.1f}s node {st['nodes']} "
                f"|I|={len(I)} |O|={len(O)} |F|={len(F)} subtree {size:,} "
                f"bound {float(lb):.8g} incumbent {float(inc['hi']):.8g}")
    kept = [(lo, S) for lo, S in kept if lo < inc["hi"]]
    open_lb = [item[0] for item in queue if item[0] < inc["hi"]]
    lows = open_lb + [lo for lo, _ in kept] + ([inc["lo"]] if inc["lo"] is not None else [])
    lw = min(lows) if lows else inc["hi"]
    st["price_cost"] = price_cost
    return {"set": inc["set"], "lo": inc["lo"], "hi": inc["hi"], "lower": lw,
            "gap": inc["hi"] - lw, "open": len(open_lb),
            "unresolved": sorted(S for _, S in kept), "stats": st,
            "seconds": time.perf_counter() - t_start}
