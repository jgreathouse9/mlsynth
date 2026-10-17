"""A best-first certified branch and bound with a hierarchy of node bounds.

Every comparison rests on a certified rational bound, as in ``certified_bnb``;
what changes is how much is spent per node. A node's bound is tried in order of
cost and stops at the first level that prunes it:

1. the monotone bound, ``g(I | F) + h(O | F)`` by Wolfe-dual certificates,
   about a millisecond;
2. the treated-side lift (``certified_dnn.treated_lift``) plus the same
   ``h(O | F)``, about a second at 30 free markets and half a minute at 40;
3. the full indicator lift, only when level 2 comes within ``full_margin`` of
   the incumbent without reaching it: about seven times the cost of level 2,
   and a few percent stronger.

A node whose subtree holds at most ``enumerate_below`` treated sets is not
bounded; its sets are priced directly, each with a certified interval. Each
lift's solution is rounded to a candidate incumbent (``I`` plus the ``r`` free
markets with the largest ``z``) and supplies the branching market (the largest
``z``). At the root the rounded set is also improved by single swaps (one
market in, one out) until none helps, and the result priced; the swaps only
propose candidates, so they cannot affect validity.

Nodes are explored best-first by their parent's bound, so at every moment

    certified lower bound = min(bounds in the queue, lo of every kept set),

and the incumbent's certified ``hi`` is an upper bound on the optimum: the
search reports a certified gap throughout, and at the end. A set is kept when
its interval overlaps the incumbent's, as in ``certified_bnb``.
"""
from __future__ import annotations

import heapq
import itertools
import time
from fractions import Fraction
from math import comb

import numpy as np

from benchmarks.studies.marex_bnb.certified_bnb import certified_fit
from benchmarks.studies.marex_bnb.certified_dnn import (indicator_lift,
                                                         lift_bound,
                                                         treated_lift)
from benchmarks.studies.marex_bnb.certify import Panel


def search(B, A, m, enumerate_below=64, full_margin=0.10, time_limit=None,
           polish=True, log=print):
    n = B.shape[1]
    panel = Panel(B - A[:, None])
    INF = Fraction(10) ** 300
    inc = {"set": None, "lo": None, "hi": INF}
    kept, seen = [], set()
    st = {"nodes": 0, "pruned": {1: 0, 2: 0, 3: 0}, "lifts": {2: 0, 3: 0},
          "lift_s": {2: 0.0, 3: 0.0}, "sets": 0, "enumerated": 0}
    t_start = time.perf_counter()

    def price(S):
        key = tuple(sorted(S))
        if key in seen:
            return
        seen.add(key)
        st["sets"] += 1
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

    def lift(level, I, O, F, h_lo):
        r = m - len(I)
        L = treated_lift(n, m, I, O, F) if level == 2 else \
            indicator_lift(n, m, I, O, F, cuts=False)
        t0 = time.perf_counter()
        lb, _, _, marg = lift_bound(panel, L, marginals=True)
        st["lift_s"][level] += time.perf_counter() - t0
        st["lifts"][level] += 1
        lb = Fraction(lb) + (h_lo if level == 2 else 0)
        score = {j: marg.get(("z", j), 0.0) for j in F} if marg else None
        if score:
            S0 = I + sorted(F, key=lambda k: (-score[k], k))[:r]
            price(S0)
            if polish and not I and not O:
                price(_swaps(B, A, S0))
        return lb, score

    def bound(I, O, F):
        """``(certified bound, branching scores)`` for a node, cheapest first."""
        lo_g, _, w = certified_fit(panel, B, A, I + F)
        lo_h, _, _ = certified_fit(panel, B, A, O + F)
        lb = lo_g + lo_h
        U = sorted(I + F)
        score = {j: w[U.index(j)] for j in F}
        if lb >= inc["hi"]:
            st["pruned"][1] += 1
            return lb, score
        lb2, sc = lift(2, I, O, F, lo_h)
        lb, score = max(lb, lb2), sc or score
        if lb >= inc["hi"]:
            st["pruned"][2] += 1
            return lb, score
        if inc["hi"] < INF and lb >= (1 - full_margin) * inc["hi"]:
            lb3, sc = lift(3, I, O, F, lo_h)
            lb, score = max(lb, lb3), sc or score
            if lb >= inc["hi"]:
                st["pruned"][3] += 1
        return lb, score

    counter = itertools.count()
    queue = [(Fraction(0), next(counter), [], [], list(range(n)))]

    def lower():
        qs = [item[0] for item in queue]
        ks = [lo for lo, _ in kept if lo < inc["hi"]]
        cand = qs + ks + ([inc["lo"]] if inc["lo"] is not None else [])
        return min(cand) if cand else inc["hi"]

    while queue:
        if time_limit and time.perf_counter() - t_start > time_limit:
            break
        prio, _, I, O, F = heapq.heappop(queue)
        if prio >= inc["hi"]:
            continue
        r = m - len(I)
        if r < 0 or r > len(F) or len(O) + len(F) - r < 1:
            continue
        if comb(len(F), r) <= enumerate_below:
            st["enumerated"] += 1
            for R in itertools.combinations(F, r):
                price(I + list(R))
            continue
        st["nodes"] += 1
        lb, score = bound(I, O, F)
        if lb >= inc["hi"]:
            continue
        j = max(F, key=lambda k: (score[k], -k))
        rest = [k for k in F if k != j]
        heapq.heappush(queue, (lb, next(counter), I + [j], O, rest))
        heapq.heappush(queue, (lb, next(counter), I, O + [j], rest))
        if log is not None:
            lw, hi = lower(), inc["hi"]
            log(f"{time.perf_counter() - t_start:8.1f}s node {st['nodes']:4d} "
                f"|I|={len(I)} |O|={len(O)} |F|={len(F)} bound {float(lb):.8g} "
                f"queue {len(queue):4d} lower {float(lw):.8g} incumbent "
                f"{float(hi):.8g} gap {float((hi - lw) / hi):.4%} "
                f"lifts {st['lifts'][2]}/{st['lifts'][3]}")
    kept = [(lo, S) for lo, S in kept if lo < inc["hi"]]
    open_lb = [item[0] for item in queue if item[0] < inc["hi"]]
    lw = min(open_lb + [lo for lo, _ in kept] + [inc["lo"]])
    return {"set": inc["set"], "lo": inc["lo"], "hi": inc["hi"], "lower": lw,
            "gap": inc["hi"] - lw, "open": len(open_lb),
            "unresolved": sorted(S for _, S in kept), "stats": st,
            "seconds": time.perf_counter() - t_start}


def _swaps(B, A, S):
    """Single-swap descent from ``S`` on the floating-point objective."""
    from benchmarks.studies.marex_bnb.bnb12 import objective
    n = B.shape[1]
    S = sorted(S)
    best = objective(B, A, S)
    improved = True
    while improved:
        improved = False
        for i in list(S):
            for j in range(n):
                if j in S:
                    continue
                T = sorted([k for k in S if k != i] + [j])
                v = objective(B, A, T)
                if v < best * (1 - 1e-12):
                    S, best, improved = T, v, True
                    break
            if improved:
                break
    return S
