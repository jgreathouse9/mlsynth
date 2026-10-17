"""A branch and bound whose pruning and reported gap rest only on certificates.

Each number the search compares is a rational bound computed exactly
(``certify``) and rounded outward:

- a node's lower bound is the larger of two certified bounds on the minimum of
  ``f`` over its subtree: the monotone bound, ``g(I | F) + h(O | F)`` with each
  side replaced by its Wolfe-dual certificate, and optionally the doubly
  nonnegative bound of ``certified_dnn``;
- a complete treated set gets a certified interval ``[lo, hi]`` for ``f``: the
  Wolfe-dual certificates of its two fits below, and the exact objective at the
  fitted weights above;
- the incumbent's value is the ``hi`` of its set, an upper bound on ``f`` there
  and so on the optimum.

A node is pruned only when its certified lower bound is at least the
incumbent's ``hi``; no tolerance is added. A set whose interval overlaps the
incumbent's is kept, in ``unresolved``: floating point cannot say which of the
two is smaller, and the gap reports it.

At termination every subtree was pruned against an upper bound no smaller than
the final ``hi``, or every set in it was given an interval. So

    f* >= lower = min(incumbent lo, min of lo over the sets kept),
    f(incumbent) <= incumbent hi,

and the incumbent is optimal to within ``gap = hi - lower``, a certified
absolute gap. A gap of zero is not claimed: the incumbent's own interval has
positive width.

Feasibility: a node is a subtree only when ``0 <= r <= |F|`` and at least one
market is left for the controls, ``|O| + |F| - r >= 1``. A node with
``r = |F|`` is the single set ``I | F``.

The tree branches on the free market the treated relaxation weights most, into
a child that excludes it and a child that includes it, as in ``bnb_binary``.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.certified_dnn import dnn_bound
from mlsynth.utils.marex_helpers.exact import side_fit


def certified_fit(panel, B, A, cols):
    """``(lo, hi, weights)``: a certified interval for the simplex fit."""
    g, w = side_fit(B, A, sorted(cols))
    w = np.asarray(w, dtype=float)
    return panel.side_lower(sorted(cols), w), panel.side_upper(sorted(cols), w), w


def search(B, A, m, use_dnn=True, dnn_min_sets=2, audit=None):
    """Certified depth-first branch and bound; returns the incumbent and a log.

    ``audit(I, O, lower)``, when given, is called on every bounded node, so a
    test can compare each certified bound with the subtree's true minimum.
    """
    n = B.shape[1]
    panel = Panel(B - A[:, None])
    inc = {"set": None, "lo": None, "hi": Fraction(10) ** 300}
    kept = []                                     # (lo, set) of unresolved sets
    st = {"nodes": 0, "pruned": 0, "dnn": 0, "dnn_pruned": 0, "sets": 0}

    def leaf(S):
        st["sets"] += 1
        C = [j for j in range(n) if j not in S]
        lo_g, hi_g, _ = certified_fit(panel, B, A, S)
        lo_h, hi_h, _ = certified_fit(panel, B, A, C)
        lo, hi = lo_g + lo_h, hi_g + hi_h
        if lo >= inc["hi"]:
            return
        if hi < inc["hi"]:
            if inc["set"] is not None:
                kept.append((inc["lo"], inc["set"]))
            inc.update(set=tuple(sorted(S)), lo=lo, hi=hi)
        else:
            kept.append((lo, tuple(sorted(S))))

    def visit(I, O, F):
        r = m - len(I)
        if r < 0 or r > len(F) or len(O) + len(F) - r < 1:
            return                                # no admissible completion
        if r == len(F):
            leaf(I + F)
            return
        st["nodes"] += 1
        lo_g, _, w = certified_fit(panel, B, A, I + F)
        lo_h, _, _ = certified_fit(panel, B, A, O + F)
        lower = lo_g + lo_h
        if use_dnn and lower < inc["hi"] and _count(len(F), r) >= dnn_min_sets:
            st["dnn"] += 1
            lb = Fraction(dnn_bound(panel, m, I, O, F)[0])
            if lb > lower:
                lower = lb
                if lower >= inc["hi"]:
                    st["dnn_pruned"] += 1
        if audit is not None:
            audit(I, O, lower)
        if lower >= inc["hi"]:
            st["pruned"] += 1
            return
        U = sorted(I + F)
        wF = {j: w[U.index(j)] for j in F}
        j = max(F, key=lambda k: (wF[k], -k))
        rest = [k for k in F if k != j]
        visit(I + [j], O, rest)
        visit(I, O + [j], rest)

    visit([], [], list(range(n)))
    # sets kept before the incumbent improved may now lie above it
    kept = [(lo, S) for lo, S in kept if lo < inc["hi"]]
    lower = min([inc["lo"]] + [lo for lo, _ in kept])
    return {"set": inc["set"], "lo": inc["lo"], "hi": inc["hi"],
            "lower": lower, "gap": inc["hi"] - lower,
            "unresolved": sorted(S for _, S in kept), "stats": st}


def _count(k, r):
    from math import comb
    return comb(k, r)
