"""Branch on exclusions: a binary include/exclude tree for MAREX's design.

The lexicographic tree of ``bnb12.py`` fixes markets into the treated set. On
GeoLift that barely constrains a design, since a fixed market can carry zero
weight; its large subtrees are never pruned. Excluding a market is what raises
a bound. This tree branches on one free market at a time, into a child that
excludes it from the treated set and a child that includes it.

A node is ``(I, O, F)``: markets fixed in, fixed out, and free; ``r = m - |I|``
remain to be chosen. Its subtree is every treated set of size ``m`` that
contains ``I`` and avoids ``O``. The branching market is the free market the
treated relaxation weights most: the simplex fit over ``I | F``, which is also
the first term of both bounds, so each node solves it once.

The bounds are those of ``bnb12.py``, which depend on ``(I, O, F, r)`` only:
``monotone`` is ``g(I | F) + h(O | F)``; ``sparse`` adds ``lam * delta^2`` with
``delta`` the distance from the relaxation's weights to the nearest weights
that use ``I`` and at most ``r`` markets of ``F``.

A node with ``r = 1`` has the sets ``I + {j}``, ``j`` in ``F``, as children.
They are priced by the library's closed-form treated bound and fitted only
under the incumbent, as the exact backend prices every set. A node with
``|F| = r`` is a single set and is fitted.

Every bounded node is checked against the minimum of its subtree, and every
run must return the enumeration's optimum.

From the repository root::

    python -m benchmarks.studies.marex_bnb.bnb_binary          # 12 markets
    python -m benchmarks.studies.marex_bnb.bnb_binary geolift  # 40 markets
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import comb

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.linalg import null_space

from benchmarks.studies.marex_bnb.bnb12 import objective, panels
from benchmarks.studies.marex_bnb.panels import RESULTS, geolift, leaves
from mlsynth.utils.marex_helpers.exact import side_fit, treated_fit_bound


def node_bound(B, A, Bc, I, O, F, r, m, kind):
    """The bound, and the free market the treated relaxation weights most."""
    U = I + F
    g, w = side_fit(B, A, U)
    w = np.asarray(w, float)
    h = side_fit(B, A, sorted(O + F))[0]
    lb = g + h
    if kind == "sparse" and len(U) > m:
        Bu = Bc[:, U]
        Z = null_space(np.ones((1, len(U))))
        lam = max(float(np.linalg.eigvalsh(Z.T @ (Bu.T @ Bu) @ Z).min()), 0.0)
        rest = np.sort(w[len(I):])[: max(len(F) - r, 0)]
        lb += lam * (float(rest @ rest) + float(rest.sum()) ** 2 / m)
    j = F[int(np.argmax(w[len(I):]))]
    return lb, j


def run(args):
    panel, kind, child_order, incumbent, m = args
    B, A, sets, f = load(panel, m)
    n = B.shape[1]
    Bc = B - A[:, None]
    member = np.zeros((len(sets), n), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    best = [incumbent, None]
    st = {"bounded": 0, "pruned": 0, "sets_cut": 0, "priced": 0, "fitted": 0,
          "max_depth": 0, "bound_s": 0.0}

    def subtree_min(I, O):
        mask = member[:, I].all(axis=1) if I else np.ones(len(sets), bool)
        if O:
            mask &= ~member[:, O].any(axis=1)
        return float(f[mask].min())

    def fit_set(S):
        st["fitted"] += 1
        value = objective(B, A, S)
        if value < best[0]:
            best[0], best[1] = value, tuple(sorted(S))

    def visit(I, O, F, depth):
        r = m - len(I)
        st["max_depth"] = max(st["max_depth"], depth)
        if r > len(F) or len(O) + len(F) - r < 1:
            return                                   # no admissible completion
        if r == len(F):
            fit_set(I + F)
            return
        if r == 1:
            S = np.array([sorted(I + [j]) for j in F])
            lbs = treated_fit_bound(B, A, S)
            st["priced"] += len(S)
            for i in np.argsort(lbs, kind="stable"):
                if lbs[i] >= best[0]:
                    break
                fit_set(list(S[i]))
            return
        t0 = time.perf_counter()
        lb, j = node_bound(B, A, Bc, I, O, F, r, m, kind)
        st["bound_s"] += time.perf_counter() - t0
        st["bounded"] += 1
        sub = subtree_min(I, O)
        assert lb <= sub * (1 + 1e-9) + 1e-9, (kind, I, O, lb, sub)
        if lb >= best[0]:
            st["pruned"] += 1
            st["sets_cut"] += comb(len(F), r)
            return
        rest = [k for k in F if k != j]
        kids = [("in", I + [j], O, rest), ("out", I, O + [j], rest)]
        if child_order == "out_first":
            kids.reverse()
        for _, I2, O2, F2 in kids:
            visit(I2, O2, F2, depth + 1)

    t0 = time.perf_counter()
    visit([], [], list(range(n)), 0)
    st["total_s"] = time.perf_counter() - t0
    fstar = float(f.min())
    S_star = tuple(int(x) for x in sets[int(np.argmin(f))])
    if np.isinf(incumbent):
        assert best[1] == S_star and abs(best[0] - fstar) <= 1e-9 * fstar, best
    start = "cold" if np.isinf(incumbent) else "at f*"
    return panel, kind, child_order, start, len(sets), st


def load(panel, m):
    if panel == "geolift":
        B, A = geolift()
        sets, g, h = leaves("geolift_m4")
        return B, A, sets, g + h
    B, A = panels()[panel]
    sets = np.array(list(itertools.combinations(range(B.shape[1]), m)))
    f = np.array([objective(B, A, S) for S in sets])
    return B, A, sets, f


def main(which):
    if which == "geolift":
        names, m = ["geolift"], 4
    else:
        names, m = list(panels()), 3
    jobs = []
    for p in names:
        _, _, _, f = load(p, m)
        fstar = float(f.min())
        for kind in ("monotone", "sparse"):
            for order in ("in_first", "out_first"):
                for inc in (np.inf, fstar * (1 + 1e-12)):
                    jobs.append((p, kind, order, inc, m))
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(run, jobs))
    print(f"{'panel':10s} {'bound':9s} {'children':10s} {'start':6s} "
          f"{'bounded':>7s} {'pruned':>6s} {'sets cut':>9s} {'priced':>7s} "
          f"{'fitted':>6s} {'depth':>5s} {'seconds':>7s}")
    rows = []
    for p, kind, order, start, nsets, st in results:
        print(f"{p:10s} {kind:9s} {order:10s} {start:6s} {st['bounded']:7d} "
              f"{st['pruned']:6d} {st['sets_cut']:9d} {st['priced']:7d} "
              f"{st['fitted']:6d} {st['max_depth']:5d} {st['total_s']:7.1f}")
        rows.append({"panel": p, "bound": kind, "children": order,
                     "start": start, "sets": nsets, **st})
    (RESULTS / f"bnb_binary_{which}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "n12")
