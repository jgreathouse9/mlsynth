"""How tight must a subtree bound be before branch and bound beats enumeration?

The tree is the lexicographic one of the k-subset branch-and-bound literature
(Brusco and Stahl 2005, ch. 2; Marti and Reinelt 2022, sec. 4.7; Cavaleiro and
Alizadeh 2022): markets carry an order, a node at depth ``d`` fixes the first
``d`` treated markets in that order, and its subtree is every treated set of
size ``m`` that starts with them. Depth ``m`` is a leaf.

A perfect bound returns each subtree's minimum ``M`` and prunes everything off
the optimum's path, which says nothing about real bounds. This measures what a
real bound needs instead. A bound that reaches a fraction ``rho`` of every
subtree's minimum prunes a node exactly when ``rho * M >= f*``, with the
incumbent already at the optimum ``f*`` (the best case). For each ``rho`` the
table gives the treated sets that still reach leaf pricing: those with no
pruned ancestor. Enumeration prices all of them.

``best_k`` is the best design with ``k`` treated markets for ``k < m``. A set
can carry markets at zero weight, so a node that fixes ``d`` markets has a
subtree minimum no larger than about ``best_{m-d}`` (exactly that, less the
small change in the control pool), which caps how far any subtree can sit
above the optimum.

From the repository root::

    python benchmarks/studies/marex_bnb/ceiling.py
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from benchmarks.studies.marex_bnb.panels import (PANELS, RESULTS, best_by_size,
                                                 leaves)

RHOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def orderings(name, sets, g, h, n):
    """Market orders to build the tree in. ``fit`` is each market's fit as a
    lone treated market, a quantity a search has before it starts."""
    single = np.array([g[(sets == j).any(axis=1) & (sets.shape[1] == 1)].min()
                       if sets.shape[1] == 1 else np.nan for j in range(n)])
    if np.isnan(single).any():
        from mlsynth.utils.marex_helpers.exact import side_fit
        make, _ = PANELS[name]
        B, A = make()
        single = np.array([side_fit(B, A, [j])[0] for j in range(n)])
    rng = np.random.default_rng(0)
    return {
        "index": np.arange(n),
        "fit_best_first": np.argsort(single, kind="stable"),
        "fit_worst_first": np.argsort(-single, kind="stable"),
        "random": rng.permutation(n),
    }


def ceiling(sets, f, order):
    """Per depth: the subtree minima over ``f*``; per rho: leaves left."""
    n = int(sets.max()) + 1
    m = sets.shape[1]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    keyed = np.sort(rank[sets], axis=1)            # each set in tree order
    fstar = float(f.min())
    ratio_by_depth = {}
    node_of = {}
    for d in range(1, m):
        prefix = keyed[:, :d]
        code = np.zeros(len(sets), dtype=np.int64)
        for c in range(d):
            code = code * n + prefix[:, c]
        uniq, inv = np.unique(code, return_inverse=True)
        mins = np.full(len(uniq), np.inf)
        np.minimum.at(mins, inv, f)
        node_of[d] = (inv, mins)
        ratio_by_depth[d] = mins / fstar
    left = {}
    nodes = {}
    for rho in RHOS:
        alive = np.ones(len(sets), dtype=bool)
        visited = 0
        for d in range(1, m):
            inv, mins = node_of[d]
            live_nodes = np.unique(inv[alive])
            visited += len(live_nodes)
            pruned = rho * mins >= fstar * (1 + 1e-12)
            alive &= ~pruned[inv]
        left[rho] = int(alive.sum())
        nodes[rho] = visited
    return fstar, ratio_by_depth, left, nodes


def main():
    out = {}
    for name, (_, m) in PANELS.items():
        sets, g, h = leaves(name)
        f = g + h
        n = int(sets.max()) + 1
        best_k = best_by_size(name, m - 1)
        out[name] = {"n": n, "m": m, "sets": len(sets), "f_star": float(f.min()),
                     "best_k_over_f_star": {k: v / float(f.min())
                                            for k, v in best_k.items()},
                     "orders": {}}
        print(f"\n{name}: n={n} m={m} sets={len(sets)} f*={f.min():.6g}")
        print("  best design with k treated, over f*: " + ", ".join(
            f"k={k}: {v / f.min():.3f}" for k, v in best_k.items()))
        for oname, order in orderings(name, sets, g, h, n).items():
            fstar, ratios, left, nodes = ceiling(sets, f, order)
            q = {d: [float(np.percentile(r, p)) for p in (10, 50, 90)]
                 + [float((r < 1.1).mean())] for d, r in ratios.items()}
            out[name]["orders"][oname] = {
                "subtree_min_over_f_star": q, "leaves_left": left,
                "nodes_bounded": nodes}
            print(f"  order {oname}:")
            for d, (p10, p50, p90, near) in q.items():
                print(f"    depth {d}: subtree min / f*  p10 {p10:.3f}  "
                      f"p50 {p50:.3f}  p90 {p90:.3f}   within 10% of f*: "
                      f"{near:.1%}")
            print("    sets left to price at rho: " + "  ".join(
                f"{r}:{left[r]}" for r in RHOS))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ceiling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
