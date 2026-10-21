"""The monotone bound, node by node, against the ceiling.

In the lexicographic tree a node at depth ``d`` fixes ``I`` (its first ``d``
treated markets in tree order) and leaves free ``F``, the markets later in the
order. Every market earlier in the order and not in ``I`` is out of the treated
set, so it is a control. A completion ``S`` satisfies ``I <= S <= I | F`` and
its control pool ``C`` satisfies ``O <= C <= O | F``, with ``O`` those earlier
markets. A simplex fit over a larger support can only fit better, so

    f(S) = g(S) + h(C) >= g(I | F) + h(O | F),

the bound of Brusco and Stahl (2005, ch. 14) applied to each side. It drops
the cardinality, so it is near zero where ``F`` is large.

This reports, per order and depth, the fraction ``bound / subtree minimum`` the
bound reaches (the ``rho`` of ``ceiling.py``), and the treated sets left to
price when every node whose bound clears ``f*`` is pruned, the incumbent again
at the optimum.

From the repository root, one or more panel names from ``panels.PANELS``::

    python -m benchmarks.studies.marex_bnb.monotone geolift_m4 county_n100_m3
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from benchmarks.studies.marex_bnb.ceiling import orderings
from benchmarks.studies.marex_bnb.panels import PANELS, RESULTS, leaves


def main(name: str = "geolift_m4"):
    from mlsynth.utils.marex_helpers.exact import side_fit
    make, m = PANELS[name]
    B, A = make()
    sets, g, h = leaves(name)
    f = g + h
    n = B.shape[1]
    fstar = float(f.min())
    out = {}
    for oname, order in orderings(name, sets, g, h, n).items():
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        keyed = np.sort(rank[sets], axis=1)
        alive = np.ones(len(sets), dtype=bool)
        per_depth = {}
        for d in range(1, m):
            code = np.zeros(len(sets), dtype=np.int64)
            for c in range(d):
                code = code * n + keyed[:, c]
            uniq, first, inv = np.unique(code, return_index=True,
                                         return_inverse=True)
            mins = np.full(len(uniq), np.inf)
            np.minimum.at(mins, inv, f)
            bound = np.empty(len(uniq))
            for u, row in enumerate(first):
                prefix = keyed[row, :d]                 # positions in the order
                last = prefix[-1]
                I = [int(order[p]) for p in prefix]
                F = [int(order[p]) for p in range(last + 1, n)]
                O = [int(order[p]) for p in range(last + 1) if p not in prefix]
                gb = side_fit(B, A, I + F)[0]
                hb = side_fit(B, A, O + F)[0] if O + F else np.inf
                bound[u] = gb + hb
            ratio = bound / mins
            pruned = bound >= fstar * (1 + 1e-9)
            assert (bound <= mins * (1 + 1e-9) + 1e-9).all(), "bound above a leaf"
            per_depth[d] = {
                "nodes": len(uniq),
                "rho_p10_p50_p90": [float(np.percentile(ratio, p))
                                    for p in (10, 50, 90)],
                "nodes_pruned": int(pruned.sum()),
            }
            alive &= ~pruned[inv]
        out[oname] = {"depths": per_depth, "sets_left": int(alive.sum()),
                      "sets": len(sets)}
        print(f"{oname:16s} sets left {alive.sum():6d} / {len(sets)} "
              f"({alive.mean():.1%})")
        for d, v in per_depth.items():
            p = v["rho_p10_p50_p90"]
            print(f"    depth {d}: {v['nodes']:5d} nodes, bound / subtree min "
                  f"p10 {p[0]:.3f} p50 {p[1]:.3f} p90 {p[2]:.3f}, pruned "
                  f"{v['nodes_pruned']}")
    (RESULTS / f"monotone_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or ["geolift_m4"]:
        print(name)
        main(name)
