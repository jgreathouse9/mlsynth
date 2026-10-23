"""Furnival and Wilson's subtree bound, transposed to MAREX's standard design.

Order the markets, and let a node be a prefix ``P`` with ``R`` the markets after
its last. Its leaves are the candidates ``P + Q`` with ``Q`` drawn from ``R``.
Every leaf ``S`` lies inside ``P + R``, and the affine treated bound only falls
as markets are added (``bordered.py``), so ``lb(S) >= lb(P + R)``. Every leaf's
control pool lies inside the markets outside ``P``, and a simplex fit only falls
as donors are added, so ``h(C(S)) >= h(all - P)``. Hence

    f(S) = g(S) + h(C(S)) >= lb(P + R) + h(all - P),

and a node whose bound clears the cutoff skips all its leaves: the bound of
Furnival and Wilson (1974) and Miller (2002, sec. 3.8), the fit with every
remaining variable included.

The cutoff is the optimum ``search.py`` recorded plus the search's margin, so
the counts are what the bound would skip once the incumbent is optimal. Each
node's bound is computed directly, a solve on up to 40 columns, which is the
cost a bound tree that reuses its parent's elimination would cut. Three market
orders: as stored, nearest the target first, and farthest first.

From the repository root, after ``search.py``::

    python benchmarks/studies/marex_bordered_pricing/subtree.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_bordered_pricing import bordered, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

#: (m, whether to add the control-side term): the control term costs a simplex
#: fit per node, so the largest size runs without it.
RUNS = ((4, False), (4, True), (5, False), (5, True), (6, False))


def affine_bound(G, cols):
    try:
        x = np.linalg.solve(G[np.ix_(cols, cols)], np.ones(len(cols)))
    except np.linalg.LinAlgError:
        return 0.0
    s = x.sum()
    return 1.0 / s if s > 0 and np.isfinite(s) else 0.0


def skipped(B, A, G, m, order, with_control, cut):
    n = B.shape[1]
    out = nodes = 0
    stack = [()]
    t0 = time.perf_counter()
    while stack:
        P = stack.pop()
        for r in range((P[-1] + 1) if P else 0, n - (m - len(P) - 1)):
            Q = P + (r,)
            need = m - len(Q)
            rest = range(r + 1, n)
            leaves = comb(len(rest), need)
            if need == 0 or leaves == 0:
                continue
            nodes += 1
            b = affine_bound(G, [order[i] for i in Q] + [order[i] for i in rest])
            if b < cut and with_control:
                pool = [order[i] for i in range(n) if i not in Q]
                b += exact.side_fit(B, A, pool)[0]
            if b >= cut:
                out += leaves
            else:
                stack.append(Q)
    return out, nodes, time.perf_counter() - t0


def main():
    B, A = panel.geolift()
    G = bordered.gram(B, A)[1]
    found = json.loads((panel.RESULTS / "search.json").read_text())
    dist = np.linalg.norm(B - A[:, None], axis=0)
    orders = {"stored": list(range(B.shape[1])),
              "nearest_first": [int(i) for i in np.argsort(dist)],
              "farthest_first": [int(i) for i in np.argsort(-dist)]}
    out = {}
    for m, with_control in RUNS:
        fstar = found[f"standard_m{m}"]["shipped"]["value"]
        cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
        total = comb(B.shape[1], m)
        for name, order in orders.items():
            k, nodes, secs = skipped(B, A, G, m, order, with_control, cut)
            key = f"m{m}_{name}" + ("_with_control" if with_control else "")
            out[key] = {"candidates": total, "skipped": k,
                        "share_skipped": round(k / total, 4), "nodes_bounded": nodes,
                        "seconds_bounding_nodes": round(secs, 1)}
            print(key, out[key], flush=True)
    (panel.RESULTS / "subtree.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
