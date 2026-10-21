"""The certified DNN bound on GeoLift's large subtrees: 40 markets, 4 treated.

The root, every fifth node of the first level of the best-first lexicographic
tree, and three include/exclude nodes around the optimum. Each line gives the
subtree's size, the ratio its bound must reach to prune it (``f*`` over the
subtree minimum), the certified and solver ratios, the solver status and the
time.

From the repository root::

    python -m benchmarks.studies.marex_bnb.certified_dnn_geolift [trace|indicator] [first k nodes]
"""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
from math import comb
from benchmarks.studies.marex_bnb.panels import geolift, leaves
from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.certified_dnn import dnn_bound
KIND = sys.argv[1] if len(sys.argv) > 1 else "trace"
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
from benchmarks.studies.marex_bnb.bnb12 import fit

B, A = geolift(); n, m = B.shape[1], 4
sets, g, h = leaves("geolift_m4"); f = g + h
fstar = f.min()
member = np.zeros((len(sets), n), bool); member[np.arange(len(sets))[:, None], sets] = True
pan = Panel(B - A[:, None])
single = np.array([fit(B, A, [j]) for j in range(n)])
order = np.argsort(single, kind="stable")          # best-first lexicographic order
nodes = [("root", [], [], list(range(n)))]
for q in range(0, 36, 5):
    I = [int(order[q])]; O = [int(order[p]) for p in range(q)]; F = [int(order[p]) for p in range(q + 1, n)]
    nodes.append((f"lex d1 q={q}", I, O, F))
S_star = sets[np.argmin(f)].tolist()
out = sorted(set(range(n)) - set(S_star))
nodes.append(("out: 6 non-optimal", [], out[:6], [j for j in range(n) if j not in out[:6]]))
nodes.append(("out: 1 optimal", [], [S_star[0]], [j for j in range(n) if j != S_star[0]]))
nodes.append(("in 1 optimal, out 1 optimal", [S_star[0]], [S_star[1]], [j for j in range(n) if j not in S_star[:2]]))
print(f"GeoLift n={n} m={m} f*={fstar:.10g}; pruning needs ratio >= f*/subtree min")
print(f"lift: {KIND}")
for lab, I, O, F in nodes[:ONLY]:
    mask = member[:, I].all(1) if I else np.ones(len(sets), bool)
    if O: mask &= ~member[:, O].any(1)
    sub = f[mask].min(); size = int(mask.sum())
    t0 = time.perf_counter()
    lb, approx, status = dnn_bound(pan, m, I, O, F, kind=KIND)
    dt = time.perf_counter() - t0
    print(f"{lab:30s} sets {size:6d}  need {fstar/sub:.3f}  certified {lb/sub:.4f}  solver {approx/sub:.4f}  "
          f"{status}  {dt:.1f} s  {'PRUNE' if lb >= fstar else ''}", flush=True)
