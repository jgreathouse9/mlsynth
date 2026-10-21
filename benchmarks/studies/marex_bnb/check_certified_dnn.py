"""Check the certified DNN bound at 12 markets against brute force.

On each 12-market panel: the root, every lexicographic node of depth one and
two, and 30 random include/exclude nodes. For each node it reports the certified
bound and the solver's value over the subtree minimum, and counts certified
bounds above the certified minimum of the subtree (the exact objective at the
best set's fitted weights), which must be zero. ``sdp_bound``, the same
relaxation without the facial reduction and the extra cuts, is solved beside it
for comparison; its value is not certified.

From the repository root::

    python -m benchmarks.studies.marex_bnb.check_certified_dnn
"""
import os, itertools, collections, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
from fractions import Fraction
from benchmarks.studies.marex_bnb.bnb12 import panels
from benchmarks.studies.marex_bnb.certify import Panel, up
from benchmarks.studies.marex_bnb.certified_dnn import dnn_bound
from benchmarks.studies.marex_bnb.sdp_bound import sdp_bound
from mlsynth.utils.marex_helpers.exact import side_fit

m = 3
rng = np.random.default_rng(1)
for name, (B, A) in panels().items():
    n = B.shape[1]
    Bc = B - A[:, None]
    pan = Panel(Bc)
    val, ub = {}, {}
    for S in itertools.combinations(range(n), m):
        C = [j for j in range(n) if j not in S]
        g, w = side_fit(B, A, list(S)); h, v = side_fit(B, A, C)
        val[S] = g + h
        ub[S] = pan.side_upper(list(S), w) + pan.side_upper(C, v)
    fstar = min(val.values())
    nodes = [([], [], list(range(n)))]
    for d in (1, 2):
        for pre in itertools.combinations(range(n), d):
            if pre[-1] > n - m + d - 1:
                continue
            I = list(pre); F = list(range(pre[-1] + 1, n))
            O = [j for j in range(pre[-1] + 1) if j not in pre]
            nodes.append((I, O, F))
    for _ in range(30):                      # binary-tree style nodes
        perm = rng.permutation(n)
        k_in = int(rng.integers(0, 3)); k_out = int(rng.integers(1, 7))
        I = sorted(perm[:k_in].tolist()); O = sorted(perm[k_in:k_in + k_out].tolist())
        F = sorted(perm[k_in + k_out:].tolist())
        nodes.append((I, O, F))
    rows, st, bad = [], collections.Counter(), 0
    t_c = t_o = 0.0
    for I, O, F in nodes:
        sub = [S for S in val if set(I) <= set(S) and not set(O) & set(S)]
        smin = min(val[S] for S in sub)
        sub_ub = min(ub[S] for S in sub)
        t0 = time.perf_counter()
        lb, approx, status = dnn_bound(pan, m, I, O, F)
        t_c += time.perf_counter() - t0
        t0 = time.perf_counter()
        old, ostat = sdp_bound(B, A, I, O, F, m - len(I), m)
        t_o += time.perf_counter() - t0
        st[status] += 1
        if Fraction(lb) > sub_ub:
            bad += 1
        rows.append((len(I), len(O), lb / smin, approx / smin, old / smin))
    R = np.array(rows)
    print(f"\n{name}: f* {fstar:.6g}, {len(nodes)} nodes, statuses {dict(st)}, "
          f"certified above the subtree's certified minimum: {bad}")
    print(f"  ms/node: certified {1000*t_c/len(nodes):.0f}, old {1000*t_o/len(nodes):.0f}")
    print(f"  root: certified {R[0,2]:.4f}  solver {R[0,3]:.4f}  old {R[0,4]:.4f}")
    for k in (1, 2):
        sel = (R[:, 0] == k)
        print(f"  |I|={k}: certified median {np.median(R[sel,2]):.4f} min {R[sel,2].min():.4f}  "
              f"solver median {np.median(R[sel,3]):.4f}  old median {np.median(R[sel,4]):.4f} min {R[sel,4].min():.4f}")
    sel = np.arange(len(R)) >= len(R) - 30
    print(f"  binary nodes: certified median {np.median(R[sel,2]):.4f} min {R[sel,2].min():.4f} "
          f"max {R[sel,2].max():.6f}; old median {np.median(R[sel,4]):.4f} max {R[sel,4].max():.6f}")
    print(f"  max certified/subtree min over all nodes {R[:,2].max():.9f}; "
          f"max (solver - certified)/f_sub {np.max(R[:,3]-R[:,2]):.2e}")
