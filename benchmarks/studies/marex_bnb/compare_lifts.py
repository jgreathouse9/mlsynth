"""Compare the certified lifts at 12 markets: strength, validity and time.

The nodes are those of ``check_certified_dnn``: the root, the lexicographic
nodes of depth one and two, and 30 random include/exclude nodes. For each lift
the line gives the certified bound over the subtree minimum at the root, the
median and least of it on each level and on the random nodes, the count of
certified bounds above the subtree's certified minimum (must be zero), the
solver statuses and the time per node.

From the repository root::

    python -m benchmarks.studies.marex_bnb.compare_lifts
"""
import collections
import itertools
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
from fractions import Fraction

from benchmarks.studies.marex_bnb.bnb12 import panels
from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.certified_dnn import (indicator_lift,
                                                         lift_bound, trace_lift)
from mlsynth.utils.marex_helpers.exact import side_fit

M = 3
LIFTS = {"trace": trace_lift, "indicator": indicator_lift}


def nodes_of(n, rng):
    nodes = [([], [], list(range(n)))]
    for d in (1, 2):
        for pre in itertools.combinations(range(n), d):
            if pre[-1] > n - M + d - 1:
                continue
            I = list(pre)
            F = list(range(pre[-1] + 1, n))
            O = [j for j in range(pre[-1] + 1) if j not in pre]
            nodes.append((I, O, F))
    for _ in range(30):
        perm = rng.permutation(n)
        k_in, k_out = int(rng.integers(0, 3)), int(rng.integers(1, 7))
        nodes.append((sorted(perm[:k_in].tolist()),
                      sorted(perm[k_in:k_in + k_out].tolist()),
                      sorted(perm[k_in + k_out:].tolist())))
    return nodes


def main():
    rng = np.random.default_rng(1)
    for name, (B, A) in panels().items():
        n = B.shape[1]
        pan = Panel(B - A[:, None])
        val, ub = {}, {}
        for S in itertools.combinations(range(n), M):
            C = [j for j in range(n) if j not in S]
            g, w = side_fit(B, A, list(S))
            h, v = side_fit(B, A, C)
            val[S] = g + h
            ub[S] = pan.side_upper(list(S), w) + pan.side_upper(C, v)
        nodes = nodes_of(n, rng)
        print(f"\n{name}: {len(nodes)} nodes")
        for lname, build in LIFTS.items():
            R, st, bad, secs = [], collections.Counter(), 0, 0.0
            for I, O, F in nodes:
                sub = [S for S in val if set(I) <= set(S) and not set(O) & set(S)]
                smin = min(val[S] for S in sub)
                t0 = time.perf_counter()
                lb, approx, status = lift_bound(pan, build(n, M, I, O, F))
                secs += time.perf_counter() - t0
                st[status] += 1
                bad += Fraction(lb) > min(ub[S] for S in sub)
                R.append((len(I), lb / smin))
            R = np.array(R)
            lev = [R[1:][(R[1:, 0] == k)][:, 1] for k in (1, 2)]
            rnd = R[-30:, 1]
            print(f"  {lname:10s} root {R[0, 1]:.4f}  d1 med {np.median(lev[0]):.4f} "
                  f"min {lev[0].min():.4f}  d2 med {np.median(lev[1]):.4f} "
                  f"min {lev[1].min():.4f}  random med {np.median(rnd):.4f} "
                  f"min {rnd.min():.4f}  above {bad}  {dict(st)}  "
                  f"{1000 * secs / len(nodes):.0f} ms/node", flush=True)


if __name__ == "__main__":
    main()
