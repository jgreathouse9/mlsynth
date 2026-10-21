"""The indicator lift's root gap on GeoLift as the treated count grows.

For m = 4, 6, 8 and 10 treated of 40 markets (91,390 to 847 million treated
sets), the root's certified lower bound is compared with an incumbent found by
rounding the lift's solution (the m markets with the largest z) and improving
it by single swaps. The lift's size depends on n, not on m, so its cost stays
flat while enumeration grows as C(40, m). The certified gap,
(incumbent - bound) / incumbent, is what a branch and bound has to close.

From the repository root::

    python -m benchmarks.studies.marex_bnb.root_gap_by_m
"""
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

from math import comb

import numpy as np

from benchmarks.studies.marex_bnb.bnb12 import objective
from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.certified_dnn import dnn_bound
from benchmarks.studies.marex_bnb.panels import geolift


def local_search(B, A, S, n):
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
    return S, best


def main(ms):
    B, A = geolift()
    n = B.shape[1]
    pan = Panel(B - A[:, None])
    print(f"GeoLift, {n} markets")
    for m in ms:
        t0 = time.perf_counter()
        lb, approx, status, marg = dnn_bound(pan, m, [], [], list(range(n)),
                                             kind="indicator", marginals=True)
        secs = time.perf_counter() - t0
        z = np.array([marg[("z", j)] for j in range(n)])
        S0 = sorted(np.argsort(-z)[:m].tolist())
        f0 = objective(B, A, S0)
        S, ub = local_search(B, A, S0, n)
        print(f"m={m:2d}  sets {comb(n, m):>12,d}  bound {lb:.6g}  rounded {f0:.6g}  "
              f"after swaps {ub:.6g}  gap {(ub - lb) / ub:.4f}  {status}  {secs:.0f} s  "
              f"z in [0.01, 0.99]: {int(np.sum((z > 0.01) & (z < 0.99)))}  set {S}",
              flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [4, 6, 8, 10])
