"""Reduced-cost fixing from one root lift: which markets can be excluded.

One lift at the root gives multipliers; for every market ``j`` the inherited
bound (``inherit.inherited_bound``) on the child that forces ``j`` into the
treated set costs one eigendecomposition. A market whose forced-in bound
reaches the optimum is excluded from the whole search, which shrinks both the
enumeration (C(n, m)) and every later lift. The true minimum with ``j`` forced
in (the library's exact search) says how many markets could be excluded at
best. With a unique optimum that is every market outside it, n - m: an exact
forced-in bound would solve the problem by fixing alone.

From the repository root::

    python -m benchmarks.studies.marex_bnb.inherit_fixing
"""
import os
import sys
import time
import warnings
from math import comb

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_bnb.certified_bnb import certified_fit
from benchmarks.studies.marex_bnb.certified_dnn import (indicator_lift,
                                                         lift_bound,
                                                         treated_lift)
from benchmarks.studies.marex_bnb.certify import Panel, down
from benchmarks.studies.marex_bnb.inherit import inherited_bound
from benchmarks.studies.marex_bnb.panels import geolift

CASES = ((6, 297926.2374, True), (8, 181943.3763, True))


def main():
    from mlsynth.utils.marex_helpers import exact as ex
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    B, A = geolift()
    n = B.shape[1]
    pan = Panel(B - A[:, None])
    for m, fstar, with_truth in CASES:
        truth = None
        if with_truth:
            truth = np.array([ex.search_cluster(
                B, A, [m], filt=SubsetFilter(forced_in=frozenset([j])),
                candidate_limit=10**9)[0].value for j in range(n)])
            could = int((truth >= fstar * (1 + 1e-9)).sum())
        for kind in ("treated", "full"):
            build = treated_lift if kind == "treated" else (
                lambda n_, m_, I, O, F: indicator_lift(n_, m_, I, O, F, cuts=False))
            L = build(n, m, [], [], list(range(n)))
            t0 = time.perf_counter()
            lb, _, _, _, mult = lift_bound(pan, L, marginals=True, multipliers=True)
            t_root = time.perf_counter() - t0
            t0 = time.perf_counter()
            inc = []
            for j in range(n):
                F = [k for k in range(n) if k != j]
                h = float(down(certified_fit(pan, B, A, F)[0])) if kind == "treated" else 0.0
                inc.append(inherited_bound(pan, L, mult, build(n, m, [j], [], F), [j], []) + h)
            t_inh = time.perf_counter() - t0
            inc = np.array(inc)
            fixed = int((inc >= fstar).sum())
            left = n - fixed
            line = (f"m={m} {kind:7s} root lift {t_root:.0f} s, 40 inherited bounds {t_inh:.1f} s: "
                    f"markets excluded {fixed}")
            if truth is not None:
                line += (f" of {could} excludable; inherited / true forced-in minimum: median "
                         f"{np.median(inc / truth):.3f}, min {np.min(inc / truth):.3f}")
            line += f"; sets left C({left},{m}) = {comb(left, m):,} of {comb(n, m):,}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
