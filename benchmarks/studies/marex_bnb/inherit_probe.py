"""Inherited bounds against fresh lifts, node by node, on real trees.

The tree branches on the market the parent's lift most wants treated (its
largest ``z``), into include and exclude children, down to ``depth``. At each
child three certified bounds are compared with the child's subtree minimum:

- ``parent``: the parent's own bound, which holds for the child trivially;
- ``inherit``: the parent's multipliers restricted to the child's lift
  (``inherit.inherited_bound``), one eigendecomposition;
- ``fresh``: the child's own lift, solved by Clarabel.

Treated lifts add the control side's certificate ``h(O | F)``, the same for
all three. Subtree minima come from brute force at 12 markets and from the
library's exact search (forced and forbidden markets) on GeoLift.

From the repository root::

    python -m benchmarks.studies.marex_bnb.inherit_probe n12 [treated|indicator] [depth]
    python -m benchmarks.studies.marex_bnb.inherit_probe geolift M [treated|indicator] [depth]
"""
import itertools
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_bnb.bnb12 import objective, panels
from benchmarks.studies.marex_bnb.certified_bnb import certified_fit
from benchmarks.studies.marex_bnb.certified_dnn import (indicator_lift,
                                                         lift_bound,
                                                         treated_lift)
from benchmarks.studies.marex_bnb.certify import Panel, down
from benchmarks.studies.marex_bnb.inherit import inherited_bound
from benchmarks.studies.marex_bnb.panels import geolift


def probe(B, A, m, kind, depth, subtree_min, label):
    n = B.shape[1]
    pan = Panel(B - A[:, None])
    build = treated_lift if kind == "treated" else indicator_lift

    def make(I, O, F):
        return build(n, m, I, O, F) if kind == "treated" else build(n, m, I, O, F, cuts=False)

    def h_lo(O, F):
        return float(down(certified_fit(pan, B, A, sorted(O + F))[0])) if kind == "treated" else 0.0

    def solve(I, O, F):
        L = make(I, O, F)
        t0 = time.perf_counter()
        lb, _, _, marg, mult = lift_bound(pan, L, marginals=True, multipliers=True)
        return L, lb + h_lo(O, F), marg, mult, time.perf_counter() - t0

    rows = []
    frontier = [([], [], list(range(n)), None)]
    for d in range(depth + 1):
        nxt = []
        for I, O, F, parent in frontier:
            r = m - len(I)
            if r < 1 or r >= len(F):
                continue
            L, lb, marg, mult, secs = solve(I, O, F)
            if parent is not None:
                pL, plb, pmult = parent
                t0 = time.perf_counter()
                inh = inherited_bound(pan, pL, pmult, L, I, O) + h_lo(O, F)
                ti = time.perf_counter() - t0
                sub = subtree_min(I, O)
                rows.append((d, len(I), len(O), plb / sub, inh / sub, lb / sub, secs, ti))
            if not mult:
                continue
            j = max(F, key=lambda k: (marg.get(("z", k), 0.0), -k))
            rest = [k for k in F if k != j]
            nxt.append((I + [j], O, rest, (L, lb, mult)))
            nxt.append((I, O + [j], rest, (L, lb, mult)))
        frontier = nxt
    R = np.array(rows)
    print(f"\n{label} ({kind} lift): {len(R)} children")
    print(f"  {'depth':>5s} {'|I|':>3s} {'|O|':>3s} {'parent':>7s} {'inherit':>8s} {'fresh':>7s} "
          f"{'fresh s':>8s} {'inherit ms':>10s}")
    for d, i, o, p, h, f, s, t in R:
        print(f"  {int(d):5d} {int(i):3d} {int(o):3d} {p:7.4f} {h:8.4f} {f:7.4f} {s:8.2f} {1000 * t:10.1f}")
    gain = (R[:, 4] - R[:, 3]) / np.maximum(R[:, 5] - R[:, 3], 1e-12)
    print(f"  inherit/fresh median {np.median(R[:, 4] / R[:, 5]):.4f}  min {np.min(R[:, 4] / R[:, 5]):.4f}; "
          f"share of the parent-to-fresh gain recovered: median {np.median(gain):.3f}; "
          f"time fresh {R[:, 6].sum():.1f} s, inherit {R[:, 7].sum():.2f} s; "
          f"any certified bound above the subtree minimum: {bool((R[:, 3:6] > 1 + 1e-9).any())}", flush=True)


def main():
    which = sys.argv[1]
    if which == "n12":
        kind = sys.argv[2] if len(sys.argv) > 2 else "treated"
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        for name, (B, A) in panels().items():
            n = B.shape[1]
            val = {S: objective(B, A, S) for S in itertools.combinations(range(n), 3)}

            def subtree_min(I, O, val=val):
                return min(v for S, v in val.items() if set(I) <= set(S) and not set(O) & set(S))
            probe(B, A, 3, kind, depth, subtree_min, name)
    else:
        m = int(sys.argv[2])
        kind = sys.argv[3] if len(sys.argv) > 3 else "treated"
        depth = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        from mlsynth.utils.marex_helpers import exact as ex
        from mlsynth.utils.syndes_helpers.exact import SubsetFilter
        B, A = geolift()

        def subtree_min(I, O):
            filt = SubsetFilter(forced_in=frozenset(I), forbidden=frozenset(O))
            return ex.search_cluster(B, A, [m], filt=filt, candidate_limit=10**9)[0].value
        probe(B, A, m, kind, depth, subtree_min, f"GeoLift m={m}")


if __name__ == "__main__":
    main()
