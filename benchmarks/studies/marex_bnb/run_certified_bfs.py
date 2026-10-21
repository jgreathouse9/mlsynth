"""Run the best-first certified search and check it against known optima.

``n12``: the four 12-market panels and ``factor_twin``, three treated, against
brute force. ``geolift M``: GeoLift with ``M`` treated; for ``M = 4`` and ``6``
the optimum is known (554,319.76 and 297,926; the library's exact search finds
both), and the run must return it with a certified gap.

From the repository root::

    python -m benchmarks.studies.marex_bnb.run_certified_bfs n12
    python -m benchmarks.studies.marex_bnb.run_certified_bfs geolift 6
    python -m benchmarks.studies.marex_bnb.run_certified_bfs geolift 8 14400 0.3

The optional third and fourth arguments are a time limit in seconds and the
margin within which the full lift is tried (``certified_bfs.search``'s
``full_margin``, default 0.10).
"""
import itertools
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

from benchmarks.studies.marex_bnb.bnb12 import objective, panels
from benchmarks.studies.marex_bnb.certified_bfs import search
from benchmarks.studies.marex_bnb.panels import geolift

KNOWN = {4: 554319.7579, 6: 297926.0}


def report(name, res, best=None):
    st = res["stats"]
    print(f"{name}: set {res['set']}  hi {float(res['hi']):.10g}  lower "
          f"{float(res['lower']):.10g}  gap {float(res['gap'] / res['hi']):.2e}  "
          f"open {res['open']}  unresolved {res['unresolved']}  nodes {st['nodes']}  "
          f"enumerated {st['enumerated']}  sets {st['sets']}  lifts {st['lifts']}  "
          f"lift s {{2: {st['lift_s'][2]:.0f}, 3: {st['lift_s'][3]:.0f}}}  pruned {st['pruned']}  "
          f"{res['seconds']:.0f} s" + (f"  known {best}" if best else ""), flush=True)


def main():
    which = sys.argv[1]
    if which == "n12":
        cases = dict(panels())
        B, A = cases["factor"]
        Bt = B.copy()
        Bt[:, 11] = Bt[:, 4]
        cases["factor_twin"] = (Bt, Bt.mean(axis=1))
        for name, (B, A) in cases.items():
            n = B.shape[1]
            val = {S: objective(B, A, S) for S in itertools.combinations(range(n), 3)}
            fstar = min(val.values())
            best = sorted(S for S, v in val.items() if v <= fstar * (1 + 1e-12))
            res = search(B, A, 3, enumerate_below=4, log=None)
            assert res["set"] in best and float(res["lower"]) <= fstar * (1 + 1e-12)
            report(name, res, best)
    else:
        m = int(sys.argv[2])
        limit = float(sys.argv[3]) if len(sys.argv) > 3 else None
        margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
        B, A = geolift()
        res = search(B, A, m, time_limit=limit, full_margin=margin)
        report(f"geolift m={m}", res, KNOWN.get(m))
        if m in KNOWN and res["open"] == 0:
            assert abs(float(res["hi"]) - KNOWN[m]) <= 1e-6 * KNOWN[m]


if __name__ == "__main__":
    main()
