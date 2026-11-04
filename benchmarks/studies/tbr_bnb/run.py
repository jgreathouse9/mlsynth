"""Reproduce the study's measurements on the GeoLift markets (#965).

    python -m benchmarks.studies.tbr_bnb.run certify [0 1 2 3]   # the certified search per group
    python -m benchmarks.studies.tbr_bnb.run lifts                 # root gap of each lift, ten geos

Timings are machine-dependent; run one process per core and cap BLAS threads
(OMP_NUM_THREADS=1), as the measurements in the README were.
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

from .hull import rss_of
from .lifts import Lift
from .panel import GROUPS, group
from .search import certify

TEN_GEO_SEED = 7
TEN_GEO_GROUPS = (("phoenix",), ("baltimore", "columbus"), ("baltimore", "minneapolis", "phoenix"),
                  ("baltimore", "columbus", "honolulu", "miami"))


def run_certify(indices):
    for i in indices:
        names = GROUPS[i]
        y, X, pool = group(names)
        out = certify(y, X)
        s = out.search
        print(f"[{'+'.join(names)}] incumbent RSS {float(out.incumbent):.10g} with "
              f"{[pool[j] for j in out.controls]} at slope {out.slope:.5g}; slope cap {float(out.slope_cap):.4g}; "
              f"{out.pairs} pairs -> hull {out.after_hull} -> shor {out.after_shor}; joint lift: {s.nodes} nodes, "
              f"{s.pruned} pruned, {s.branched} branched, {len(s.open)} open; "
              f"{'CERTIFIED OPTIMAL' if out.certified else 'NOT CERTIFIED'} in {out.seconds:.0f}s", flush=True)


def run_lifts(ratio=1.25):
    """Root gap of each lift level on the ten-geo subset, against enumeration."""
    import pandas as pd
    from benchmarks.cases.tbrmm import _scoring_window
    panel = _scoring_window()
    raw = pd.read_csv("basedata/geolift_test_data.csv")
    keep = sorted(np.random.default_rng(TEN_GEO_SEED).choice(sorted(raw["location"].unique()), size=10,
                                                              replace=False).tolist())
    wide = panel[panel["geo"].isin(keep)].pivot(index="date", columns="geo", values="Y").sort_index()
    geos = list(wide.columns); Y = wide.to_numpy(float); M = Y - Y.mean(0)
    for names in TEN_GEO_GROUPS:
        T = [geos.index(g) for g in names]; pool = [j for j in range(len(geos)) if j not in T]
        y, X = M[:, T].sum(1), M[:, pool]; p = X.shape[1]
        best, C = min((rss_of(y, X, set(c)), c) for r in range(1, p + 1) for c in combinations(range(p), r))
        x = X[:, list(C)].sum(1); b = float((y @ x) / (x @ x))
        edges = [b / 1e3]
        while edges[-1] < b * 1e3:
            edges.append(edges[-1] * ratio)
        pieces = [(0.0, edges[0])] + list(zip(edges[:-1], edges[1:]))
        row = []
        for level in ("shor", "persp", "rlt", "full"):
            L = Lift(y, X, level)
            lb = min(L.piece(lo, hi, m) for lo, hi in pieces for m in range(1, p + 1))
            row.append(f"{level} {(best - lb) / best:+.2%}")
        print(f"{'+'.join(names)}: exact {best:.6g} | " + "  ".join(row), flush=True)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["certify"]
    if cmd == "certify":
        run_certify([int(a) for a in rest] or range(len(GROUPS)))
    elif cmd == "lifts":
        run_lifts()
    else:
        raise SystemExit(__doc__)
