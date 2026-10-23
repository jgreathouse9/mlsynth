"""The library's branch and bound on the study's panels, and MAREX end to end.

``mlsynth.utils.marex_helpers.branch`` is the study's persp+N search moved into
the exact backend, which hands it any cluster above twenty million treated sets.
This runs it on the panels of ``results/persp_n.md`` and checks each answer
against the optima found there (the library's enumeration with its limit
raised, and the study's certified search). Then GeoLift with ten treated
through ``MAREX`` itself, which the exact backend refused before the search.

From the repository root::

    python -m benchmarks.studies.marex_bnb.library_search
"""
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import pandas as pd

from benchmarks.studies.marex_bnb.panels import ROOT, geolift, walmart
from mlsynth import MAREX
from mlsynth.utils.marex_helpers.branch import branch_and_bound
from mlsynth.utils.marex_helpers.exact import count_candidates
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

#: Optima from results/persp_n.md: value and treated set (column positions).
KNOWN = {("geolift", 8): (181943.3763, (0, 2, 10, 12, 18, 20, 24, 27)),
         ("geolift", 10): (127404.7767, (1, 21, 22, 23, 28, 29, 30, 35, 37, 39)),
         ("walmart", 6): (0.03101918869, (1, 2, 11, 26, 31, 39)),
         ("walmart", 8): (0.02248592328, (1, 2, 11, 13, 26, 31, 33, 39))}


def search(name, m):
    B, A = walmart() if name == "walmart" else geolift()
    total = count_candidates(B.shape[1], [m], SubsetFilter())
    t0 = time.perf_counter()
    res = branch_and_bound(B, A, [m], filt=SubsetFilter())
    secs = time.perf_counter() - t0
    d, c = res.designs[0], res.certificate
    value, treated = KNOWN[(name, m)]
    agree = d.treated == treated and abs(d.value - value) <= 1e-9 * value
    print(f"{name} m={m}: {total:,} sets, {secs:.1f} s, {res.lifts} bounds, "
          f"{res.priced:,} sets priced; {d.value:.10g} at {d.treated}; "
          f"certified [{c.lower:.10g}, {c.upper:.10g}], relative gap "
          f"{c.gap / c.upper:.1e}, {len(c.unresolved)} unresolved; "
          f"{'agrees' if agree else 'DIFFERS'} with results/persp_n.md", flush=True)


def marex_geolift(m):
    d = pd.read_csv(ROOT / "basedata" / "geolift_market_data.csv")
    times = sorted(d["date"].unique())
    t0 = time.perf_counter()
    res = MAREX({"df": d, "outcome": "Y", "unitid": "location", "time": "date",
                 "T0": 90, "m_eq": m, "backend": "exact", "certify": True,
                 "inference": False, "display_graph": False}).fit()
    secs = time.perf_counter() - t0
    cert = res.certificate
    value = KNOWN[("geolift", m)][0]
    agree = abs(res.globres.objective - value) <= 1e-9 * value
    print(f"MAREX GeoLift m={m} ({len(times)} days, T0=90): {secs:.1f} s, "
          f"objective {res.globres.objective:.10g}, treated "
          f"{sorted(map(str, res.assignment['treated']))}; certificate "
          f"[{cert.lower:.10g}, {cert.upper:.10g}]; "
          f"{'agrees' if agree else 'DIFFERS'} with results/persp_n.md", flush=True)


def main(which):
    if which in ("all", "search"):
        for name, m in (("geolift", 8), ("geolift", 10), ("walmart", 6), ("walmart", 8)):
            search(name, m)
    if which in ("all", "marex"):
        marex_geolift(10)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
