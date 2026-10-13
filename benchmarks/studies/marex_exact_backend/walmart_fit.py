"""The Walmart example of docs/marex.rst, timed end to end with each backend.

The whole ``MAREX(...).fit()``, inference included: the search five times after
a warm-up, the MIQP once. Run from the repository root.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from mlsynth import MAREX
df = pd.read_csv("basedata/walmart_weekly_sales.csv")
base = {"df": df, "outcome": "sales", "unitid": "store", "time": "week",
        "T0": 128, "blank_periods": 28, "T_post": 15, "m_eq": 2,
        "design": "standard", "standardize": True, "inference": True,
        "display_graph": False}
MAREX({**base, "backend": "exact"}).fit()          # warm-up
for backend, reps in (("exact", 5), ("mip", 1)):
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = MAREX({**base, "backend": backend}).fit()
        runs.append(time.perf_counter() - t0)
    print(backend, "end-to-end fit median", round(float(np.median(runs)), 3), "s",
          "treated", sorted(int(u) for u in r.treated_units),
          "p", round(r.globres.inference.global_p_value, 3),
          "objective", r.globres.objective, "load", os.getloadavg()[0], flush=True)
