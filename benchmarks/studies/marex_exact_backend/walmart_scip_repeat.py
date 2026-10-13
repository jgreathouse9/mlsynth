"""The Walmart MIQP's design step in a fresh process, on the estimator's panel.

``timing.py`` timed this solve at 46 s; a fresh process on bitwise-identical
input took 80 s and stopped at a different incumbent. Rerun under fixed
``PYTHONHASHSEED`` values to see whether the variation follows the hash seed.
Run from the repository root.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import time, warnings
import numpy as np, pandas as pd, cvxpy as cp
warnings.filterwarnings("ignore")
from mlsynth.utils.marex_helpers.optimization import solve_design
from mlsynth.utils.marex_helpers.setup import prepare_marex_panel
df = pd.read_csv("basedata/walmart_weekly_sales.csv")
panel = prepare_marex_panel(df, outcome="sales", unitid="store", time="week", cluster=None,
                            T0=128, inference=True, blank_periods=28, T_post=15)
print("file order of stores:", list(panel.Y_full.index[:8]), "...")
t0 = time.perf_counter()
r = solve_design(panel.Y_full, 128, panel.clusters, blank_periods=28, m_eq=2,
                 standardize=True, solver=cp.SCIP)
print("MIQP design step, file order:", round(time.perf_counter() - t0, 2), "s",
      "objective", r["objective"], flush=True)
