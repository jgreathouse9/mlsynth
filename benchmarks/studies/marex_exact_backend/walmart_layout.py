"""The Walmart MIQP on one set of values in two memory layouts.

Plain arrays: a DataFrame copies into pandas' own layout, so only an ndarray
reaches solve_design in the order it was built in. Run from the repository root.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import time, warnings
import numpy as np, pandas as pd, cvxpy as cp
warnings.filterwarnings("ignore")
from mlsynth.utils.marex_helpers.optimization import solve_design
from mlsynth.utils.marex_helpers.formulation import (
    prepare_clusters, prepare_fit_slices, compute_cluster_means_members,
    build_membership_mask)
from mlsynth.utils.marex_helpers.optimization import _augment_fit
df = pd.read_csv("basedata/walmart_weekly_sales.csv")
vals = df.pivot(index="store", columns="week", values="sales").to_numpy()
z = np.zeros(vals.shape[0], dtype=int)
arrays = {"C order": np.ascontiguousarray(vals), "F order": np.asfortranarray(vals)}
fits = {}
for name, Y in arrays.items():
    Yn, cl, N, labs, K, l2k = prepare_clusters(Y, z)
    Y_fit, _, _ = prepare_fit_slices(Yn, 128, 28)
    X = _augment_fit(Y_fit, None, 1.0, True)
    M = build_membership_mask(cl, l2k, N, K)
    Xbar, _ = compute_cluster_means_members(X, M, labs)
    fits[name] = (np.asarray(X), np.asarray(Xbar, dtype=float))
print("predictors bitwise equal:", bool((fits["C order"][0] == fits["F order"][0]).all()),
      "| target bitwise equal:", bool((fits["C order"][1] == fits["F order"][1]).all()),
      "| max target diff:", float(np.abs(fits["C order"][1] - fits["F order"][1]).max()), flush=True)
for name, Y in arrays.items():
    t0 = time.perf_counter()
    r = solve_design(Y, 128, z, blank_periods=28, m_eq=2, standardize=True, solver=cp.SCIP)
    print(name, "MIQP design step:", round(time.perf_counter() - t0, 2), "s",
          "objective", repr(r["objective"]), flush=True)
