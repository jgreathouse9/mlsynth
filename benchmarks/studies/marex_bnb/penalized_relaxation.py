"""How tight is the plain continuous relaxation of MAREX's penalized design?

The design is that of Abadie and Zhao (2026, online appendix, section OA.1),
which adapts the penalty of Abadie and L'Hour (2021) to the choice of markets:

f(S) = min ||Bc w||^2 + ||Bc v||^2 + lam * a'w + lam * a'v, a_j = ||x_j - xbar||^2,
w on the simplex over S, v over the complement. The relaxation keeps w <= z,
v <= 1 - z, sum z = m with z in [0, 1]: one convex QP, no lift. The exact
optimum is the library's search; with a penalty on one cluster it fits most
candidates (#777), so the run takes minutes per lambda.

From the repository root::

    python -m benchmarks.studies.marex_bnb.penalized_relaxation
"""
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import cvxpy as cp
import numpy as np
from mlsynth.utils.marex_helpers.exact import search_cluster
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

from benchmarks.studies.marex_bnb.panels import geolift, walmart


def relax(Bc, a, lam, m):
    T, n = Bc.shape
    s = np.sqrt(np.max(a))                     # scale for the solver only
    Bs, as_ = Bc / s, a / s**2
    w, v, z = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True), cp.Variable(n)
    obj = cp.sum_squares(Bs @ w) + cp.sum_squares(Bs @ v) + lam * as_ @ w + lam * as_ @ v
    cons = [cp.sum(w) == 1, cp.sum(v) == 1, z >= 0, z <= 1, cp.sum(z) == m, w <= z, v <= 1 - z]
    p = cp.Problem(cp.Minimize(obj), cons)
    p.solve(solver=cp.CLARABEL)
    return p.value * s**2, int((w.value > 1e-6).sum()), int((v.value > 1e-6).sum())


def panels():
    B, A = geolift()
    yield "GeoLift (n=40, T=90)", B, A, [4, 6]
    B, A = walmart()
    yield "Walmart, covariates (n=45, T=104)", B, A, [2, 4]


for name, B, A, ms in panels():
    Bc = B - A[:, None]
    a = (Bc ** 2).sum(axis=0)
    print(name, flush=True)
    for m in ms:
        for lam in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
            t = time.perf_counter()
            best = search_cluster(B, A, [m], filt=SubsetFilter(),
                                  linear_treated=lam * a if lam else None,
                                  linear_control=lam * a if lam else None)[0]
            t_exact = time.perf_counter() - t
            t = time.perf_counter()
            r, kw, kv = relax(Bc, a, lam, m)
            t_rel = time.perf_counter() - t
            nz_w = int((best.w > 1e-9).sum())
            print(f"  m={m} lam={lam:<6} exact {best.value:.6g} ({t_exact:.2f} s, "
                  f"{nz_w} treated with weight)  relaxation {max(r, 0):.6g} = "
                  f"{max(r, 0) / best.value:6.1%} ({t_rel:.2f} s; support w {kw}, v {kv})",
                  flush=True)
