import os, sys, time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.argv = ["rlt_scale.py", "/home/user/wt-4a", "24", "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read().split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.sparse as sp, scipy.linalg as la
A = g["A"].tocsc(); m = A.shape[0]
J = np.arange(min(A.shape[1], 13000)); Aj = A[:, J]; th = np.random.default_rng(0).uniform(0.1, 10, J.size)
t = time.perf_counter(); Ms = Aj.multiply(th) @ Aj.T; t1 = time.perf_counter() - t
t = time.perf_counter(); M = Ms.toarray(); t2 = time.perf_counter() - t
M[np.diag_indices(m)] += 1e-6 * M.diagonal().max() + 1.0
t = time.perf_counter(); F = la.cho_factor(M, lower=True, check_finite=False); t3 = time.perf_counter() - t
t = time.perf_counter(); la.cho_solve(F, np.ones(m), check_finite=False); la.cho_solve(F, np.ones(m), check_finite=False); t4 = time.perf_counter() - t
t = time.perf_counter(); A.T @ np.ones(m); t5 = time.perf_counter() - t
print(f"rows {m:,}; sparse A Theta A' {t1:.2f} s ({Ms.nnz:,} nonzeros, {Ms.nnz / m**2:.2%}); to dense {t2:.2f} s "
      f"({m * m * 8 / 1e9:.2f} GB); dense Cholesky {t3:.2f} s ({m**3 / 3 / t3 / 1e9:.0f} GFLOP/s); two solves {t4:.2f} s; "
      f"full pricing A'y {t5 * 1e3:.1f} ms")
