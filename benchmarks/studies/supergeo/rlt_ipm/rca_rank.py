import os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
n = sys.argv[2]
sys.argv = ["rlt_scale.py", sys.argv[1], n, "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read().split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.sparse as sp, scipy.linalg as la
A, lhs, rhs = g["A"].tocsc(), g["lhs"], g["rhs"]; m = A.shape[0]; eq = lhs == rhs; ge = ~eq; nge = int(ge.sum())
Af = sp.hstack([A, sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))]).tocsc()
R, piv = la.qr(Af.toarray().T, mode="r", pivoting=True, check_finite=False)
dg = np.abs(np.diag(R)); tol = max(Af.shape) * np.finfo(float).eps * dg[0]
rk = int((dg > tol).sum())
# rows that touch only group columns of a single group (cover * x_H rows no partner covers): A row with one nonzero
nnz_row = np.diff(A.tocsr().indptr)
print(f"n={n}: full standard-form A {m:,} rows, rank {rk:,} ({m - rk:,} dependent); rows with a single nonzero "
      f"(-x_H alone: unit i not coverable by H's partners) {int((nnz_row == 1).sum()):,}")
