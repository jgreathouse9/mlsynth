"""RCA link: can the frozen support satisfy A_J v = b, v >= 0, and what is the restricted LP's optimum? And does A_J
have dependent rows (and is b consistent with them)?
Loads a dump from sparse_ipm2_dbg.py (J, v, s, y), rebuilds the same standard-form LP, and solves the restricted LP
with HiGHS (simplex). Reports its status and value against the full LP reference 4.344031 and the frozen DOBJ
4.344024121, the numerical rank of A_J (pivoted QR of A_J', tolerance max(dim) eps |R_11|), and the least-squares
residual of A_J v = b over the support without the sign constraint (zero if b is in the range of A_J).
python -u rca24_restricted.py <worktree> <dump.npz>"""
import os, sys, time
for v_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v_] = "1"
wt, dump = sys.argv[1], sys.argv[2]
sys.argv = ["rlt_scale.py", wt, "24", "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read().split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.sparse as sp, scipy.linalg as la, highspy
A, lhs, rhs, cost, C, P = g["A"].tocsc(), g["lhs"], g["rhs"], g["cost"], g["C"], g["P"]
m = A.shape[0]; eq = lhs == rhs; ge = ~eq; nge = int(ge.sum())
S_ = sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))
Af = sp.hstack([A, S_]).tocsc(); b = np.where(eq, rhs, lhs); cf = np.r_[cost, np.zeros(nge)]
d = np.load(dump); J = d["J"]
Aj = Af[:, J].tocsc()
print(f"dump {dump}: support {J.size:,} columns ({int(((J >= C) & (J < C + P)).sum()):,} pairs), rows {m:,}")
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
h.setOptionValue("solver", "simplex")
lp = highspy.HighsLp(); lp.num_col_ = J.size; lp.num_row_ = m
lp.col_cost_ = cf[J]; lp.col_lower_ = np.zeros(J.size); lp.col_upper_ = np.full(J.size, highspy.kHighsInf)
lp.row_lower_ = b; lp.row_upper_ = b
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = Aj.indptr; lp.a_matrix_.index_ = Aj.indices; lp.a_matrix_.value_ = Aj.data
h.passModel(lp)
t = time.perf_counter(); h.run(); s = time.perf_counter() - t
st = h.modelStatusToString(h.getModelStatus()); obj = h.getInfo().objective_function_value
print(f"restricted LP (HiGHS simplex): {st} in {s:.1f} s, value {obj:.9f}; full LP reference 4.344031, frozen DOBJ "
      f"4.344024121")
t = time.perf_counter()
Ad = Aj.toarray()
R, piv = la.qr(Ad.T, mode="r", pivoting=True, check_finite=False)
dg = np.abs(np.diag(R)); tol = max(Ad.shape) * np.finfo(float).eps * dg[0]
rank = int((dg > tol).sum())
print(f"numerical rank of A_J: {rank:,} of {m:,} rows ({m - rank:,} dependent); |R_ii| smallest above tol "
      f"{dg[dg > tol].min():.2e}, largest below {dg[dg <= tol].max() if (dg <= tol).any() else 0:.2e}; {time.perf_counter() - t:.1f} s")
xls, res, rk, sv = la.lstsq(Ad, b, check_finite=False, lapack_driver="gelsd")
print(f"least-squares residual |A_J v - b|_inf without signs: {float(np.max(np.abs(Ad @ xls - b))):.2e}")
