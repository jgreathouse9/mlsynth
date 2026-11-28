"""RCA rung 4: facial reduction of the pair relaxation (no library change), after Im and Wolkowicz (arXiv 2023), whose
two steps are (1) restrict to the minimal face so a strictly feasible point exists, (2) restore full row rank.
Standard form as sparse_ipm2.py builds it (A v = b, v >= 0).
Step 1: the variables that can be positive in some feasible point, found by repeated maximum-support LPs:
  max sum_{j in U} t_j  s.t.  A v = b, v >= 0, 0 <= t_j <= v_j, t_j <= 1  (j in U),
U starting as every column; the columns with t_j > TOL at the optimum are positive somewhere and leave U; repeat
until a round finds none. Z = U at the end: zero in every feasible solution (up to TOL). HiGHS interior point.
Step 2: on the columns outside Z, the rows of a pivoted QR of A' whose pivots are above max(dim) eps |R_11|.
Saved: Z, the kept rows, and the counts, to <out.npz>, for sparse_ipm_rca.py (FR=<out.npz>).
python -u rca_facial.py <worktree> <n> <out.npz>"""
import os, sys, time
for v_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v_] = "1"
wt, n, outp = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["rlt_scale.py", wt, n, "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read().split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.sparse as sp, scipy.linalg as la, highspy
A, lhs, rhs, C, P = g["A"].tocsc(), g["lhs"], g["rhs"], g["C"], g["P"]
m = A.shape[0]; eq = lhs == rhs; ge = ~eq; nge = int(ge.sum())
Af = sp.hstack([A, sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))]).tocsc()
b = np.where(eq, rhs, lhs); NV = Af.shape[1]
kind = np.array(["x"] * C + ["w"] * P + ["s"] * nge)
TOL = 1e-7
U = np.ones(NV, bool); rnd = 0; t0 = time.perf_counter()
while True:
    rnd += 1
    Uj = np.flatnonzero(U); k = Uj.size
    # columns [v (NV) | t (k)]; rows: A v = b; t - v_U <= 0
    T = sp.hstack([-sp.identity(NV, format="csc")[Uj], sp.identity(k, format="csc")]).tocsc()
    M = sp.vstack([sp.hstack([Af, sp.csc_matrix((m, k))]), T]).tocsc()
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("solver", "ipm"); h.setOptionValue("run_crossover", "off")
    lp = highspy.HighsLp(); lp.num_col_ = NV + k; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.r_[np.zeros(NV), -np.ones(k)]
    lp.col_lower_ = np.zeros(NV + k); lp.col_upper_ = np.r_[np.full(NV, highspy.kHighsInf), np.ones(k)]
    lp.row_lower_ = np.r_[b, np.full(k, -highspy.kHighsInf)]; lp.row_upper_ = np.r_[b, np.zeros(k)]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr; lp.a_matrix_.index_ = M.indices; lp.a_matrix_.value_ = M.data
    h.passModel(lp); h.run()
    st = h.modelStatusToString(h.getModelStatus())
    t = np.asarray(h.getSolution().col_value)[NV:]
    pos = Uj[t > TOL]
    print(f"round {rnd}: {st}; {k:,} columns not yet shown positive, {pos.size:,} positive now; "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    U[pos] = False
    if pos.size == 0:
        break
Z = np.flatnonzero(U)
print(f"zero in every feasible solution: {Z.size:,} (x {int((kind[Z] == 'x').sum()):,}, w {int((kind[Z] == 'w').sum()):,}, "
      f"surplus {int((kind[Z] == 's').sum()):,})", flush=True)
keepc = np.flatnonzero(~U)
Ar = Af[:, keepc].tocsr()
live = np.flatnonzero(np.diff(Ar.indptr) > 0)
assert np.all(np.abs(b[np.setdiff1d(np.arange(m), live)]) < 1e-12)
t = time.perf_counter()
R, piv = la.qr(Ar[live].toarray().T, mode="r", pivoting=True, check_finite=False)
dg = np.abs(np.diag(R)); tol = max(Ar.shape) * np.finfo(float).eps * dg[0]; rk = int((dg > tol).sum())
rows = np.sort(live[piv[:rk]])
print(f"rows: {m:,}, {live.size:,} not empty after step 1, rank {rk:,}: {m - rk:,} removed ({time.perf_counter() - t:.1f} s)")
np.savez(outp, Z=Z, rows=rows, m=m, NV=NV)
