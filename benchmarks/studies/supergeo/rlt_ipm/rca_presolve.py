"""RCA rung 4: is the pair relaxation handed to the interior point strictly feasible, and do its dependent rows come
from variables fixed at zero in every feasible solution? (no library change)

Standard form as sparse_ipm2.py builds it: A v = b, v >= 0 (group columns x, pair columns w, one surplus per >= row;
x <= 1 and w <= 1 are implied). Two presolve rules of Andersen and Andersen (Math. Program. 71 (1995) 221-245),
applied until nothing changes, with bounds l = 0, u = +inf:
  singleton row (rule (v), p. 224): a row with one active entry fixes that variable at b_i / a_ik (here 0);
  forcing constraint (rule (x), p. 227): a row with b_i = 0 whose active coefficients all have one sign forces
  every variable in it to 0;
  an empty row with b_i != 0 is infeasible, with b_i = 0 it is dropped.
Reported: the variables fixed, by type; the rows dropped; the reduced matrix's numerical rank (pivoted QR, as in
rca24_restricted.py); and strict feasibility of the reduced LP, max t s.t. A v = b, v >= t, t <= 1 (HiGHS):
t* > 0 means a strictly feasible point exists (Slater), t* = 0 that some variable is zero in every feasible solution
(Im and Wolkowicz, arXiv 2023: every basic feasible solution is then degenerate).
python -u rca_presolve.py <worktree> <n>"""
import os, sys, time
for v_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v_] = "1"
wt, n = sys.argv[1], sys.argv[2]
sys.argv = ["rlt_scale.py", wt, n, "none", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlt_scale.py")).read().split('if SOLVER == "clarabel":')[0]
g = {}; exec(compile(src, "rlt_scale.py", "exec"), g)
import numpy as np, scipy.sparse as sp, scipy.linalg as la, highspy
A, lhs, rhs, C, P = g["A"].tocsc(), g["lhs"], g["rhs"], g["C"], g["P"]
m = A.shape[0]; eq = lhs == rhs; ge = ~eq; nge = int(ge.sum())
Af = sp.hstack([A, sp.csc_matrix((-np.ones(nge), (np.flatnonzero(ge), np.arange(nge))), shape=(m, nge))]).tocsr()
b = np.where(eq, rhs, lhs); NV = Af.shape[1]
kind = np.array(["x"] * C + ["w"] * P + ["s"] * nge)
fixed = np.zeros(NV, bool); row_live = np.ones(m, bool)
indptr, idx, val = Af.indptr, Af.indices, Af.data
rounds = 0; t0 = time.perf_counter()
while True:
    rounds += 1; changed = False
    for i in np.flatnonzero(row_live):
        cols = idx[indptr[i]:indptr[i + 1]]; a = val[indptr[i]:indptr[i + 1]]
        act = ~fixed[cols]; cols, a = cols[act], a[act]
        if cols.size == 0:
            assert abs(b[i]) < 1e-12, f"row {i}: empty with b = {b[i]} (infeasible)"
            row_live[i] = False; changed = True; continue
        if cols.size == 1:
            val_ = b[i] / a[0]
            assert abs(val_) < 1e-12, f"singleton row {i} fixes a variable at {val_} (not 0)"
            fixed[cols[0]] = True; row_live[i] = False; changed = True; continue
        if abs(b[i]) < 1e-12 and (np.all(a > 0) or np.all(a < 0)):
            fixed[cols] = True; row_live[i] = False; changed = True
    if not changed:
        break
fx = {k: int((fixed & (kind == k)).sum()) for k in "xws"}
print(f"n={n}: {m:,} rows, {NV:,} columns (x {C:,}, w {P:,}, surplus {nge:,}); presolve {rounds} rounds, "
      f"{time.perf_counter() - t0:.1f} s: fixed at 0: x {fx['x']:,}, w {fx['w']:,}, surplus {fx['s']:,}; rows dropped "
      f"{int((~row_live).sum()):,}")
keepc = np.flatnonzero(~fixed); keepr = np.flatnonzero(row_live)
Ar = Af[keepr][:, keepc].tocsc(); br = b[keepr]
t = time.perf_counter()
R, piv = la.qr(Ar.toarray().T, mode="r", pivoting=True, check_finite=False)
dg = np.abs(np.diag(R)); tol = max(Ar.shape) * np.finfo(float).eps * dg[0]; rk = int((dg > tol).sum())
print(f"reduced: {Ar.shape[0]:,} rows x {Ar.shape[1]:,} columns, rank {rk:,} ({Ar.shape[0] - rk:,} dependent), "
      f"{time.perf_counter() - t:.1f} s")
# strict feasibility: max t s.t. Ar v - t * 0 = br, v - t >= 0, t <= 1  (variables v, t)
nc = Ar.shape[1]
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("solver", "ipm")
lp = highspy.HighsLp(); lp.num_col_ = nc + 1
lp.col_cost_ = np.r_[np.zeros(nc), -1.0]; lp.col_lower_ = np.r_[np.full(nc, -highspy.kHighsInf), -highspy.kHighsInf]
lp.col_upper_ = np.r_[np.full(nc, highspy.kHighsInf), 1.0]
M = sp.vstack([sp.hstack([Ar, sp.csc_matrix((Ar.shape[0], 1))]),
               sp.hstack([sp.identity(nc, format="csc"), -np.ones((nc, 1))])]).tocsc()
lp.num_row_ = M.shape[0]
lp.row_lower_ = np.r_[br, np.zeros(nc)]; lp.row_upper_ = np.r_[br, np.full(nc, highspy.kHighsInf)]
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = M.indptr; lp.a_matrix_.index_ = M.indices; lp.a_matrix_.value_ = M.data
h.passModel(lp); t = time.perf_counter(); h.run()
print(f"strict feasibility of the reduced LP: {h.modelStatusToString(h.getModelStatus())}, max t = "
      f"{-h.getInfo().objective_function_value:.3e} ({time.perf_counter() - t:.1f} s)")
