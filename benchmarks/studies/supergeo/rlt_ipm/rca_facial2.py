"""RCA rung 4, certified: dual-side facial reduction of the pair relaxation (no library change), after Im and
Wolkowicz (arXiv 2023): {v : A v = b, v >= 0} has no strictly feasible point iff some y gives z = A'y >= 0, z != 0,
b'y = 0; then z'v = b'y = 0 for every feasible v, so v_j = 0 wherever z_j > 0, and y is the certificate (an
exposing vector). rca_facial.py found the zeros from the primal side, whose tolerance let feasible columns through at
24 markets (its reduced LP's optimum came out at 4.5918, above the full LP's 4.344031); here each zero carries y.

Round k, on the columns F not yet shown zero:  max sum_{j in F} t_j  s.t.  A_F' y - t >= 0,  b'y = 0,  0 <= t <= 1,
y free (HiGHS). Z_k = {j in F : (A'y)_j > TAU}; checked: min_{j in F} (A'y)_j >= -EPS |y|_inf and |b'y| <= EPS |b||y|.
Repeat on F minus Z_k until a round finds none. Then on F: strict feasibility max t s.t. A_F v = b, v >= t, t <= 1;
t* > 0 means the face is reached (a relative-interior point exists); t* = 0 that zeros remain undetected.
Then the rows of a pivoted QR of A_F' (full row rank). Saved to <out.npz>: Z, rows, and each round's y and Z_k.
python -u rca_facial2.py <worktree> <n> <out.npz>   (env HIGHS_LOG=<file>: HiGHS's iteration log for each round)"""
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
TAU, EPS = 1e-6, 1e-9
INF = highspy.kHighsInf


def solve(h, lp):
    h.passModel(lp); h.run()
    return h.modelStatusToString(h.getModelStatus())


F = np.ones(NV, bool); Ys, Zs = [], []; t0 = time.perf_counter(); rnd = 0
while True:
    rnd += 1
    Fj = np.flatnonzero(F); k = Fj.size
    AF = Af[:, Fj].tocsc()
    # columns [y (m) | t (k)]; rows: A_F' y - t >= 0 (k rows), b'y = 0 (1 row)
    M = sp.vstack([sp.hstack([AF.T, -sp.identity(k)]), sp.hstack([sp.csr_matrix(b[None, :]), sp.csr_matrix((1, k))])]).tocsc()
    h = highspy.Highs(); h.setOptionValue("threads", 1)
    if os.environ.get("HIGHS_LOG"):   # the solver's own log, appended as it runs (CLAUDE.md invariant 10)
        h.setOptionValue("output_flag", True); h.setOptionValue("log_to_console", False)
        h.setOptionValue("log_file", os.environ["HIGHS_LOG"])
    else:
        h.setOptionValue("output_flag", False)
    # interior point, no crossover: the exposing-vector LP is degenerate and dual simplex stalls in phase 1 on it
    # (24 markets: ~17,000 primal infeasibilities still open after 37 s); the certificate checks apply to the y returned
    h.setOptionValue("solver", "ipm"); h.setOptionValue("run_crossover", "off")
    lp = highspy.HighsLp(); lp.num_col_ = m + k; lp.num_row_ = k + 1
    lp.col_cost_ = np.r_[np.zeros(m), -np.ones(k)]
    lp.col_lower_ = np.r_[np.full(m, -INF), np.zeros(k)]; lp.col_upper_ = np.r_[np.full(m, INF), np.ones(k)]
    lp.row_lower_ = np.r_[np.zeros(k), 0.0]; lp.row_upper_ = np.r_[np.full(k, INF), 0.0]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr; lp.a_matrix_.index_ = M.indices; lp.a_matrix_.value_ = M.data
    st = solve(h, lp)
    y = np.asarray(h.getSolution().col_value)[:m]
    z = AF.T @ y
    ny = max(float(np.max(np.abs(y))), 1e-300)
    ok = float(z.min()) >= -EPS * ny * 10 and abs(float(b @ y)) <= EPS * ny * float(np.abs(b).sum()) * 10
    Zk = Fj[z > TAU * ny]
    print(f"round {rnd}: {st}; {k:,} columns left; exposing vector |y| {ny:.2e}, min z {float(z.min()):.2e}, b'y "
          f"{float(b @ y):.2e} (certificate {'valid' if ok else 'INVALID'}); {Zk.size:,} new zeros (x "
          f"{int((kind[Zk] == 'x').sum()):,}, w {int((kind[Zk] == 'w').sum()):,}, s {int((kind[Zk] == 's').sum()):,}); "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    if Zk.size == 0 or not ok:
        break
    Ys.append(y); Zs.append(Zk); F[Zk] = False
Fj = np.flatnonzero(F); AF = Af[:, Fj].tocsc(); nc = Fj.size
Z = np.flatnonzero(~F)
print(f"certified zeros: {Z.size:,} of {NV:,} (x {int((kind[Z] == 'x').sum()):,}, w {int((kind[Z] == 'w').sum()):,}, "
      f"s {int((kind[Z] == 's').sum()):,}) in {len(Ys)} rounds", flush=True)
# strict feasibility on the face: max t s.t. A_F v = b, v - t >= 0, t <= 1
M = sp.vstack([sp.hstack([AF, sp.csc_matrix((m, 1))]), sp.hstack([sp.identity(nc, format="csc"), -np.ones((nc, 1))])]).tocsc()
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
lp = highspy.HighsLp(); lp.num_col_ = nc + 1; lp.num_row_ = M.shape[0]
lp.col_cost_ = np.r_[np.zeros(nc), -1.0]; lp.col_lower_ = np.full(nc + 1, -INF); lp.col_upper_ = np.r_[np.full(nc, INF), 1.0]
lp.row_lower_ = np.r_[b, np.zeros(nc)]; lp.row_upper_ = np.r_[b, np.full(nc, INF)]
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = M.indptr; lp.a_matrix_.index_ = M.indices; lp.a_matrix_.value_ = M.data
st = solve(h, lp)
print(f"strict feasibility on the face: {st}, max t = {-h.getInfo().objective_function_value:.3e}", flush=True)
Ar = AF.tocsr(); live = np.flatnonzero(np.diff(Ar.indptr) > 0)
assert np.all(np.abs(b[np.setdiff1d(np.arange(m), live)]) < 1e-12)
R, piv = la.qr(Ar[live].toarray().T, mode="r", pivoting=True, check_finite=False)
dg = np.abs(np.diag(R)); tol = max(Ar.shape) * np.finfo(float).eps * dg[0]; rk = int((dg > tol).sum())
rows = np.sort(live[piv[:rk]])
print(f"rows: {m:,}, {live.size:,} not empty on the face, rank {rk:,}: {m - rk:,} removed", flush=True)
np.savez(outp, Z=Z, rows=rows, m=m, NV=NV, Y=np.array(Ys, dtype=object), Zs=np.array(Zs, dtype=object), allow_pickle=True)
