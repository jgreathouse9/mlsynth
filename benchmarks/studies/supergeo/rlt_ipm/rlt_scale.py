"""The pair-product (first-level RLT) relaxation at a chosen arm size, built with vectorised indexing and solved
through highspy 1.16 (PDLP by default), with the weak-duality bound from the returned duals (no library change).

Same LP as rlt2.py and rlt_hs.py (which builds it with Python loops; at 40 markets the two must agree): over the groups
kept at the optimum z (root reduced cost <= z - L) and the disjoint pairs of them whose reduced costs fit in the gap,
  cover sum_{G ni i} x_G = 1;  cardinality sum x >= k;  cover * x_H: sum_{G ni i, G in Pair(H)} w_GH - x_H = 0 for
  every H and every unit i not in H;  cardinality * x_H: sum_{G in Pair(H)} w_GH - (k - 1) x_H >= 0;  0 <= x, w <= 1.
Bound: for row duals y clipped to their allowed signs, B(y) = sum_r (y_r > 0 ? y_r lhs_r : y_r rhs_r)
+ sum_v min(r_v l_v, r_v u_v), r = c - A'y; valid at any y, up to float rounding (not claimed rigorous here).
Panel: make_seasonal_sales_panel (seed PANEL_SEED, default 0, T = 60, design window 42), Q = 2; for a seed other than 0
the optimum z comes from columns.solve_exact. Optima (solve_exact): 40 4.889056,
60 5.759575, 80 6.260568. Single-threaded.
python -u rlt_scale.py <worktree> <n> <solver> <time_limit_s> [<option=value> ...]"""
import os, sys, time, resource
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
wt, n, SOLVER, TL = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
extra = [a.split("=", 1) for a in sys.argv[5:]]
sys.path.insert(0, wt)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from fractions import Fraction
import highspy
from mlsynth.utils.pangeo_helpers import columns, make_seasonal_sales_panel, prepare_pangeo_inputs
OPT = {20: 5.0738116454316975, 24: 4.533603317537888, 30: 4.383420575945047, 40: 4.889056469467156, 60: 5.7595746451645775, 80: 6.260567565706034}
Q = 2
if int(os.environ.get("PANEL_SEED", "0")) == 0 and n in OPT:
    z = OPT[n]
else:   # other seeds: the optimum from solve_exact (prepared below, before the LP is built)
    z = None
T0 = time.perf_counter()
def say(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.perf_counter() - T0:7.1f}s, peak {rss:.1f} GB] {m}", flush=True)

PANEL_SEED = int(os.environ.get("PANEL_SEED", "0"))
panel = make_seasonal_sales_panel(units_per_arm=n, arms=("A",), T=60, seed=PANEL_SEED)
inp = prepare_pangeo_inputs(df=panel, outcome="sales", arm="arm", unitid="unit", time="time")
idx = np.asarray(inp.arm_units["A"]); Y = inp.Y[:, :42]
pool = columns.score_columns(idx, Y, Q); k = columns.min_groups(n, Q, 1); root = columns.root_lp(pool, k)
if z is None:
    z = float(columns.solve_exact(idx, Y, Q)[1]["objective_value"])
L = float(root.bound)
zf = Fraction(z) * (1 + Fraction(1, 10**9))
# GROUP_GAP_FACTOR > 1 keeps the groups whose reduced cost fits in that multiple of the gap (more groups, more rows,
# and pairs growing with their square): the scaling direction, since pairs per row grow with the group count.
GROUP_GAP_FACTOR = float(os.environ.get("GROUP_GAP_FACTOR", "1"))
keep = np.flatnonzero(root.reduced_costs <= columns._up(Fraction(GROUP_GAP_FACTOR) * (zf - Fraction(root.bound))))
rc = np.asarray(root.reduced_costs, float)[keep]; c = np.asarray(pool.cost, float)[keep]
mem = np.asarray(pool.members[keep], np.int64); C = keep.size; W = mem.shape[1]
inc = columns._incidence(mem, n).astype(bool)
gap = z * (1 + 1e-9) - L
# PAIR_GAP_FACTOR > 1 keeps the pairs whose reduced costs fit in that multiple of the gap; the groups (and so the
# rows) are unchanged. The LP stays a valid relaxation, only weaker (the scaling test: pairs grow, rows fixed).
PAIR_GAP_FACTOR = float(os.environ.get("PAIR_GAP_FACTOR", "1"))
# pairs: disjoint (bit masks, n <= 62 not needed: compare incidence blocks) and fitting in the gap
PA_l, PB_l = [], []
incf = inc.astype(np.float32)
order = np.argsort(rc)
for a0 in range(0, C, 2048):
    blk = np.arange(a0, min(C, a0 + 2048))
    ov = incf[blk] @ incf.T                                  # overlaps
    ok = (ov == 0) & ((rc[blk, None] + rc[None, :]) <= PAIR_GAP_FACTOR * gap)
    ok &= (np.arange(C)[None, :] > blk[:, None])
    a, b = np.nonzero(ok)
    PA_l.append(blk[a]); PB_l.append(b)
PA = np.concatenate(PA_l); PB = np.concatenate(PB_l); P = PA.size
say(f"n={n}: {C:,} groups kept, {P:,} pairs that fit; group LP bound L {L:.6f}, optimum {z:.6f} ({1 - L / z:.2%} gap)")

# row ids: cover rows 0..n-1, cardinality n, then for each H its units not in H, then the cardinality * x_H rows
out_mask = ~inc
cnt = out_mask.sum(axis=1)
start = n + 1 + np.r_[0, np.cumsum(cnt)[:-1]]
rowid = np.full((C, n), -1, np.int64)
rr, uu = np.nonzero(out_mask)
rowid[rr, uu] = start[rr] + (np.cumsum(out_mask, axis=1)[rr, uu] - 1)
cr0 = n + 1 + int(cnt.sum())
R_ = []; C_ = []; V_ = []
# x columns: cover, cardinality, -1 in its own cover * x_H rows, -(k-1) in its cardinality * x_H row
gr, gu = np.nonzero(inc); R_.append(gu); C_.append(gr); V_.append(np.ones(gr.size))
R_.append(np.full(C, n)); C_.append(np.arange(C)); V_.append(np.ones(C))
R_.append(rowid[rr, uu]); C_.append(rr); V_.append(-np.ones(rr.size))
R_.append(cr0 + np.arange(C)); C_.append(np.arange(C)); V_.append(np.full(C, -(k - 1.0)))
# w columns: row (A, i) for i in B, row (B, i) for i in A, and the two cardinality * x rows
pc = C + np.arange(P)
for s in range(W):
    ub = mem[PB, s]; v = ub >= 0
    R_.append(rowid[PA[v], ub[v]]); C_.append(pc[v]); V_.append(np.ones(int(v.sum())))
    ua = mem[PA, s]; v = ua >= 0
    R_.append(rowid[PB[v], ua[v]]); C_.append(pc[v]); V_.append(np.ones(int(v.sum())))
R_.append(cr0 + PA); C_.append(pc); V_.append(np.ones(P))
R_.append(cr0 + PB); C_.append(pc); V_.append(np.ones(P))
Rn = cr0 + C
A = sp.csc_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))), shape=(Rn, C + P))
del R_, C_, V_
A.sum_duplicates()
lhs = np.zeros(Rn); rhs = np.zeros(Rn)
lhs[:n] = rhs[:n] = 1.0; lhs[n] = k; rhs[n] = np.inf; rhs[cr0:] = np.inf
nz = np.diff(A.tocsr().indptr) > 0
A = A.tocsr()[nz].tocsc(); lhs = lhs[nz]; rhs = rhs[nz]
cost = np.r_[c, np.zeros(P)]; lo = np.zeros(C + P); up = np.ones(C + P)
say(f"LP: {C + P:,} columns, {A.shape[0]:,} rows, {A.nnz:,} nonzeros")

if SOLVER == "clarabel":
    import clarabel
    eq = lhs == rhs; ge = ~eq
    Ae = A.tocsr()[eq]; Ag = A.tocsr()[ge]
    N = C + P; I = sp.identity(N, format="csr")
    M = sp.vstack([Ae, -Ag, -I, I]).tocsc()
    b = np.r_[rhs[eq], -lhs[ge], np.zeros(N), np.ones(N)]
    st = clarabel.DefaultSettings(); st.verbose = True; st.time_limit = TL; st.max_threads = 1
    for kk, vv in extra:
        setattr(st, kk, float(vv) if ("." in vv or "e" in vv) else int(vv))
    cones = [clarabel.ZeroConeT(int(eq.sum())), clarabel.NonnegativeConeT(int(ge.sum()) + 2 * N)]
    t = time.perf_counter()
    solver = clarabel.DefaultSolver(sp.csc_matrix((N, N)), cost, M, b, cones, st)
    sol = solver.solve(); s = time.perf_counter() - t
    zz = np.asarray(sol.z); y = np.zeros(A.shape[0])
    y[np.flatnonzero(eq)] = -zz[:int(eq.sum())]; y[np.flatnonzero(ge)] = zz[int(eq.sum()):int(eq.sum()) + int(ge.sum())]
    one = np.isinf(rhs); y = np.where(one, np.maximum(y, 0.0), y)
    r = cost - A.T @ y
    B = float(np.sum(np.where(y > 0, y * lhs, y * np.where(one, 0.0, rhs))) + np.sum(np.minimum(r * lo, r * up)))
    say(f"clarabel {dict(extra)}: {sol.status} in {s:.1f} s, objective {sol.obj_val:.6f}, {sol.iterations} "
        f"iterations; bound {B:.6f} ({1 - B / z:.2%} below the optimum), closes {(B - L) / (z - L):.1%} of the "
        f"group LP's gap")
    sys.exit(0)
h = highspy.Highs(); h.setOptionValue("output_flag", False)
h.setOptionValue("threads", 1); h.setOptionValue("solver", SOLVER); h.setOptionValue("time_limit", TL)
for kk, vv in extra:
    try:
        vv = float(vv) if ("." in vv or "e" in vv) else int(vv)
    except ValueError:
        pass
    h.setOptionValue(kk, vv)
lp = highspy.HighsLp()
lp.num_col_ = C + P; lp.num_row_ = A.shape[0]
lp.col_cost_ = cost; lp.col_lower_ = lo; lp.col_upper_ = up
lp.row_lower_ = lhs; lp.row_upper_ = np.where(np.isinf(rhs), highspy.kHighsInf, rhs)
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
h.passModel(lp)
t = time.perf_counter(); h.run(); s = time.perf_counter() - t
status = h.modelStatusToString(h.getModelStatus())
sol = h.getSolution(); info = h.getInfo()
y = np.asarray(sol.row_dual, float)
one = np.isinf(rhs)
y = np.where(one, np.maximum(y, 0.0), y)
r = cost - A.T @ y
B = float(np.sum(np.where(y > 0, y * lhs, y * np.where(one, 0.0, rhs))) + np.sum(np.minimum(r * lo, r * up)))
say(f"{SOLVER} {dict(extra)}: {status} in {s:.1f} s, objective {info.objective_function_value:.6f}, dual valid "
    f"{sol.dual_valid}; bound {B:.6f} ({1 - B / z:.2%} below the optimum), closes {(B - L) / (z - L):.1%} of the "
    f"group LP's gap; iterations pdlp {info.pdlp_iteration_count} ipm {info.ipm_iteration_count}")
