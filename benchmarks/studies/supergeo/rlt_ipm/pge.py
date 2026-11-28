"""Pair-gap elementary inequalities: the part of the pair-product (first-level RLT) relaxation that needs no pair
variables, measured against it at 40 markets (no library change).

Sherali and Lee (Discrete Appl. Math. 68 (1996)): the first level keeps, for every column H and every row i that H does
not cover, x_H = sum of w_GH over the columns G covering i that share no row with H (eqs. (12)-(14), p. 158); it
implies Balas's elementary inequality x_H <= sum of those x_G (Proposition 2, p. 159), and with w_GH = 0 fixed for
the G outside a set Q, x_H <= sum over Q of x_G is valid (Proposition 4, p. 160). Here Q is the partners that fit in
the gap with H, as rlt2.py keeps them: groups G disjoint from H with rc_G + rc_H <= z - L. A design costing at most z
that holds H covers i with such a G (its cost is at least L + rc_G + rc_H), so the rows hold for every design costing
at most z, and min(z, LP) bounds the optimum. Families, each added in full to the group LP over the kept groups:
  bal    x_H <= sum_{G ni i, G disjoint H} x_G               (Balas, no gap test)
  pge    x_H <= sum_{G ni i, G in Pair(H)} x_G               (gap test)
  card   (k - 1) x_H <= sum_{G in Pair(H)} x_G               (the cardinality row times x_H, projected)
  both   pge + card
Panel and arm exactly as rlt2.py: make_seasonal_sales_panel (seed 0, T = 60, design window 42), Q = 2, 40 markets,
optimum 4.889056, group LP 4.165353, RLT 4.623153 (63.3% of the gap) in 8,392 s. Single-threaded.
python -u pge.py <worktree> <out.json> [n]"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
wt, outp = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
sys.path.insert(0, wt)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from fractions import Fraction
from scipy.optimize import linprog
from mlsynth.utils.pangeo_helpers import columns, make_seasonal_sales_panel, prepare_pangeo_inputs
OPT = {40: 4.889056}
RLT = {40: 4.623153}
T0 = time.perf_counter()
def say(m):
    print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)

Q = 2
panel = make_seasonal_sales_panel(units_per_arm=n, arms=("A",), T=60, seed=0)
inp = prepare_pangeo_inputs(df=panel, outcome="sales", arm="arm", unitid="unit", time="time")
idx = np.asarray(inp.arm_units["A"]); Y = inp.Y[:, :42]
pool = columns.score_columns(idx, Y, Q); k = columns.min_groups(n, Q, 1); root = columns.root_lp(pool, k)
z = OPT[n]; L = float(root.bound)
zf = Fraction(z) * (1 + Fraction(1, 10**9))
keep = np.flatnonzero(root.reduced_costs <= columns._up(zf - Fraction(root.bound)))
rc = np.asarray(root.reduced_costs, float)[keep]; c = np.asarray(pool.cost, float)[keep]
mem = pool.members[keep]; C = keep.size
inc = columns._incidence(mem, n).astype(bool)                       # groups x units
gap = z * (1 + 1e-9) - L
disj = ~(inc.astype(np.int32) @ inc.T.astype(np.int32)).astype(bool)  # C x C, True when disjoint
np.fill_diagonal(disj, False)
fit = disj & ((rc[:, None] + rc[None, :]) <= gap)
say(f"n={n}: {C:,} kept groups, {int(disj.sum()) // 2:,} disjoint pairs, {int(fit.sum()) // 2:,} that fit in the gap "
    f"(rlt2.py: 199,142); L {L:.6f}")

INC = sp.csr_matrix(inc.astype(float))
A_cov = INC.T.tocsr()
card = sp.csr_matrix(-np.ones((1, C)))


def elementary(adj):
    """Rows x_H - sum_{G ni i, adj[H, G]} x_G <= 0 for every H and every unit i not in H."""
    rows, cols, vals = [], [], []
    r = 0
    for h in range(C):
        part = np.flatnonzero(adj[h])
        pin = inc[part]                                   # partners x units
        for i in np.flatnonzero(~inc[h]):
            g = part[pin[:, i]]
            rows.append(r); cols.append(h); vals.append(1.0)
            rows += [r] * g.size; cols += g.tolist(); vals += [-1.0] * g.size
            r += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, C))


def cardrows(adj):
    M = sp.csr_matrix(-adj.astype(float))
    return (M + sp.diags(np.full(C, float(k - 1)))).tocsr()


fams = {}
t = time.perf_counter(); fams["bal"] = elementary(disj); tb = time.perf_counter() - t
t = time.perf_counter(); fams["pge"] = elementary(fit); tp = time.perf_counter() - t
fams["card"] = cardrows(fit)
fams["both"] = sp.vstack([fams["pge"], fams["card"]]).tocsr()
say(f"rows built: bal {fams['bal'].shape[0]:,} ({tb:.1f} s), pge {fams['pge'].shape[0]:,} ({tp:.1f} s), "
    f"card {fams['card'].shape[0]:,}")

out = dict(n=n, z=z, L=L, rlt=RLT[n], groups=C, pairs_fit=int(fit.sum()) // 2, runs=[])
t = time.perf_counter()
g = linprog(c, A_ub=card, b_ub=[-float(k)], A_eq=A_cov, b_eq=np.ones(n), bounds=(0, 1), method="highs")
glp = float(g.fun)
say(f"group LP {glp:.6f} ({1 - glp / z:.2%} below the optimum) in {time.perf_counter() - t:.2f} s")
for name in ("bal", "pge", "card", "both"):
    A = sp.vstack([card, fams[name]]).tocsr()
    b = np.r_[-float(k), np.zeros(fams[name].shape[0])]
    for meth in ("highs-ds", "highs-ipm"):
        t = time.perf_counter()
        r = linprog(c, A_ub=A, b_ub=b, A_eq=A_cov, b_eq=np.ones(n), bounds=(0, 1), method=meth)
        s = time.perf_counter() - t
        v = float(r.fun) if r.status == 0 else float("nan")
        x = r.x if r.x is not None else np.zeros(C)
        frac = int(((x > 1e-7) & (x < 1 - 1e-7)).sum())
        rec = dict(family=name, method=meth, rows=int(fams[name].shape[0]), value=v, status=int(r.status), s=s,
                   closed=(v - glp) / (z - glp), of_rlt=(v - glp) / (RLT[n] - glp), fractional=frac)
        out["runs"].append(rec)
        say(f"{name:5s} {meth:9s}: LP {v:.6f} ({1 - v / z:.2%} below), closes {rec['closed']:.1%} of the gap "
            f"({rec['of_rlt']:.1%} of what RLT closes); {frac} fractional; status {r.status} in {s:.2f} s")
        json.dump(out, open(outp, "w"), indent=1)
say("done")
