"""Monotone dual ascent on the pair-product (first-level RLT) relaxation, by blocks, at 40 markets (no library change).

The relaxation is rlt2.py's: group variables x, and a pair variable for each kept disjoint pair that fits in the gap,
written as two copies u_{G|H} (in H's rows) and u_{H|G} (in G's rows) that are equal. Block H, for x_H = 1, is the LP
  min sum_G d_{G|H} y_G  s.t.  sum_{G ni i, G in Pair(H)} y_G = 1 (i not in H),  sum y >= k - 1,  y >= 0,
and the bound is the group LP with costs c_H + v_H (v_H the block's value; x_H = 0 when the block is infeasible).
Every cost move below rewrites the objective by an identity that holds on the relaxation's feasible set, so each
iterate's bound is a valid lower bound, and none lowers a block value or a master cost, so the bound never falls:
  (a) master slack into the block: x_H's master reduced cost cbar_H > 0 moves to H's copies, using
      x_H = (1 / (n - |H|)) sum_G |G| u_{G|H} (the block's cover rows summed): d_{G|H} += cbar_H |G| / (n - |H|),
      c_H -= cbar_H. Block H's value rises by exactly cbar_H.
  (b) block slack across the pair: with r^H_G >= 0 the block's LP reduced cost after (a), move phi r^H_G from
      d_{G|H} to d_{H|G} (u_{G|H} = u_{H|G}). Block H keeps its value (its dual stays feasible); block G's costs rise.
This is the scheme of Adams and Johnson's level-1 dual ascent for the QAP as Adams, Guignard, Hahn and Hightower (EJOR
180 (2007)) describe it, reduced cost moved between a block and its mirror; the moves here are derived for this
relaxation, not copied. Floats; rigour not claimed in this measurement (block reduced costs clipped at 0).
python -u rlt_da.py <worktree> <out.json> <iterations> <phi>"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
wt, outp, ITERS, PHI = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
sys.path.insert(0, wt)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from fractions import Fraction
from scipy.optimize import linprog
from mlsynth.utils.pangeo_helpers import columns, make_seasonal_sales_panel, prepare_pangeo_inputs
n, Q, z, RLT = 40, 2, 4.889056, 4.623153
T0 = time.perf_counter()
def say(m):
    print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)

panel = make_seasonal_sales_panel(units_per_arm=n, arms=("A",), T=60, seed=0)
inp = prepare_pangeo_inputs(df=panel, outcome="sales", arm="arm", unitid="unit", time="time")
idx = np.asarray(inp.arm_units["A"]); Y = inp.Y[:, :42]
pool = columns.score_columns(idx, Y, Q); k = columns.min_groups(n, Q, 1); root = columns.root_lp(pool, k)
L = float(root.bound)
zf = Fraction(z) * (1 + Fraction(1, 10**9))
keep = np.flatnonzero(root.reduced_costs <= columns._up(zf - Fraction(root.bound)))
rc = np.asarray(root.reduced_costs, float)[keep]; c = np.asarray(pool.cost, float)[keep].copy()
mem = pool.members[keep]; C = keep.size
inc = columns._incidence(mem, n).astype(bool)
size = inc.sum(axis=1).astype(float)
gap = z * (1 + 1e-9) - L
disj = ~(inc.astype(np.int32) @ inc.T.astype(np.int32)).astype(bool)
np.fill_diagonal(disj, False)
fit = disj & ((rc[:, None] + rc[None, :]) <= gap)
say(f"{C:,} groups, {int(fit.sum()) // 2:,} pairs; group LP bound {L:.6f}")

# d[h] holds the costs of block h's copies, aligned with part[h]; mirror[h][j] = position of h in block part[h][j]
part = [np.flatnonzero(fit[h]) for h in range(C)]
pos = [dict(zip(p.tolist(), range(p.size))) for p in part]
mirror = [np.array([pos[g][h] for g in part[h]], np.int64) for h in range(C)]
Ablk = [sp.csr_matrix(inc[part[h]][:, ~inc[h]].T.astype(float)) for h in range(C)]
d = [np.zeros(p.size) for p in part]
A_cov = sp.csr_matrix(inc.T.astype(float)); card = sp.csr_matrix(-np.ones((1, C)))


def blocks():
    v = np.full(C, np.inf); r = [None] * C
    for h in range(C):
        if part[h].size == 0:
            continue
        A = Ablk[h]
        res = linprog(d[h], A_ub=-np.ones((1, part[h].size)), b_ub=[-(k - 1.0)], A_eq=A, b_eq=np.ones(A.shape[0]),
                      bounds=(0, None), method="highs-ds")
        if res.status != 0:
            continue
        v[h] = res.fun
        alpha = np.asarray(res.eqlin.marginals); m_ub = float(res.ineqlin.marginals[0])
        r[h] = np.maximum(d[h] - A.T @ alpha + m_ub, 0.0)
    return v, r


hist = []; best = -np.inf; t0 = time.perf_counter()
for it in range(ITERS):
    v, r = blocks()
    feas = np.isfinite(v)
    cost = c + np.where(feas, v, 0.0)
    g = linprog(cost, A_ub=card, b_ub=[-float(k)], A_eq=A_cov, b_eq=np.ones(n),
                bounds=list(zip(np.zeros(C), np.where(feas, 1.0, 0.0))), method="highs-ds")
    assert g.status == 0, g.message
    val = float(g.fun); best = max(best, val)
    pi = np.asarray(g.eqlin.marginals); s_m = float(g.ineqlin.marginals[0])
    cbar = cost - A_cov.T @ pi + s_m
    el = time.perf_counter() - t0
    rec = dict(it=it, value=val, closed=(val - L) / (z - L), of_rlt=(val - L) / (RLT - L), seconds=el,
               infeasible_blocks=int((~feas).sum()))
    hist.append(rec)
    if it < 5 or it % 5 == 0:
        say(f"it {it:4d}: bound {val:.6f} (closes {rec['closed']:.1%} of the gap, {rec['of_rlt']:.1%} of RLT's), "
            f"{rec['infeasible_blocks']} infeasible blocks; blocks with v > 0: {int((v[feas] > 1e-9).sum())}, sum v "
            f"{float(v[feas].sum()):.4f}, master slack pushed {float(cbar[feas & (cbar > 1e-12)].sum()):.4f}, "
            f"master support {int((g.x > 1e-9).sum())}; {el:.0f} s")
        json.dump(dict(n=n, z=z, L=L, rlt=RLT, phi=PHI, hist=hist), open(outp, "w"), indent=1)
    # (a) master slack into the blocks
    for h in np.flatnonzero(feas & (cbar > 1e-12)):
        shift = cbar[h] * size[part[h]] / (n - size[h])
        d[h] += shift; r[h] = r[h] + shift; c[h] -= cbar[h]
    # (b) block slack across each pair, from the snapshot r
    moves = [PHI * r[h] if r[h] is not None else None for h in range(C)]
    for h in range(C):
        if moves[h] is None:
            continue
        d[h] -= moves[h]
        for j, gq in enumerate(part[h]):
            d[gq][mirror[h][j]] += moves[h][j]
json.dump(dict(n=n, z=z, L=L, rlt=RLT, phi=PHI, hist=hist), open(outp, "w"), indent=1)
say(f"done: bound {best:.6f}, closes {(best - L) / (z - L):.1%} of the gap ({(best - L) / (RLT - L):.1%} of RLT's)")
