"""Lagrangian bound for the pair-product (first-level RLT) relaxation, by blocks, at 40 markets (no library change).

Adams, Guignard, Hahn and Hightower (EJOR 180 (2007)) solve the level-1 and level-2 RLT relaxations of the QAP, which
are large and degenerate, by a Lagrangian relaxation whose subproblem is block-diagonal, with dual ascent; every
iterate's value is a valid bound. The same split here. rlt2.py's LP: group variables x, pair variables w_GH over the
kept disjoint pairs that fit in the gap,
  cover              sum_{G ni i} x_G = 1;   cardinality  sum x >= k
  cover * x_H        sum_{G ni i, G in Pair(H)} w_GH = x_H      every H, every unit i not in H
  cardinality * x_H  sum_{G in Pair(H)} w_GH >= (k - 1) x_H     every H
Copy w_GH into u_{G|H} (in H's rows) and u_{H|G} (in G's rows) and dualize u_{G|H} = u_{H|G} with multiplier mu_GH.
For fixed mu the problem splits: H's block, with u = x_H y, is the LP
  v_H(mu) = min sum_G d_{G|H} y_G  s.t.  sum_{G ni i, G in Pair(H)} y_G = 1 (i not in H),  sum y >= k - 1,  y >= 0
with d_{G|H} = +mu_GH and d_{H|G} = -mu_GH (orientation fixed by G < H), infinite if infeasible (x_H fixed at 0); then
  Lag(mu) = min sum_H (c_H + v_H(mu)) x_H  s.t.  cover, cardinality, 0 <= x <= 1,
a group LP. Lag(mu) <= RLT LP for every mu, Lag(0) is the group LP, and max_mu Lag(mu) is the RLT LP (LP duality).
Step: the volume algorithm (Barahona and Anbil, Math. Program. 87 (2000)), since at mu = 0 every block value is 0
and its solution arbitrary, so a single subgradient s_GH = x_H y^H_G - x_G y^G_H is a poor direction: the direction
is the running average sbar = alpha s + (1 - alpha) sbar, the trial point mu_hat + t sbar with
t = f (z - Lag(mu_hat)) / |sbar|^2, a serious step (mu_hat moves) when the trial improves, f times 1.1 after a
serious step whose subgradient agrees with sbar, times 0.66 after <patience> trials without improvement. alpha = 0.1,
f starts at <theta0>. (Not read: the volume paper; this is the scheme as the volume tree in ntree.py uses it.) Reported: the best bound against time, iterations, against the RLT LP 4.623153 (8,392
s) and the optimum 4.889056. Panel and arm as rlt2.py. Single-threaded.
python -u rlt_lag.py <worktree> <out.json> <iterations> <theta0> <patience>"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
wt, outp, ITERS, THETA0, PAT = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
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
rc = np.asarray(root.reduced_costs, float)[keep]; c = np.asarray(pool.cost, float)[keep]
mem = pool.members[keep]; C = keep.size
inc = columns._incidence(mem, n).astype(bool)
gap = z * (1 + 1e-9) - L
disj = ~(inc.astype(np.int32) @ inc.T.astype(np.int32)).astype(bool)
np.fill_diagonal(disj, False)
fit = disj & ((rc[:, None] + rc[None, :]) <= gap)
iu, ju = np.nonzero(np.triu(fit))                 # pairs G < H: iu = G, ju = H
P = iu.size
pid = -np.ones((C, C), np.int64); pid[iu, ju] = np.arange(P); pid[ju, iu] = np.arange(P)
say(f"{C:,} groups, {P:,} pairs (rlt2.py: 199,142); group LP bound L {L:.6f}")

# block H: partners, their sign on mu (+1 when the partner G < H, i.e. d_{G|H} = +mu; -1 when G > H), rows
blocks = []
for h in range(C):
    part = np.flatnonzero(fit[h])
    out_units = np.flatnonzero(~inc[h])
    A = sp.csr_matrix(inc[part][:, out_units].T.astype(float))         # units not in H x partners
    sign = np.where(part < h, 1.0, -1.0)
    blocks.append((part, pid[h, part], sign, A, out_units.size))
A_cov = sp.csr_matrix(inc.T.astype(float)); card = sp.csr_matrix(-np.ones((1, C)))


def evaluate(mu):
    v = np.zeros(C); Y_ = []
    feas = np.ones(C, bool)
    for h, (part, pp, sign, A, m) in enumerate(blocks):
        if part.size == 0:
            feas[h] = False; Y_.append(None); continue
        d = sign * mu[pp]
        r = linprog(d, A_ub=-np.ones((1, part.size)), b_ub=[-(k - 1.0)], A_eq=A, b_eq=np.ones(m), bounds=(0, None),
                    method="highs-ds")
        if r.status != 0:
            feas[h] = False; Y_.append(None); continue
        v[h] = r.fun; Y_.append(r.x)
    cost = c + v
    ub = np.where(feas, 1.0, 0.0)
    g = linprog(cost, A_ub=card, b_ub=[-float(k)], A_eq=A_cov, b_eq=np.ones(n),
                bounds=list(zip(np.zeros(C), ub)), method="highs-ds")
    assert g.status == 0, g.message
    x = g.x
    s = np.zeros(P)
    for h, (part, pp, sign, A, m) in enumerate(blocks):
        if Y_[h] is None or x[h] <= 0:
            continue
        # u_{G|H} enters s_GH with +1 when G < H (d = +mu), the copy u_{H|G} with -1
        np.add.at(s, pp, sign * x[h] * Y_[h])
    return float(g.fun), s, int((~feas).sum())


mu_hat = np.zeros(P); f = THETA0; red = 0; ALPHA = 0.1
best, sbar, _ = evaluate(mu_hat); nfix = 0
hist = []
t0 = time.perf_counter()
for it in range(ITERS):
    nn = float(sbar @ sbar)
    if nn < 1e-18:
        say("direction zero: optimal"); break
    mu = mu_hat + f * (z - best) / nn * sbar
    val, s, nfix = evaluate(mu)
    if val > best + 1e-9:
        if float(s @ sbar) > 0:
            f = min(f * 1.1, 2.0)
        mu_hat = mu; best = val; red = 0
    else:
        red += 1
        if red >= PAT:
            f *= 0.66; red = 0
    sbar = ALPHA * s + (1 - ALPHA) * sbar
    el = time.perf_counter() - t0
    rec = dict(it=it, value=val, best=best, closed=(best - L) / (z - L), of_rlt=(best - L) / (RLT - L), f=f,
               s_norm=float(np.linalg.norm(sbar)), fixed_zero=nfix, seconds=el)
    hist.append(rec)
    if it < 5 or it % 10 == 0:
        say(f"it {it:4d}: trial {val:.6f}, best {best:.6f} (closes {rec['closed']:.1%} of the gap, {rec['of_rlt']:.1%} "
            f"of RLT's), |sbar| {rec['s_norm']:.3f}, f {f:.4f}, {nfix} groups with an infeasible block, {el:.0f} s")
        json.dump(dict(n=n, z=z, L=L, rlt=RLT, pairs=P, hist=hist), open(outp, "w"), indent=1)
json.dump(dict(n=n, z=z, L=L, rlt=RLT, pairs=P, hist=hist), open(outp, "w"), indent=1)
say(f"done: best {best:.6f}, closes {(best - L) / (z - L):.1%} of the gap ({(best - L) / (RLT - L):.1%} of RLT's)")
