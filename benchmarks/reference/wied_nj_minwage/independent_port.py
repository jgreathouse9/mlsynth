"""Independent port of Wied's DR-SC, written from the paper's equations and
using statsmodels' Probit for the DR step -- NOT the author's probit_fit.

  eq (4) : per-cell probit DR of 1{Y<=y} on X
  eq (6) : Gram matrix G and cross-vector c, averaged over pre-periods and grid
  eq (7) : closed-form min-norm weights under 1'w = 1
  eq (9) : counterfactual DR parameter as the weighted donor combination
  eq (2) : f_t(x) = mean_l [ Lambda(x'th1) - Lambda(x'th0) ]^2
"""
import json, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
import statsmodels.api as sm

# The estimation sample and its standardisation constants sit beside this script,
# so it runs from any working directory. Both are the float64 files
# ``provenance.txt`` describes; the Gram matrix has kappa ~ 9.6e4 and a float32
# round-trip moves the donor weights by 40x the tolerance they reproduce at.
_DIR = Path(__file__).resolve().parent

d = pd.read_parquet(_DIR / "nj_estimation_sample.parquet")
d["STATEFIP"] = d["STATEFIP"].astype(int); d["t"] = d["t"].astype(int)
meta = json.loads((_DIR / "nj_meta.json").read_text())
SP = {k: meta[k] for k in ("me","se","mx","sx")}
NJ, PRE, POST = 34, [1, 2, 3], 4
states = [NJ] + sorted(s for s in d.STATEFIP.unique() if s != NJ)

# grid: 38 quantiles on [0.10, 0.90], ties collapsed  (paper: m = 32 active)
grid = np.unique(np.round(np.quantile(d.logwage.values, np.linspace(.10, .90, 38)), 8))
m, p = len(grid), 4

def design(g):
    X = np.empty((len(g), 4)); X[:, 0] = 1.0
    X[:, 1] = g.e_std.values; X[:, 2] = g.x_std.values; X[:, 3] = g.x_std.values ** 2
    return X

t0 = time.time(); theta = {}
for si, s in enumerate(states):
    for t in (1, 2, 3, 4):
        g = d[(d.STATEFIP == s) & (d.t == t)]
        X, y = design(g), g.logwage.values
        th = np.zeros((m, p))
        for j, yy in enumerate(grid):
            z = (y <= yy).astype(float)
            if z.min() == z.max():                      # degenerate cell
                th[j] = [8.0 if z[0] == 1 else -8.0, 0, 0, 0]; continue
            th[j] = sm.Probit(z, X).fit(disp=0, warn_convergence=False).params
        theta[(si, t)] = th
print(f"DR via statsmodels: {time.time()-t0:.0f}s, m={m}", flush=True)

# eq (6): G and c
J = len(states) - 1
G = np.zeros((J, J)); c = np.zeros(J)
for t in PRE:
    for a in range(J):
        c[a] += np.sum(theta[(0, t)] * theta[(a+1, t)])
        for b in range(J):
            G[a, b] += np.sum(theta[(a+1, t)] * theta[(b+1, t)])
G /= (len(PRE) * m); c /= (len(PRE) * m)

# eq (7): closed-form weights
one = np.ones(J); Gi = np.linalg.inv(G); Gic, Gi1 = Gi @ c, Gi @ one
w = Gic - Gi1 * ((one @ Gic - 1.0) / (one @ Gi1))

# eq (9) + eq (2)
th0 = sum(w[a] * theta[(a+1, POST)] for a in range(J))
def eval_point(ed, ex):
    es = (ed - SP["me"]) / SP["se"]; xs = (ex - SP["mx"]) / SP["sx"]
    return np.array([1.0, es, xs, xs**2])

print(f"kappa(G) = {np.linalg.cond(G):.5g}   (paper 9.6e4)")
print(f"sum(w)   = {w.sum():.9f}")
FIPS = {12:"Florida",36:"New York",46:"South Dakota",32:"Nevada",29:"Missouri"}
PAPER = {"Florida":.515,"New York":.479,"South Dakota":-.255,"Nevada":.199,"Missouri":-.196}
don = states[1:]
print("\n=== Table 1: independent port vs paper ===")
for a in np.argsort(-np.abs(w))[:5]:
    nm = FIPS.get(don[a], str(don[a]))
    print(f"  {nm:<14}{w[a]:>9.4f}   paper {PAPER.get(nm, float('nan')):>7.3f}")
PAPER_F = {"x0":0.58,"x10":1.71,"xhl":1.55,"xlh":0.61,"x90":0.21}
EVAL = [("x0",12,10),("x10",10,2),("xhl",16,2),("xlh",10,37),("x90",16,37)]
print("\n=== Table 2 f-hat: independent port vs paper ===")
for nm, ed, ex in EVAL:
    x = eval_point(ed, ex)
    dd = norm.cdf(theta[(0, POST)] @ x) - norm.cdf(th0 @ x)
    print(f"  {nm:5s}{np.mean(dd**2)*1e3:8.3f}   paper {PAPER_F[nm]:6.2f}")
