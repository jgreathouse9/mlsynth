"""Which simplex solver reaches the optimum on real panels, and is its point feasible?

minnorm.simplex_optimum_is_unique reports True on Basque, West Germany and
Proposition 99 -- including the two with more donors than pre-periods -- so
there is one minimiser and a solver returning different weights has not found
it. This prints the objective at each returned point, the excess over the best,
and two feasibility readings the objective alone hides: whether the weights sum
to one, and whether the excluded donors are at exactly zero.

That second reading is the one that matters for reporting. A synthetic control
names a donor pool, and an interior-point solver approaches the boundary
without reaching it, so a donor that should carry nothing carries 1e-8 instead.
Whether it appears in the pool then depends on a threshold, which is how this
library ended up printing predictors_kept = 6 in one place and 5 in another.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cvxpy as cp

from mlsynth.utils.datautils import dataprep
from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.solvers.minnorm import simplex_optimum_is_unique
from mlsynth.utils.solvers.simplex import simplex_lstsq
from mlsynth.utils.sparse_sc_helpers.inner import solve_w

def cvxpy_solve(B, a):
    w = cp.Variable(B.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(a - B @ w)), [cp.sum(w) == 1]).solve()
    return np.asarray(w.value).ravel()

SPECS = []
b = pd.read_csv("basedata/basque_data.csv")
b["treat"] = ((b.regionname == "Basque Country (Pais Vasco)") & (b.year >= 1970)).astype(int)
SPECS.append(("Basque", b, "regionname", "year", "gdpcap", "treat"))
g = pd.read_csv("basedata/germany_augmented.csv")
g["Reunification"] = ((g.country == "West Germany") & (g.year >= 1990)).astype(int)
SPECS.append(("West Germany", g, "country", "year", "gdp", "Reunification"))
p = pd.read_csv("basedata/augmented_cali_long.csv")
p["treated"] = ((p.state == "California") & (p.year >= 1989)).astype(int)
SPECS.append(("Prop 99", p, "state", "year", "cigsale", "treated"))

for name, df, uid, tcol, out, tr in SPECS:
    prep = dataprep(df, unit_id_column_name=uid, time_period_column_name=tcol,
                    outcome_column_name=out, treatment_indicator_column_name=tr)
    T0 = int(prep["pre_periods"])
    B = np.asarray(prep["donor_matrix"], float)[:T0]
    a = np.asarray(prep["y"], float).ravel()[:T0]
    sols = {
        "active-set": np.asarray(solve_simplex_qp(B, a)).ravel(),
        "FISTA":      np.asarray(simplex_lstsq(B, a)).ravel(),
        "cvxpy":      cvxpy_solve(B, a),
        "Clarabel":   np.asarray(solve_w(np.ones(T0), a, B)).ravel(),
    }
    objs = {k: float(np.sum((a - B @ w) ** 2)) for k, w in sols.items()}
    best = min(objs.values())
    uniq = simplex_optimum_is_unique(B, a, sols["active-set"])
    print(f"\n{name}  (J={B.shape[1]}, T0={T0}, unique minimiser: {uniq})")
    print(f"  {'solver':<11} {'objective':>15} {'excess':>12} {'sum w':>9} "
          f"{'min w':>11} {'max|dw| vs best':>16}")
    ref = min(objs, key=objs.get)
    for k, w in sols.items():
        print(f"  {k:<11} {objs[k]:>15.9g} {objs[k]-best:>12.3e} "
              f"{w.sum():>9.6f} {w.min():>11.3e} "
              f"{np.max(np.abs(w - sols[ref])):>16.2e}")
    print(f"  lowest objective: {ref}")
