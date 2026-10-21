"""Does SCIP recognise MAREX's objective as a second-order cone?

cvxpy does not hand SCIP a convex QP. Its SCIP interface rewrites each
``sum_squares`` into auxiliary variables tied by linear equalities plus one
quadratic row,

    s_1^2 + ... + s_n^2 - t^2 <= 0,   t >= 0,

which is the Lorentz cone written as an indefinite quadratic. SCIP has a
dedicated handler for that shape, ``nlhdlr_soc``, carrying the highest detect
priority of any nonlinear handler (100, against 50 for ``convex`` and 0 for
``default``). It disaggregates an n-term cone into n small cones plus one
linear row.

On MAREX's program it takes nothing, and every row goes to ``default``, which
builds one auxiliary per square, cuts each from below by tangents, and sums
them in one linear row. Why, is ``causes.py`` and ``minimal.py``: cvxpy's
encoding ties the cone's right side to one of its left components, and
presolve plus the simplifier cancel the pair. What it costs is ``damage.py``
and ``timing.py``: the cone's disaggregation is not a stronger relaxation -- the
root bound is no higher -- but it is about a third cheaper in LP iterations
and cuts, and about 14% faster on the geometric mean, unevenly.

    python detect.py results/detect.csv
"""
from __future__ import annotations

import os
import sys

import cvxpy as cp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments import handler_rows
from panels import program


def measure(J: int, m_eq: int) -> dict:
    prob = program(J=J, m_eq=m_eq)
    prob.solve(solver=cp.SCIP)
    model = prob.solver_stats.extra_stats["model"]
    rows = handler_rows(model)
    return dict(
        J=J, m_eq=m_eq, objective=round(float(prob.value), 6),
        nodes=model.getNNodes(), lp_iters=model.getNLPIterations(),
        root_bound=round(model.getDualboundRoot(), 6),
        soc_detects=rows["soc"][0] if rows["soc"] else None,
        convex_detects=rows["convex"][0] if rows["convex"] else None,
        default_detects=rows["default"][0] if rows["default"] else None,
        nonlinear_cuts=rows["nonlinear_cuts"],
        nonlinear_cuts_applied=rows["nonlinear_applied"],
    )


def main(out: str) -> None:
    rows = [measure(J, m) for J, m in ((12, 3), (16, 4), (20, 5))]
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/detect.csv")
