"""The first bisection, kept as the record of a wrong turn.

``detect.py`` establishes that SCIP's SOC handler declines MAREX's cones. This
script was the first attempt to find out why: rebuild the cone directly in
PySCIPOpt and add MAREX's features one at a time, looking for the one that
makes the handler decline.

``bare``
    One cone, 7 terms, t free and carrying the objective.
``t_tied``
    t pinned to the objective variable by ``t - obj == 1``, the half of cvxpy's
    encoding that ties the cone's right side to the objective.
``wide``
    The cone widened to 20 terms, MAREX's fit-window length.
``varbound``
    The disjointness ``w_j <= z_j``, ``v_j <= 1 - z_j`` and the cardinality
    constraint added around it.
``two_cones``
    Both cones, sharing the w and v variables, as the standard design emits.

Every case detects, and the conclusion first drawn from that -- that the cause
was in cvxpy's coefficients and not in the shape -- was wrong. It is the
shape. cvxpy writes ``sum_squares(r) <= x`` as ``||(2r, 1 - x)|| <= 1 + x``,
so one of the cone's *left* components, ``1 - x``, is affine in the same
variable as its right side, ``1 + x``. Every case here encodes a standard cone
``sum s_i^2 <= t^2`` with no such left component, so none of them could reach
the cause: presolve merges the tied pair and the simplifier cancels their
squares, and a bisection that never includes the pair cannot implicate it. The
case named ``t_tied`` has the half of the tie that does no harm.

``minimal.py`` is the reproducer that does reach it, with a twin that differs
in that one feature, and ``causes.py`` counts the two mechanisms involved.
The cases below still run and still detect; they answer a question the
diagnosis did not need answered.

    python feature_bisect.py results/feature_bisect.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments import handler_rows


def soc_detects(model) -> int | None:
    rows = handler_rows(model)
    return rows["soc"][1] if rows["soc"] else None


def build(ncones: int, nterms: int, J: int, m_eq: int, varbound: bool,
          tie_t: bool, seed: int = 0) -> Model:
    model = Model()
    model.hideOutput()
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(nterms, J)) + 20.0
    target = Y.mean(axis=1)

    w = [model.addVar(lb=0, ub=1, vtype="C") for _ in range(J)]
    v = [model.addVar(lb=0, ub=1, vtype="C") for _ in range(J)]
    model.addCons(quicksum(w) == 1)
    model.addCons(quicksum(v) == 1)
    if varbound:
        z = [model.addVar(vtype="B") for _ in range(J)]
        model.addCons(quicksum(z) == m_eq)
        for j in range(J):
            model.addCons(w[j] <= z[j])
            model.addCons(v[j] <= 1 - z[j])

    for side in ([w, v][:ncones]):
        t = model.addVar(lb=0, vtype="C")
        if tie_t:
            obj = model.addVar(lb=0, obj=1, vtype="C")
            model.addCons(t - obj == 1)
        else:
            model.setObjective(t, "minimize")
        s = [model.addVar(lb=None, vtype="C") for _ in range(nterms)]
        for i in range(nterms):
            model.addCons(
                s[i] == quicksum(Y[i, j] * side[j] for j in range(J)) - target[i])
        model.addCons(quicksum(x * x for x in s) - t * t <= 0)
    return model


CASES = {
    "bare":      dict(ncones=1, nterms=6,  J=4,  m_eq=2, varbound=False, tie_t=False),
    "t_tied":    dict(ncones=1, nterms=6,  J=4,  m_eq=2, varbound=False, tie_t=True),
    "wide":      dict(ncones=1, nterms=19, J=12, m_eq=3, varbound=False, tie_t=True),
    "varbound":  dict(ncones=1, nterms=19, J=12, m_eq=3, varbound=True,  tie_t=True),
    "two_cones": dict(ncones=2, nterms=19, J=12, m_eq=3, varbound=True,  tie_t=True),
}


def main(out: str) -> None:
    rows = []
    for name, kw in CASES.items():
        model = build(**kw)
        model.optimize()
        rows.append(dict(case=name, **kw, soc_detects=soc_detects(model),
                         nodes=model.getNNodes()))
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/feature_bisect.csv")
