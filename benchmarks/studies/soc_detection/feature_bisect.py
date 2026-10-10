"""Which feature of MAREX's program makes the SOC handler decline it?

``detect.py`` establishes that it does decline. This builds the same algebraic
shape directly in PySCIPOpt, feature by feature, to find the one that matters.
Each case emits the same cone, ``sum_i s_i^2 - t^2 <= 0`` with ``t >= 0``, and
differs only in what surrounds it.

The cases, and what each one rules out if the cone is still detected:

``bare``
    One cone, 7 terms, t free and carrying the objective. The control.
``t_tied``
    t pinned to the objective variable by ``t - obj == 1``, as cvxpy does
    rather than letting t carry the objective itself. Rules out the
    aggregation of that equality turning ``t^2`` into an offset square, which
    the handler's own source says it does not detect.
``wide``
    The cone widened to 20 terms, MAREX's fit-window length. Rules out size.
``varbound``
    The disjointness ``w_j <= z_j``, ``v_j <= 1 - z_j`` and the cardinality
    constraint added around it. Rules out the integer structure.
``two_cones``
    Both cones, sharing the w and v variables, as the standard design emits.
    Rules out the interaction between them.

Every case detects. So the cause is not any of these, and the difference lives
in the coefficients cvxpy emits rather than in the shape. ``reread.py``
narrows it further: the written model reproduces the zero, so it is in the
model and not in how cvxpy builds it.

    python feature_bisect.py results/feature_bisect.csv
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum


def soc_detects(model) -> int | None:
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
        path = tf.name
    try:
        model.writeStatistics(path)
        text = open(path).read()
    finally:
        os.unlink(path)
    m = re.search(r"\n *soc *: *(\d+) *(\d+)", text)
    return int(m.group(1)) if m else None


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
            t.setAttr if False else None
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
