"""The smallest model that reproduces the failure, and its one-feature twin.

``build(n, tied=True)`` writes ``sum_i r_i^2 <= x`` the way cvxpy does, as the
Lorentz cone

    sum_i s_i^2 + u^2 <= tau^2,   s_i = 2 r_i,   u = 1 - x,   tau = 1 + x,

the identity for a rotated cone. ``u`` and ``tau`` are affine in the same
variable ``x``. Presolve aggregates one into the other and the simplifier
expands the square of the resulting sum, so ``u^2 - tau^2`` becomes ``-4x`` and
the row reaches the nonlinear handlers with no negative square left.

``build(n, tied=False)`` changes one thing: ``u`` is defined from a variable of
its own, ``u = 1 - y``. Nothing ties it to ``tau``, so there is nothing for
presolve to merge, the negative square survives, and the SOC handler takes the
row. No integers, no MAREX, no cvxpy -- the mechanism is the encoding and SCIP.

    python minimal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from pyscipopt import Model, quicksum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments import handler_rows, negative_squares, presolved_rows


def build(n: int = 6, k: int = 3, tied: bool = True, seed: int = 0,
          params: dict | None = None) -> Model:
    """``min x  s.t.  sum_i (a_i'w - b_i)^2 <= x``, ``w`` on the simplex."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, k))
    b = rng.normal(size=n)
    model = Model()
    model.hideOutput()
    for key, val in (params or {}).items():
        model.setParam(key, val)
    w = [model.addVar(lb=0, ub=1, name=f"w{j}") for j in range(k)]
    model.addCons(quicksum(w) == 1)
    x = model.addVar(lb=None, obj=1, name="x")
    tau = model.addVar(lb=0, name="tau")
    u = model.addVar(lb=None, name="u")
    s = [model.addVar(lb=None, name=f"s{i}") for i in range(n)]
    model.addCons(tau - x == 1)
    if tied:
        model.addCons(u + x == 1)
    else:
        y = model.addVar(lb=None, name="y")
        model.addCons(u + y == 1)
    for i in range(n):
        model.addCons(s[i] == 2 * (quicksum(a[i, j] * w[j] for j in range(k))
                                   - b[i]))
    model.addCons(quicksum(v * v for v in s) + u * u - tau * tau <= 0)
    return model


def observe(model: Model) -> dict:
    """Negative squares after presolve, then the handlers' participations."""
    model.presolve()
    rows = presolved_rows(model)
    neg = sum(negative_squares(r) for r in rows)
    model.optimize()
    stats = handler_rows(model)
    return dict(neg_squares=neg, soc=stats["soc"][1] if stats["soc"] else 0,
                status=model.getStatus(), objective=model.getObjVal(),
                row=rows[0] if rows else "")


if __name__ == "__main__":
    from causes import CONFIGS
    for tied in (True, False):
        for name, params in CONFIGS.items():
            obs = observe(build(tied=tied, params=params))
            print(f"tied={tied!s:5s} {name:15s} neg_squares={obs['neg_squares']} "
                  f"soc={obs['soc']}  {obs['row'][:70]}")
