"""Count the causes: correct each mechanism independently and see what moves.

cvxpy writes ``sum_squares(r) <= x`` as the Lorentz cone

    || (2r, 1 - x) ||  <=  1 + x,

the standard identity for a rotated cone. Its right side ``1 + x`` and one of
its left components ``1 - x`` are affine in the same variable. Two SCIP
mechanisms then act on that in sequence:

aggregation
    Presolve reads the two defining equalities and substitutes one auxiliary
    for the other, ``s_rhs = 2 - s_lhs``.
expansion
    The expression simplifier expands a square of a sum
    (``expr/pow/expandmaxexponent``, default 2, ``expr_pow.c`` line 1537), so
    ``s_lhs^2 - (2 - s_lhs)^2`` becomes ``-4 + 4 s_lhs`` and the only negative
    square in the row is gone.

Each is switched off on its own, on the written MAREX model, and the presolved
row is read back. If the handler participates when either one is removed, both
are necessary and the failure is their conjunction.

    python causes.py results/causes.csv
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
from pyscipopt import Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments import handler_rows, negative_squares, presolved_rows

HERE = os.path.dirname(os.path.abspath(__file__))
CIP = os.path.join(HERE, "results", "marex.cip")

CONFIGS = {
    "baseline":       {},
    "no_aggregation": {"presolving/donotaggr": True},
    "no_expansion":   {"expr/pow/expandmaxexponent": 1},
    "neither":        {"presolving/donotaggr": True,
                       "expr/pow/expandmaxexponent": 1},
}

def presolve(params: dict) -> list[str]:
    model = Model()
    model.hideOutput()
    model.readProblem(CIP)
    for key, val in params.items():
        model.setParam(key, val)
    model.presolve()
    return presolved_rows(model)


def solve(params: dict) -> dict:
    model = Model()
    model.hideOutput()
    model.readProblem(CIP)
    for key, val in params.items():
        model.setParam(key, val)
    t0 = time.perf_counter()
    model.optimize()
    secs = time.perf_counter() - t0
    rows = handler_rows(model)
    return dict(objective=round(model.getObjVal(), 6),
                status=model.getStatus(), nodes=model.getNNodes(),
                lp_iters=model.getNLPIterations(), secs=round(secs, 3),
                soc=rows["soc"][1] if rows["soc"] else None,
                default=rows["default"][1] if rows["default"] else None,
                nonlinear_cuts=rows["nonlinear_cuts"])


def main(out: str) -> None:
    records = []
    for name, params in CONFIGS.items():
        rows = presolve(params)
        rec = dict(config=name, **{k.split("/")[-1]: v for k, v in params.items()})
        rec["neg_squares_after_presolve"] = sum(negative_squares(r) for r in rows)
        rec["first_row_head"] = rows[0][:90] if rows else ""
        rec.update(solve(params))
        records.append(rec)
    frame = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(frame.drop(columns=["first_row_head"]).to_string(index=False))
        print()
        for r in records:
            print(f"{r['config']:15s} {r['first_row_head']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/causes.csv")
