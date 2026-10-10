"""Write MAREX's program to a file, then solve the file. The zero reproduces.

``feature_bisect.py`` rules out the shape. This rules out the build path: the
program is written to a CIP file and handed back to a fresh SCIP with no cvxpy
and no mlsynth in the process. If detection were being suppressed by how cvxpy
adds the constraints -- the order, the interface, some residual state on the
model -- a re-read would detect. It does not, so the cause is in the model
itself, and the written file is a self-contained reproducer.

That file is what to send upstream. It needs nothing from this repository.

    python reread.py results/marex.cip
"""
from __future__ import annotations

import os
import sys

import cvxpy as cp
from pyscipopt import Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import handler_rows
from panels import program


def main(out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    prob = program(J=12, m_eq=3)
    prob.solve(solver=cp.SCIP)
    built = prob.solver_stats.extra_stats["model"]
    rows_built = handler_rows(built)
    built.writeProblem(out, trans=False)

    fresh = Model()
    fresh.hideOutput()
    fresh.readProblem(out)
    fresh.optimize()
    rows_fresh = handler_rows(fresh)

    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    for label, rows, model in (("as built ", rows_built, built),
                               ("re-read  ", rows_fresh, fresh)):
        print(f"{label} soc={rows['soc']}  default={rows['default']}  "
              f"nodes={model.getNNodes()}  obj={model.getObjVal():.6f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/marex.cip")
