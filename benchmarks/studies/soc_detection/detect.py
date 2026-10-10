"""Does SCIP recognise MAREX's objective as a second-order cone?

cvxpy does not hand SCIP a convex QP. Its SCIP interface rewrites each
``sum_squares`` into auxiliary variables tied by linear equalities plus one
quadratic row,

    s_1^2 + ... + s_n^2 - t^2 <= 0,   t >= 0,

which is the Lorentz cone written as an indefinite quadratic. SCIP has a
dedicated handler for that shape, ``nlhdlr_soc``, carrying the highest detect
priority of any nonlinear handler (100, against 50 for ``convex`` and 0 for
``default``). It disaggregates an n-term cone into n small cones plus one
linear row, a stronger and far more compact relaxation than linearizing each
square separately.

On MAREX's program it detects nothing. Every detection goes to ``default``,
which separates the same two constraints with thousands of gradient cuts.

    python detect.py results/detect.csv
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

import cvxpy as cp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import program


def handler_rows(model) -> dict:
    """``(detects, detectall)`` per nonlinear handler, from SCIP's statistics.

    ``writeStatistics`` is used instead of ``printStatistics`` because the
    latter writes from C and does not reach a Python-level stdout redirect.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
        path = tf.name
    try:
        model.writeStatistics(path)
        text = open(path).read()
    finally:
        os.unlink(path)
    out = {}
    for name in ("soc", "convex", "quadratic", "default"):
        m = re.search(rf"\n *{name} *: *(\d+) *(\d+)", text)
        out[name] = (int(m.group(1)), int(m.group(2))) if m else None
    # The Constraints table's `nonlinear` row: Number MaxNumber #Separate
    # #Propagate #EnfoLP #EnfoRelax #EnfoPS #Check #ResProp Cutoffs DomReds
    # Cuts Applied Conss Children. Split the fields instead of counting them
    # in a regex -- the Number column can carry a `+` suffix.
    out["nonlinear_cuts"] = out["nonlinear_applied"] = None
    for line in text.split("\n"):
        if line.strip().startswith("nonlinear ") and ":" in line:
            f = line.split(":", 1)[1].split()
            if len(f) >= 13 and f[0].rstrip("+").isdigit():
                out["nonlinear_cuts"] = int(f[11])
                out["nonlinear_applied"] = int(f[12])
                break
    return out


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
