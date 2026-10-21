"""Does the missing detection cost anything?

A handler that declines a row is a value that is wrong. Whether it did damage
is a separate, measured question, and one instance cannot answer it: SCIP's
node and iteration counts move by large factors under a reordering of rows and
columns that changes nothing about the problem, so a single pair of solves has
no power to attribute a difference to the handler.

So every MAREX program is solved under each configuration of ``causes.py`` and
under several permutations of its rows and columns, and the configurations are
compared pairwise on the same (program, permutation). The root dual bound is
recorded beside the effort counts, because it measures the relaxation each
configuration builds and is far less sensitive to the search path than nodes
or iterations are.

    python damage.py results/damage.csv
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import cvxpy as cp
import numpy as np
import pandas as pd
from pyscipopt import Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causes import CONFIGS
from instruments import handler_rows
from panels import program

PANELS = [(J, m, seed) for J, m in ((12, 3), (16, 4), (20, 5))
          for seed in (3, 11, 42)]
ORDERS = (None, 1, 2, 3, 4, 5)          # None: as written; k: permutation seed k


def write_cip(J: int, m: int, seed: int, directory: str) -> str:
    prob = program(J=J, m_eq=m, seed=seed)
    prob.solve(solver=cp.SCIP)
    path = os.path.join(directory, f"marex_J{J}_m{m}_s{seed}.cip")
    prob.solver_stats.extra_stats["model"].writeProblem(path, trans=False,
                                                        verbose=False)
    return path


def solve(path: str, params: dict, order) -> dict:
    model = Model()
    model.hideOutput()
    model.readProblem(path)
    for key, val in params.items():
        model.setParam(key, val)
    if order is not None:
        model.setParam("randomization/permutationseed", order)
        model.setParam("randomization/permuteconss", True)
        model.setParam("randomization/permutevars", True)
    t0 = time.perf_counter()
    model.optimize()
    secs = time.perf_counter() - t0
    rows = handler_rows(model)
    return dict(status=model.getStatus(), objective=model.getObjVal(),
                nodes=model.getNNodes(), lp_iters=model.getNLPIterations(),
                secs=secs, root_bound=model.getDualboundRoot(),
                soc=rows["soc"][1] if rows["soc"] else None,
                nonlinear_cuts=rows["nonlinear_cuts"])


def main(out: str) -> None:
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for J, m, seed in PANELS:
            path = write_cip(J, m, seed, tmp)
            for order in ORDERS:
                for name, params in CONFIGS.items():
                    rec = dict(J=J, m=m, seed=seed,
                               order=-1 if order is None else order,
                               config=name)
                    rec.update(solve(path, params, order))
                    records.append(rec)
                    print(f"J={J} seed={seed} order={rec['order']} {name:15s} "
                          f"nodes={rec['nodes']:5d} iters={rec['lp_iters']:7d} "
                          f"root={rec['root_bound']:.4f} soc={rec['soc']}",
                          flush=True)
    frame = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/damage.csv")
