"""The weakly targeted design: the leaf changes, the bound survives.

MAREX's ``design="weakly_targeted"`` minimises

    ||A - B w||^2 + beta * ||B w - B v||^2.

The second term couples ``w`` and ``v``, so naming the treated set no longer
splits the objective into two independent fits: the per-candidate problem is
one quadratic program over the product of the two simplices. But the coupling
term is non-negative, so ``f(S) >= g(S)``, the same treated-side fit the
standard design prunes on, and the search survives with a different leaf.
How many leaves survive depends on ``beta``: the larger it is, the more of the
objective the bound cannot see.

This runs the search with a cvxpy leaf against SCIP's MIQP for four values of
``beta`` on one panel.

    python weakly_targeted.py results/weakly_targeted.csv
"""
from __future__ import annotations

import os
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import _prepared, fit_matrices
from search import all_subsets, fit_value

from mlsynth.utils.marex_helpers.formulation import (
    build_constraints,
    build_objective,
    init_cvxpy_variables,
    precompute_distances,
)

J, M, SEED = 12, 3, 3


def scip(beta: float) -> tuple[float, float]:
    Y_fit, Xbar, members, labels, Mmask, N, K = _prepared(J, SEED)
    D1, D2 = precompute_distances(Y_fit, Xbar, members)
    w, v, z = init_cvxpy_variables(N, K, boolean=True)
    cons = build_constraints(w, v, z, Mmask, members, labels, M, None, None,
                             None, None, True)
    obj = build_objective(Y_fit, Xbar, members, w, v, z, "weakly_targeted",
                          beta, 0.0, 0.0, 0.0, 0.0, 0.0, D1, D2)
    prob = cp.Problem(obj, cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.SCIP)
    return float(prob.value), time.perf_counter() - t0


def coupled_leaf(B, A, S, T, beta: float) -> float:
    wS = cp.Variable(len(S), nonneg=True)
    vT = cp.Variable(len(T), nonneg=True)
    st, sc = B[:, S] @ wS, B[:, T] @ vT
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A - st)
                                  + beta * cp.sum_squares(st - sc)),
                      [cp.sum(wS) == 1, cp.sum(vT) == 1])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value)


def main(out: str) -> None:
    B, A = fit_matrices(J, SEED)
    allj = np.arange(J)
    subs = all_subsets(J, M)
    g = np.array([fit_value(B, A, S) for S in subs])
    order = np.argsort(g, kind="stable")
    rows = []
    for beta in (1e-6, 0.01, 1.0, 5.0):
        ref, scip_secs = scip(beta)
        t0 = time.perf_counter()
        best, best_S, leaves = np.inf, None, 0
        for i in order:
            if g[i] >= best:
                break
            S = subs[i]
            total = coupled_leaf(B, A, S, np.setdiff1d(allj, S), beta)
            leaves += 1
            if total < best:
                best, best_S = total, S
        rows.append(dict(beta=beta, scip_objective=ref, search_objective=best,
                         match=abs(best - ref) <= 1e-4 * (1 + abs(ref)),
                         leaves=leaves, candidates=len(subs),
                         treated=tuple(int(j) for j in best_S),
                         search_secs=time.perf_counter() - t0,
                         scip_secs=scip_secs))
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/weakly_targeted.csv")
