"""The bounds' table: each bound over its subtree's minimum price, at random nodes.

Eight nodes per panel, drawn as the study drew them (seed 0). Shor's relaxation
alone is solved here on the node's parts divided by their largest entry; the
lift is ``bounds.lift_upper``, certified from its multipliers.

From the repository root::

    python -m benchmarks.studies.syndes_global_bnb.ratios factor 24 4
    python -m benchmarks.studies.syndes_global_bnb.ratios noise 24 4
    python -m benchmarks.studies.syndes_global_bnb.ratios geolift 24 4
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np

from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

from .bounds import lift_upper, node_lower, split, trs_upper
from .quality import price_all
from .run import PANELS

DEPTHS = (0, 2, 4, 6, 3, 5, 7, 1)


def shor_upper(R, I, O, F, K) -> float:
    """Shor's relaxation without the products, solved on unit-scale parts. The
    solver's value, not certified: the table compares tightness only."""
    import cvxpy as cp
    c, b, A = split(R, I, O, F)
    n, s = len(F), 2 * (K - len(I)) - len(F)
    unit = max(abs(c), float(np.abs(b).max(initial=0.0)), float(np.abs(A).max()))
    X = cp.Variable((n + 1, n + 1), symmetric=True)
    x, XX = X[0, 1:], X[1:, 1:]
    cons = [X >> 0, X[0, 0] == 1, cp.diag(XX) == 1, cp.sum(x) == s,
            cp.sum(XX, axis=1) == s * x]
    prob = cp.Problem(cp.Maximize((c + 2 * b @ x + cp.trace(A @ XX)) / unit), cons)
    prob.solve(solver="CLARABEL")
    return unit * float(prob.value)


def main(argv):
    name, N, K = argv[0], int(argv[1]), int(argv[2])
    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(PANELS[name](N)))
    sets, price, _ = price_all(eng, N, K)
    member = np.zeros((len(sets), N), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    best = float(price.min())
    rng = np.random.default_rng(0)
    for d in DEPTHS:
        units = rng.permutation(N)
        I, O = [], []
        for u in units[:d]:
            (I if (rng.random() < K / N and len(I) < K) else O).append(int(u))
        F = [int(u) for u in units[d:]]
        mask = np.ones(len(sets), dtype=bool)
        for u in I:
            mask &= member[:, u]
        for u in O:
            mask &= ~member[:, u]
        sub = float(price[mask].min())
        ratio = lambda U: node_lower(eng.alpha, U) / sub
        t = time.perf_counter()
        lift = lift_upper(eng.R, I, O, F, K)
        lift_s = time.perf_counter() - t
        print(f"{name} I={len(I)} O={len(O)}  trs {ratio(trs_upper(eng.R, I, O, F, K)[0]):.4f}"
              f"  shor {ratio(shor_upper(eng.R, I, O, F, K)):.4f}"
              f"  lift {ratio(lift):.4f} ({lift_s:.2f}s)  submin/f* {sub / best:.4f}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
