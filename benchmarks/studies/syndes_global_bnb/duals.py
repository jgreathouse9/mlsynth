"""The regularised dual against the interior-point lift, node by node.

``ratios``: at random nodes of a panel small enough to price every set, each
bound over the subtree's minimum price, and its seconds::

    python -m benchmarks.studies.syndes_global_bnb.duals ratios factor 24 4

``nodes``: at the root and random nodes of a panel too large for that, each
bound over the lift's, its seconds and evaluations, and dual fixing: with the
incumbent at the panel's optimum, the free units whose inherited child bound
(the parent's multipliers read on the child, one eigendecomposition) already
prunes one side::

    python -m benchmarks.studies.syndes_global_bnb.duals nodes geolift 40 10 237545.0339
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
from .dual import dual_upper, restrict, theta
from .quality import price_all
from .run import PANELS

DEPTHS = (0, 2, 4, 6, 3, 5, 7, 1)


def _timed(f, *a, **k):
    t = time.perf_counter()
    out = f(*a, **k)
    return out, time.perf_counter() - t


def _node(rng, N, K, d):
    units = rng.permutation(N)
    I, O = [], []
    for u in units[:d]:
        (I if (rng.random() < K / N and len(I) < K) else O).append(int(u))
    return I, O, [int(u) for u in units[d:]]


def ratios(name, N, K):
    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(PANELS[name](N)))
    sets, price, _ = price_all(eng, N, K)
    member = np.zeros((len(sets), N), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    rng = np.random.default_rng(0)
    for d in DEPTHS:
        I, O, F = _node(rng, N, K, d)
        mask = np.ones(len(sets), dtype=bool)
        for u in I:
            mask &= member[:, u]
        for u in O:
            mask &= ~member[:, u]
        sub = float(price[mask].min())
        ratio = lambda U: node_lower(eng.alpha, U) / sub
        lift, t_lift = _timed(lift_upper, eng.R, I, O, F, K)
        prod, t_prod = _timed(dual_upper, eng.R, I, O, F, K, cuts="products")
        tri, t_tri = _timed(dual_upper, eng.R, I, O, F, K, cuts="triangles")
        print(f"{name} I={len(I)} O={len(O)}  trs {ratio(trs_upper(eng.R, I, O, F, K)[0]):.4f}"
              f"  lift {ratio(lift):.4f} ({t_lift:.2f}s)  products {ratio(prod):.4f} ({t_prod:.2f}s)"
              f"  triangles {ratio(tri):.4f} ({t_tri:.2f}s)", flush=True)


def nodes(name, N, K, optimum):
    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(PANELS[name](N)))
    goal = 4.0 * eng.alpha / optimum          # a node prunes when its U is at or under this
    rng = np.random.default_rng(0)
    for d in (0, 2, 4, 6, 8):
        I, O, F = _node(rng, N, K, d)
        lift, t_lift = _timed(lift_upper, eng.R, I, O, F, K)
        line = f"{name} n={len(F)} I={len(I)}  lift {t_lift:.2f}s  trs/lift {trs_upper(eng.R, I, O, F, K)[0] / lift:.4f}"
        for cuts in ("products", "triangles"):
            (U, info), t = _timed(dual_upper, eng.R, I, O, F, K, cuts=cuts, info=True)
            line += (f"  {cuts} {U / lift:.5f} ({t:.2f}s, {info.evaluations} ev, "
                     f"{len(info.state[1])} cuts)")
        # dual fixing from the converged triangle dual
        c, b, A = split(eng.R, I, O, F)
        unit = max(abs(c), float(np.abs(b).max(initial=0.0)), float(np.abs(A).max()))
        t = time.perf_counter()
        fixed = both = 0
        for u in F:
            cF = [f for f in F if f != u]
            pruned = []
            for cI, cO in ((I + [u], O), (I, O + [u])):
                r = K - len(cI)
                if r < 0 or r > len(cF):
                    pruned.append(True)
                    continue
                y, cc, cv = restrict(info.state, F, cF)
                c2, b2, A2 = split(eng.R, cI, cO, cF)
                Uc = unit * theta(c2 / unit, b2 / unit, A2 / unit, 2 * r - len(cF), y, cc, cv, 1.0)[0]
                pruned.append(Uc <= goal)
            fixed += any(pruned)
            both += all(pruned)
        line += (f"  pruned {U <= goal}  fixing: {fixed} of {len(F)} units fix a side"
                 f" ({both} both) in {time.perf_counter() - t:.3f}s")
        print(line, flush=True)


if __name__ == "__main__":
    mode, name, N, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    if mode == "ratios":
        ratios(name, N, K)
    else:
        nodes(name, N, K, float(sys.argv[5]))
