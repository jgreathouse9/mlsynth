"""How much the node bounds prune, before any search is built.

For each panel, every treated set of size ``K`` is priced by the two-way closed
form (``gram.py``), which equals the design's value whenever its sign conditions
are slack. The include/exclude tree over the units, in a fixed order, is then
walked with the incumbent already at the best price, the best case for any
search: a node is pruned when its bound reaches the incumbent. Reported per
panel and bound:

- nodes bounded, and the sets left to price (those under no pruned node);
- per depth, the median of the node bound over its subtree's minimum price.

From the repository root::

    python -m benchmarks.studies.syndes_global_bnb.quality
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from itertools import combinations
from math import comb

import numpy as np

from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem, signs_from_subsets

from .bounds import node_lower, spectral_upper, trs_upper

BOUNDS = {
    "spectral": lambda R, I, O, F, K: spectral_upper(R, I, O, F, K),
    "trs": lambda R, I, O, F, K: trs_upper(R, I, O, F, K)[0],
    "min": lambda R, I, O, F, K: min(trs_upper(R, I, O, F, K)[0],
                                      spectral_upper(R, I, O, F, K)),
}


def factor(N, T=30, rank=4, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def common_shock(N, T=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1)) + rng.standard_normal((T, N))


def geolift(N=40):
    import pandas as pd
    df = pd.read_csv("basedata/geolift_market_data.csv")
    Y = df.pivot(index="date", columns="location", values="Y").to_numpy(float)
    return Y[:90, :N]


def price_all(eng, N, K):
    sets = np.array(list(combinations(range(N), K)), dtype=np.int64)
    lower = np.empty(len(sets))
    exact = np.empty(len(sets), dtype=bool)
    for a in range(0, len(sets), 20_000):
        out = eng.bound(signs_from_subsets(N, sets[a:a + 20_000]))
        lower[a:a + 20_000] = out.lower
        exact[a:a + 20_000] = out.exact
    return sets, lower, exact


def walk(eng, sets, price, K, order, bound):
    """The tree with the incumbent at the best price; returns the counts and
    the bound-to-subtree-minimum ratios by depth."""
    N = eng.R.shape[0]
    member = np.zeros((len(sets), N), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    best = float(price.min())
    ratios = {}
    counts = {"bounded": 0, "pruned": 0, "left": 0}

    def visit(depth, I, O, mask):
        n_sets = int(mask.sum())
        if n_sets == 0:
            return
        if depth == N or len(I) == K:
            counts["left"] += n_sets
            return
        F = [u for u in order[depth:]]
        lb = node_lower(eng.alpha, bound(eng.R, I, O, F, K))
        counts["bounded"] += 1
        ratios.setdefault(depth, []).append(lb / float(price[mask].min()))
        if lb >= best * (1 - 1e-12):
            counts["pruned"] += n_sets
            return
        u = order[depth]
        visit(depth + 1, I + [u], O, mask & member[:, u])
        visit(depth + 1, I, O + [u], mask & ~member[:, u])

    visit(0, [], [], np.ones(len(sets), dtype=bool))
    return counts, {d: float(np.median(v)) for d, v in sorted(ratios.items())}


RHOS = (0.98, 0.99, 0.995, 0.999, 1.0)


def ceiling(sets, price, N, K, order):
    """Sets left to price, per ``rho``, for a bound that reaches ``rho`` of every
    subtree's minimum price, with the incumbent at the best price."""
    member = np.zeros((len(sets), N), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    best = float(price.min())
    left = {}
    for rho in RHOS:
        total = 0
        stack = [(0, 0, np.ones(len(sets), dtype=bool))]
        while stack:
            depth, k, mask = stack.pop()
            if not mask.any():
                continue
            if depth == N or k == K:
                total += int(mask.sum())
                continue
            if rho * float(price[mask].min()) >= best * (1 - 1e-12) and rho < 1.0:
                continue
            if rho == 1.0 and float(price[mask].min()) > best * (1 + 1e-12):
                continue
            u = order[depth]
            stack.append((depth + 1, k + 1, mask & member[:, u]))
            stack.append((depth + 1, k, mask & ~member[:, u]))
        left[rho] = total
    return left


def run(name, Y, K, lam=None):
    N = Y.shape[1]
    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))
    t = time.perf_counter()
    sets, price, exact = price_all(eng, N, K)
    t_price = time.perf_counter() - t
    star = int(np.argmin(price))
    print(f"{name}: N={N} K={K} sets={len(sets):,} price {t_price:.2f}s "
          f"({1e6 * t_price / len(sets):.2f} us/set); best {price[star]:.6g} "
          f"exact={bool(exact[star])}; root Rayleigh/best "
          f"{eng.global_lower_bound(K) / price[star]:.4f}")
    orders = {"index": list(range(N))}
    _, x = trs_upper(eng.R, [], [], list(range(N)), K)
    orders["relaxed"] = [int(u) for u in np.argsort(-x)]
    for oname, order in orders.items():
        left = ceiling(sets, price, N, K, order)
        print(f"  order={oname:7s} ceiling: sets left at rho " + " ".join(
            f"{rho}:{v:,d} ({v / len(sets):.1%})" for rho, v in left.items()))
        if CEILING_ONLY:
            continue
        for bname, bound in BOUNDS.items():
            t = time.perf_counter()
            counts, ratios = walk(eng, sets, price, K, order, bound)
            secs = time.perf_counter() - t
            shown = " ".join(f"{d}:{r:.3f}" for d, r in ratios.items() if d <= 8)
            print(f"  order={oname:7s} bound={bname:8s} bounded={counts['bounded']:6d} "
                  f"left={counts['left']:8,d} ({counts['left'] / len(sets):.1%}) "
                  f"{secs:6.1f}s  bound/submin by depth: {shown}")
    sys.stdout.flush()


PANELS = {
    "factor": [(f"factor N={N} K={K} seed={s}", lambda N=N, s=s: factor(N, seed=s), K)
               for N, K in [(24, 4), (30, 5), (40, 4)] for s in range(2)],
    "noise": [(f"noise N={N} K={K} seed={s}", lambda N=N, s=s: common_shock(N, seed=s), K)
              for N, K in [(24, 4), (30, 5)] for s in range(2)],
    "geolift": [("geolift N=40 K=4", lambda: geolift(40), 4)],
}


CEILING_ONLY = False

if __name__ == "__main__":
    args = sys.argv[1:]
    if "ceiling" in args:
        CEILING_ONLY = True
        args.remove("ceiling")
    groups = args or list(PANELS)
    for g in groups:
        for name, make, K in PANELS[g]:
            run(name, make(), K)
