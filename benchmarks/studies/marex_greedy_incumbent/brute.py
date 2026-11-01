"""The base design's objective on every treated set of a panel, by brute force.

``g`` is the treated fit on every nonempty set of markets, indexed by bitmask,
so ``f(S) = g[S] + g[~S]`` for every treated set at once. At 18 markets that is
262,143 simplex fits, about ten seconds on four processes, cached under
``results/`` (ignored by git).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def _fits(args):
    from mlsynth.utils.marex_helpers.exact import side_fit
    from mlsynth.utils.marex_helpers.formulation import carries_weight
    B, A, masks = args
    n = B.shape[1]
    g = np.empty(len(masks))
    weighted = np.empty(len(masks), dtype=bool)
    for r, mask in enumerate(masks):
        value, w = side_fit(B, A, [j for j in range(n) if mask >> j & 1])
        g[r], weighted[r] = value, carries_weight(w)
    return g, weighted


def treated_fits(name: str, B: np.ndarray, A: np.ndarray, workers: int = 4):
    """``g`` and whether its fit weights every market, on every bitmask."""
    path = RESULTS / f"brute_{name}.npz"
    if path.exists():
        z = np.load(path)
        if z["B"].shape == B.shape and np.array_equal(z["B"], B):
            return z["g"], z["weighted"]
    n = B.shape[1]
    blocks = np.array_split(np.arange(1, 1 << n), 64)
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_fits, [(B, A, b) for b in blocks]))
    g = np.full(1 << n, np.nan)
    weighted = np.zeros(1 << n, dtype=bool)
    g[1:] = np.concatenate([p[0] for p in parts])
    weighted[1:] = np.concatenate([p[1] for p in parts])
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(path, B=B, g=g, weighted=weighted)
    return g, weighted


def optima(g: np.ndarray, weighted: np.ndarray, n: int
           ) -> Dict[int, Tuple[float, Tuple[int, ...]]]:
    """The least objective and its treated set at every size ``1..n-1``."""
    full = (1 << n) - 1
    masks = np.arange(1, full)
    size = np.array([bin(int(m)).count("1") for m in masks])
    f = g[masks] + g[full ^ masks]
    out = {}
    for m in range(1, n):
        f_m = np.where((size == m) & weighted[masks], f, np.inf)
        k = int(np.argmin(f_m))
        out[m] = (float(f_m[k]), tuple(j for j in range(n) if masks[k] >> j & 1))
    return out


def sets_of_size(n: int, m: int) -> np.ndarray:
    """Every treated set of size ``m``, one row each, in increasing order."""
    import itertools
    return np.array(list(itertools.combinations(range(n), m)), dtype=np.int64)


if __name__ == "__main__":
    from .panels import PANELS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for name, make in PANELS.items():
        B, A = make()
        g, weighted = treated_fits(name, B, A)
        best = optima(g, weighted, B.shape[1])
        print(name, {m: round(v, 6) for m, (v, _) in best.items()})
