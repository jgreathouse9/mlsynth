"""The panels every script in this study measures on, and their exact leaves.

A panel is what MAREX's exact backend hands ``search_cluster`` for one cluster:
the markets as the columns of ``B`` over the fit window, and their mean as the
target ``A``. ``leaves`` evaluates the standard design's objective
``f(S) = g(S) + h(C(S))`` exactly on every treated set of size ``m`` and caches
the result, since every later measurement reads it.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent / "results"


def geolift(T0: int = 90):
    """GeoLift's 40 markets as one cluster over the first ``T0`` days."""
    d = pd.read_csv(ROOT / "basedata" / "geolift_market_data.csv")
    Y = d.pivot(index="date", columns="location", values="Y").sort_index()
    B = Y.to_numpy(float)[:T0]
    return B, B.mean(axis=1)


def walmart():
    """Walmart's 45 stores as MAREX's search sees them in the paper's design.

    The configuration is ``benchmarks/cases/marex_walmart.py``'s: fit weeks
    1..100, the four store covariates averaged over them, standardised. The
    columns of ``B`` and the target ``A`` are captured from the exact backend's
    call to ``search_cluster`` (so they are exactly what the library searches),
    and cached in ``results/walmart_panel.npz``.
    """
    path = RESULTS / "walmart_panel.npz"
    if path.exists():
        z = np.load(path)
        return z["B"], z["A"]
    import warnings

    import mlsynth.utils.marex_helpers.exact as ex
    from mlsynth import MAREX
    from benchmarks.cases.marex_walmart import _BASE, _CFG

    seen = []
    original = ex.search_cluster

    def capture(B, A, sizes, **kw):
        seen.append((np.array(B), np.array(A)))
        return original(B, A, sizes, **kw)

    ex.search_cluster = capture
    try:
        df = pd.read_csv(_BASE / "walmart_weekly_sales_covariates.csv")
        cfg = {k: v for k, v in _CFG.items() if k != "program_type"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MAREX({"df": df, **cfg, "backend": "exact"}).fit()
    finally:
        ex.search_cluster = original
    B, A = seen[0]
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(path, B=B, A=A)
    return B, A


def factor(J: int, T: int, seed: int, n_good: int = 4, hi: float = 3.0):
    """The test suite's factor panel: ``n_good`` markets track the mean loading
    with almost no noise, the rest are noisy (``mlsynth/tests``' ``_panel``)."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.4, 1.6, size=(2, J))
    load[:, :n_good] = load.mean(axis=1, keepdims=True)
    scale = np.full(J, hi)
    scale[:n_good] = 0.05
    B = f @ load + rng.normal(size=(T, J)) * scale + 20.0
    return B, B.mean(axis=1)


def noise(J: int, T: int, seed: int):
    """Markets with a common shock and no structure: no market is special."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, J)) + rng.normal(size=(T, 1)) * 2.0
    return B, B.mean(axis=1)


def county(n: int, seed: int = 0):
    """``n`` counties drawn from the BLS county unemployment panel (193
    counties, 36 months, all used as the fit window). More markets than
    periods, the regime where the exact backend's candidate limit binds."""
    d = pd.read_csv(ROOT / "basedata" / "spsydid_bls_county_subset.csv")
    d["t"] = d["year"] * 12 + d["month"]
    Y = d.pivot(index="t", columns="FIPS", values="Unemployment_Rate").sort_index()
    cols = np.random.default_rng(seed).choice(Y.shape[1], n, replace=False)
    B = Y.to_numpy(float)[:, np.sort(cols)]
    return B, B.mean(axis=1)


PANELS = {
    "geolift_m4": (lambda: geolift(), 4),
    "county_n40_m3": (lambda: county(40), 3),
    "county_n70_m3": (lambda: county(70), 3),
    "county_n100_m3": (lambda: county(100), 3),
    "factor_J24_m4": (lambda: factor(24, 40, seed=1), 4),
    "factor_J30_m4": (lambda: factor(30, 40, seed=2), 4),
    "noise_J24_m4": (lambda: noise(24, 40, seed=3), 4),
}


def _fits(args):
    from mlsynth.utils.marex_helpers.exact import side_fit
    B, A, block = args
    n = B.shape[1]
    out = np.empty((len(block), 2))
    for r, S in enumerate(block):
        C = [j for j in range(n) if j not in S]
        out[r, 0] = side_fit(B, A, list(S))[0]
        out[r, 1] = side_fit(B, A, C)[0]
    return out


def leaves(name: str, workers: int = 4):
    """Every treated set of the panel's size, with ``g``, ``h`` and ``f``."""
    path = RESULTS / f"leaves_{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["sets"], z["g"], z["h"]
    make, m = PANELS[name]
    B, A = make()
    sets = np.array(list(itertools.combinations(range(B.shape[1]), m)))
    blocks = np.array_split(sets, max(1, len(sets) // 2000))
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_fits, [(B, A, b) for b in blocks]))
    gh = np.vstack(parts)
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(path, sets=sets, g=gh[:, 0], h=gh[:, 1])
    return sets, gh[:, 0], gh[:, 1]


def best_by_size(name: str, k_max: int):
    """``min f`` over treated sets of each size ``1..k_max``."""
    from mlsynth.utils.marex_helpers.exact import side_fit
    make, _ = PANELS[name]
    B, A = make()
    n = B.shape[1]
    out = {}
    for k in range(1, k_max + 1):
        best = np.inf
        for S in itertools.combinations(range(n), k):
            C = [j for j in range(n) if j not in S]
            best = min(best, side_fit(B, A, list(S))[0] + side_fit(B, A, C)[0])
        out[k] = best
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for name in PANELS:
        s, g, h = leaves(name)
        print(name, len(s), "sets, best f", float((g + h).min()))
