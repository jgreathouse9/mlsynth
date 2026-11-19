"""Enumeration, the truth the study's bounds are checked against, on panels small enough for it."""
from itertools import combinations

import numpy as np


def panel(seed: int, n: int = 30, p: int = 6, scale: float = 40.0, noise: float = 4.0):
    """A centred treatment aggregate on ``p`` centred candidate controls."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * scale
    k = int(rng.integers(1, p)) if p > 1 else 1
    y = X[:, :k].sum(axis=1) * rng.uniform(0.3, 1.5) + noise * rng.normal(size=n)
    return y - y.mean(), X - X.mean(axis=0)


def piece_optimum(y, X, lo, hi, m, *, include=None, exclude=None):
    """min over |C| = m and b in [lo, hi] of ||y - b X u_C||^2, by enumeration."""
    best = np.inf
    for C in combinations(range(X.shape[1]), m):
        if include is not None and not set(include) <= set(C):
            continue
        if exclude is not None and set(exclude) & set(C):
            continue
        x = X[:, list(C)].sum(axis=1)
        b = np.clip((y @ x) / (x @ x), lo, hi)
        best = min(best, float(((y - b * x) ** 2).sum()))
    return best


def positive_optimum(y, X):
    """The best RSS over control groups whose best-fit slope is positive, and the group."""
    best, arg = np.inf, None
    for r in range(1, X.shape[1] + 1):
        for C in combinations(range(X.shape[1]), r):
            x = X[:, list(C)].sum(axis=1)
            b = (y @ x) / (x @ x)
            if b > 0:
                v = float(((y - b * x) ** 2).sum())
                if v < best:
                    best, arg = v, set(C)
    return best, arg


def geo_panel(seed: int, n: int = 40, p: int = 7, rho: float = 0.5):
    """Centred series of ``p`` geos (periods by geos) on one trend, with geo sizes
    that differ and AR(1) noise, so that treatment groups differ in fit and some
    designs fail the gates."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n).cumsum()
    size = rng.uniform(1.0, 6.0, p)
    e = np.zeros((n, p))
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(size=p)
    Y = (f[:, None] * rng.uniform(0.6, 1.4, p) + 2.0 * e) * size
    return Y - Y.mean(axis=0)


def wide_panel(seed: int, n: int = 10, p: int = 12):
    """geo_panel's series shifted to be positive, with more geos than periods: every
    entry of the Gram matrix M'M is positive and M'M is singular."""
    M = geo_panel(seed, n=n, p=p)
    return M - M.min() + 1.0


def raw_panel(seed: int, n: int = 24, p: int = 6, level: float = 40.0):
    """geo_panel's series on positive levels, uncentred: periods by geos, every
    geo's mean ``level`` times a size in [1, 3], the panel the relative key reads."""
    M = geo_panel(seed, n=n, p=p)
    rng = np.random.default_rng(seed + 7919)
    return M + level * rng.uniform(1.0, 3.0, p)


def mirror_panel(seed: int, n: int = 24, p: int = 6):
    """A raw panel in which geo 2 moves against geos 0 and 1, so the relative key's
    best design has a negative slope (checked by the tests that use it)."""
    rng = np.random.default_rng(seed)
    s = np.sin(np.arange(n) / 2.0) * 10
    M = 60 + rng.normal(size=(n, p)) * 4
    M[:, 0] = 60 + s + 0.2 * rng.normal(size=n)
    M[:, 1] = 60 + s + 0.2 * rng.normal(size=n)
    M[:, 2] = 90 - s + 0.1 * rng.normal(size=n)
    return M


def relative_optimum(M, k: int, n_test: int):
    """The relative key's best design with ``k`` treatment geos over the raw panel M,
    by enumeration of every treatment group and control group, scored by the
    library's score_split: (key, treatment, controls). Designs the library scores
    -inf (an exact fit) never win."""
    from mlsynth.utils.tbr_helpers.design.objective import score_split

    p = M.shape[1]
    best = (-np.inf, None, None)
    for T in combinations(range(p), k):
        y = M[:, list(T)].sum(axis=1)
        rest = [j for j in range(p) if j not in T]
        for r in range(1, len(rest) + 1):
            for C in combinations(rest, r):
                v = score_split(y, M[:, list(C)].sum(axis=1), objective="relative", n_test=n_test).key[0]
                if v > best[0]:
                    best = (v, set(T), set(C))
    return best
