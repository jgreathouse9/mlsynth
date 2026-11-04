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
