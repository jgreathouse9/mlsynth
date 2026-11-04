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


def panel_ar(seed: int, n: int = 40, p: int = 6, rho: float = 0.6):
    """A panel whose treatment aggregate carries AR(1) noise, so that the
    Durbin-Watson and A/A gates fail for some designs and pass for others."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n).cumsum()
    X = f[:, None] * rng.uniform(0.5, 1.5, p) + rng.normal(size=(n, p)) * 3
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal()
    k = int(rng.integers(1, p))
    y = X[:, :k].sum(axis=1) * rng.uniform(0.3, 1.5) + e * 4
    return y - y.mean(), X - X.mean(axis=0)


def gated_optimum(y, X, n_test: int):
    """The lowest-RSS positive-slope control group that passes the reference key's
    four gates, by enumeration, and the gates of the RSS-best design; the group is
    None when no design passes."""
    from mlsynth.utils.tbr_helpers.design.objective import score_split

    rows = []
    for r in range(1, X.shape[1] + 1):
        for C in combinations(range(X.shape[1]), r):
            x = X[:, list(C)].sum(axis=1)
            b = (y @ x) / (x @ x)
            if b > 0:
                rows.append((float(((y - b * x) ** 2).sum()), set(C)))
    rows.sort(key=lambda t: t[0])
    for rss, C in rows:
        d = score_split(y, X[:, sorted(C)].sum(axis=1), objective="reference", n_test=n_test).detail
        if d["corr_test"] and d["aa_test"] and d["bb_test"] and d["dw_test"]:
            return rss, C
    return None, None


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


def treatment_optimum(M, k: int, n_test: int):
    """The reference key's best design with ``k`` treatment geos, by enumeration of
    every treatment group and control group, among designs that pass the four gates:
    (key, treatment, controls), or (None, None, None) when no design passes."""
    from mlsynth.utils.tbr_helpers.design.objective import score_split

    p = M.shape[1]
    best = (None, None, None)
    for T in combinations(range(p), k):
        y = M[:, list(T)].sum(axis=1)
        rest = [j for j in range(p) if j not in T]
        for r in range(1, len(rest) + 1):
            for C in combinations(rest, r):
                s = score_split(y, M[:, list(C)].sum(axis=1), objective="reference", n_test=n_test)
                if all(s.key[:4]) and (best[0] is None or s.key > best[0]):
                    best = (s.key, set(T), set(C))
    return best
