"""Spike: the signed-sum inversion with a linear trend in the contrast.

When the design leaves a trend in the contrast, ``e_t = a + theta t + eta_t``,
the errors are not exchangeable, every reference block sits earlier on the trend
than the post window, and the inversion fails (``blocks_weights.py``). At the
true ``theta`` the detrended errors ``a + eta_t`` are exchangeable again, a
constant ``a`` changing nothing, so the trend is a nuisance parameter to invert
over: the first edition of Vovk, Gammerman and Shafer's detrending
transformations, and the filter of Chernozhukov, Wuthrich and Zhu's Remark 2.

Two arms:

* ``exact`` -- Bonferroni. A permutation confidence set for ``theta`` from the
  blank periods alone, with statistic ``|sum_t (t - tbar) (g_t - theta t)|``
  (exact under exchangeable ``eta``), at level ``alpha / 2``; then the union over
  that set of the signed-sum bands at level ``alpha / 2`` on the detrended blank
  and post series. Misses only when one of the two misses.
* ``plugin`` -- the least-squares slope on the blank periods, the series
  detrended by it, and the signed-sum inversion at level ``alpha``. Valid only
  approximately, as the estimated slope is treated as known.

Every reference sum is linear in ``theta``, so both p-values are evaluated on a
grid of slopes in one pass. The exact arm's band is the hull of the bands at
seven slopes spread over the slope set, which can be a little narrower than the
union over the whole set; its coverage is exact on the grid.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.detrend [draws] [n] [law]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.candidate2 import _ar
from benchmarks.studies.marex_cumulative_conformal.inversion import (
    Reference, band, blocks, recency_weights)

H = 12
LAWS = ("gauss", "drift", "steep", "drift_ar5", "curved", "sorted")


def errors(law, N, rng):
    t = np.arange(N, dtype=float)
    if law == "gauss":
        return rng.standard_normal(N)
    if law == "drift":
        return rng.standard_normal(N) + 0.01 * t
    if law == "steep":
        return rng.standard_normal(N) + 0.05 * t
    if law == "drift_ar5":
        return _ar(0.5, N, rng) + 0.01 * t
    if law == "curved":
        return rng.standard_normal(N) + 0.0004 * t ** 2
    if law == "sorted":
        # a sorted uniform level, the rearrangement the design study used
        return rng.standard_normal(N) + np.sort(rng.uniform(0.0, 3.0, N))
    raise ValueError(law)


def slope_grid(blank, half_width=8.0, points=401):
    t = np.arange(blank.size, dtype=float)
    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, blank, rcond=None)
    resid = blank - X @ coef
    se = np.sqrt(resid @ resid / max(blank.size - 2, 1) / ((t - t.mean()) ** 2).sum())
    return coef[1], np.linspace(coef[1] - half_width * se, coef[1] + half_width * se, points)


def slope_pvalues(blank, thetas, scheme, R, rng):
    """p-value of each slope, from permuting the detrended blank errors."""
    n = blank.size
    t = np.arange(n, dtype=float)
    c = t - t.mean()
    if scheme == "cyclic":
        perm = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    else:
        perm = np.vstack([np.arange(n)[None, :],
                          rng.random((R, n)).argsort(axis=1)])
    # sum_t c_t (g - theta t)_{pi(t)} = A_pi - theta B_pi
    A = blank[perm] @ c
    B = t[perm] @ c
    S = np.abs(A[None, :] - thetas[:, None] * B[None, :])
    return np.mean(S >= S[:, :1] - 1e-9 * (1 + S[:, :1]), axis=1)


def sum_pvalues(blank, post_gaps, thetas, C, idx, w=None):
    """p-value of (C, theta) for each theta, signed sum on the detrended series."""
    n = blank.size
    v = np.concatenate([blank, post_gaps])
    t = np.arange(v.size, dtype=float)
    a0 = v[idx].sum(axis=1)                    # contrast sums
    T = t[idx].sum(axis=1)                     # trend sums
    r = (idx >= n).sum(axis=1) / post_gaps.size
    S = np.abs(a0[None, :] - thetas[:, None] * T[None, :] - r[None, :] * C)
    obs = S[:, :1]
    hit = S >= obs - 1e-9 * (1 + np.abs(a0).max())
    if w is None:
        return np.mean(hit, axis=1)
    return hit @ w / w.sum()


def band_at(blank, post_gaps, theta, idx, alpha, w=None):
    n = blank.size
    v = np.concatenate([blank, post_gaps]) - theta * np.arange(n + post_gaps.size)
    ref = Reference(a=v[idx].sum(axis=1), r=(idx >= n).sum(axis=1) / post_gaps.size,
                    G=float(v[n:].sum()), H=post_gaps.size, w=w)
    return band(ref, alpha)


def run(n, law, draws, seed, alphas=(0.05, 0.2), R=999):
    rng = np.random.default_rng(seed)
    noise = np.std([(_ar(0.5, n + H, rng) if law == "drift_ar5"
                     else rng.standard_normal(n + H))[n:].sum() for _ in range(20000)])
    cover, width, unb = {}, {}, {}
    arms = [("cyclic", "cyclic", None), ("subsets", "subsets", None)]
    for _ in range(draws):
        u = errors(law, n + H, rng)
        blank, post = u[:n], u[n:]
        C = float(H)
        gaps = post + 1.0
        theta_hat, grid = slope_grid(blank)
        for name, scheme, rho in arms:
            idx = blocks(n, H, scheme, max_subsets=R, rng=rng)
            w = None if rho is None else recency_weights(idx, n, rho)
            p_none = sum_pvalues(blank, gaps, np.array([0.0]), C, idx, w)[0]
            p_plug = sum_pvalues(blank, gaps, np.array([theta_hat]), C, idx, w)[0]
            p_slope = slope_pvalues(blank, grid, scheme, R, rng)
            p_joint = sum_pvalues(blank, gaps, grid, C, idx, w)
            for a in alphas:
                ok = p_slope > a / 2
                edge = ok[0] or ok[-1]
                covered = bool(np.any(ok & (p_joint > a / 2)))
                for arm, hit in ((f"{name}", p_none > a),
                                 (f"{name}+plugin", p_plug > a),
                                 (f"{name}+exact", covered)):
                    cover[(arm, a)] = cover.get((arm, a), 0) + hit
                if a == alphas[0]:
                    for arm, b in ((f"{name}", band_at(blank, gaps, 0.0, idx, a, w)),
                                   (f"{name}+plugin", band_at(blank, gaps, theta_hat, idx, a, w))):
                        _w = b.upper - b.lower
                        if np.isfinite(_w):
                            width.setdefault(arm, []).append(_w / noise)
                        else:
                            unb[arm] = unb.get(arm, 0) + 1
                    arm = f"{name}+exact"
                    if not ok.any() or edge:
                        unb[arm] = unb.get(arm, 0) + 1
                        continue
                    picks = np.unique(np.linspace(np.flatnonzero(ok)[0],
                                                  np.flatnonzero(ok)[-1], 7).round().astype(int))
                    bs = [band_at(blank, gaps, grid[i], idx, a / 2, w) for i in picks]
                    lo = min(b.lower for b in bs if not b.empty)
                    hi = max(b.upper for b in bs if not b.empty)
                    if np.isfinite(hi - lo):
                        width.setdefault(arm, []).append((hi - lo) / noise)
                    else:
                        unb[arm] = unb.get(arm, 0) + 1
    for a in alphas:
        cells = []
        for name, *_ in arms:
            for arm in (name, f"{name}+plugin", f"{name}+exact"):
                s = f"{arm}={cover[(arm, a)] / draws:.3f}"
                if a == alphas[0]:
                    wv = width.get(arm, [])
                    s += (f" w{np.median(wv):.2f}" if wv else " w-")
                    if unb.get(arm):
                        s += f" inf{unb[arm] / draws:.2f}"
                cells.append(s)
        print(f"n={n} H={H} {law} alpha={a}: " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    sizes = (int(sys.argv[2]),) if len(sys.argv) > 2 else (24, 48, 96)
    laws = (sys.argv[3],) if len(sys.argv) > 3 else LAWS
    for n in sizes:
        for law in laws:
            run(n, law, draws, seed=1000 * n + LAWS.index(law))
