"""A band for the total effect by inverting a constant-effect sharp null.

MAREX fits its weights on the fitting periods alone, so the synthetic contrast
``g_t = w'Y_t - v'Y_t`` on the blank and post periods needs no refit under a
null. Under ``H0: tau_t = C / H`` for every post period the errors on those
``N = n + H`` periods are known,

    e_t = g_t                 (blank)
    e_t = g_t - C / H         (post),

and Abadie and Zhao's Theorem 2 compares a post-block statistic with the same
statistic on rearranged blocks. With the signed sum ``|sum_{t in B} e_t|`` the
observed value is ``|G - C|``, ``G`` the post sum of the contrast, whatever the
path: the null spreads ``C`` evenly, a path spreads it some other way, and the
post sum cannot tell them apart. A reference block ``k`` holding ``n_k`` post
periods has sum ``a_k - (n_k / H) C``, with ``a_k`` its sum of contrasts.

So each reference block counts towards the p-value on the set of ``C`` where

    |a_k - r_k C| >= |G - C|,     r_k = n_k / H,

and squaring turns that into the sign of a product of two linear functions,
with roots ``(G - a_k) / (1 - r_k)`` and ``(G + a_k) / (1 + r_k)``. The p-value
is a step function of ``C`` with at most two steps per block, and the band
(every ``C`` with ``p > alpha``) is read off exactly at those breakpoints.

For ``r_k < 1`` the reference sum grows more slowly than ``|G - C|``, so every
block but the observed one stops counting far enough out: the band is bounded
once ``1 / K <= alpha`` for ``K`` reference blocks, with no condition on how many
horizon-length windows the blank period holds.

Schemes:

* ``cyclic`` -- the ``N`` cyclic shifts of the blank-then-post series, CWZ's
  ``Pi_->``; time order survives inside a block.
* ``subsets`` -- every ``H``-subset of the ``N`` periods when there are at most
  ``max_subsets`` of them, else the observed block plus ``max_subsets`` drawn
  uniformly; Abadie and Zhao's scheme.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Optional, Tuple

import numpy as np

SCHEMES = ("cyclic", "subsets")


@dataclass(frozen=True)
class Reference:
    """Reference blocks as sums of the contrast and shares of post periods.

    Row 0 is the observed post block, so a p-value is a mean over rows.
    """

    a: np.ndarray
    r: np.ndarray
    G: float
    H: int


def blocks(n: int, H: int, scheme: str, *, max_subsets: int = 999,
           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Index matrix, one reference block per row; row 0 is the post block."""
    N = n + H
    post = np.arange(n, N)
    if scheme == "cyclic":
        return (n + np.arange(N)[:, None] + np.arange(H)[None, :]) % N
    if scheme != "subsets":
        raise ValueError(f"scheme must be one of {SCHEMES}; got {scheme!r}")
    if comb(N, H) <= max_subsets + 1:
        rows = np.array(list(combinations(range(N), H)))
        is_post = np.all(rows == post[None, :], axis=1)
        return np.vstack([post[None, :], rows[~is_post]])
    rng = np.random.default_rng() if rng is None else rng
    drawn = np.sort(rng.random((max_subsets, N)).argpartition(H, axis=1)[:, :H], axis=1)
    return np.vstack([post[None, :], drawn])


def reference(blank: np.ndarray, gaps: np.ndarray, idx: np.ndarray) -> Reference:
    blank = np.asarray(blank, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    n, H = blank.size, gaps.size
    v = np.concatenate([blank, gaps])
    return Reference(a=v[idx].sum(axis=1), r=(idx >= n).sum(axis=1) / H,
                     G=float(gaps.sum()), H=H)


def pvalue(ref: Reference, C) -> np.ndarray:
    """p-value of ``H0: tau_t = C / H`` at each ``C``."""
    C = np.atleast_1d(np.asarray(C, dtype=float))
    obs = np.abs(ref.G - C)
    S = np.abs(ref.a[None, :] - ref.r[None, :] * C[:, None])
    # the tolerance absorbs rounding in sums of the same numbers taken in a
    # different order, which is what makes a block equal the observed one
    tol = 1e-9 * (1.0 + np.abs(ref.a).max() + np.abs(C)[:, None])
    return np.mean(S >= obs[:, None] - tol, axis=1)


def breakpoints(ref: Reference) -> np.ndarray:
    r, a, G = ref.r, ref.a, ref.G
    keep = r < 1.0
    lo = (G - a[keep]) / (1.0 - r[keep])
    hi = (G + a) / (1.0 + r)
    return np.unique(np.concatenate([lo, hi]))


@dataclass(frozen=True)
class Band:
    lower: float
    upper: float
    gaps: bool          # accepted points separated by rejected ones
    empty: bool
    n_reference: int


def band(ref: Reference, alpha: float) -> Band:
    """Every ``C`` with ``p(C) > alpha``, as its hull and whether it has holes."""
    b = breakpoints(ref)
    span = max(1.0, float(np.abs(b).max()) if b.size else 1.0)
    mids = (b[:-1] + b[1:]) / 2.0
    # the step function is constant between breakpoints, may jump at them, and
    # is constant beyond the outermost ones
    pts = np.concatenate([[b[0] - span], b, mids, [b[-1] + span]]) if b.size \
        else np.array([ref.G])
    order = np.argsort(pts, kind="stable")
    pts = pts[order]
    ok = pvalue(ref, pts) > alpha
    if not ok.any():
        return Band(np.nan, np.nan, False, True, ref.a.size)
    first, last = np.flatnonzero(ok)[[0, -1]]
    lower = -np.inf if first == 0 and b.size else float(pts[first])
    upper = np.inf if last == pts.size - 1 and b.size else float(pts[last])
    holes = bool((~ok[first:last + 1]).any())
    return Band(lower, upper, holes, False, ref.a.size)


def interval(blank, gaps, alpha: float, scheme: str = "cyclic", *,
             max_subsets: int = 999, rng=None) -> Tuple[Band, Reference]:
    idx = blocks(np.asarray(blank).size, np.asarray(gaps).size, scheme,
                 max_subsets=max_subsets, rng=rng)
    ref = reference(blank, gaps, idx)
    return band(ref, alpha), ref
