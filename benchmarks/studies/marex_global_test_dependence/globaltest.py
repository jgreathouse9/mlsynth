"""MAREX's global test with three reference sets.

Abadie and Zhao's statistic is the mean absolute contrast over the post periods,
eq. (16). Their p-value, eq. (17), ranks it among the same statistic over every
``H``-subset of the blank and post periods, which is exact when the periods are
exchangeable (Theorem 2). The reference sets here, built with the
``marex_cumulative_conformal`` study's ``blocks``:

- ``subsets``: every ``H``-subset (or 999 drawn uniformly), eq. (17) as MAREX
  ships it;
- ``cyclic``: the ``N = n + H`` cyclic shifts of the blank-then-post series,
  Chernozhukov, Wuthrich and Zhu's (2021) moving-block permutations, valid for
  stationary weakly dependent residuals. Each reference block keeps its periods
  in time order, so serial dependence inside it is the post block's;
- ``cyclic`` with ``rho``: the same shifts weighted ``rho`` to the distance from
  the post window (Barber, Candes, Ramdas and Tibshirani 2023), fixed in advance;
- ``windows``: the post block against the ``n - H + 1`` windows of ``H``
  consecutive blank periods. No reference block holds a post period, so an
  effect moves only the observed statistic. The windows do not form a group with
  the post block, so the test is approximately valid under stationarity, not
  exact under exchangeability.

The observed block is row 0 of every reference and is counted, so a p-value is
never below ``1 / N`` (cyclic), ``1 / (1 + B)`` (drawn subsets) or
``1 / (n - H + 2)`` (windows).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.inversion import blocks, recency_weights

SCHEMES = ("subsets", "cyclic", "windows")
STATS = ("abs", "signed")


def global_pvalue(blank: np.ndarray, post: np.ndarray, scheme: str, *, stat: str = "abs",
                  rho: Optional[float] = None, max_subsets: int = 999,
                  rng: Optional[np.random.Generator] = None) -> float:
    """``stat="abs"`` is eq. (16), the mean absolute contrast over a block;
    ``stat="signed"`` is the absolute value of its mean, CWZ's Remark 1 statistic
    for an average effect."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}; got {scheme!r}")
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}; got {stat!r}")
    blank, post = np.asarray(blank, dtype=float), np.asarray(post, dtype=float)
    n, H = blank.size, post.size
    if scheme == "windows":
        if n < H:
            raise ValueError(f"blank windows need a blank at least as long as the post ({n} < {H})")
        idx = np.vstack([np.arange(n, n + H)[None, :],
                         np.arange(n - H + 1)[:, None] + np.arange(H)[None, :]])
    else:
        idx = blocks(n, H, scheme, max_subsets=max_subsets, rng=rng)
    v = np.concatenate([blank, post])
    S = np.abs(v)[idx].mean(axis=1) if stat == "abs" else np.abs(v[idx].mean(axis=1))
    w = np.ones(len(S)) if rho is None else recency_weights(idx, n, rho)
    tol = 1e-12 * (1.0 + S[0])
    return float(np.sum(w * (S >= S[0] - tol)) / np.sum(w))
