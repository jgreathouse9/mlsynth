"""Sufficient statistics ``V`` and ``b`` of the least-squares contrast (EC.2).

With features ``X_0(t) = 1`` and ``X_k(t) = sum_{t_l < t, e_l = k} psi_k(t - t_l)``,
the per-node loss is ``1/2 theta' V theta - b_e' theta`` where

    V_{kk'} = 1/n sum_j int_0^{T_j} X_k X_k' dt,
    b_{e,k} = 1/n sum_j sum_{i: e_i = e} X_k(t_i).

The kernel is exponential, ``psi_k(t) = beta_k exp(-beta_k t)`` (integrates to
one, as Sec. 2.3 requires), with a decay per source type. ``V`` therefore does
not depend on the target node and is shared by all nodes. Between consecutive
events every ``X_k`` decays geometrically, so each integral has a closed form
and one sweep over event slots, vectorised across paths, computes everything
exactly in ``O(total events * p^2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .paths import Path, validate_paths


def as_decay(decay, p: int) -> np.ndarray:
    beta = np.broadcast_to(np.asarray(decay, dtype=float), (p,)).copy()
    if np.any(~np.isfinite(beta)) or np.any(beta <= 0):
        raise ValueError("kernel decay rates must be positive and finite")
    return beta


@dataclass(frozen=True)
class SufficientStats:
    """Sums (not means) over a set of paths, so folds can be added and subtracted."""

    G: np.ndarray  # (p+1, p+1): sum_j int X X' dt, X = (1, X_1..X_p)
    B: np.ndarray  # (p, p+1): row e = sum over type-e events of X(t_i)
    n: int

    @property
    def V(self) -> np.ndarray:
        return self.G / self.n

    def b(self, e: int) -> np.ndarray:
        return self.B[e] / self.n

    def __add__(self, other: "SufficientStats") -> "SufficientStats":
        return SufficientStats(self.G + other.G, self.B + other.B, self.n + other.n)

    def __sub__(self, other: "SufficientStats") -> "SufficientStats":
        return SufficientStats(self.G - other.G, self.B - other.B, self.n - other.n)


def sufficient_stats(paths: Sequence[Path], p: int, decay) -> SufficientStats:
    validate_paths(paths, p)
    beta = as_decay(decay, p)
    n = len(paths)
    M = max(len(path) for path in paths)
    # Slot k holds the k-th event; slot m_j (and any later slot) holds T_j, so
    # the final sweep integrates each path out to the end of its window.
    times = np.empty((n, M + 1))
    types = np.full((n, M + 1), -1, dtype=int)
    for j, path in enumerate(paths):
        m = len(path)
        times[j, :m] = path.times
        times[j, m:] = path.T
        types[j, :m] = path.types

    pair = beta[:, None] + beta[None, :]
    G = np.zeros((p + 1, p + 1))
    B = np.zeros((p, p + 1))
    x = np.zeros((n, p))  # X(t) just after the previous slot
    prev = np.zeros(n)
    for k in range(M + 1):
        dt = times[:, k] - prev
        decay_k = np.exp(-beta[None, :] * dt[:, None])
        # int_0^dt x_a x_b exp(-(beta_a + beta_b) s) ds
        fac = -np.expm1(-pair[None] * dt[:, None, None]) / pair
        G[1:, 1:] += np.einsum("na,nb,nab->ab", x, x, fac)
        G[0, 1:] += np.einsum("na,na->a", x, -np.expm1(-beta[None, :] * dt[:, None]) / beta)
        G[0, 0] += dt.sum()
        x = x * decay_k  # left limit X(t_k-)
        ty = types[:, k]
        has = ty >= 0
        if has.any():
            rows = np.column_stack([np.ones(has.sum()), x[has]])
            np.add.at(B, ty[has], rows)
            x[has, ty[has]] += beta[ty[has]]
        prev = times[:, k]
    G[1:, 0] = G[0, 1:]
    return SufficientStats(G, B, n)
