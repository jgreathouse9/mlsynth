"""Staggered-adoption factor-model DGP for SSC examples and tests.

Reproduces the simulation design of Cao, Lu & Wu (2026, Section 3):

    y_{i,t} = tau_{i,t} d_{i,t} + lambda_i' f_t + alpha_i + xi_t + c + eps_{i,t}

with ``r`` AR(1) common factors ``f_t`` and time effect ``xi_t`` (each
``g_t = 0.5 g_{t-1} + N(0,1)``), unit factor loadings and fixed effects drawn
from ``U[-sqrt(3), sqrt(3)]``, intercept ``c``, and ``N(0,1)`` idiosyncratic
noise. Treatment is staggered; the dynamic effect grows with event time,
``tau_{i,t} = base + max(e_{i,t}, 0)`` where ``e_{i,t} = t - t_i``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ar1(T: int, rng: np.random.Generator, rho: float = 0.5) -> np.ndarray:
    g = np.zeros(T)
    for t in range(1, T):
        g[t] = rho * g[t - 1] + rng.normal()
    return g


def simulate_ssc_panel(
    n_units: int = 33,
    n_never: int = 3,
    T0: int = 42,
    S: int = 7,
    n_factors: int = 3,
    base_effect: float = 1.0,
    intercept: float = 5.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a staggered-adoption panel in the Cao-Lu-Wu regime.

    Parameters
    ----------
    n_units : int
        Total number of units.
    n_never : int
        Number of never-treated units (the rest adopt at staggered times within
        the post window).
    T0 : int
        Clean pre-treatment periods (before any adoption).
    S : int
        Post-treatment periods (the adoption window).
    n_factors : int
        Number of AR(1) common factors ``r``.
    base_effect : float
        Constant part of the dynamic effect; the full effect at event time
        ``e`` is ``base_effect + e``.
    intercept : float
        Common intercept ``c``.
    seed : int
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Long panel with columns ``unit``, ``time``, ``Y``, ``treated``.
    """
    rng = np.random.default_rng(seed)
    N, T = n_units, T0 + S
    rt3 = np.sqrt(3.0)

    F = np.column_stack([_ar1(T, rng) for _ in range(n_factors)])   # (T, r)
    Lam = rng.uniform(-rt3, rt3, size=(N, n_factors))               # (N, r)
    alpha = rng.uniform(-rt3, rt3, size=N)
    xi = _ar1(T, rng)
    Y0 = (Lam @ F.T) + alpha[:, None] + xi[None, :] + intercept \
        + rng.normal(size=(N, T))                                   # untreated

    # Staggered adoption: first n_never units never treated; the rest adopt at
    # times spread across the post window [T0, T0 + S - 1].
    n_treated = N - n_never
    adopt = np.full(N, -1, dtype=int)
    treated_units = np.arange(n_never, N)
    offsets = (np.arange(n_treated) * S) // max(n_treated, 1)        # 0..S-1
    for k, i in enumerate(treated_units):
        adopt[i] = T0 + int(offsets[k])

    D = np.zeros((N, T), dtype=int)
    Y = Y0.copy()
    for i in range(N):
        if adopt[i] >= 0:
            D[i, adopt[i]:] = 1
            for t in range(adopt[i], T):
                e = t - adopt[i]                                     # 0-based
                Y[i, t] += base_effect + e                          # dynamic effect

    rows = []
    for i in range(N):
        name = f"n{i}" if adopt[i] < 0 else f"u{i}"
        for t in range(T):
            rows.append((name, t, float(Y[i, t]), int(D[i, t])))
    return pd.DataFrame(rows, columns=["unit", "time", "Y", "treated"])
