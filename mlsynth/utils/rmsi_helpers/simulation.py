"""Block-adoption causal DGP with side information for RMSI examples/tests.

Mirrors the regime RMSI targets: an outcome matrix whose untreated potential
outcomes are driven by (nonlinear) functions of unit covariates ``X`` and time
covariates ``Z`` plus a residual low-rank part, with a block of treated units
adopting at a common period. Returns a tidy long panel (with the covariate
columns) ready for :class:`mlsynth.RMSI`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_rmsi_panel(
    n_units: int = 40,
    n_treated: int = 8,
    T0: int = 20,
    n_post: int = 11,
    d_unit: int = 2,
    d_time: int = 2,
    att: float = 5.0,
    noise: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a block, side-information, low-rank causal panel.

    Parameters
    ----------
    n_units, n_treated : int
        Total units and the number treated (treated units adopt at ``T0``).
    T0, n_post : int
        Pre- and post-treatment period counts.
    d_unit, d_time : int
        Numbers of unit (row) and time (column) covariates.
    att : float
        Constant additive treatment effect on treated post cells.
    noise : float
        Idiosyncratic-noise standard deviation.
    seed : int
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Long panel with columns ``unit``, ``time``, ``Y``, ``treated``,
        ``x0..`` (unit covariates), ``z0..`` (time covariates).
    """
    rng = np.random.default_rng(seed)
    N, T = n_units, T0 + n_post

    X = rng.uniform(-1.0, 1.0, size=(N, d_unit))
    Z = rng.uniform(-1.0, 1.0, size=(T, d_time))

    # Nonlinear interaction (M1), plus residual low-rank (M4).
    G1 = np.column_stack([np.sin(X[:, 0]), X[:, min(1, d_unit - 1)] ** 2])
    Q1 = np.column_stack([Z[:, 0], np.cos(Z[:, min(1, d_time - 1)])])
    W = rng.normal(size=(N, 2))
    V = rng.normal(size=(T, 2))
    M = G1 @ Q1.T + 0.7 * (W @ V.T)
    Y = M + rng.normal(scale=noise, size=(N, T))

    treated_units = set(range(n_units - n_treated, n_units))   # last units treated
    D = np.zeros((N, T), dtype=int)
    for i in treated_units:
        D[i, T0:] = 1
        Y[i, T0:] += att

    rows = []
    for i in range(N):
        name = f"t{i}" if i in treated_units else f"c{i}"
        for t in range(T):
            row = {"unit": name, "time": t, "Y": float(Y[i, t]),
                   "treated": int(D[i, t])}
            for k in range(d_unit):
                row[f"x{k}"] = float(X[i, k])
            for k in range(d_time):
                row[f"z{k}"] = float(Z[t, k])
            rows.append(row)
    return pd.DataFrame(rows)
