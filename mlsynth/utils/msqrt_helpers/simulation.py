"""High-dimensional, multiple-treated factor DGP for MSQRT examples/tests.

Mirrors the regime MSQRT (Shen, Song & Abadie 2025) targets: many
disaggregated units, several treated simultaneously (block design), outcomes
driven by a low-rank latent-factor structure so that each treated unit is a
sparse combination of donors. Returns a tidy long panel ready for
:class:`mlsynth.MSQRT`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_msqrt_panel(
    n_treated: int = 5,
    n_control: int = 40,
    T0: int = 60,
    n_post: int = 20,
    n_factors: int = 3,
    att: float = 2.0,
    noise: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a block, multiple-treated, low-rank panel.

    Parameters
    ----------
    n_treated, n_control : int
        Counts of treated and never-treated units.
    T0, n_post : int
        Pre- and post-treatment period counts (common adoption at ``T0``).
    n_factors : int
        Rank of the latent-factor structure.
    att : float
        Constant additive treatment effect applied to treated post cells.
    noise : float
        Idiosyncratic-noise standard deviation.
    seed : int
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Long panel with columns ``unit``, ``time``, ``Y``, ``treated``.
    """
    rng = np.random.default_rng(seed)
    N = n_treated + n_control
    T = T0 + n_post

    # Low-rank structure: time factors (T x k) and unit loadings (N x k).
    F = rng.normal(size=(T, n_factors))
    Lam = rng.normal(size=(N, n_factors))
    unit_level = rng.normal(scale=2.0, size=N)
    time_level = np.cumsum(rng.normal(scale=0.2, size=T))

    Y = (Lam @ F.T) + unit_level[:, None] + time_level[None, :]
    Y += rng.normal(scale=noise, size=(N, T))

    treated_idx = set(range(n_treated))   # first n_treated units are treated
    D = np.zeros((N, T), dtype=int)
    for i in treated_idx:
        D[i, T0:] = 1
        Y[i, T0:] += att

    rows = []
    for i in range(N):
        name = f"t{i}" if i in treated_idx else f"c{i}"
        for t in range(T):
            rows.append((name, t, float(Y[i, t]), int(D[i, t])))
    return pd.DataFrame(rows, columns=["unit", "time", "Y", "treated"])
