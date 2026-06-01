"""Data-generating processes for MSQRT examples, tests and replication.

Implements the regime of Shen, Song & Abadie (2025): many disaggregated donor
units following an AR(1) process, with several treated units adopting at the
same period (a block design), each treated unit a **sparse, convex** combination
of the donors. These are the building blocks of the paper's Monte-Carlo study
(see :mod:`mlsynth.utils.msqrt_helpers.replication`); :func:`simulate_msqrt_panel`
wraps them into a tidy long panel -- with an injected treatment effect -- ready
for :class:`mlsynth.MSQRT`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ar1_panel(n: int, T: int, rng: np.random.Generator, *, burn: int = 50) -> np.ndarray:
    """Simulate the paper's AR(1) outcome panel, time-major.

    Each unit follows ``Y_{i,t} = 0.1 * c_i + 0.9 * Y_{i,t-1} + Z_{i,t}`` with
    ``c_i in {1, ..., 10}`` (cycled across units) and ``Z_{i,t} ~ N(0, 1)``, so
    the stationary mean of unit ``i`` is ``c_i`` and the series is persistent.

    Parameters
    ----------
    n : int
        Number of units.
    T : int
        Number of retained periods (after burn-in).
    rng : numpy.random.Generator
        Random generator (drives the innovations).
    burn : int, optional
        Burn-in periods discarded so the panel starts near stationarity
        (default ``50``).

    Returns
    -------
    numpy.ndarray, shape (T, n)
        Time-major outcome matrix (rows are periods, columns are units).
    """
    c = (np.arange(n) % 10 + 1).astype(float)        # c_i in {1, ..., 10}
    Y = np.zeros((n, T + burn))
    Y[:, 0] = c                                      # start at the stationary mean
    for t in range(1, T + burn):
        Y[:, t] = 0.1 * c + 0.9 * Y[:, t - 1] + rng.normal(size=n)
    return Y[:, burn:].T                             # drop burn-in -> (T, n)


def random_theta(n: int, m: int, s: int, rng: np.random.Generator) -> np.ndarray:
    """Random sparse weight matrix with ``s`` total non-zeros, columns summing to 1.

    The ``s`` non-zeros are spread across the ``m`` columns as evenly as possible
    (at least one per column); within a column the non-zero weights are drawn
    uniformly and normalised to sum to one (a convex combination of donors).

    Parameters
    ----------
    n : int
        Number of donors (rows of ``Theta``).
    m : int
        Number of treated units (columns of ``Theta``).
    s : int
        Total number of non-zero entries across the whole matrix.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray, shape (n, m)
        Sparse, column-stochastic weight matrix.
    """
    Theta = np.zeros((n, m))
    base, extra = divmod(s, m)
    per_col = np.full(m, base, dtype=int)
    per_col[:extra] += 1
    per_col = np.clip(per_col, 1, n)
    for j in range(m):
        idx = rng.choice(n, size=int(per_col[j]), replace=False)
        w = rng.random(int(per_col[j]))
        Theta[idx, j] = w / w.sum()                  # column sums to 1
    return Theta


def simulate_msqrt_panel(
    n_treated: int = 5,
    n_control: int = 40,
    T0: int = 100,
    n_post: int = 10,
    nonzeros_per_unit: int = 5,
    att: float = 2.0,
    noise: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a block, multiple-treated panel in the Shen-Song-Abadie regime.

    Donors follow the AR(1) process of :func:`ar1_panel`; each treated unit is a
    sparse convex combination of the donors (:func:`random_theta`) plus Gaussian
    noise -- i.e. ``Y_treated = X @ Theta + E`` -- so the synthetic-control model
    is well specified. A constant treatment effect ``att`` is then added to the
    treated units' post-treatment cells; all treated units adopt at period
    ``T0`` (a block design).

    Parameters
    ----------
    n_treated, n_control : int
        Counts of treated and never-treated (donor) units.
    T0, n_post : int
        Pre- and post-treatment period counts (common adoption at ``T0``).
    nonzeros_per_unit : int
        Approximate number of active donors per treated unit (controls the
        sparsity of ``Theta``); capped at ``n_control``.
    att : float
        Constant additive treatment effect applied to treated post-period cells.
    noise : float
        Standard deviation of the idiosyncratic error ``E``.
    seed : int
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Long panel with columns ``unit`` (``c*`` donors, ``t*`` treated),
        ``time``, ``Y``, ``treated``.
    """
    rng = np.random.default_rng(seed)
    T = T0 + n_post

    X = ar1_panel(n_control, T, rng)                 # (T, n_control)
    s = int(min(nonzeros_per_unit, n_control)) * n_treated
    Theta = random_theta(n_control, n_treated, s, rng)
    Y_treated = X @ Theta + rng.normal(scale=noise, size=(T, n_treated))
    Y_treated[T0:] += att                            # inject the effect post-T0

    # Assemble the long panel: donors then treated, all units over all periods.
    rows = []
    for j in range(n_control):
        for t in range(T):
            rows.append((f"c{j}", t, float(X[t, j]), 0))
    for j in range(n_treated):
        for t in range(T):
            d = int(t >= T0)
            rows.append((f"t{j}", t, float(Y_treated[t, j]), d))
    return pd.DataFrame(rows, columns=["unit", "time", "Y", "treated"])
