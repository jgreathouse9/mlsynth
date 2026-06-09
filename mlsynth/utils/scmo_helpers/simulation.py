"""Monte-Carlo DGPs for the Synthetic Control with Multiple Outcomes (SCMO)
benchmarks.

Two data-generating processes from the two SCMO papers, kept here (not inline in
the benchmark cases) per the benchmarking definition-of-done:

* :func:`simulate_tian` -- the shared-predictor factor model of Tian, Lee &
  Panchenko (2026, *Econometrics Journal*, Section 3, eq. 3.1), which is also the
  ``Simulation1.R`` DGP of the Sun, Ben-Michael & Feller (2025, REStat)
  replication package. Outcomes share the unit predictors ``mu_i`` but draw
  outcome-specific time loadings, so matching on more outcomes sharpens the
  identification of the common ``mu_i``. Used for the *concatenated* variant.

* :func:`simulate_sun` -- the common/idiosyncratic factor model of Sun,
  Ben-Michael & Feller (2025, REStat, supplement Appendix D, eqs. 16 and the
  ``rho``-mixed model). A single common factor (``rho = 1``) is the regime where
  averaging across outcomes cancels idiosyncratic noise and helps; purely
  idiosyncratic loadings (``rho = 0``) is the adversarial regime where averaging
  hurts. Used for the *averaged* variant.

Both return a list of ``K`` outcome matrices (each ``(N, TT)`` with ``TT = T0 +
1``), the unit count, the post-period count, and the treated-unit row index;
:func:`to_panel` reshapes them into the long DataFrame SCMO consumes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def simulate_tian(
    rng: np.random.Generator,
    T0: int,
    K: int,
    *,
    N0: int = 29,
    f: int = 4,
    tau: float = 0.0,
) -> Tuple[List[np.ndarray], int, int, int]:
    """Tian-Lee-Panchenko (2026) Section-3 DGP (== Sun et al. ``Simulation1.R``).

    ``Y_{i,t,k} = mu_i' lambda_{t,k} + eps_{i,t,k}`` with ``mu_i = (1, x_i)``,
    ``x_i ~ U[-1, 1]^f`` drawn **once** (shared across the ``K`` outcomes so they
    share the unit predictors), ``lambda_{t,k} ~ N(0, 1)`` and
    ``eps ~ N(0, 1)`` drawn per outcome/period. Unit 0 is treated; it receives
    ``tau`` at the single post-period ``t = T0``.

    Parameters
    ----------
    rng : np.random.Generator
    T0 : int
        Number of pre-treatment periods (post is a single period).
    K : int
        Number of related outcomes.
    N0 : int, default 29
        Donor count (paper: 29, so ``N = 30``).
    f : int, default 4
        Number of latent predictors (the factor dimension).
    tau : float, default 0.0
        Treatment effect injected at the post-period (paper: 0, a null DGP).

    Returns
    -------
    (outcomes, N, TT, treated_idx)
    """
    N = 1 + N0
    TT = T0 + 1
    X = np.empty((N, f))
    for r in range(f):
        X[0, r] = rng.uniform(-1, 1)
        X[1:, r] = rng.uniform(-1, 1, N0)
    X = np.column_stack([np.ones(N), X])  # (N, 1 + f) shared predictors
    outcomes: List[np.ndarray] = []
    for _ in range(K):
        beta = rng.normal(0.0, 1.0, size=(1 + f, TT))
        Yk = X @ beta + rng.normal(0.0, 1.0, size=(N, TT))
        Yk[0, T0] += tau
        outcomes.append(Yk)
    return outcomes, N, TT, 0


def simulate_sun(
    rng: np.random.Generator,
    T0: int,
    K: int,
    *,
    N: int = 50,
    rho: float = 1.0,
    tau: float = 0.0,
) -> Tuple[List[np.ndarray], int, int, int]:
    """Sun-Ben-Michael-Feller (2025) Appendix-D factor DGP.

    Common-factor model ``Y_{i,t,k}(0) = rho * phi_i mu_t + (1 - rho) *
    phi_{i,k} mu_{t,k} + eps`` with ``eps ~ N(0, 1)``. The common loadings
    ``phi_i`` are evenly spaced on ``[1, 5]`` and ``mu_t`` evenly on
    ``[0.5, 1]``; the treated unit is the one with the **second-largest** loading
    (a hard-to-fit extreme unit, Ferman 2021 style). At ``rho = 1`` all ``K``
    outcomes share one factor (averaging cancels noise -> helps); at ``rho = 0``
    each outcome has its own loadings (averaging mixes unrelated signals ->
    hurts).

    Returns ``(outcomes, N, TT, treated_idx)``.
    """
    TT = T0 + 1
    phi = np.linspace(1.0, 5.0, N)
    mu = np.linspace(0.5, 1.0, TT)
    treated_idx = int(np.argsort(phi)[-2])  # second-largest loading
    common = np.outer(phi, mu)              # (N, TT), shared across outcomes
    outcomes: List[np.ndarray] = []
    for _ in range(K):
        if rho < 1.0:
            phik = rng.uniform(1.0, 5.0, N)
            muk = rng.uniform(0.5, 1.0, TT)
            idio = np.outer(phik, muk)
        else:
            idio = 0.0
        Yk = rho * common + (1.0 - rho) * idio + rng.normal(0.0, 1.0, size=(N, TT))
        Yk[treated_idx, T0] += tau
        outcomes.append(Yk)
    return outcomes, N, TT, treated_idx


def to_panel(
    outcomes: List[np.ndarray], N: int, TT: int, treated_idx: int
) -> pd.DataFrame:
    """Reshape a list of ``K`` ``(N, TT)`` outcome matrices into the long panel
    SCMO consumes: columns ``unit``, ``time``, ``treat`` (the treated unit at the
    final period), and ``y0 .. y{K-1}`` (the outcomes; ``y0`` is the primary).
    """
    K = len(outcomes)
    rows = []
    for i in range(N):
        for t in range(TT):
            row = {"unit": f"u{i:02d}", "time": t,
                   "treat": int(i == treated_idx and t == TT - 1)}
            for k in range(K):
                row[f"y{k}"] = float(outcomes[k][i, t])
            rows.append(row)
    return pd.DataFrame(rows)
