"""Seasonal factor-model simulator of sales-like panel data for PANGEO.

Generates a balanced panel of "sales" with the structure typical of geo
marketing data, drawing on the factor-model DGPs used throughout mlsynth:

.. math::

   Y_{it} = \\underbrace{\\lambda_t^\\top \\mu_i}_{\\text{low-rank factors}}
          + \\underbrace{a_i \\sin(2\\pi t / s + \\phi_i)}_{\\text{seasonality}}
          + \\underbrace{\\gamma_i + \\beta_i t}_{\\text{unit level + trend}}
          + \\varepsilon_{it},

across several **non-overlapping treatment arms**. The design problem is
prospective, so the generated panel is *pre-treatment only* -- it is the
historical window a designer would use to build balanced supergeo pairs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_seasonal_sales_panel(
    units_per_arm: int = 5,
    arms: Tuple[str, ...] = ("A", "B", "C"),
    T: int = 156,
    n_factors: int = 2,
    season_period: int = 52,
    noise: float = 0.05,
    seed: int = 0,
    covariates: bool = False,
) -> pd.DataFrame:
    """Simulate a seasonal, multi-arm, sales-like pre-treatment panel.

    Parameters
    ----------
    units_per_arm : int
        Number of geos (markets) eligible for each arm.
    arms : tuple of str
        Arm labels; each unit is eligible for exactly one arm. Arms occupy
        non-overlapping geos.
    T : int
        Number of pre-treatment periods (e.g. weeks; default 3 years).
    n_factors : int
        Rank of the common low-rank factor structure.
    season_period : int
        Seasonal cycle length (e.g. 52 weeks).
    noise : float
        Idiosyncratic noise scale, relative to the signal.
    seed : int
        RNG seed.
    covariates : bool
        If ``True``, also emit time-invariant baseline ``population`` and
        ``income`` columns (correlated with the unit's level and factor
        loadings) for PANGEO's covariate-balancing option.

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``unit``, ``time``, ``sales``, ``arm``
        (plus ``population`` and ``income`` when ``covariates=True``).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Common low-rank factor structure (shared across all units).
    F = np.cumsum(rng.standard_normal((T, n_factors)) * 0.3, axis=0)
    F += np.linspace(0.0, 1.0, T)[:, None]            # mild common drift

    rows = []
    for arm in arms:
        # Geos within an arm occupy one region and share the seasonal cycle
        # (a region peaks at the same time of year); only the amplitude and
        # a small phase jitter vary across geos. Unit-level differences come
        # mainly from the factor loadings, level and trend -- so balanced,
        # genuinely parallel supergeo pairs are achievable.
        arm_phase = rng.uniform(0, 2 * np.pi)
        for u in range(units_per_arm):
            mu = rng.standard_normal(n_factors)        # unit factor loadings
            level = 10.0 + rng.standard_normal() * 2.0  # baseline sales level
            trend = rng.standard_normal() * 0.01        # gentle unit trend
            amp = 1.0 + rng.uniform(0, 1.0)             # seasonal amplitude
            phase = arm_phase + rng.normal(0, 0.15)     # near-shared phase
            signal = (
                F @ mu
                + amp * np.sin(2 * np.pi * t / season_period + phase)
                + level
                + trend * t
            )
            y = signal + rng.standard_normal(T) * noise * np.std(signal)
            name = f"{arm}{u}"
            extra = {}
            if covariates:
                # Baseline characteristics correlated with the unit's latent
                # level / loadings, so balancing them is non-trivial.
                extra["population"] = float(
                    50_000 + 8_000 * level + 5_000 * rng.standard_normal())
                extra["income"] = float(
                    40_000 + 6_000 * mu[0] + 3_000 * rng.standard_normal())
            for ti in range(T):
                rows.append({"unit": name, "time": int(ti),
                             "sales": float(y[ti]), "arm": arm, **extra})
    return pd.DataFrame(rows)
