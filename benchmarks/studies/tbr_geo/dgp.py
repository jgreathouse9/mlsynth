"""Kerman, Wang & Vaver (2017) section 5.1, written from the paper.

    m_i  ~ lognormal, normalised so that sum_i m_i = 1
    y_it = m_i (0.5 W_t + 0.5 Z_it)

W_t is the seasonal component common to every geo and Z_it the geo's own noise;
both have mean 1 and standard deviations c_w and c_z. The paper sets

    rho = c_w^2 / (c_w^2 + c_z^2)   and   c^2 = c_w^2 + c_z^2

so c_w^2 = rho c^2 and c_z^2 = (1 - rho) c^2. Under this parametrisation
Cor(y_it, y_jt) = rho exactly, while the realised coefficient of variation of
y_it is c/2, since the two components each enter at weight 0.5.

The shipped ``matched_markets/examples/data_simulator.py`` is NOT this design:
it uses linear geo sizes and power-law heteroskedasticity with no common
component, so it produces no cross-geo correlation and cannot sweep rho.

One parameter the paper does not give is the lognormal's shape. SIGMA_LOG
below is a documented choice, not a recovered value; ``sweep_sigma_log`` in the
driver measures how much it moves the answer.
"""
from __future__ import annotations

import numpy as np

SIGMA_LOG = 1.0


def geo_sizes(n_geos: int, rng: np.random.Generator,
              sigma_log: float = SIGMA_LOG) -> np.ndarray:
    m = rng.lognormal(mean=0.0, sigma=sigma_log, size=n_geos)
    return m / m.sum()


def panel(n_geos: int, n_pre: int, n_test: int, rho: float, c: float,
          rng: np.random.Generator, sigma_log: float = SIGMA_LOG) -> np.ndarray:
    """Return a (n_pre + n_test, n_geos) array of geo time series."""
    T = n_pre + n_test
    m = geo_sizes(n_geos, rng, sigma_log)
    c_w, c_z = np.sqrt(rho) * c, np.sqrt(1.0 - rho) * c
    W = 1.0 + c_w * rng.normal(size=(T, 1))
    Z = 1.0 + c_z * rng.normal(size=(T, n_geos))
    return m * (0.5 * W + 0.5 * Z)
