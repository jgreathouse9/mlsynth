"""Conditional-moment estimation for the out-of-sample CFPT band.

The out-of-sample error of an SC prediction is the post-treatment residual
``u_i(Ti+k)``. Cattaneo et al. bound it with a sub-Gaussian concentration
inequality that needs two conditional moments per treated unit: the mean
``E[u_it | H]`` and a variance proxy ``sigma_it^2`` (the optimal sub-Gaussian
parameter, which equals ``V[u_it | H]`` under strict sub-Gaussianity). Both are
estimated from the *pre-treatment* residuals of the synthetic-control fit.

For the block, short-post settings MSQRT targets, a per-unit constant proxy
(the residual mean and variance over the pre-period) is the natural default;
``assume_zero_mean`` imposes the common ``E[u|H] = 0`` simplification.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def conditional_moments(
    pre_residuals: np.ndarray,
    *,
    assume_zero_mean: bool = False,
) -> Tuple[float, float]:
    """Per-unit ``(mean, variance-proxy)`` from a vector of pre-period residuals.

    Parameters
    ----------
    pre_residuals : np.ndarray
        Synthetic-control residuals ``u_hat_it`` over the pre-treatment periods
        for one treated unit.
    assume_zero_mean : bool
        If True, fix ``E[u|H] = 0`` and estimate the variance proxy as the mean
        squared residual; otherwise demean first.
    """
    r = np.asarray(pre_residuals, dtype=float).ravel()
    if r.size == 0:
        return 0.0, 0.0
    mu = 0.0 if assume_zero_mean else float(r.mean())
    sigma2 = float(np.mean((r - mu) ** 2))
    return mu, sigma2


def unit_moments(
    pre_residuals: np.ndarray,
    unit_names,
    *,
    assume_zero_mean: bool = False,
) -> Tuple[Dict, Dict]:
    """Stack :func:`conditional_moments` over the columns of a residual matrix.

    Parameters
    ----------
    pre_residuals : np.ndarray
        ``(T0, m)`` pre-period residuals (one column per treated unit).
    unit_names : sequence
        Length-``m`` treated-unit identifiers (the dict keys).

    Returns
    -------
    (mu, sigma) : (dict, dict)
        ``{unit: mean}`` and ``{unit: sigma}`` (standard deviation, not
        variance).
    """
    R = np.asarray(pre_residuals, dtype=float)
    mu, sigma = {}, {}
    for j, name in enumerate(unit_names):
        m, s2 = conditional_moments(R[:, j], assume_zero_mean=assume_zero_mean)
        mu[name] = m
        sigma[name] = float(np.sqrt(max(s2, 0.0)))
    return mu, sigma
