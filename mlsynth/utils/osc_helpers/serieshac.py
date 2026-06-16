"""Fixed-smoothing Series-HAC variance and the Sun (2013) bandwidth for OSC.

These are the inference primitives of the Orthogonalized Synthetic Control: an
orthonormal-series long-run-variance estimator and a CPE-optimal smoothing
parameter, feeding a t-test whose reference distribution is t with the smoothing
parameter as degrees of freedom (so size is controlled without a consistent
variance). Pure linear algebra -- solver-independent -- so these are the
functions pinned value-for-value against the reference.
"""
from __future__ import annotations

import numpy as np


def orthonormal_basis(x, j: int):
    """Orthonormal Fourier basis function phi_j on [0, 1].

    Even ``j`` -> sqrt(2) sin(2 pi j x); odd ``j`` -> sqrt(2) cos(2 pi j x).
    """
    raise NotImplementedError


def series_hac_variance(preg, postg, eta, h: int):
    """Orthonormal-series (fixed-smoothing) variance of the orthogonalized ATT.

    Parameters
    ----------
    preg : ndarray (Q-1, T0)
        Pre-period orthogonalized moment residuals (per instrument).
    postg : ndarray (T1,)
        Post-period orthogonalized moment residuals (treated gap minus ATT).
    eta : ndarray (Q,)
        Instrument/moment weights (last entry the post-moment weight).
    h : int
        Smoothing parameter (number of basis terms).
    """
    raise NotImplementedError


def cpe_optimal_h(preg, p: int = 1, sig: float = 0.05) -> int:
    """CPE-optimal smoothing parameter K via Sun (2013), from the pre-residuals."""
    raise NotImplementedError


def ttest_pvalue(beta_hat, V, h, n, beta0: float = 0.0):
    """Two-sided p-value: t_n = sqrt(n)(beta_hat - beta0)/sqrt(V), df = h."""
    raise NotImplementedError


def ttest_ci(beta_hat, V, h, alpha: float):
    """(1 - alpha) t-interval for the ATT using the Series-HAC variance."""
    raise NotImplementedError
