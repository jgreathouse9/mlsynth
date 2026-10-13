"""Theorem 1's variance for the localized estimand, and the pointwise band.

Two components. :math:`\\Sigma_1` propagates the sampling error in the estimated
loading through the kernel-weighted post-period aggregation; :math:`\\Sigma_2`
is the contribution of the treated unit's own idiosyncratic errors over the
effective smoothing window. Both are built from the pre-period residuals, which
are refit here period by period with the Su and Wang (2017) boundary kernel.

That boundary kernel divides the Epanechnikov weight by its integral over the
truncated support. The normalizer is one scalar multiplying the whole kernel
vector for a given period, and it enters only through the weighted least-squares
solve in :func:`~.numerics.weighted_ls`, which is invariant to a positive
rescaling of the weights. It therefore cannot move a reported number, and the
reference implementation's left-branch form is kept as written.

The pointwise band is a different object from the interval on the localized
estimand: it requires the treated unit's loading path to be independent and
normal across periods, which is a stronger assumption than Theorem 1 needs. The
interval on the estimate is the defensible output.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...exceptions import MlsynthDataError
from .loadings import half_epanechnikov, post_period_kernel
from .numerics import reference_pinv, weighted_ls

__all__ = ["boundary_kernel", "pre_period_residuals", "atel_variance",
           "pointwise_variance"]


def boundary_kernel(
    periods: np.ndarray, t: int, n_pre: int, window: int, bandwidth: float
) -> np.ndarray:
    """Su and Wang (2017) boundary kernel centred on period ``t``.

    Inside the interior the weight is the plain Epanechnikov. In the two
    boundary regions it is divided by the kernel's integral over the part of the
    support that remains, which keeps the weights' order uniform in ``t``.
    """
    scaled = (np.asarray(periods, dtype=float) - t) / window
    base = 0.75 * (1.0 - scaled**2) * (np.abs(scaled) <= 1.0)
    if t < window:
        a = t / window
        return base / ((a + a**3 / 3.0 + 2.0 / 3.0) * 0.75)
    if t > n_pre - window:
        a = (1.0 - t / n_pre) / bandwidth
        return base / ((a - a**3 / 3.0 + 2.0 / 3.0) * 0.75)
    return base


def pre_period_residuals(
    treated_outcome: np.ndarray, factors_pre: np.ndarray, bandwidth: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Fitted values and residuals over the pre-period, one local fit per period.

    Returns
    -------
    tuple of np.ndarray
        ``(fitted, residuals)``, each shape ``(T0,)``.
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    F0 = np.asarray(factors_pre, dtype=float)
    T0 = F0.shape[0]
    window = int(np.floor(T0 * float(bandwidth)))
    if window < 1:
        raise MlsynthDataError(
            f"Bandwidth {bandwidth} leaves an empty smoothing window over "
            f"{T0} pre-periods."
        )
    periods = np.arange(1, T0 + 1)
    fitted = np.empty(T0)
    for t in periods:
        kernel = boundary_kernel(periods, int(t), T0, window, float(bandwidth))
        design = np.hstack([F0, F0 * ((periods - t) / T0)[:, None]])
        fitted[t - 1] = (design @ weighted_ls(design, kernel, y[:T0]))[t - 1]
    return fitted, y[:T0] - fitted


def _loading_precision(factors_pre: np.ndarray, pre_kernel: np.ndarray) -> np.ndarray:
    """``(F0' K F0)^+``, the inverse information in the loading fit."""
    return reference_pinv(factors_pre.T @ (pre_kernel[:, None] * factors_pre))


def _residual_meat(
    factors_pre: np.ndarray, pre_kernel: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    """``F0' K diag(u^2) K F0``, the sandwich's middle term."""
    scale = (pre_kernel**2) * (residuals**2)
    return factors_pre.T @ (scale[:, None] * factors_pre)


def atel_variance(
    treated_outcome: np.ndarray,
    factors_pre: np.ndarray,
    factors_post: np.ndarray,
    bandwidth: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Theorem 1's variance, with the pre-period fit and residuals.

    Returns
    -------
    tuple
        ``(variance, fitted_pre, residuals_pre)``.
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    F0 = np.asarray(factors_pre, dtype=float)
    F1 = np.asarray(factors_post, dtype=float)
    T0 = F0.shape[0]
    n_total = y.size
    T1 = n_total - T0
    pre_window = int(np.floor(T0 * float(bandwidth)))
    post_window = int(np.floor(T1 * float(bandwidth)))
    if post_window < 1:
        raise MlsynthDataError(
            f"Bandwidth {bandwidth} leaves an empty post-period window over "
            f"{T1} post-periods."
        )

    fitted, residuals = pre_period_residuals(y, F0, bandwidth)
    periods = np.arange(1, T0 + 1)
    pre_kernel = half_epanechnikov((periods - T0) / pre_window, "left")
    post_kernel = post_period_kernel(T0, n_total, post_window)

    aggregated = (F1.T * post_kernel).sum(axis=1) / post_window
    precision = _loading_precision(F0, pre_kernel)
    meat = _residual_meat(F0, pre_kernel, residuals)
    sigma_loading = float(aggregated @ precision @ meat @ precision @ aggregated)

    tail = residuals[T0 - pre_window :]
    tail_kernel = half_epanechnikov(
        (np.arange(T0 - pre_window + 1, T0 + 1) - T0) / pre_window, "left"
    )
    sigma_idiosyncratic = float(np.mean(tail**2 * tail_kernel**2) / post_window)
    return sigma_loading + sigma_idiosyncratic, fitted, residuals


def pointwise_variance(
    factors: np.ndarray,
    loadings: np.ndarray,
    residuals_pre: np.ndarray,
    bandwidth: float,
    n_post: int,
) -> np.ndarray:
    """Per-period variance of the counterfactual path, shape ``(T1,)``.

    This supports a band around the counterfactual, under the stronger
    independent-normal assumption on the loading path noted in the module
    docstring.
    """
    F = np.asarray(factors, dtype=float)
    beta = np.asarray(loadings, dtype=float)
    residuals = np.asarray(residuals_pre, dtype=float).ravel()
    n_total = F.shape[0]
    T0 = n_total - int(n_post)
    F0, F1 = F[:T0], F[T0:]
    pre_window = int(np.floor(T0 * float(bandwidth)))
    post_window = int(np.floor(int(n_post) * float(bandwidth)))

    periods = np.arange(1, T0 + 1)
    pre_kernel = half_epanechnikov((periods - T0) / pre_window, "left")
    post_kernel = post_period_kernel(T0, n_total, post_window)

    centred = beta - (beta * post_kernel).sum(axis=1, keepdims=True) / post_window
    centred = centred * np.sqrt(post_kernel)
    from_loading = np.einsum("tj,jk,tk->t", F1, centred @ centred.T, F1) / post_window

    precision = _loading_precision(F0, pre_kernel)
    meat = _residual_meat(F0, pre_kernel, residuals)
    sandwich = precision @ meat @ precision
    from_residuals = np.einsum("tj,jk,tk->t", F1, sandwich, F1) / pre_window
    return from_loading + from_residuals
