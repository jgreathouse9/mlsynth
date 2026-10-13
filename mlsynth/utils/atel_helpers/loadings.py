"""Bandwidth selection and the time-varying loading.

The loading on the treated unit moves with time, so it is fit by local linear
regression on the pre-period and carried into the post-period. Because the
estimand sits at :math:`T_0`, which is an endpoint of the sample, the kernel is
one-sided with its mass doubled instead of renormalised by its truncated
integral -- the treatment the reference implementation uses on the estimation
path.

The bandwidth comes from leave-one-out cross-validation over a fixed grid
(paper eq. 20). The kernel there is centred on the last pre-period for every
held-out t, so the criterion scores prediction at the point the estimand is
localised at and not at each t in turn.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError
from .numerics import weighted_ls

__all__ = [
    "BANDWIDTH_GRID",
    "half_epanechnikov",
    "post_period_kernel",
    "cv_bandwidth",
    "local_linear_loadings",
]

#: The reference implementation's bandwidth grid, 0.30 to 0.95 in steps of 0.05.
BANDWIDTH_GRID = np.round(np.linspace(0.30, 0.95, 14), 10)


def half_epanechnikov(u: np.ndarray, side: str) -> np.ndarray:
    """Epanechnikov weight on one side of its support, mass doubled.

    Parameters
    ----------
    u : np.ndarray
        Scaled distance from the localization point.
    side : {"left", "right"}
        ``"left"`` keeps ``u >= -1`` (pre-period), ``"right"`` keeps ``u <= 1``
        (post-period).

    Returns
    -------
    np.ndarray
        ``1.5 (1 - u^2)`` inside the half support, zero outside.
    """
    u = np.asarray(u, dtype=float)
    keep = (u >= -1.0) if side == "left" else (u <= 1.0)
    return 0.75 * (1.0 - u**2) * keep * 2.0


def _local_linear_design(factors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``[F, F * offset]``: the level block and the slope block."""
    return np.hstack([factors, factors * offsets[:, None]])


def post_period_kernel(n_pre: int, n_total: int, scaled_window: int) -> np.ndarray:
    """Localization weights for each post-period (paper eq. 5)."""
    lead = np.arange(n_pre + 1, n_total + 1)
    return half_epanechnikov((lead - n_pre) / scaled_window, "right")


def cv_bandwidth(
    treated_pre: np.ndarray, factors_pre: np.ndarray
) -> float:
    """Leave-one-out bandwidth over :data:`BANDWIDTH_GRID`.

    Parameters
    ----------
    treated_pre : np.ndarray
        Treated outcomes over the pre-period, shape ``(T0,)``.
    factors_pre : np.ndarray
        Estimated factors over the pre-period, shape ``(T0, J)``.

    Returns
    -------
    float
        The grid point minimising the criterion. Ties take the smallest.
    """
    y = np.asarray(treated_pre, dtype=float).ravel()
    F = np.asarray(factors_pre, dtype=float)
    T0 = y.size
    if F.shape[0] != T0:
        raise MlsynthDataError(
            f"The pre-period factors have {F.shape[0]} rows for {T0} treated "
            "pre-period observations."
        )
    idx = np.arange(1, T0 + 1)
    design = _local_linear_design(F, idx / T0 - 1.0)
    scores = np.empty(BANDWIDTH_GRID.size)
    for i, h in enumerate(BANDWIDTH_GRID):
        window = int(np.floor(T0 * h))
        if window < 1:
            scores[i] = np.inf
            continue
        total = 0.0
        for t in range(T0):
            kernel = half_epanechnikov((idx - T0) / window, "left").copy()
            kernel[t] = 0.0
            beta = weighted_ls(design, kernel, y)
            total += (y[t] - (design @ beta)[t]) ** 2
        scores[i] = total / T0
    return float(BANDWIDTH_GRID[int(np.argmin(scores))])


def local_linear_loadings(
    treated_outcome: np.ndarray, factors_pre: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Loadings for each post-period, shape ``(J, T1)``.

    The local linear fit at :math:`T_0` gives a level and a slope per factor;
    the slope carries the loading forward as ``(T0 - t) / T1`` for each
    post-period t, so the loading is extrapolated linearly in rescaled time.
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    F0 = np.asarray(factors_pre, dtype=float)
    n_total = y.size
    T0, n_factors = F0.shape
    T1 = n_total - T0
    if T1 < 1:
        raise MlsynthDataError("ATEL needs at least one post-treatment period.")
    window = int(np.floor(T0 * float(bandwidth)))
    if window < 1:
        raise MlsynthDataError(
            f"Bandwidth {bandwidth} leaves an empty smoothing window over "
            f"{T0} pre-periods."
        )
    idx = np.arange(1, T0 + 1)
    kernel = half_epanechnikov((idx - T0) / window, "left")
    design = _local_linear_design(F0, idx / T0 - 1.0)
    beta = weighted_ls(design, kernel, y[:T0])
    lead = np.arange(T0 + 1, n_total + 1)
    level = beta[:n_factors][:, None]
    slope = beta[n_factors:][:, None]
    return level + slope * ((T0 - lead) / T1)[None, :]
