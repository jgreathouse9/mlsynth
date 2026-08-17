"""Frozen containers for conformal prediction bands.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
https://doi.org/10.1080/01621459.2021.1920957
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CumulativeConformalBand:
    """Conformal band for a cumulative (summed-over-horizon) treatment effect.

    The band is symmetric: ``[point - half_width, point + half_width]``. When the
    calibration set is too small for the ``ceil((m + 1)(1 - alpha))``-th order
    statistic to exist, ``half_width`` is ``+inf`` and the endpoints are infinite --
    an honest statement that the requested level is unreachable at this sample
    size, rather than a silently under-covering guess.

    Parameters
    ----------
    point : float
        Cumulative effect estimate, ``sum_k (y_{T0+k} - synthetic_{T0+k})`` over
        the horizon.
    lower, upper : float
        Band endpoints, ``point -/+ half_width``.
    half_width : float
        Split-conformal half-width of the centred conformity scores.
    n_scores : int
        Number of conformity scores (calibration blocks) the band was built from.
    alpha : float
        Target miscoverage level; the band targets ``1 - alpha`` coverage.
    horizon : int
        Number of post-treatment periods accumulated. ``0`` when the band was
        built from precomputed scores by the caller.
    """

    point: float
    lower: float
    upper: float
    half_width: float
    n_scores: int
    alpha: float
    horizon: int = 0
