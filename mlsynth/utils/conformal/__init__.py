"""Conformal prediction bands for synthetic control.

Conformal inference asks one question of a fitted control: how large are the
errors it makes on data it did not see, and how far must a band reach to cover
``1 - alpha`` of them? The answer is an order statistic of conformity scores, and
this package keeps that answer -- and the score constructions feeding it -- in one
place, so bands cannot drift apart between estimators.

Contents:

* :mod:`.quantile` -- :func:`split_conformal_quantile`, the finite-sample order
  statistic every band is calibrated by.
* :mod:`.scores` -- conformity-score constructions. Currently
  :func:`rolling_origin_block_sums`, the out-of-sample cumulative errors a
  cumulative band needs.
* :mod:`.cumulative` -- the band for a cumulative (total) effect:
  :func:`cumulative_conformal_interval` (pure combiner) and
  :func:`cumulative_conformal_from_refit` (single-treated-unit convenience).
* :mod:`.structure` -- :class:`CumulativeConformalBand`.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

from .cumulative import cumulative_conformal_from_refit, cumulative_conformal_interval
from .quantile import split_conformal_quantile
from .scores import rolling_origin_block_sums
from .structure import CumulativeConformalBand

__all__ = [
    "CumulativeConformalBand",
    "cumulative_conformal_from_refit",
    "cumulative_conformal_interval",
    "rolling_origin_block_sums",
    "split_conformal_quantile",
]
