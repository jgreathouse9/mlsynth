"""Conformal prediction bands for synthetic control.

Conformal inference asks one question of a fitted control: how large are the
errors it makes on data it did not see, and how far must a band reach to cover
``1 - alpha`` of them? The answer is an order statistic of conformity scores, and
this package keeps that answer -- and the score constructions feeding it -- in one
place, so bands cannot drift apart between estimators.

Contents:

* :mod:`.quantile` -- :func:`split_conformal_quantile`, the finite-sample order
  statistic every band is calibrated by.
* :mod:`.inversion` -- :func:`confidence_set_bounds`, the duality rule a
  test-inversion band is read off with: the nulls a level-alpha test does not
  reject.
* :mod:`.scores` -- conformity-score constructions. Currently
  :func:`rolling_origin_block_sums`, the out-of-sample cumulative errors a
  cumulative band needs.
* :mod:`.cumulative` -- the band for a cumulative (total) effect:
  :func:`cumulative_conformal_interval` (pure combiner) and
  :func:`cumulative_conformal_from_refit` (single-treated-unit convenience).
* :mod:`.refit` -- :func:`conformal_refit_gaps`, the refit a test-inversion
  procedure performs under a candidate null, and the one place the choice
  between a simplex and a ridge-augmented control is made.
* :mod:`.structure` -- :class:`CumulativeConformalBand`.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

from .cumulative import cumulative_conformal_from_refit, cumulative_conformal_interval
from .inversion import confidence_set_bounds
from .permutation import BLOCK_STATISTICS, moving_block_pvalue
from .quantile import split_conformal_quantile
from .refit import CONFORMAL_REFIT_RULES, conformal_refit_gaps
from .scores import MIN_TRAIN_PERIODS, rolling_origin_block_sums
from .structure import CumulativeConformalBand

__all__ = [
    "BLOCK_STATISTICS",
    "CONFORMAL_REFIT_RULES",
    "MIN_TRAIN_PERIODS",
    "CumulativeConformalBand",
    "confidence_set_bounds",
    "conformal_refit_gaps",
    "cumulative_conformal_from_refit",
    "cumulative_conformal_interval",
    "moving_block_pvalue",
    "rolling_origin_block_sums",
    "split_conformal_quantile",
]
