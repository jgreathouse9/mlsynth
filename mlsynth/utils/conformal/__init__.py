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
* :mod:`.scores` -- conformity-score constructions, all sharing one
  :func:`origin_schedule` so they calibrate on the same windows:
  :func:`rolling_origin_block_sums` (one summed score per window, what a
  split-conformal band needs), :func:`rolling_origin_period_errors` (the same
  windows kept per period, what a resampled band needs), and
  :func:`rolling_origin_counterfactual_errors` (the same again for an estimator
  that returns a counterfactual instead of donor weights).
* :mod:`.cumulative` -- the band for a cumulative (total) effect:
  :func:`cumulative_conformal_interval` (pure combiner) and
  :func:`cumulative_conformal_from_refit` (single-treated-unit convenience).
* :mod:`.cumulative_inversion` -- the same total by the other route:
  :func:`cumulative_conformal_by_inversion` inverts a constant-effect null over
  the whole post block, so its reference set is the ``T0 + H`` periods and not a
  count of calibration windows. It restricts the effect's shape where
  :mod:`.cumulative` restricts the window count.
* :mod:`.refit` -- :func:`conformal_refit_gaps`, the refit a test-inversion
  procedure performs under a candidate null, and the one place the choice
  between a simplex and a ridge-augmented control is made.
* :mod:`.resample` -- the third route from scores to a band: resample the errors
  into whole paths (:func:`block_error_paths`) and let the caller accumulate them,
  which is what a running total needs. :func:`resample_cumulative_paths` composes
  it with the calibration pass in one call.
* :mod:`.ensemble` -- a fourth route to out-of-sample errors:
  :func:`bootstrap_loo_errors` fits an ensemble on bootstrap samples of the
  training periods and scores each period from the members that did not sample
  it, so a calibration series comes back without withholding periods to buy it.
* :mod:`.structure` -- :class:`CumulativeConformalBand`.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

from .cumulative import cumulative_conformal_from_refit, cumulative_conformal_interval
from .cumulative_inversion import (CONFORMAL_PERMUTATIONS,
                                   CumulativeInversionBand,
                                   cumulative_conformal_by_inversion)
from .ensemble import (AGGREGATIONS, DEFAULT_MEMBERS, EnsembleErrors,
                       bootstrap_loo_errors)
from .inversion import confidence_set_bounds
from .permutation import BLOCK_STATISTICS, moving_block_pvalue
from .quantile import split_conformal_quantile
from .refit import CONFORMAL_REFIT_RULES, conformal_refit_gaps
from .resample import (WHOLE_HORIZON, block_error_paths,
                       resample_cumulative_paths,
                       resample_cumulative_paths_from_weights,
                       resolve_block)
from .scores import (MIN_TRAIN_PERIODS, origin_schedule,
                     rolling_origin_block_sums,
                     rolling_origin_counterfactual_errors,
                     rolling_origin_period_errors)
from .structure import CumulativeConformalBand

__all__ = [
    "AGGREGATIONS",
    "BLOCK_STATISTICS",
    "CONFORMAL_PERMUTATIONS",
    "CONFORMAL_REFIT_RULES",
    "CumulativeInversionBand",
    "DEFAULT_MEMBERS",
    "EnsembleErrors",
    "MIN_TRAIN_PERIODS",
    "CumulativeConformalBand",
    "confidence_set_bounds",
    "bootstrap_loo_errors",
    "conformal_refit_gaps",
    "cumulative_conformal_by_inversion",
    "cumulative_conformal_from_refit",
    "cumulative_conformal_interval",
    "moving_block_pvalue",
    "WHOLE_HORIZON",
    "block_error_paths",
    "origin_schedule",
    "resample_cumulative_paths",
    "resample_cumulative_paths_from_weights",
    "resolve_block",
    "rolling_origin_counterfactual_errors",
    "rolling_origin_block_sums",
    "rolling_origin_period_errors",
    "split_conformal_quantile",
]
