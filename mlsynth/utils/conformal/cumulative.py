"""Conformal interval for the cumulative (total) treatment effect.

mlsynth reports the cumulative effect ``C = sum_k (y_{T0+k} - synthetic_{T0+k})``
as a point estimate. This module gives it a band calibrated *for the sum*: the
half-width comes from conformity scores that are themselves cumulative errors over
windows of the same length, so the accumulation of period-to-period dependence is
measured rather than assumed.

Two entry points:

* :func:`cumulative_conformal_interval` -- the pure combiner. Given a point
  estimate and a vector of conformity scores, it centres the scores and returns
  the band. Callers with their own scoring pass (a pooled multi-unit refit, say)
  use this directly.
* :func:`cumulative_conformal_from_refit` -- the single-treated-unit convenience:
  builds rolling-origin scores via
  :func:`~mlsynth.utils.conformal.scores.rolling_origin_block_sums`, then combines.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .quantile import split_conformal_quantile
from .scores import rolling_origin_block_sums
from .structure import CumulativeConformalBand


def _validate_alpha(alpha) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise MlsynthConfigError(f"alpha must be a number in (0, 1); got {alpha!r}.")
    if not 0.0 < float(alpha) < 1.0:
        raise MlsynthConfigError(f"alpha must lie in the open interval (0, 1); got {alpha!r}.")
    return float(alpha)


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MlsynthConfigError(f"{name} must be a positive integer; got {value!r}.")
    if int(value) < 1:
        raise MlsynthConfigError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def cumulative_conformal_interval(
    point: float,
    scores: np.ndarray,
    *,
    alpha: float = 0.1,
    horizon: int = 0,
) -> CumulativeConformalBand:
    """Combine a cumulative point estimate with conformity scores into a band.

    The scores are centred before the order statistic is taken, so a systematic
    offset in the calibration windows widens nothing: the band measures spread
    about the typical cumulative error, not distance from zero.

    Parameters
    ----------
    point : float
        Cumulative effect estimate over the horizon.
    scores : array-like, shape ``(m,)``
        Conformity scores -- cumulative out-of-sample errors over calibration
        windows of the same length as the horizon.
    alpha : float, optional
        Target miscoverage (default ``0.1``, a 90% band).
    horizon : int, optional
        Horizon the point estimate accumulates, recorded on the band.

    Raises
    ------
    MlsynthConfigError
        If ``alpha`` is not in ``(0, 1)``.
    MlsynthDataError
        If ``scores`` is empty or contains non-finite values.
    """
    alpha = _validate_alpha(alpha)
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise MlsynthDataError(
            "cumulative conformal inference needs at least one conformity score; "
            "got an empty score vector."
        )
    if not np.all(np.isfinite(s)):
        raise MlsynthDataError(
            "conformity scores must all be finite; got "
            f"{int((~np.isfinite(s)).sum())} non-finite value(s)."
        )
    half = float(split_conformal_quantile(s - s.mean(), alpha=alpha))
    return CumulativeConformalBand(
        point=float(point),
        lower=float(point) - half,
        upper=float(point) + half,
        half_width=half,
        n_scores=int(s.size),
        alpha=alpha,
        horizon=int(horizon),
    )


def cumulative_conformal_from_refit(
    y: np.ndarray,
    Y0: np.ndarray,
    pre_periods: int,
    horizon: int,
    weight_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    alpha: float = 0.1,
    min_train_frac: float = 0.3,
) -> CumulativeConformalBand:
    """Cumulative conformal band for one treated unit, calibrated by refitting.

    Fits the control on the full pre-period for the point estimate, then calibrates
    by refitting at sliding non-overlapping origins within the pre-period.

    Parameters
    ----------
    y : array-like, shape ``(T,)``
        Treated-unit outcome over the full sample.
    Y0 : array-like, shape ``(T, J)``
        Donor outcomes aligned to ``y``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    horizon : int
        Number of post-periods to accumulate, ``L``.
    weight_fn : callable
        ``weight_fn(y_train, Y0_train) -> w``; the estimator's own refit.
    alpha : float, optional
        Target miscoverage (default ``0.1``).
    min_train_frac : float, optional
        Earliest calibration origin as a fraction of ``T0`` (default ``0.3``).

    Raises
    ------
    MlsynthConfigError
        If ``alpha``, ``horizon``, ``pre_periods`` or ``min_train_frac`` is invalid.
    MlsynthDataError
        If ``Y0`` is not two-dimensional, its rows do not match ``y``, or the
        horizon runs past the end of the sample.
    """
    alpha = _validate_alpha(alpha)
    horizon = _validate_positive_int(horizon, "horizon")
    pre_periods = _validate_positive_int(pre_periods, "pre_periods")
    if isinstance(min_train_frac, bool) or not isinstance(
        min_train_frac, (int, float, np.floating, np.integer)
    ):
        raise MlsynthConfigError(
            f"min_train_frac must be a number in (0, 1); got {min_train_frac!r}."
        )
    if not 0.0 < float(min_train_frac) < 1.0:
        raise MlsynthConfigError(
            f"min_train_frac must lie in the open interval (0, 1); got {min_train_frac!r}."
        )

    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    if Y0.ndim != 2:
        raise MlsynthDataError(
            f"Y0 must be a two-dimensional donor matrix (T x J); got {Y0.ndim} dimension(s)."
        )
    if Y0.shape[0] != y.shape[0]:
        raise MlsynthDataError(
            f"y has {y.shape[0]} periods but Y0 has {Y0.shape[0]} rows; they must match."
        )
    if pre_periods + horizon > y.shape[0]:
        raise MlsynthDataError(
            f"pre_periods + horizon ({pre_periods} + {horizon}) runs past the end of the "
            f"sample ({y.shape[0]} periods); there is no full post-window to accumulate."
        )

    w_full = np.asarray(weight_fn(y[:pre_periods], Y0[:pre_periods]), dtype=float).ravel()
    post = slice(pre_periods, pre_periods + horizon)
    point = float(np.sum(y[post] - Y0[post] @ w_full))

    scores = rolling_origin_block_sums(
        y, Y0, pre_periods, horizon, weight_fn, min_train_frac=float(min_train_frac)
    )
    if scores.size == 0:
        # A valid configuration can still leave no room for a calibration window
        # (a late first origin, or a horizon nearly as long as the pre-period).
        # That is a statement about the sample, not a caller error.
        return CumulativeConformalBand(
            point=point, lower=-np.inf, upper=np.inf, half_width=float("inf"),
            n_scores=0, alpha=alpha, horizon=horizon,
        )
    return cumulative_conformal_interval(point, scores, alpha=alpha, horizon=horizon)
