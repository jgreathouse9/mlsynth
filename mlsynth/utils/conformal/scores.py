"""Conformity scores for a cumulative effect: rolling-origin out-of-sample block sums.

A cumulative band needs scores on the *sum* over an ``L``-period window, and those
scores have to be honest about the fit. Scoring a window the control was fitted on
understates the error (the fit already chased that window); rescaling a per-period
band by ``L`` assumes the periods accumulate independently. Both distort the sum.

The construction here does neither. It slides an origin across the pre-period and,
at each origin, refits the control on the data strictly before it and reads the
next ``L`` periods -- a genuine out-of-sample window of exactly the length the
band is for. Origins step by ``L``, so the windows do not overlap and the scores
stay exchangeable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

#: Shortest training block a calibration refit may use. A control fitted on a
#: handful of periods interpolates them and predicts nothing, so the window it
#: scores would report the fit's degeneracy rather than the method's error.
MIN_TRAIN_PERIODS = 10


def rolling_origin_block_sums(
    y: np.ndarray,
    Y0: np.ndarray,
    pre_periods: int,
    horizon: int,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    *,
    min_train_frac: float = 0.3,
) -> np.ndarray:
    """Out-of-sample cumulative errors over non-overlapping pre-period windows.

    Parameters
    ----------
    y : np.ndarray, shape ``(T,)``
        Treated-unit outcome over the full sample.
    Y0 : np.ndarray, shape ``(T, J)``
        Donor outcomes aligned to ``y``.
    pre_periods : int
        Number of pre-treatment periods ``T0``; origins are drawn from within it.
    horizon : int
        Window length ``L`` -- both the block length and the origin stride.
    weight_fn : callable
        ``weight_fn(keep_idx) -> w``, the estimator's own refit on the periods
        indexed by ``keep_idx``. Called once per origin. Indices rather than
        sliced arrays, matching :func:`~mlsynth.utils.inferutils.debiased_sc_ttest`:
        an estimator with covariates has to subset those by the same periods, and
        only the caller knows how.
    min_train_frac : float, optional
        Earliest origin as a fraction of ``T0`` (default ``0.3``). The first
        origin is ``max(MIN_TRAIN_PERIODS, T0 * min_train_frac)``, so short
        panels still train on an absolute minimum of periods.

    Returns
    -------
    np.ndarray, shape ``(m,)``
        One cumulative out-of-sample error per origin. Empty when the pre-period
        admits no origin.

    Notes
    -----
    Callers whose refit produces several treated units at once (a pooled fit) can
    build their own score vector per unit from a single pass and hand it to
    :func:`~mlsynth.utils.conformal.cumulative.cumulative_conformal_interval`
    directly; this helper is the single-treated-unit path.
    """
    start = max(MIN_TRAIN_PERIODS, int(pre_periods * float(min_train_frac)))
    scores = []
    for origin in range(start, int(pre_periods) - int(horizon) + 1, int(horizon)):
        w = np.asarray(weight_fn(np.arange(origin)), dtype=float).ravel()
        block = slice(origin, origin + int(horizon))
        scores.append(float(np.sum(y[block] - Y0[block] @ w)))
    return np.asarray(scores, dtype=float)
