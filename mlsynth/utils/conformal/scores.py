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


def origin_schedule(pre_periods: int, horizon: int, min_train_frac: float):
    """The origins a calibration pass scores, in increasing order.

    One definition of the schedule, so every reader of it -- the summing reader,
    the per-period reader, and the PDA reader that asks its estimator for a
    counterfactual instead of a weight vector -- calibrates on the same windows.

    Parameters
    ----------
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    horizon : int
        Window length ``L``; also the stride, so the windows do not overlap and
        the scores stay exchangeable.
    min_train_frac : float
        Earliest origin as a fraction of ``T0``, floored at
        :data:`MIN_TRAIN_PERIODS` absolute periods.

    Returns
    -------
    range
        The origins, empty when the pre-period admits none.
    """
    start = max(MIN_TRAIN_PERIODS, int(pre_periods * float(min_train_frac)))
    return range(start, int(pre_periods) - int(horizon) + 1, int(horizon))


def _rolling_origin_blocks(
    y: np.ndarray,
    Y0: np.ndarray,
    pre_periods: int,
    horizon: int,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    min_train_frac: float,
) -> list:
    """One array of ``horizon`` out-of-sample per-period errors per origin.

    The refit at each origin sees only periods strictly before it, so the errors
    carry no in-sample optimism. Callers reduce these however their band needs:
    :func:`rolling_origin_block_sums` sums each block into a conformity score,
    :func:`rolling_origin_period_errors` concatenates them.
    """
    blocks = []
    for origin in origin_schedule(pre_periods, horizon, min_train_frac):
        w = np.asarray(weight_fn(np.arange(origin)), dtype=float).ravel()
        block = slice(origin, origin + int(horizon))
        blocks.append(np.asarray(y[block] - Y0[block] @ w, dtype=float))
    return blocks


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
    blocks = _rolling_origin_blocks(y, Y0, pre_periods, horizon, weight_fn,
                                    min_train_frac)
    return np.asarray([float(np.sum(b)) for b in blocks], dtype=float)


def rolling_origin_period_errors(
    y: np.ndarray,
    Y0: np.ndarray,
    pre_periods: int,
    horizon: int,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    *,
    min_train_frac: float = 0.3,
) -> np.ndarray:
    """The same pass, kept per period: the blocks concatenated in origin order.

    :func:`rolling_origin_block_sums` reduces each window to the sum of its
    residuals, which is the conformity score a split-conformal band needs. A
    resampled band needs what the sum destroys -- the ordering of the periods
    inside a window, and across consecutive windows -- so this returns them.

    Because origins step by a whole horizon and are visited in increasing order,
    the result is a contiguous stretch of out-of-sample periods: origin ``o``
    scores ``o .. o + L - 1`` and the next origin picks up at ``o + L``. Serial
    correlation in the returned series is therefore the panel's own, and a block
    bootstrap over it means something.

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
        indexed by ``keep_idx``, as in :func:`rolling_origin_block_sums`.
    min_train_frac : float, optional
        Earliest origin as a fraction of ``T0`` (default ``0.3``), floored at
        ``MIN_TRAIN_PERIODS`` absolute periods.

    Returns
    -------
    np.ndarray, shape ``(m * L,)``
        The per-period errors, in time order. Empty when the pre-period admits no
        origin.

    Raises
    ------
    MlsynthConfigError
        If ``horizon`` is not a positive integer, or ``min_train_frac`` is not a
        number in ``[0, 1]``.
    MlsynthDataError
        If ``Y0`` does not align with ``y``, or ``pre_periods`` exceeds the sample.
    """
    from ...exceptions import MlsynthConfigError, MlsynthDataError

    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) \
            or int(horizon) < 1:
        raise MlsynthConfigError(
            f"horizon must be an integer of at least 1; got {horizon!r}."
        )
    if isinstance(min_train_frac, bool) \
            or not isinstance(min_train_frac, (int, float, np.floating)) \
            or not 0.0 <= float(min_train_frac) <= 1.0:
        raise MlsynthConfigError(
            f"min_train_frac must be a number in [0, 1]; got {min_train_frac!r}."
        )

    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    if Y0.ndim != 2 or Y0.shape[0] != y.size:
        raise MlsynthDataError(
            f"Y0 must align with y on the time axis: y has {y.size} periods, Y0 "
            f"has shape {Y0.shape}."
        )
    if isinstance(pre_periods, bool) or not isinstance(pre_periods, (int, np.integer)) \
            or int(pre_periods) < 1:
        raise MlsynthConfigError(
            f"pre_periods must be an integer of at least 1; got {pre_periods!r}."
        )
    if int(pre_periods) > y.size:
        raise MlsynthDataError(
            f"pre_periods ({int(pre_periods)}) exceeds the {y.size} periods "
            "supplied; there is no pre-period to calibrate on."
        )

    blocks = _rolling_origin_blocks(y, Y0, int(pre_periods), int(horizon),
                                    weight_fn, float(min_train_frac))
    if not blocks:
        return np.asarray([], dtype=float)
    return np.concatenate(blocks)
