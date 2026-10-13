"""Rolling-origin holdout: does the HCW fit survive periods it did not see?

HCW fits an unrestricted OLS on the pre-period and extrapolates it past the
treatment date. The pre-period fit statistic cannot say whether that
extrapolation is sound, because it is the quantity the regression minimised: a
four-point fit interpolates its four points whatever the underlying relation is.
Separating the two needs periods the regression did not see.

This slides an origin across the pre-period, refits on the periods strictly
before it, and scores the next ``horizon`` periods out of sample. Origins step
by a whole horizon, so the windows do not overlap. The summary is the ratio of
held-out to in-sample root mean squared error: near one, the fit extrapolates
about as well as it interpolates; much above one, the donor coefficients are
chasing the pre-period and the post-period counterfactual rests on nothing the
data has tested.

The construction is the same one :mod:`mlsynth.utils.conformal.scores` uses to
calibrate a cumulative band, with one deliberate difference. That module floors
the training block at :data:`~mlsynth.utils.conformal.scores.MIN_TRAIN_PERIODS`
(ten), because a band calibrated on a degenerate fit reports the degeneracy
instead of the method's error. A diagnostic wants the opposite: a pre-period too
short to extrapolate is the condition being reported, so the floor here is the
identification requirement alone -- enough rows to fit the widest donor model
with a residual degree of freedom.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError
from ..structures import PDAHoldout

#: Shortest training block, before the donor count is taken into account. An
#: intercept, one donor and one residual degree of freedom.
MIN_HOLDOUT_TRAIN = 3


def holdout_schedule(pre_periods: int, horizon: int,
                     min_train: int = MIN_HOLDOUT_TRAIN) -> range:
    """The origins a holdout pass scores, in increasing order.

    Parameters
    ----------
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    horizon : int
        Window length, and the stride, so windows do not overlap.
    min_train : int, optional
        Earliest origin, i.e. the shortest training block admitted.

    Returns
    -------
    range
        The origins; empty when the pre-period cannot spare a whole window.
    """
    return range(int(min_train), int(pre_periods) - int(horizon) + 1, int(horizon))


def _rmse(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float).ravel()
    return float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")


def rolling_origin_holdout(
    y: np.ndarray,
    X: np.ndarray,
    pre_periods: int,
    horizon: int,
    fit: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    *,
    min_train: Optional[int] = None,
) -> PDAHoldout:
    """Score the HCW fit on pre-periods it was not trained on.

    Parameters
    ----------
    y : np.ndarray, shape ``(T,)``
        Treated-unit outcome over the full sample.
    X : np.ndarray, shape ``(T, J)``
        Donor outcomes aligned to ``y``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    horizon : int
        Periods scored per origin.
    fit : callable
        ``fit(y, X, origin) -> counterfactual``, the estimator's own refit
        trained on periods ``[0, origin)`` and returning a counterfactual over
        every period. Exceptions from it propagate: an origin that cannot be fit
        is a finding, not a row to drop.
    min_train : int, optional
        Shortest training block. Defaults to
        ``max(MIN_HOLDOUT_TRAIN, J + 2)``, which leaves the widest donor model a
        residual degree of freedom at every origin.

    Returns
    -------
    PDAHoldout
        Per-period out-of-sample errors in time order, with the two root mean
        squared errors and their ratio.

    Raises
    ------
    MlsynthConfigError
        If ``horizon`` or ``pre_periods`` is not a positive integer.
    MlsynthDataError
        If ``X`` does not align with ``y``, or ``pre_periods`` exceeds the sample.

    Warns
    -----
    UserWarning
        When the pre-period admits no origin, so nothing is scored.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) \
            or int(horizon) < 1:
        raise MlsynthConfigError(
            f"horizon must be an integer of at least 1; got {horizon!r}.")
    if isinstance(pre_periods, bool) \
            or not isinstance(pre_periods, (int, np.integer)) \
            or int(pre_periods) < 1:
        raise MlsynthConfigError(
            f"pre_periods must be an integer of at least 1; got {pre_periods!r}.")

    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise MlsynthDataError(
            f"X must align with y on the time axis: y has {y.size} periods, X "
            f"has shape {X.shape}.")
    T0 = int(pre_periods)
    if T0 > y.size:
        raise MlsynthDataError(
            f"pre_periods ({T0}) exceeds the {y.size} periods supplied; there "
            "is no pre-period to hold out from.")

    horizon = int(horizon)
    floor = int(min_train) if min_train is not None \
        else max(MIN_HOLDOUT_TRAIN, X.shape[1] + 2)

    in_sample = _rmse(y[:T0] - np.asarray(fit(y, X, T0), dtype=float).ravel()[:T0])

    origins = list(holdout_schedule(T0, horizon, floor))
    if not origins:
        warnings.warn(
            f"The pre-period is too short to hold out from: {T0} periods, a "
            f"{floor}-period training floor and a horizon of {horizon} leave no "
            "origin to score, so the HCW extrapolation is untested.",
            UserWarning, stacklevel=2)
        return PDAHoldout(
            horizon=horizon, min_train=floor,
            origins=np.asarray([], dtype=int), errors=np.asarray([], dtype=float),
            held_out_rmse=float("nan"), in_sample_rmse=in_sample,
            ratio=float("nan"), n_origins=0)

    blocks = []
    for origin in origins:
        cf = np.asarray(fit(y, X, int(origin)), dtype=float).ravel()
        window = slice(int(origin), int(origin) + horizon)
        blocks.append(y[window] - cf[window])
    errors = np.concatenate(blocks)

    held_out = _rmse(errors)
    ratio = held_out / in_sample if in_sample > 0 else float("nan")
    return PDAHoldout(
        horizon=horizon, min_train=floor,
        origins=np.asarray(origins, dtype=int), errors=errors,
        held_out_rmse=held_out, in_sample_rmse=in_sample,
        ratio=float(ratio), n_origins=len(origins))
