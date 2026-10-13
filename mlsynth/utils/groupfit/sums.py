"""The regression's sufficient statistics, and whether they identify it."""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from .structures import GroupSums


def group_sums(y: np.ndarray, x: np.ndarray) -> GroupSums:
    """The centred sums of squares and cross products of ``y`` and ``x``.

    Centred in one pass about each series' own mean. The uncentred alternative,
    ``n sum(x^2) - (sum x)^2``, is the naive variance formula: on a group
    aggregate of 44,000 varying by 500 it carries nine digits of mean square
    against four of signal, and the difference shows up in the third significant
    figure of the slope's variance.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.size != x.size:
        raise MlsynthDataError(
            f"the response and the regressor cover different windows: "
            f"{y.size} and {x.size} period(s).")
    if y.size == 0:
        raise MlsynthDataError("the fitting window is empty.")
    n = int(y.size)
    sum_x, sum_y = float(x.sum()), float(y.sum())
    dx, dy = x - sum_x / n, y - sum_y / n
    return GroupSums(n=n, sum_x=sum_x, sum_y=sum_y, s_xx=float(dx @ dx),
                     s_xy=float(dx @ dy), s_yy=float(dy @ dy))


def is_identified(sums: GroupSums) -> bool:
    """True when the regressor varies, so the slope has a unique answer.

    ``S_xx`` is zero exactly when the regressor is constant through the window,
    and then ``beta = S_xy / S_xx`` is nothing over nothing. What to do in that
    case differs by estimator -- hold the slope at one, take the minimum-norm
    solution, refuse the design -- so this package reports the condition and
    every function in it requires that the condition holds.
    """
    return sums.s_xx > 0.0


def _require_identified(sums: GroupSums) -> None:
    """Raise unless the regressor varies. The message names the caller's choice."""
    if not is_identified(sums):
        raise MlsynthEstimationError(
            "the regressor is constant through the fitting window, so the "
            "slope is not identified: every slope fits equally well. Decide "
            "what to do about it at the call site -- hold the slope at one, "
            "take the minimum-norm solution, or refuse the design.")
