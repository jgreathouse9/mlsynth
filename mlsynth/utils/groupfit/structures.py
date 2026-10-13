"""What the two-group regression is, as data.

Both records are frozen. ``GroupSums`` is the regression's sufficient statistics
and ``TwoGroupFit`` is what fitting them produces; everything either one carries
is something a caller would otherwise recompute with a second pass over the
window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupSums:
    """The centred sums of squares and cross products of two group aggregates.

    Every scalar the fit, the covariance and the prediction variance need is a
    function of these six numbers, so they are computed once and passed along.
    Centred about the mean: the uncentred forms are the naive variance and
    covariance formulas and they cancel at the scale a group aggregate has.

    ``s_xx > 0`` is the regression's identification condition, and
    :func:`~mlsynth.utils.groupfit.sums.is_identified` is the name for it.
    """

    n: int
    sum_x: float
    sum_y: float
    s_xx: float
    s_xy: float
    s_yy: float

    @property
    def x_bar(self) -> float:
        """The regressor's mean over the fitting window."""
        return self.sum_x / self.n

    @property
    def y_bar(self) -> float:
        """The response's mean over the fitting window."""
        return self.sum_y / self.n


@dataclass(frozen=True)
class TwoGroupFit:
    """``y_t = alpha + beta x_t + eps_t`` fitted on two group aggregates.

    ``resid`` is the fitting window's residual series, carried because the
    diagnostics that read it -- CUSUM, Durbin-Watson, a Breusch-Godfrey
    regression, a long-run variance -- would otherwise each form it again, and
    because ``sigma_sq`` is computed from it so the two cannot disagree.
    """

    alpha: float
    beta: float
    sigma_sq: float
    df: int
    resid: np.ndarray
    sums: GroupSums

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The fitted line at any regressor values, in or out of the window."""
        return self.alpha + self.beta * np.asarray(x, dtype=float)
