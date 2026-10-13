"""Least squares on the two group aggregates, from the centred sums.

One regressor and an intercept, so the normal equations have a closed form and
nothing is solved. The two functions are separate because a caller that has
already formed the sums -- to check identification, or because it is scoring many
candidate groupings against one response -- should not form them again.
"""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError
from .structures import GroupSums, TwoGroupFit
from .sums import _require_identified, group_sums

#: Two parameters, so a window of two leaves no residual and no scale.
MIN_WINDOW = 3


def fit_on_sums(sums: GroupSums, y: np.ndarray, x: np.ndarray) -> TwoGroupFit:
    """Fit ``y = alpha + beta x`` given sums already computed from ``y`` and ``x``.

    Requires an identified design; see
    :func:`~mlsynth.utils.groupfit.sums.is_identified` for why the degenerate
    case is the caller's to decide.

    The residual is formed as ``(y - ybar) - beta (x - xbar)`` and not as
    ``y - (alpha + beta x)``. The two are the same residual on paper. The second
    subtracts two numbers of the regressor's own magnitude, where the first works
    in the deviations, which is the whole of the signal when a group aggregate's
    spread is small against its level: on a series near 1e6 varying in its
    eleventh significant figure the uncentred form returns a residual variance
    that is noise, and the centred one returns the right number.
    """
    if sums.n < MIN_WINDOW:
        raise MlsynthDataError(
            f"fitting an intercept and a slope needs at least {MIN_WINDOW} "
            f"periods to leave a residual scale; the window has {sums.n}.")
    _require_identified(sums)
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    beta = sums.s_xy / sums.s_xx
    alpha = sums.y_bar - beta * sums.x_bar
    resid = (y - sums.y_bar) - beta * (x - sums.x_bar)
    df = sums.n - 2
    return TwoGroupFit(alpha=alpha, beta=beta,
                       sigma_sq=float(resid @ resid) / df, df=df,
                       resid=resid, sums=sums)


def fit_two_group(y: np.ndarray, x: np.ndarray) -> TwoGroupFit:
    """Form the sums and fit, for a caller with nothing to reuse."""
    return fit_on_sums(group_sums(y, x), y, x)
