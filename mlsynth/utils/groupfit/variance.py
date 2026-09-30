"""The unscaled covariance, and the prediction variance built from it.

The prediction variance is the term that sets how wide a projected interval is,
and it is the part of this arithmetic where a mistake is silent: a wrong point
estimate usually looks wrong, a wrong interval width does not. Two estimators in
the library form it independently, which is the reason this module exists.
"""
from __future__ import annotations

import numpy as np

from .structures import GroupSums
from .sums import _require_identified


def unscaled_cov(sums: GroupSums) -> np.ndarray:
    """``(X'X)^-1`` for ``X = [1, x]``, in closed form about the mean.

    The Gram matrix is two by two, so the inverse is three divisions:

        [[1/n + xbar^2 / S_xx,  -xbar / S_xx],
         [       -xbar / S_xx,     1 / S_xx]]

    Written about ``S_xx`` and not about the determinant
    ``n sum(x^2) - (sum x)^2``. Maximum relative error over 200 draws of 90
    periods against the same formula in float128:

        level    spread   cond(X'X)   centred    determinant   pinv
        4.4e4    5e2      2.7e13      7.3e-16    4.9e-12       2.4e-07
        4.4e4    1.1e4    5.7e10      6.4e-16    1.4e-14       6.8e-10
        1e6      1e2      1.5e20      7.7e-16    1.0e-07       1.0
        1e6      1e0      1.0e24      7.3e-16    1.4e-03       1.0

    A decomposition is the worst of the three: ``np.linalg.pinv`` discards a
    singular value below ``2 eps sigma_max``, which on two columns is the slope's
    variance entire, so ``v_beta`` comes back zero and a projected interval stops
    responding to where the test window sits.
    """
    _require_identified(sums)
    n, s_xx, x_bar = sums.n, sums.s_xx, sums.x_bar
    return np.array([[1.0 / n + x_bar * x_bar / s_xx, -x_bar / s_xx],
                     [-x_bar / s_xx, 1.0 / s_xx]])


def prediction_variance(sums: GroupSums, x_bar):
    """``[1, x_bar]' (X'X)^-1 [1, x_bar]``, the variance of a fitted value.

    ``x_bar`` may be one regressor value or an array of them, and the result has
    its shape. A caller projecting over a growing window passes the running mean
    at every horizon and gets the whole profile in one call.

    Expanding the quadratic form collapses the three entries of the covariance
    into two terms::

        1 / n + (x_bar - xbar_pre)^2 / S_xx

    which is the form used here. It says what the quantity means: the fitted
    value is most precise at the window's own mean and loses precision with the
    square of how far the point being predicted sits from it, scaled by how much
    the regressor moved. Computing it this way also avoids forming the
    covariance and contracting it, which is three divisions and two products of
    numbers that nearly cancel.

    Both aggregations give the same number: the group size scales ``S_xx`` by its
    square and the squared deviation by the same factor.
    """
    _require_identified(sums)
    gap = np.asarray(x_bar, dtype=float) - sums.x_bar
    out = 1.0 / sums.n + gap * gap / sums.s_xx
    return float(out) if out.ndim == 0 else out
