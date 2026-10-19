"""How many blank and fitting periods MAREX's inference needs.

With ``inference=True`` the pre-period splits into a fitting window, where the
design is chosen, and a blank window held out of it (Abadie and Zhao's ``E``
and ``B``). Two floors decide the default split.

The blank floor. The pointwise band's half-width is the
``ceil((Tb + 1)(1 - alpha))``-th order statistic of the absolute blank-period
effects, the split-conformal quantile (Lei et al. 2018). It exists only when
that rank is at most ``Tb``, which needs ``Tb >= ceil(1 / alpha) - 1``: 19 blank
periods at ``alpha = 0.05``, 9 at 0.10. Below it the band is unbounded.

The fitting floor, ``max(2 r, 10)``, with ``r`` the effective rank of the
pre-period panel. Abadie and Zhao's Assumption 2 asks for at least as many
fitting periods as factors. A design fitted on fewer periods than it has
factors to match interpolates, and the blank effects then describe the
interpolation instead of the out-of-sample error: on the panels of #744, five
fitting periods fitted the pre-period exactly and left the cumulative
false-positive rate at 0.38 against a nominal 0.05.

``r`` is Ahn and Horenstein's (2013) eigenvalue-ratio estimate: the ``k`` that
maximises ``mu_k / mu_(k+1)`` among the eigenvalues of the panel with unit and
period means removed. A share-of-variance threshold counts noise dimensions as
factors once the noise is a visible part of the variance; the ratio looks for
the gap after the factors and does not.

References
----------
Ahn, S. C. and A. R. Horenstein (2013). Eigenvalue ratio test for the number of
factors. Econometrica 81(3), 1203-1227.

Lei, J., M. G'Sell, A. Rinaldo, R. J. Tibshirani and L. Wasserman (2018).
Distribution-free predictive inference for regression. Journal of the American
Statistical Association 113(523), 1094-1111.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import MlsynthConfigError

#: Largest factor count the eigenvalue ratio considers.
KMAX = 8
#: Fewest fitting periods the default leaves, whatever the rank.
MIN_FIT_PERIODS = 10
#: Share of the pre-period the default blank window aims for.
BLANK_SHARE = 0.3


def min_blank_periods(alpha: float) -> int:
    """Fewest blank periods at which the split-conformal quantile exists.

    Computed with the arithmetic of
    :func:`~mlsynth.utils.conformal.quantile.split_conformal_quantile`, so the
    two agree at every level, including where ``1 / alpha`` is an integer.
    """
    n = max(int(math.floor(1.0 / alpha)) - 2, 1)
    while int(np.ceil((n + 1) * (1.0 - alpha))) > n:
        n += 1
    return n


def effective_rank(Y_pre, kmax: int = KMAX) -> int:
    """Number of factors in a ``(units, periods)`` panel, by eigenvalue ratio.

    Unit and period means are removed first: a common trend is fitted by any
    weights that sum to one, so it is not a factor the fitting window has to
    resolve. Removing both means leaves at most ``m = min(N, T) - 1``
    dimensions. A panel whose centred rank is below ``m`` has exact low rank,
    and that rank is returned; otherwise the ratio picks the gap among the first
    ``min(kmax, m - 1)`` eigenvalues.
    """
    Y = np.asarray(Y_pre, dtype=float)
    N, T = Y.shape
    m = min(N, T) - 1
    if m <= 0:
        return 0
    X = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + Y.mean()
    mu = np.linalg.svd(X, compute_uv=False)[:m] ** 2
    tol = (max(N, T) * np.finfo(float).eps * np.linalg.norm(Y)) ** 2
    rank = int(np.sum(mu > tol))
    if rank < m or m == 1:
        return rank
    k_hi = min(int(kmax), m - 1)
    return int(np.argmax(mu[:k_hi] / mu[1:k_hi + 1])) + 1


def default_blank_periods(Y_pre, alpha: float) -> int:
    """The default blank window for a ``(units, T0)`` pre-period panel.

    ``BLANK_SHARE`` of the pre-period, raised to the blank floor and capped so
    the fitting window keeps its floor.

    Raises
    ------
    MlsynthConfigError
        When the pre-period is shorter than the two floors together.
    """
    Y = np.asarray(Y_pre, dtype=float)
    T0 = Y.shape[1]
    tb = min_blank_periods(alpha)
    r = effective_rank(Y)
    te = max(2 * r, MIN_FIT_PERIODS)
    if T0 < tb + te:
        raise MlsynthConfigError(
            f"inference=True needs at least {tb} blank periods at alpha={alpha}, "
            f"the fewest at which the conformal quantile exists (ceil(1/alpha) - "
            f"1), and at least {te} fitting periods, max(2r, "
            f"{MIN_FIT_PERIODS}) with r={r} the effective rank of the pre-period "
            f"panel: {tb + te} pre-periods in all, and T0={T0}. A larger alpha "
            f"lowers the blank floor (alpha=0.1 needs {min_blank_periods(0.1)}). "
            f"An explicit blank_periods is honoured below the floor, with an "
            f"unbounded pointwise band beside the permutation p-values, which "
            f"do not need it.")
    return min(max(int(BLANK_SHARE * T0), tb), T0 - te)
