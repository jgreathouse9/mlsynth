"""Sieve bases and the diversified weights they build.

ATEL approximates the time-varying factor loading by a sieve in the observed
covariates: the loading on factor j at time t is a smooth function of
:math:`X_{it}`, and the sieve replaces that function by a finite basis. The
basis values themselves become the diversified weights
:math:`W_{it}^{(j)}` that :mod:`.factors` averages donor outcomes against.

Three bases follow Lee (2026) and the author's toolbox: B-spline (the paper's
recommendation), trigonometric, and polynomial.

One layout detail decides which weight series the projection sees.
:func:`construct_weights` emits ``J * P`` blocks of ``T`` columns each, ordered
by (basis j, covariate p) with p inner, and :func:`~.factors.diversified_factors`
consumes the first ``J`` of them. With ``P`` covariates the consumed set is a
complete set of (basis, covariate) pairs exactly when ``J`` is a multiple of
``P``; :class:`~.config.ATELConfig` refuses the other cases, because there the
estimate depends on the order the covariates are named.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from ...exceptions import MlsynthConfigError, MlsynthDataError

__all__ = [
    "bspline_weights",
    "trig_weights",
    "poly_weights",
    "basis_values",
    "construct_weights",
]


def _check_width(J: int) -> None:
    if not isinstance(J, (int, np.integer)) or J <= 1:
        raise MlsynthConfigError(
            f"A sieve basis needs a width of at least 2; got {J!r}."
        )


def bspline_weights(x: np.ndarray, J: int) -> np.ndarray:
    """B-spline basis of width ``J`` evaluated at ``x``.

    The spline order and the breakpoints follow the width: ``J = 2`` is linear
    with no interior knot, ``J = 3`` quadratic with none, and ``J >= 4`` is
    cubic over ``J - 2`` breakpoints spanning the range of ``x``. The order
    therefore changes with ``J`` even when the projection goes on to consume
    only the first few basis functions.

    Parameters
    ----------
    x : np.ndarray
        Covariate values, pooled over every unit and period.
    J : int
        Basis width, at least 2.

    Returns
    -------
    np.ndarray
        Shape ``(x.size, J)``; rows sum to one.

    Raises
    ------
    MlsynthConfigError
        If ``J < 2``.
    MlsynthDataError
        If ``x`` is constant, which leaves the knot vector degenerate.
    """
    _check_width(J)
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = float(x.min()), float(x.max())
    if not np.isfinite([lo, hi]).all():
        raise MlsynthDataError("Covariate values must be finite.")
    if hi <= lo:
        raise MlsynthDataError(
            "A B-spline basis needs a covariate with range; this one is "
            f"constant at {lo}."
        )
    if J == 2:
        order, n_breaks = 2, 2
    elif J == 3:
        order, n_breaks = 3, 2
    else:
        order, n_breaks = 4, J - 2
    breaks = (
        np.array([lo, hi]) if n_breaks == 2 else np.linspace(lo, hi, n_breaks)
    )
    knots = np.concatenate(
        [np.repeat(breaks[0], order), breaks[1:-1], np.repeat(breaks[-1], order)]
    )
    return BSpline.design_matrix(x, knots, order - 1, extrapolate=False).toarray()


def trig_weights(x: np.ndarray, J: int) -> np.ndarray:
    """Trigonometric basis: an intercept then ``floor((J-1)/2)`` cos/sin pairs.

    An intercept plus complete pairs fills an odd number of columns, so an even
    width leaves the last column at zero. Whether such a column reaches the
    projection depends on ``J`` and the covariate count together, and
    :mod:`.factors` refuses a consumed block that is identically zero.
    """
    _check_width(J)
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = float(x.min()), float(x.max())
    z = np.full(x.size, 0.5) if hi <= lo else (x - lo) / (hi - lo)
    W = np.zeros((x.size, J))
    W[:, 0] = 1.0
    col = 1
    for k in range(1, (J - 1) // 2 + 1):
        if col < J:
            W[:, col] = np.cos(2.0 * np.pi * k * z)
            col += 1
        if col < J:
            W[:, col] = np.sin(2.0 * np.pi * k * z)
            col += 1
    return W


def poly_weights(x: np.ndarray, J: int) -> np.ndarray:
    """Monomial basis ``[x, x^2, ..., x^J]``, with no intercept."""
    _check_width(J)
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x ** (j + 1) for j in range(J)])


_BASES = {
    "bspline": bspline_weights,
    "trigonometric": trig_weights,
    "polynomial": poly_weights,
}


def basis_values(x: np.ndarray, J: int, basis: str) -> np.ndarray:
    """Dispatch to one of the three sieve bases."""
    try:
        fn = _BASES[basis]
    except KeyError:
        raise MlsynthConfigError(
            f"Unknown basis {basis!r}; choose one of {sorted(_BASES)}."
        ) from None
    return fn(x, J)


def construct_weights(X: np.ndarray, J: int, basis: str) -> np.ndarray:
    """Diversified weights for every unit, shape ``(N + 1, T * J * P)``.

    The basis is fit on each covariate pooled over every unit and period, the
    treated unit included, so the treated unit's covariates move the knots.
    Blocks are ordered by (basis j, covariate p) with p inner.

    Parameters
    ----------
    X : np.ndarray
        Covariate cube ``(N + 1, T, P)``. A 2-D array is read as ``P = 1``.
    J : int
        Sieve width.
    basis : {"bspline", "trigonometric", "polynomial"}

    Returns
    -------
    np.ndarray
        Shape ``(N + 1, T * J * P)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.ndim != 3:
        raise MlsynthDataError(
            f"The covariate cube must be 2-D or 3-D; got {X.ndim} dimensions."
        )
    n_units, T, P = X.shape
    if P == 0:
        raise MlsynthDataError(
            "ATEL builds its diversified weights from covariates, so at least "
            "one covariate is required."
        )
    if not np.isfinite(X).all():
        raise MlsynthDataError("The covariate cube contains non-finite values.")
    per_cov = [basis_values(X[:, :, p].ravel(order="F"), J, basis) for p in range(P)]
    blocks = [
        per_cov[p][:, j].reshape(n_units, T, order="F")
        for j in range(J)
        for p in range(P)
    ]
    return np.hstack(blocks)
