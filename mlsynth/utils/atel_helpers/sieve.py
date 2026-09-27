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
    "hadamard_weights",
    "tile_unit_weights",
    "weight_conditioning",
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


def hadamard_weights(n_units: int, n_columns: int) -> np.ndarray:
    """Deterministic sign weights, Fan and Liao (2022) Section 4.4.

    The first column is all ones and column k alternates blocks of ``k - 1``
    ones and minus ones, which is their simulation choice (i). Being
    deterministic, these satisfy the independence half of Assumption 2.1 by
    construction, and the bounded-entries half exactly. What they cannot
    guarantee is the rank condition against the unobserved loadings, since they
    carry no information about them -- which is a reason to take more columns
    than factors, not fewer.

    Parameters
    ----------
    n_units : int
        Number of units the weights apply to.
    n_columns : int
        Number of weight series, at least 2.

    Returns
    -------
    np.ndarray
        Shape ``(n_units, n_columns)``, entries in ``{-1, +1}``.
    """
    if n_columns < 2:
        raise MlsynthConfigError(
            f"Hadamard weights need at least 2 columns; got {n_columns}."
        )
    if n_units < 1:
        raise MlsynthDataError(f"Need at least one unit; got {n_units}.")
    rows = np.arange(int(n_units))
    W = np.ones((int(n_units), int(n_columns)))
    for k in range(2, int(n_columns) + 1):
        W[:, k - 1] = np.where((rows // (k - 1)) % 2 == 0, 1.0, -1.0)
    return W


def tile_unit_weights(unit_weights: np.ndarray, n_periods: int) -> np.ndarray:
    """Lay unit-level weights out in the block layout the projection reads.

    A weight that does not move with time still has to be presented as one
    block of ``n_periods`` identical columns per weight series, because
    :func:`~.factors.diversified_factors` slices by period block.
    """
    W = np.asarray(unit_weights, dtype=float)
    if W.ndim != 2:
        raise MlsynthDataError("Unit weights must be 2-D (units, columns).")
    if n_periods < 1:
        raise MlsynthDataError(f"Need at least one period; got {n_periods}.")
    return np.hstack([np.repeat(W[:, [j]], n_periods, axis=1)
                      for j in range(W.shape[1])])


def weight_conditioning(
    donor_weights: np.ndarray, n_factors: int, n_periods: int
) -> float:
    """Smallest eigenvalue of ``W_t' W_t / N`` over the periods, Assumption 2.1(ii).

    Fan and Liao require the weights to be non-degenerate among themselves,
    uniformly. Evaluated on the cross-section actually projected at each period,
    this is a number the caller can read: near zero means two weight series are
    carrying the same information and the projection has fewer usable directions
    than it appears to.

    This says nothing about the rank condition ``rank(W'B / N) = r``, which
    involves the unobserved loadings and stays undiagnosable.
    """
    W = np.asarray(donor_weights, dtype=float)
    n_donors = W.shape[0]
    worst = np.inf
    for t in range(int(n_periods)):
        block = np.column_stack(
            [W[:, j * int(n_periods) + t] for j in range(int(n_factors))]
        )
        worst = min(worst, float(np.linalg.eigvalsh(block.T @ block / n_donors).min()))
    return worst
