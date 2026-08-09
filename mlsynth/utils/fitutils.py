"""Vectorized goodness-of-fit / loss primitives.

Bite-sized, pure functions for the *loss* side of estimator reporting --
how well the counterfactual tracks the treated unit -- kept separate from the
treatment-*effect* primitives in :mod:`mlsynth.utils.effectutils`. Each is a
dot-product over a residual/outcome vector and returns a raw (unrounded)
value; callers round only for display.

Notation (Shi--Huang): ``r`` is a residual (gap) vector ``y - y_hat``.
"""

from __future__ import annotations

import numpy as np


def _ravel(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, dtype=float).ravel()


def rmse(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r / r.size)) if r.size else float("nan")


def std(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(v)) if v.size else float("nan")


def r_squared(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("nan")
    y_c = y - y.mean()
    denom = float(y_c @ y_c)
    # A series that is constant to machine precision does not center to
    # exactly zero: subtracting the mean leaves rounding residue of order
    # eps*|y| per element, so `denom != 0` lets a flat series through and
    # divides by ~1e-20. A flat pre-period is an ordinary panel -- a count
    # that stays at zero, a rate that does not move -- and the result was a
    # silent 1.0 or a value of order -1e19 depending on the residuals. Compare
    # against the noise floor of the centering itself instead.
    noise_floor = y.size * np.finfo(float).eps ** 2 * float(y @ y)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)
