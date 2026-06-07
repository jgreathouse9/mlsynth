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
    return float(1.0 - (r @ r) / denom) if denom != 0 else float("nan")
