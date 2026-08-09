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


from mutmut.mutation.trampoline import wrap_in_trampoline as _mutmut_mutated, MutantDict
mutants_x__ravel__mutmut: MutantDict = {}  # type: ignore


@_mutmut_mutated(mutants_x__ravel__mutmut)
def _ravel(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, dtype=float).ravel()


def x__ravel__mutmut_orig(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, dtype=float).ravel()


def x__ravel__mutmut_1(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(None, dtype=float).ravel()


def x__ravel__mutmut_2(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, dtype=None).ravel()


def x__ravel__mutmut_3(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(dtype=float).ravel()


def x__ravel__mutmut_4(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, ).ravel()

mutants_x__ravel__mutmut['_mutmut_orig'] = x__ravel__mutmut_orig # type: ignore # mutmut generated
mutants_x__ravel__mutmut['x__ravel__mutmut_1'] = x__ravel__mutmut_1 # type: ignore # mutmut generated
mutants_x__ravel__mutmut['x__ravel__mutmut_2'] = x__ravel__mutmut_2 # type: ignore # mutmut generated
mutants_x__ravel__mutmut['x__ravel__mutmut_3'] = x__ravel__mutmut_3 # type: ignore # mutmut generated
mutants_x__ravel__mutmut['x__ravel__mutmut_4'] = x__ravel__mutmut_4 # type: ignore # mutmut generated
mutants_x_rmse__mutmut: MutantDict = {}  # type: ignore


@_mutmut_mutated(mutants_x_rmse__mutmut)
def rmse(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r / r.size)) if r.size else float("nan")


def x_rmse__mutmut_orig(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r / r.size)) if r.size else float("nan")


def x_rmse__mutmut_1(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = None
    return float(np.sqrt(r @ r / r.size)) if r.size else float("nan")


def x_rmse__mutmut_2(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(None)
    return float(np.sqrt(r @ r / r.size)) if r.size else float("nan")


def x_rmse__mutmut_3(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(None) if r.size else float("nan")


def x_rmse__mutmut_4(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(None)) if r.size else float("nan")


def x_rmse__mutmut_5(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r * r.size)) if r.size else float("nan")


def x_rmse__mutmut_6(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r / r.size)) if r.size else float(None)


def x_rmse__mutmut_7(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r / r.size)) if r.size else float("XXnanXX")


def x_rmse__mutmut_8(residuals: np.ndarray) -> float:
    """Root-mean-square error of a residual vector, ``sqrt(r . r / n)``."""
    r = _ravel(residuals)
    return float(np.sqrt(r @ r / r.size)) if r.size else float("NAN")

mutants_x_rmse__mutmut['_mutmut_orig'] = x_rmse__mutmut_orig # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_1'] = x_rmse__mutmut_1 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_2'] = x_rmse__mutmut_2 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_3'] = x_rmse__mutmut_3 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_4'] = x_rmse__mutmut_4 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_5'] = x_rmse__mutmut_5 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_6'] = x_rmse__mutmut_6 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_7'] = x_rmse__mutmut_7 # type: ignore # mutmut generated
mutants_x_rmse__mutmut['x_rmse__mutmut_8'] = x_rmse__mutmut_8 # type: ignore # mutmut generated
mutants_x_std__mutmut: MutantDict = {}  # type: ignore


@_mutmut_mutated(mutants_x_std__mutmut)
def std(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(v)) if v.size else float("nan")


def x_std__mutmut_orig(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(v)) if v.size else float("nan")


def x_std__mutmut_1(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = None
    return float(np.std(v)) if v.size else float("nan")


def x_std__mutmut_2(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(None)
    return float(np.std(v)) if v.size else float("nan")


def x_std__mutmut_3(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(None) if v.size else float("nan")


def x_std__mutmut_4(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(None)) if v.size else float("nan")


def x_std__mutmut_5(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(v)) if v.size else float(None)


def x_std__mutmut_6(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(v)) if v.size else float("XXnanXX")


def x_std__mutmut_7(values: np.ndarray) -> float:
    """Population standard deviation ``sqrt(Var)`` of a vector."""
    v = _ravel(values)
    return float(np.std(v)) if v.size else float("NAN")

mutants_x_std__mutmut['_mutmut_orig'] = x_std__mutmut_orig # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_1'] = x_std__mutmut_1 # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_2'] = x_std__mutmut_2 # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_3'] = x_std__mutmut_3 # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_4'] = x_std__mutmut_4 # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_5'] = x_std__mutmut_5 # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_6'] = x_std__mutmut_6 # type: ignore # mutmut generated
mutants_x_std__mutmut['x_std__mutmut_7'] = x_std__mutmut_7 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut: MutantDict = {}  # type: ignore


@_mutmut_mutated(mutants_x_r_squared__mutmut)
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


def x_r_squared__mutmut_orig(observed: np.ndarray, residuals: np.ndarray) -> float:
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


def x_r_squared__mutmut_1(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = None
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


def x_r_squared__mutmut_2(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(None)
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


def x_r_squared__mutmut_3(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = None
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


def x_r_squared__mutmut_4(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(None)
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


def x_r_squared__mutmut_5(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size != 0:
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


def x_r_squared__mutmut_6(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 1:
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


def x_r_squared__mutmut_7(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float(None)
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


def x_r_squared__mutmut_8(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("XXnanXX")
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


def x_r_squared__mutmut_9(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("NAN")
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


def x_r_squared__mutmut_10(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("nan")
    y_c = None
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


def x_r_squared__mutmut_11(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("nan")
    y_c = y + y.mean()
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


def x_r_squared__mutmut_12(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("nan")
    y_c = y - y.mean()
    denom = None
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


def x_r_squared__mutmut_13(observed: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination ``1 - r . r / (y_c . y_c)``.

    ``y_c`` is the centered observed vector; returns ``nan`` when the observed
    series is empty or has zero variance.
    """
    y = _ravel(observed)
    r = _ravel(residuals)
    if y.size == 0:
        return float("nan")
    y_c = y - y.mean()
    denom = float(None)
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


def x_r_squared__mutmut_14(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = None
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_15(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = y.size * np.finfo(float).eps ** 2 / float(y @ y)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_16(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = y.size / np.finfo(float).eps ** 2 * float(y @ y)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_17(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = y.size * np.finfo(float).eps * 2 * float(y @ y)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_18(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = y.size * np.finfo(None).eps ** 2 * float(y @ y)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_19(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = y.size * np.finfo(float).eps ** 3 * float(y @ y)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_20(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    noise_floor = y.size * np.finfo(float).eps ** 2 * float(None)
    if denom <= noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_21(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    if denom < noise_floor:
        return float("nan")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_22(observed: np.ndarray, residuals: np.ndarray) -> float:
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
        return float(None)
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_23(observed: np.ndarray, residuals: np.ndarray) -> float:
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
        return float("XXnanXX")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_24(observed: np.ndarray, residuals: np.ndarray) -> float:
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
        return float("NAN")
    return float(1.0 - (r @ r) / denom)


def x_r_squared__mutmut_25(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    return float(None)


def x_r_squared__mutmut_26(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    return float(1.0 + (r @ r) / denom)


def x_r_squared__mutmut_27(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    return float(2.0 - (r @ r) / denom)


def x_r_squared__mutmut_28(observed: np.ndarray, residuals: np.ndarray) -> float:
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
    return float(1.0 - (r @ r) * denom)

mutants_x_r_squared__mutmut['_mutmut_orig'] = x_r_squared__mutmut_orig # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_1'] = x_r_squared__mutmut_1 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_2'] = x_r_squared__mutmut_2 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_3'] = x_r_squared__mutmut_3 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_4'] = x_r_squared__mutmut_4 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_5'] = x_r_squared__mutmut_5 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_6'] = x_r_squared__mutmut_6 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_7'] = x_r_squared__mutmut_7 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_8'] = x_r_squared__mutmut_8 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_9'] = x_r_squared__mutmut_9 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_10'] = x_r_squared__mutmut_10 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_11'] = x_r_squared__mutmut_11 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_12'] = x_r_squared__mutmut_12 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_13'] = x_r_squared__mutmut_13 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_14'] = x_r_squared__mutmut_14 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_15'] = x_r_squared__mutmut_15 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_16'] = x_r_squared__mutmut_16 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_17'] = x_r_squared__mutmut_17 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_18'] = x_r_squared__mutmut_18 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_19'] = x_r_squared__mutmut_19 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_20'] = x_r_squared__mutmut_20 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_21'] = x_r_squared__mutmut_21 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_22'] = x_r_squared__mutmut_22 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_23'] = x_r_squared__mutmut_23 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_24'] = x_r_squared__mutmut_24 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_25'] = x_r_squared__mutmut_25 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_26'] = x_r_squared__mutmut_26 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_27'] = x_r_squared__mutmut_27 # type: ignore # mutmut generated
mutants_x_r_squared__mutmut['x_r_squared__mutmut_28'] = x_r_squared__mutmut_28 # type: ignore # mutmut generated
