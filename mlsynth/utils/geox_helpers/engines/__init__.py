"""Pluggable scoring engines for the GEOX design.

Everything in the design except the fit and the p-value is estimator-independent
-- candidate nomination, the backtest windows, effect injection, power, MDE,
ranking, the constraint layer, the plots. This package is the seam that makes
that explicit, following the dispatcher shape PDA and SPILLSYNTH already use: a
subpackage per engine exposing uniform module-level functions, selected by a
config string.

An engine supplies five things.

``fit_once(y, Y0, n_pre, start, end, n_tr) -> EngineFit``
    Fit on the pre-period and predict across the whole panel.

``att(fit, y, start, end) -> float``
    Mean gap over the treatment window. Shared arithmetic, but it hangs off the
    engine so an estimator reporting on a different scale can override it.

``sweep_p_values(fit, y, Y0, n_pre, start, end, effect_sizes, ...) -> dict``
    The whole effect grid at once. The grid belongs to the engine because the
    two inference procedures differ in what they can hoist: a placebo standard
    error is invariant to the injected effect and is drawn once for the grid,
    while a conformal p-value re-permutes against the treated series and has to
    be recomputed per effect size. Keeping the grid here keeps that cost
    decision inside the module that owns it.

    The dict carries ``tau``, ``p_value`` and ``sigma`` (``None`` where the
    procedure has no standard error), plus ``tau0`` -- the ATT with nothing
    injected. ``tau0`` is the backtest's estimation error: the injected truth is
    ``e * mean(y[post])`` and the estimate is ``tau0 + e * mean(y[post])``, so
    the difference is ``tau0`` at every effect size. It is returned instead of
    being read off the grid because the grid is user-supplied and need not
    contain zero.

``point_inference(fit, y, Y0, n_pre, start, end, ...) -> (p_value, details)``
    A single window's test, for the realized readout.

``detection_boundary(fit, y, Y0, n_pre, start, end, *, alpha, ...) -> (up, down)``
    The effect sizes at which this backtest starts detecting, in each
    direction. Closed form where the p-value is analytic in the effect, and
    ``(nan, nan)`` where it is not -- reported absent instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from ....exceptions import MlsynthConfigError


@dataclass
class EngineFit:
    """One engine's fit on a pseudo-experiment's pre-period.

    The fields every engine supplies. ``time_weights`` is ``None`` for
    estimators that weight donors alone; ``extras`` carries whatever is specific
    to the engine (SDID's ridge ``zeta``, an augmented estimator's intercept and
    penalty) so the shared pipeline never has to know which is which.
    """

    counterfactual: np.ndarray
    donor_weights: np.ndarray
    pre_rmspe: float
    scaled_l2: float
    time_weights: Optional[np.ndarray] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Engine:
    """One resolved scoring engine: its name and its five functions."""

    name: str
    fit_once: Callable[..., EngineFit]
    att: Callable[..., float]
    sweep_p_values: Callable[..., Dict[str, Any]]
    point_inference: Callable[..., Any]
    detection_boundary: Callable[..., Any]


def placebo_detection_boundary(att_0: float, baseline: float, sigma, alpha: float):
    """Effects at which a placebo-tested backtest starts detecting.

    Detection is ``2(1 - Phi(|tau| / sigma)) < alpha`` and the analytic shortcut
    gives ``tau = att_0 + e * baseline``, linear in the effect, so the crossings
    solve directly:

        up   = ( z * sigma - att_0) / baseline
        down = (-z * sigma - att_0) / baseline

    with ``z`` the two-sided normal critical value. ``att_0`` is the backtest's
    own placebo effect and is generally nonzero, so the interval sits off centre
    and the two directions differ -- a design already drifting negative needs
    less of a push downward than upward.
    """
    from scipy.stats import norm

    if sigma is None or not np.isfinite(sigma) or sigma <= 0 or baseline == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return ((z * sigma - att_0) / baseline, (-z * sigma - att_0) / baseline)


def placebo_interval(att_value: float, sigma, alpha: float):
    """``att +/- z sigma``, the interval the placebo test inverts.

    Detection is ``2(1 - Phi(|att| / sigma)) < alpha``, so the set of nulls the
    test does not reject is exactly this interval -- excluding zero and
    rejecting are the same statement, which is what keeps the reported interval
    and the reported p-value from being two different tests.

    ``(nan, nan)`` when sigma is unusable, so an interval is absent instead of
    degenerate.
    """
    from scipy.stats import norm

    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(att_value - z * sigma), float(att_value + z * sigma)


def resolve_engine(name: str) -> Engine:
    """The engine registered under ``name``.

    Raises
    ------
    MlsynthConfigError
        If ``name`` is not a registered engine.
    """
    if name == "sdid":
        from .sdid import ENGINE as _sdid
        return _sdid
    if name == "augsynth":
        from .augsynth import ENGINE as _augsynth
        return _augsynth
    raise MlsynthConfigError(
        f"unknown engine {name!r}; available engines are {sorted(ENGINE_NAMES)}.")


ENGINE_NAMES = frozenset({"sdid", "augsynth"})

__all__ = ["Engine", "EngineFit", "ENGINE_NAMES", "resolve_engine",
           "placebo_interval", "placebo_detection_boundary"]
