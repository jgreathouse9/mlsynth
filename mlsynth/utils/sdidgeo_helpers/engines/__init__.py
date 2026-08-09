"""Pluggable scoring engines for the SDIDGEO design.

Everything in the design except the fit and the p-value is estimator-independent
-- candidate nomination, the backtest windows, effect injection, power, MDE,
ranking, the constraint layer, the plots. This package is the seam that makes
that explicit, following the dispatcher shape PDA and SPILLSYNTH already use: a
subpackage per engine exposing uniform module-level functions, selected by a
config string.

An engine supplies four things.

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

``point_inference(fit, y, Y0, n_pre, start, end, ...) -> (p_value, details)``
    A single window's test, for the realized readout.
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
    """One resolved scoring engine: its name and its four functions."""

    name: str
    fit_once: Callable[..., EngineFit]
    att: Callable[..., float]
    sweep_p_values: Callable[..., Dict[str, Any]]
    point_inference: Callable[..., Any]


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

__all__ = ["Engine", "EngineFit", "ENGINE_NAMES", "resolve_engine"]
