"""Typed, NumPy-first result containers for Partially Pooled SCM (staggered).

PPSCM ports augsynth::multisynth (Ben-Michael, Feller & Rothstein 2022): a
partially-pooled synthetic control for staggered adoption that interpolates,
via ``nu``, between a separate SCM per treated unit (``nu`` small) and a fully
pooled SCM (``nu`` large), on top of two-way fixed effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class PPSCMInputs:
    """Preprocessed staggered panel (the only pandas touchpoint is ``setup``).

    Parameters
    ----------
    Xy : np.ndarray
        Full outcome matrix, shape ``(n, T)`` (units x all periods).
    trt : np.ndarray
        Adoption index per unit (position in ``time_labels``); ``inf`` for
        never-treated controls.
    n_pre : int
        Number of pre-treatment periods (columns before the last adoption).
    time_labels : np.ndarray
        Sorted time labels, length ``T``.
    units : np.ndarray
        Unit labels, length ``n``.
    outcome : str
        Outcome column name.
    intervention_time : Any
        The last adoption time (pre/post split point).
    """

    Xy: np.ndarray
    trt: np.ndarray
    n_pre: int
    time_labels: np.ndarray
    units: np.ndarray
    outcome: str
    intervention_time: Any

    @property
    def n(self) -> int:
        return int(self.Xy.shape[0])

    @property
    def treated_units(self) -> np.ndarray:
        return self.units[np.isfinite(self.trt)]

    @property
    def control_units(self) -> np.ndarray:
        return self.units[~np.isfinite(self.trt)]


@dataclass(frozen=True)
class PPSCMDesign:
    """The fitted design: pooling level and balance diagnostics."""

    nu_used: float
    lam: float
    fixedeff: bool
    time_cohort: bool
    n_leads: int
    n_lags: int
    global_l2: float
    ind_l2: float
    scaled_global_l2: float
    scaled_ind_l2: float

    @property
    def pct_improve_global(self) -> float:
        return 100.0 * (1.0 - self.scaled_global_l2)

    @property
    def pct_improve_ind(self) -> float:
        return 100.0 * (1.0 - self.scaled_ind_l2)


@dataclass(frozen=True)
class PPSCMEventStudy:
    """Relative-time (time-since-treatment) average ATT path."""

    horizons: np.ndarray          # 0, 1, ..., n_leads-1
    tau: np.ndarray               # n1-weighted average effect per horizon
    se: np.ndarray
    ci: np.ndarray                # (H, 2)


@dataclass(frozen=True)
class PPSCMInference:
    """Overall (post-period average) ATT and its inference."""

    att: float
    se: float
    ci: Tuple[float, float]
    method: str


@dataclass(frozen=True)
class PPSCMResults:
    """Top-level container returned by :meth:`mlsynth.PPSCM.fit`."""

    inputs: PPSCMInputs
    design: PPSCMDesign
    event_study: PPSCMEventStudy
    inference: PPSCMInference
    donor_weights: Dict[Any, Dict[Any, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def att(self) -> float:
        return self.inference.att

    @property
    def nu(self) -> float:
        return self.design.nu_used
