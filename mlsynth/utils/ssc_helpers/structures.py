"""Frozen dataclasses for the SSC (Staggered Synthetic Control) estimator.

Cao, Lu & Wu (2026), *Synthetic Control Inference for Staggered Adoption*
(The Econometrics Journal). SSC estimates heterogeneous, dynamic treatment
effects under staggered adoption by modelling each unit's untreated outcome as
an intercept plus a simplex synthetic control on **all other units** (so
not-yet-treated units are valid donors), then jointly estimating every
unit x time effect by GLS and aggregating to event-time / overall ATT. Inference
is Andrews' (2003) end-of-sample stability test, calibrated on pre-treatment
residual windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SSCInputs:
    """Preprocessed staggered panel for SSC.

    Attributes
    ----------
    Y : np.ndarray
        Outcomes, shape ``(N, T1)`` (``T1 = T0 + S``).
    D : np.ndarray
        Treatment indicators, shape ``(N, T1)`` (``1`` where treated;
        absorbing: once 1, stays 1).
    T0 : int
        Number of "clean" pre-treatment periods (before *any* unit is treated);
        treatment first appears at column ``T0``.
    unit_names : list
    time_labels : np.ndarray
    treated_idx : np.ndarray
        Indices of ever-treated units.
    adoption : np.ndarray
        Per-unit first-treated column index (``-1`` for never-treated).
    """

    Y: np.ndarray
    D: np.ndarray
    T0: int
    unit_names: List[Any]
    time_labels: np.ndarray
    treated_idx: np.ndarray
    adoption: np.ndarray

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T1(self) -> int:
        return self.Y.shape[1]

    @property
    def S(self) -> int:
        """Number of post-treatment periods (from first adoption onward)."""
        return self.Y.shape[1] - self.T0

    @property
    def K(self) -> int:
        """Number of treated unit-period cells (length of ``tau``)."""
        return int(self.D[:, self.T0:].sum())


@dataclass(frozen=True)
class SSCInference:
    """Andrews end-of-sample stability inference for an SSC aggregate.

    The reference distribution is formed from ``T0 - S`` pre-treatment residual
    windows (the placebo "effects" under the null); a band is the point estimate
    plus the lower/upper quantiles of that mean-zero distribution.

    Attributes
    ----------
    method : str
        ``"andrews_eos"``.
    alpha : float
        Two-sided level (e.g. 0.1 -> 90% band).
    n_placebo : int
        Number of pre-treatment placebo windows.
    """

    method: str
    alpha: float
    n_placebo: int


@dataclass(frozen=True)
class SSCBand:
    """Point estimate, prediction band and p-value for one aggregate effect.

    Attributes
    ----------
    label : Any
        Event time (int) or ``None`` for the overall ATT.
    point : float
        ``L @ tau_hat``.
    lower, upper : float
        End-of-sample band endpoints.
    p_value : float
        Two-sided p-value for ``H0: effect = 0`` (Andrews test).
    n_cells : int
        Number of treated cells entering this aggregate.
    """

    label: Any
    point: float
    lower: float
    upper: float
    p_value: float
    n_cells: int

    @property
    def significant(self) -> bool:
        return (self.lower > 0.0) or (self.upper < 0.0)

    @property
    def ci(self):
        return (self.lower, self.upper)


@dataclass(frozen=True)
class SSCResults:
    """Top-level container returned by :meth:`mlsynth.SSC.fit`.

    Attributes
    ----------
    inputs : SSCInputs
    tau : np.ndarray
        Per-treated-cell individual treatment effects, length ``K``.
    index : np.ndarray
        ``(K, 3)`` rows ``[post_period s (1-based), unit_index, event_time e
        (0-based)]`` aligning with ``tau``.
    att : float
        Overall ATT (mean of ``tau``).
    att_band : SSCBand
        Overall ATT with its end-of-sample band and p-value.
    event_att : dict
        ``{event_time e: ATT_e}`` (event-study point estimates).
    event_bands : dict
        ``{event_time e: SSCBand}``.
    effects : np.ndarray
        ``(N, S)`` per-cell effects placed on the post-period grid (NaN where
        a unit is untreated at that post period).
    a_hat : np.ndarray
        Per-unit synthetic-control intercepts, length ``N``.
    B_hat : np.ndarray
        ``(N, N)`` synthetic-control weight matrix (row ``i`` = donor weights
        for unit ``i``; zero diagonal).
    weights : object
        :class:`mlsynth.config_models.WeightsResults` -- per-treated-unit donor
        weights plus a summary.
    residuals : np.ndarray
        ``(N, T0)`` pre-treatment prediction errors.
    inference : SSCInference, optional
    metadata : dict
    """

    inputs: SSCInputs
    tau: np.ndarray
    index: np.ndarray
    att: float
    att_band: Optional[SSCBand]
    event_att: Dict[int, float]
    event_bands: Dict[int, SSCBand]
    effects: np.ndarray
    a_hat: np.ndarray
    B_hat: np.ndarray
    weights: Any
    residuals: np.ndarray
    inference: Optional[SSCInference] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
