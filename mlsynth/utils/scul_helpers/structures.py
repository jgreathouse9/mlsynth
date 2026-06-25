"""Frozen, NumPy-first containers for Synthetic Control Using Lasso (SCUL).

Everything below is pure NumPy; the only DataFrame touchpoint is
:func:`mlsynth.utils.scul_helpers.setup.prepare_scul_inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)


@dataclass(frozen=True)
class SCULInputs:
    """Preprocessed, NumPy-only panel for the SCUL engine.

    Parameters
    ----------
    treated_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Length-``J`` donor-unit labels.
    y : np.ndarray
        Treated-unit outcome series, shape ``(T,)``.
    donor_matrix : np.ndarray
        Wide donor pool, shape ``(T, P)`` -- one block of all donor units per
        donor variable (the high-dimensional, multi-type pool).
    col_unit : np.ndarray
        Length-``P`` donor-unit label owning each column of ``donor_matrix``.
    col_variable : Sequence
        Length-``P`` variable name for each column.
    donor_outcome : np.ndarray
        Outcome series of each donor unit, shape ``(T, J)`` -- the placebo
        targets for inference.
    T0 : int
        Number of pre-treatment periods.
    time_labels : Sequence
        Length-``T`` period labels.
    metadata : dict
        Free-form provenance.
    """

    treated_name: Any
    donor_names: Sequence
    y: np.ndarray
    donor_matrix: np.ndarray
    col_unit: np.ndarray
    col_variable: Sequence
    donor_outcome: np.ndarray
    T0: int
    time_labels: Sequence
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_pool(self) -> int:
        return int(self.donor_matrix.shape[1])


@dataclass(frozen=True)
class SCULFit:
    """A single SCUL fit."""

    counterfactual: np.ndarray       # (T,)
    gap: np.ndarray                  # (T,)
    att: float
    weights: np.ndarray              # (P,)
    intercept: float
    ridge_lambda: float
    cohens_d: float                  # pre-period unit-free fit
    donor_weights: Dict[Any, float]  # nonzero pool columns -> weight
    p_value: Optional[float] = None
    n_placebo: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SCULResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SCUL.fit` (an
    :class:`~mlsynth.config_models.EffectResult`)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SCULInputs
    fit: SCULFit
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "SCULResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard
            return self
        fit = self.fit
        labels = np.asarray(self.inputs.time_labels)
        T0, T = self.inputs.T0, self.inputs.T
        ts = TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(fit.counterfactual, dtype=float),
            estimated_gap=np.asarray(fit.gap, dtype=float),
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None),
        )
        object.__setattr__(self, "effects", EffectsResults(att=float(fit.att)))
        object.__setattr__(self, "time_series", ts)
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in fit.donor_weights.items()}))
        object.__setattr__(self, "fit_diagnostics",
                           FitDiagnosticsResults(rmse_pre=float(fit.cohens_d)))
        if fit.p_value is not None:
            object.__setattr__(self, "inference", InferenceResults(
                p_value=fit.p_value, method="placebo"))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="SCUL",
            is_recommended=True,
            parameters_used={
                "ridge_lambda": float(fit.ridge_lambda),
                "cohens_d": float(fit.cohens_d),
                "n_pool": int(self.inputs.n_pool),
                "n_selected": int(np.sum(np.abs(fit.weights) > 1e-10)),
                "intercept": float(fit.intercept),
            },
        ))
        return self


SCULResults.model_rebuild()
