"""Frozen, NumPy-first containers for Synthetic Control with Temporal Aggregation.

SCTA (Sun, Ben-Michael & Feller 2024) balances the treated unit against donors
on a stacked matching vector ``[aggregate block means | disaggregated pre
outcomes]``, weighted by a fixed diagonal ``V`` parameterised by ``nu``.
Everything below is pure NumPy; the only DataFrame touchpoint is
:func:`mlsynth.utils.scta_helpers.setup.prepare_scta_inputs`.
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
class SCTAInputs:
    """Preprocessed, NumPy-only panel for the SCTA engine.

    Parameters
    ----------
    treated_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Length-``J`` donor labels (column order of ``donor_matrix``).
    y : np.ndarray
        Treated-unit outcome series, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes, shape ``(T, J)`` (rows = periods).
    T0 : int
        Number of pre-treatment periods.
    block_length : int
        Aggregation block length ``K`` (high-frequency periods per aggregate).
    time_labels : Sequence
        Length-``T`` period labels.
    metadata : dict
        Free-form provenance.
    """

    treated_name: Any
    donor_names: Sequence
    y: np.ndarray
    donor_matrix: np.ndarray
    T0: int
    block_length: int
    time_labels: Sequence
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_donors(self) -> int:
        return int(self.donor_matrix.shape[1])

    @property
    def n_blocks(self) -> int:
        return int(self.T0 // self.block_length)


@dataclass(frozen=True)
class SCTAFit:
    """A single SCTA fit at a given ``nu``."""

    nu: float
    weights: np.ndarray              # (n_donors,)
    counterfactual: np.ndarray       # (T,)
    gap: np.ndarray                  # (T,)
    att: float
    pre_rmse: float                  # RMSE over the disaggregated pre-periods
    rmse_dis: float                  # disaggregated pre-period RMSE (frontier x)
    rmse_agg: float                  # aggregated pre-period RMSE (frontier y)
    donor_weights: Dict[Any, float]
    att_se: Optional[float] = None
    ci: Tuple[float, float] = (float("nan"), float("nan"))
    p_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SCTAResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SCTA.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: it lifts the headline
    ``fit`` into the standardized sub-models so the flat accessors resolve
    through the base contract, and keeps the imbalance-frontier diagnostic in
    :attr:`frontier`.

    Parameters
    ----------
    inputs : SCTAInputs
        Preprocessed panel.
    fit : SCTAFit
        The headline fit (at the scalar ``nu``).
    frontier : list of dict, optional
        One ``{"nu", "rmse_dis", "rmse_agg", "att"}`` point per requested
        ``nu`` when an imbalance frontier was traced.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SCTAInputs
    fit: SCTAFit
    frontier: Optional[List[Dict[str, float]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "SCTAResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard; effects is None at construction
            return self
        fit = self.fit
        labels = np.asarray(self.inputs.time_labels)
        T0 = self.inputs.T0
        T = self.inputs.T
        ts = TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(fit.counterfactual, dtype=float),
            estimated_gap=np.asarray(fit.gap, dtype=float),
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None),
        )
        inf = None
        if fit.att_se is not None or fit.p_value is not None:
            lo, hi = fit.ci
            inf = InferenceResults(
                standard_error=fit.att_se,
                ci_lower=None if np.isnan(lo) else float(lo),
                ci_upper=None if np.isnan(hi) else float(hi),
                p_value=fit.p_value,
                method=fit.metadata.get("inference_method"),
            )
        object.__setattr__(self, "effects", EffectsResults(att=float(fit.att)))
        object.__setattr__(self, "time_series", ts)
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in fit.donor_weights.items()}))
        object.__setattr__(self, "fit_diagnostics",
                           FitDiagnosticsResults(rmse_pre=float(fit.pre_rmse)))
        if inf is not None:
            object.__setattr__(self, "inference", inf)
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="SCTA",
            is_recommended=True,
            parameters_used={
                "nu": float(fit.nu),
                "block_length": int(self.inputs.block_length),
                "n_blocks": int(self.inputs.n_blocks),
                "augment": fit.metadata.get("augment"),
            },
        ))
        return self


SCTAResults.model_rebuild()
