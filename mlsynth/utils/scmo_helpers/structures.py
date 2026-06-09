"""Frozen, NumPy-first containers for Synthetic Control with Multiple Outcomes.

SCMO (Tian-Lee-Panchenko 2024; Sun-Ben-Michael-Feller 2025) builds the
synthetic control by matching the treated unit to donors on a **matching
matrix** ``Z`` assembled from one or more related outcomes/predictors --
optionally across several pre-treatment periods -- rather than a single
outcome's long trajectory. Everything below is pure NumPy; the only
DataFrame touchpoint is :func:`mlsynth.utils.scmo_helpers.setup.prepare_scmo_inputs`.

Units and time are addressed through :class:`IndexSet` (immutable
label<->integer maps) so downstream code never reaches back into pandas.
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

# IndexSet currently lives in fast_scm_helpers.structure on this branch;
# on main it is re-homed to helperutils -- a one-line import swap at sync.
from ..fast_scm_helpers.structure import IndexSet


# Weighting schemes.
CONCATENATED = "concatenated"   # Tian-Lee-Panchenko: stack all standardized columns
AVERAGED = "averaged"           # Sun-Ben-Michael-Feller: match the average of outcomes
SEPARATE = "separate"           # conventional per-outcome baseline
MA = "MA"                       # model-average of concatenated & averaged


@dataclass(frozen=True)
class SCMOInputs:
    """Preprocessed, NumPy-only panel for the SCMO engine.

    Parameters
    ----------
    unit_index : IndexSet
        All ``N`` units; row order of ``Y`` and ``Z``.
    time_index : IndexSet
        All ``T`` periods; column order of ``Y``.
    treated_idx : int
        Row index (into ``unit_index``) of the treated unit.
    donor_idx : np.ndarray
        Row indices of the donor pool.
    Y : np.ndarray
        Primary-outcome panel, shape ``(N, T)`` (rows = units).
    T0 : int
        Number of pre-treatment periods.
    Z : np.ndarray
        Standardized matching matrix, shape ``(N, P)`` (rows = units).
    predictor_labels : Sequence
        Length-``P`` labels for the columns of ``Z``.
    metadata : dict
        Free-form provenance (spec, demean flag, dropped columns, ...).
    """

    unit_index: IndexSet
    time_index: IndexSet
    treated_idx: int
    donor_idx: np.ndarray
    Y: np.ndarray
    T0: int
    Z: np.ndarray
    predictor_labels: Sequence
    col_period: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.Y.shape[1])

    @property
    def n_donors(self) -> int:
        return int(self.donor_idx.shape[0])

    @property
    def donor_labels(self) -> np.ndarray:
        return self.unit_index.get_labels(self.donor_idx)

    @property
    def treated_label(self) -> Any:
        return self.unit_index.get_labels([self.treated_idx])[0]

    @property
    def y_treated(self) -> np.ndarray:
        return self.Y[self.treated_idx]

    @property
    def Y_donors(self) -> np.ndarray:
        return self.Y[self.donor_idx]

    @property
    def Z_treated(self) -> np.ndarray:
        return self.Z[self.treated_idx]

    @property
    def Z_donors(self) -> np.ndarray:
        return self.Z[self.donor_idx]


@dataclass(frozen=True)
class SCMOMethodFit:
    """A single weighting-scheme fit (concatenated / averaged / separate / MA)."""

    name: str
    weights: np.ndarray              # (n_donors,) donor weights
    counterfactual: np.ndarray       # (T,)
    gap: np.ndarray                  # (T,)
    att: float
    pre_rmse: float
    donor_weights: Dict[Any, float]
    att_se: Optional[float] = None
    ci: Tuple[float, float] = (float("nan"), float("nan"))
    p_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SCMOResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SCMO.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    a dispatcher over the weighting schemes in ``fits``, it lifts the selected
    scheme's quantities into the standardized sub-models so the flat accessors
    (``att`` / ``counterfactual`` / ``gap`` / ``donor_weights`` / ``pre_rmse``
    / ``att_ci``) resolve through the base contract. The SCMO-specific fields
    below keep every per-scheme fit available.

    Parameters
    ----------
    inputs : SCMOInputs
        Preprocessed panel.
    fits : dict
        ``{scheme_name: SCMOMethodFit}`` for every weighting scheme run.
    selected_variant : str
        Which scheme drives the standardized surface / flat accessors.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SCMOInputs
    fits: Dict[str, SCMOMethodFit]
    selected_variant: str = CONCATENATED
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "SCMOResults":
        """Born-standardized lift of the selected scheme into the shared
        sub-models. Uses ``object.__setattr__`` because the model is frozen.
        """
        if self.effects is not None or not self.fits:
            return self
        fit = self._primary
        labels = np.asarray(self.inputs.time_index.labels)
        T0 = self.inputs.T0
        T = self.inputs.T
        ts = TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y_treated, dtype=float),
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
            method_name=f"SCMO ({self.selected_variant})", is_recommended=True))
        return self

    @property
    def _primary(self) -> SCMOMethodFit:
        return self.fits.get(self.selected_variant, next(iter(self.fits.values())))

    def att_by_method(self) -> Dict[str, float]:
        return {name: fit.att for name, fit in self.fits.items()}


# Resolve forward references (module uses ``from __future__ import annotations``).
SCMOResults.model_rebuild()
