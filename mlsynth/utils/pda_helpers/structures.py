"""Frozen, NumPy-first containers for the Panel Data Approach (PDA).

PDA (Hsiao, Ching & Wan 2012) predicts a treated unit's untreated
counterfactual by a linear regression on the control units fit over the
pre-treatment window, then extrapolates out-of-sample. ``mlsynth`` exposes
three high-dimensional PDA variants, each in its own subpackage with the
inference theory from its own paper:

* ``l2``    -- L2-relaxation (Shi & Wang 2024).
* ``lasso`` -- L1/LASSO (Li & Bell 2017).
* ``fs``    -- forward-selected PDA (Shi & Huang 2023).

Everything below is pure NumPy; units/time are addressed through
:class:`IndexSet`. The only DataFrame touchpoint is ``setup``.
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

from ..fast_scm_helpers.structure import IndexSet

# Method names.
L2 = "l2"
LASSO = "lasso"
FS = "fs"


@dataclass(frozen=True)
class PDAInputs:
    """Preprocessed, NumPy-only panel for the PDA engine.

    Parameters
    ----------
    unit_index : IndexSet
        All ``N`` donor units (column order of ``X``).
    time_index : IndexSet
        All ``T`` periods (row order of ``y`` and ``X``).
    y : np.ndarray
        Treated-unit outcome over all periods, shape ``(T,)``.
    X : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    T0 : int
        Number of pre-treatment periods (``T1``); post is ``T2 = T - T0``.
    treated_label : Any
        Identifier of the treated unit.
    metadata : dict
        Free-form provenance.
    """

    unit_index: IndexSet
    time_index: IndexSet
    y: np.ndarray
    X: np.ndarray
    T0: int
    treated_label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def T2(self) -> int:
        return self.T - self.T0

    @property
    def n_donors(self) -> int:
        return int(self.X.shape[1])

    @property
    def donor_labels(self) -> np.ndarray:
        return np.asarray(self.unit_index.labels)


@dataclass(frozen=True)
class PDAMethodFit:
    """A single PDA-variant fit (l2 / lasso / fs)."""

    name: str
    beta: np.ndarray                 # donor coefficients
    intercept: float
    counterfactual: np.ndarray       # (T,)
    gap: np.ndarray                  # (T,)  = y - counterfactual
    att: float                       # mean post-period gap
    att_se: float
    ci: Tuple[float, float]
    p_value: float
    donor_weights: Dict[Any, float]
    selected_donors: Optional[List[Any]] = None      # fs / lasso support
    prediction_intervals: Optional[Dict[str, Any]] = None  # Jiang et al. (2025) PIs
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDAResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.PDA.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    a dispatcher over the PDA variants in ``fits``, it lifts the selected
    variant's quantities into the standardized sub-models so the flat accessors
    (``att`` / ``att_ci`` / ``counterfactual`` / ``gap`` / ``donor_weights`` /
    ``pre_rmse``) resolve through the base contract. ``donor_weights`` here are
    the regression coefficients (PDA is a regression/factor counterfactual, not
    a simplex donor average). The PDA-specific fields keep every variant fit.

    Parameters
    ----------
    inputs : PDAInputs
        Preprocessed panel.
    fits : dict
        ``{variant_name: PDAMethodFit}`` for every variant run (l2 / lasso / fs).
    selected_variant : str
        Which variant drives the standardized surface / flat accessors.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: PDAInputs
    fits: Dict[str, PDAMethodFit]
    selected_variant: str = FS
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "PDAResults":
        """Born-standardized lift of the selected variant into the shared
        sub-models. Uses ``object.__setattr__`` because the model is frozen.
        """
        if self.effects is not None or not self.fits:
            return self
        fit = self._primary
        labels = np.asarray(self.inputs.time_index.labels)
        T0 = self.inputs.T0
        T = self.inputs.T
        gap = np.asarray(fit.gap, dtype=float)
        pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2))) if T0 > 0 else float("nan")
        lo, hi = fit.ci
        inf = InferenceResults(
            standard_error=fit.att_se,
            ci_lower=None if (lo is None or np.isnan(lo)) else float(lo),
            ci_upper=None if (hi is None or np.isnan(hi)) else float(hi),
            p_value=None if (fit.p_value is None or np.isnan(fit.p_value)) else float(fit.p_value),
            method=f"pda_{fit.name}",
        )
        object.__setattr__(self, "effects", EffectsResults(
            att=float(fit.att), att_std_err=fit.att_se))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(fit.counterfactual, dtype=float),
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in fit.donor_weights.items()},
            summary_stats={"constraint": "unconstrained regression coefficients"}))
        object.__setattr__(self, "fit_diagnostics",
                           FitDiagnosticsResults(rmse_pre=pre_rmse))
        object.__setattr__(self, "inference", inf)
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name=f"PDA ({self.selected_variant})", is_recommended=True))
        return self

    @property
    def _primary(self) -> PDAMethodFit:
        return self.fits.get(self.selected_variant, next(iter(self.fits.values())))

    @property
    def att_se(self) -> float:
        return self._primary.att_se

    def att_by_method(self) -> Dict[str, float]:
        return {name: fit.att for name, fit in self.fits.items()}

    def se_by_method(self) -> Dict[str, float]:
        return {name: fit.att_se for name, fit in self.fits.items()}


# Resolve forward references (module uses ``from __future__ import annotations``).
PDAResults.model_rebuild()
