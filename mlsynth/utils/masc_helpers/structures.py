"""Typed result containers for the MASC estimator.

Kellogg, Mogstad, Pouliot, and Torgovitsky (2021), *Combining Matching
and Synthetic Control to Trade off Biases from Extrapolation and
Interpolation*. The estimator forms a convex combination of a
nearest-neighbour matching weight vector and a synthetic-control
simplex weight vector, with both tuning parameters (``m``, ``phi``)
selected by rolling-origin cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)


@dataclass(frozen=True)
class MASCInputs:
    """Pre-pivoted inputs for a single-treated-unit MASC fit.

    Covariate panels (when supplied) are stored as full ``(T, J + 1, P)``
    tensors where the last axis indexes predictors and the second axis
    indexes units with the treated unit in slot 0; this lets each CV
    fold aggregate covariates over its own pre-window (matching the
    R reference, which re-averages within every fold).
    """

    Y_treated: np.ndarray         # (T,)
    Y_donors: np.ndarray          # (T, J)
    treated_label: Any
    donor_labels: Tuple[Any, ...]
    time_index: np.ndarray
    intervention_time: Any
    treatment_period: int         # 1-indexed position in `time_index`
    T: int                        # total periods
    T0: int                       # pre-period length
    T1: int                       # post-period length
    J: int                        # number of donors
    cov_treated_panel: Optional[np.ndarray] = None   # (T, P)
    cov_donors_panel: Optional[np.ndarray] = None    # (T, J, P)
    covariate_names: Tuple[Any, ...] = ()
    covariate_windows: Optional[dict] = None  # name -> (start, end) inclusive

    @property
    def has_covariates(self) -> bool:
        return self.cov_treated_panel is not None and self.cov_treated_panel.size > 0


@dataclass(frozen=True)
class MASCFit:
    """Single MASC point-estimate fit."""

    att: float
    weights: np.ndarray           # (J,) -- phi * match + (1-phi) * sc
    weights_match: np.ndarray     # (J,) -- nearest-neighbour weights
    weights_sc: np.ndarray        # (J,) -- simplex SC weights
    phi_hat: float
    m_hat: int
    counterfactual: np.ndarray    # (T,)
    gap: np.ndarray               # (T,) -- treated - counterfactual
    pre_rmse: float
    cv_error: float               # min CV error at (m_hat, phi_hat)
    cv_error_by_fold: np.ndarray  # (len(folds),) at the selected (m_hat, phi_hat)
    cv_grid: np.ndarray           # (len(m_grid), 3) -- columns: m, phi, cv_error
    donor_weights: dict = field(default_factory=dict)


class MASCResults(BaseEstimatorResults):
    """Top-level container returned by ``MASC.fit``.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    it lifts the single ``fit`` into the standardized sub-models so the flat
    accessors (``att`` / ``counterfactual`` / ``gap`` / ``donor_weights`` /
    ``pre_rmse``) resolve through the base contract. The MASC-specific blend
    detail stays on ``fit`` and the convenience properties below.

    Parameters
    ----------
    inputs : MASCInputs
        Pre-pivoted inputs.
    fit : MASCFit
        The single MASC point-estimate fit (blended matching + SC weights,
        the CV-selected ``m`` / ``phi``, counterfactual, gap).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MASCInputs
    fit: MASCFit

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "MASCResults":
        """Born-standardized lift of ``fit`` into the shared sub-models.
        Uses ``object.__setattr__`` because the model is frozen.
        """
        if self.effects is not None:
            return self
        f = self.fit
        labels = np.asarray(self.inputs.time_index)
        T0 = self.inputs.T0
        T = self.inputs.T
        object.__setattr__(self, "effects", EffectsResults(att=float(f.att)))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.Y_treated, dtype=float),
            counterfactual_outcome=np.asarray(f.counterfactual, dtype=float),
            estimated_gap=np.asarray(f.gap, dtype=float),
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in f.donor_weights.items()},
            summary_stats={"constraint": "simplex (matching+SC blend)",
                           "phi_hat": float(f.phi_hat), "m_hat": int(f.m_hat)}))
        object.__setattr__(self, "fit_diagnostics",
                           FitDiagnosticsResults(rmse_pre=float(f.pre_rmse)))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="MASC", is_recommended=True))
        return self

    @property
    def weights_vector(self) -> np.ndarray:
        """The blended MASC weight vector ``phi*match + (1-phi)*SC``."""
        return self.fit.weights

    @property
    def phi_hat(self) -> float:
        return self.fit.phi_hat

    @property
    def m_hat(self) -> int:
        return self.fit.m_hat


# Resolve forward references (module uses ``from __future__ import annotations``).
MASCResults.model_rebuild()
