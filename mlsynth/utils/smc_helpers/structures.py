"""Typed result containers for the SMC estimator (Zhu 2023).

``SMCInputs`` holds the pivoted panel; ``SMCFit`` the single deterministic
point estimate; ``SMCResults`` lifts the fit into the standardized
``BaseEstimatorResults`` sub-models so the flat accessors resolve through the
shared contract.
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
class SMCInputs:
    """Pre-pivoted inputs for a single-treated-unit SMC fit."""

    Y_treated: np.ndarray             # (T,)
    Y_donors: np.ndarray              # (T, J)
    treated_label: Any
    donor_labels: Tuple[Any, ...]
    time_index: np.ndarray            # (T,)
    intervention_time: Any
    treatment_period: int             # 1-indexed position in ``time_index``
    T: int
    T0: int
    T1: int
    J: int
    cov_treated: Optional[np.ndarray] = None    # (P,) pre-period covariate means
    cov_donors: Optional[np.ndarray] = None     # (P, J)
    covariate_names: Tuple[Any, ...] = ()

    @property
    def has_covariates(self) -> bool:
        return self.cov_treated is not None and self.cov_treated.size > 0


@dataclass(frozen=True)
class SMCFit:
    """Single deterministic SMC point estimate."""

    att: float
    weights: np.ndarray               # (J,) combined coefficients theta * w
    theta: np.ndarray                 # (J,) per-donor univariate OLS coefficients
    w: np.ndarray                     # (J,) box-[0, 1] synthesis weights
    bias: float
    sigma2: float
    counterfactual: np.ndarray        # (T,)
    gap: np.ndarray                   # (T,) treated - counterfactual
    pre_rmse: float
    n_matching_rows: int
    n_covariates: int
    donor_weights: dict = field(default_factory=dict)


class SMCResults(BaseEstimatorResults):
    """Top-level container returned by ``SMC.fit`` (an ``EffectResult``).

    Lifts the single ``fit`` into the standardized sub-models so the flat
    accessors (``att`` / ``counterfactual`` / ``gap`` / ``donor_weights`` /
    ``pre_rmse``) resolve through the base contract. SMC-specific detail
    (the ``theta`` rescalings, the box weights, ``sigma^2``) stays on ``fit``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SMCInputs
    fit: SMCFit

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "SMCResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard
            return self
        f = self.fit
        labels = np.asarray(self.inputs.time_index)
        T0, T = self.inputs.T0, self.inputs.T
        object.__setattr__(self, "effects", EffectsResults(att=float(f.att)))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.Y_treated, dtype=float),
            counterfactual_outcome=np.asarray(f.counterfactual, dtype=float),
            estimated_gap=np.asarray(f.gap, dtype=float),
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in f.donor_weights.items()},
            summary_stats={
                "constraint": "box [0,1] on w; combined theta*w may extrapolate",
                "n_matching_rows": int(f.n_matching_rows),
                "n_covariates": int(f.n_covariates),
                "sigma2": float(f.sigma2)}))
        object.__setattr__(self, "fit_diagnostics",
                           FitDiagnosticsResults(rmse_pre=float(f.pre_rmse)))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="SMC", is_recommended=True))
        return self

    @property
    def weights_vector(self) -> np.ndarray:
        """Combined donor coefficients ``theta * w`` (may be negative)."""
        return self.fit.weights

    @property
    def box_weights(self) -> np.ndarray:
        """The synthesis weights ``w`` on the box ``[0, 1]``."""
        return self.fit.w


# Resolve forward references (module uses ``from __future__ import annotations``).
SMCResults.model_rebuild()
