"""Typed containers for the BFSC estimator (Pinkney 2021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import ConfigDict, model_validator

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
class BFSCInputs:
    """Panel arrays fed into the BFSC sampler (treated unit is row 0)."""

    Y: np.ndarray                 # (J, T) treated first, then donors
    y_target: np.ndarray          # (T,) treated outcome, all periods
    T0: int
    T: int
    J: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray


@dataclass(frozen=True)
class BFSCPosterior:
    """Posterior draws relevant to the counterfactual and diagnostics."""

    counterfactual: np.ndarray    # (n_draws, T) treated counterfactual, unscaled
    sigma: np.ndarray             # (n_draws,) idiosyncratic scale (scaled units)
    n_factors: int
    n_draws: int
    accept_prob: float
    n_divergent: int
    max_rhat: float


@dataclass(frozen=True)
class BFSCInference:
    """Posterior summaries of the counterfactual, ATT, and credible bands."""

    counterfactual_mean: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    att_mean: float
    att_lower: float
    att_upper: float
    att_samples: np.ndarray
    ci_alpha: float


class BFSCResults(BaseEstimatorResults):
    """Top-level container returned by ``BFSC.fit`` (an ``EffectResult``)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: BFSCInputs
    posterior: BFSCPosterior
    inference_detail: BFSCInference

    @model_validator(mode="after")
    def _populate(self) -> "BFSCResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard
            return self
        inf = self.inference_detail
        labels = np.asarray(self.inputs.time_labels)
        T0, T = self.inputs.T0, self.inputs.T
        y_obs = np.asarray(self.inputs.y_target, dtype=float)
        cf = np.asarray(inf.counterfactual_mean, dtype=float)
        gap = y_obs - cf
        pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2))) if T0 > 0 else float("nan")
        att_se = float(np.std(inf.att_samples)) if inf.att_samples.size else None

        object.__setattr__(self, "effects", EffectsResults(
            att=None if np.isnan(inf.att_mean) else float(inf.att_mean),
            att_std_err=att_se))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=y_obs,
            counterfactual_outcome=cf,
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={},   # a factor model has no explicit donor weights
            summary_stats={
                "model": "Bayesian latent factor (horseshoe+ loadings)",
                "n_factors": int(self.posterior.n_factors),
                "sigma_post_mean": float(np.mean(self.posterior.sigma)),
                "nuts_accept_prob": float(self.posterior.accept_prob),
                "nuts_divergences": int(self.posterior.n_divergent),
                "max_rhat": float(self.posterior.max_rhat)}))
        object.__setattr__(self, "fit_diagnostics", FitDiagnosticsResults(
            rmse_pre=None if np.isnan(pre_rmse) else float(pre_rmse)))
        object.__setattr__(self, "inference", InferenceResults(
            standard_error=att_se,
            ci_lower=None if np.isnan(inf.att_lower) else float(inf.att_lower),
            ci_upper=None if np.isnan(inf.att_upper) else float(inf.att_upper),
            confidence_level=float(1.0 - inf.ci_alpha),
            method="bayesian_posterior",
            details=inf))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="BFSC", is_recommended=True))
        return self


BFSCResults.model_rebuild()
