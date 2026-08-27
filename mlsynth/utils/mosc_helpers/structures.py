"""Typed containers for the MOSC estimator.

Everything crossing a stage boundary is one of these, so a caller reading a
result never has to unpack a tuple positionally to find out what it holds.
"""

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
class MOSCInputs:
    """The panel MOSC fits, oriented ``(T, N)`` with the treated unit first."""

    panel: np.ndarray             # (T, N) outcomes, treated unit in column 0
    y_target: np.ndarray          # (T,) treated outcome, all periods
    pre_periods: int
    total_periods: int
    n_units: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray


@dataclass(frozen=True)
class MOSCPosterior:
    """Draws of the estimated confounding structure and of the counterfactual."""

    loadings: np.ndarray          # (S, K, N) per-unit latent vectors
    counterfactual: np.ndarray    # (S, T_post) treated counterfactual, outcome scale
    n_factors: int
    n_draws: int
    factor_model: str


@dataclass(frozen=True)
class MOSCDiagnostics:
    """What the fit says about whether its own assumptions hold.

    These are the quantities a caller acts on, so they are typed fields and not
    messages.

    ``residual_autocorrelation`` is the one that decides the outcome scale.
    Equations 12 and 19 require the latent factors to render a unit's outcomes
    conditionally independent, so what has to be near zero is the correlation
    left after conditioning. A cumulative series carries it at 0.2 to 0.45 and
    its first difference at 0.07 to 0.17, measured on the authors' own panels;
    ``outcome_scale="difference"`` is the repair.

    ``pearson_dispersion`` is 1 under a well-specified Poisson model and reports
    on the likelihood, not on the scale. It moves in both directions: an
    overdispersed panel drives it above 1, while a smooth series that a rank-``K``
    model fits almost exactly drives it below. Read it as a warning that the
    count assumption is doing badly, in whichever direction it departs.
    """

    heldout_log_density: float
    pearson_dispersion: float
    residual_autocorrelation: float
    outcome_scale: str
    n_heldout_cells: int


@dataclass(frozen=True)
class MOSCInference:
    """Posterior summaries of the counterfactual, the ATT and its band."""

    counterfactual_mean: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    att_mean: float
    att_lower: float
    att_upper: float
    att_samples: np.ndarray
    ci_alpha: float


class MOSCResults(BaseEstimatorResults):
    """Top-level container returned by ``MOSC.fit`` (an ``EffectResult``)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MOSCInputs
    posterior: MOSCPosterior
    diagnostics: MOSCDiagnostics
    inference_detail: MOSCInference

    @model_validator(mode="after")
    def _populate(self) -> "MOSCResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard
            return self

        detail = self.inference_detail
        labels = np.asarray(self.inputs.time_labels)
        pre, total = self.inputs.pre_periods, self.inputs.total_periods
        observed = np.asarray(self.inputs.y_target, dtype=float)
        counterfactual = np.asarray(detail.counterfactual_mean, dtype=float)
        gap = observed - counterfactual
        pre_rmse = float(np.sqrt(np.mean(gap[:pre] ** 2))) if pre > 0 else float("nan")
        att_se = float(np.std(detail.att_samples)) if detail.att_samples.size else None

        object.__setattr__(self, "effects", EffectsResults(
            att=None if np.isnan(detail.att_mean) else float(detail.att_mean),
            att_std_err=att_se))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=observed,
            counterfactual_outcome=counterfactual,
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[pre] if pre < total else None)))
        object.__setattr__(self, "weights", WeightsResults(
            # A statement, not an absence: MOSC adjusts for latent loadings and
            # has no donor weights at all, which is what a caller needs to know.
            donor_weights={},
            weights_at=["posterior"],
            summary_stats={
                "model": f"probabilistic factor model ({self.posterior.factor_model})",
                "n_factors": int(self.posterior.n_factors),
                "n_draws": int(self.posterior.n_draws),
                "outcome_scale": self.diagnostics.outcome_scale}))
        object.__setattr__(self, "fit_diagnostics", FitDiagnosticsResults(
            rmse_pre=None if np.isnan(pre_rmse) else float(pre_rmse)))
        object.__setattr__(self, "inference", InferenceResults(
            standard_error=att_se,
            ci_lower=None if np.isnan(detail.att_lower) else float(detail.att_lower),
            ci_upper=None if np.isnan(detail.att_upper) else float(detail.att_upper),
            confidence_level=float(1.0 - detail.ci_alpha),
            method="bayesian_posterior",
            details=detail))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="MOSC",
            is_recommended=True,
            additional_outputs={
                "heldout_log_density": float(self.diagnostics.heldout_log_density),
                "pearson_dispersion": float(self.diagnostics.pearson_dispersion),
                "residual_autocorrelation": float(self.diagnostics.residual_autocorrelation),
                "outcome_scale": self.diagnostics.outcome_scale,
                "effect_sign_convention": "observed minus counterfactual (equation 43)"}))
        return self


MOSCResults.model_rebuild()
