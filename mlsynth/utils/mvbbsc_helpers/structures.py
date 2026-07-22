"""Typed containers for the MVBBSC estimator (Martinez & Vives-i-Bastida 2024)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
class MVBBSCInputs:
    """Panel arrays fed into the MVBBSC sampler.

    Parameters
    ----------
    y_target : np.ndarray
        Length-``T`` treated outcome over all periods.
    X_all : np.ndarray
        Shape ``(T, N)`` donor matrix over all periods.
    T0 : int
        Number of pre-treatment periods.
    T : int
        Total number of periods.
    N : int
        Number of donor units.
    treated_unit_name : str
    donor_names : Sequence
    time_labels : np.ndarray
    """

    y_target: np.ndarray
    X_all: np.ndarray
    T0: int
    T: int
    N: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray


@dataclass(frozen=True)
class MVBBSCPosterior:
    """Posterior draws relevant to the counterfactual, weights, and diagnostics.

    Parameters
    ----------
    counterfactual : np.ndarray
        Shape ``(n_draws, T)`` posterior-predictive counterfactual draws on the
        outcome scale (back-transformed from the standardized fit).
    weights : np.ndarray
        Shape ``(n_draws, N)`` simplex donor-weight draws.
    sigma : np.ndarray
        Length-``n_draws`` idiosyncratic scale (in standardized units).
    n_draws : int
    accept_prob : float
    n_divergent : int
    max_rhat : float
    """

    counterfactual: np.ndarray
    weights: np.ndarray
    sigma: np.ndarray
    n_draws: int
    accept_prob: float
    n_divergent: int
    max_rhat: float


@dataclass(frozen=True)
class MVBBSCInference:
    """Posterior summaries of the counterfactual, ATT, and credible bands."""

    counterfactual_mean: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    att_mean: float
    att_lower: float
    att_upper: float
    att_samples: np.ndarray
    ci_alpha: float


class MVBBSCResults(BaseEstimatorResults):
    """Top-level container returned by ``MVBBSC.fit`` (an ``EffectResult``).

    Populates the standardized sub-models so the flat accessors (``att`` /
    ``att_ci`` / ``counterfactual`` / ``gap`` / ``donor_weights`` /
    ``pre_rmse``) resolve through the base contract. ``att`` is the posterior
    mean ATT and ``att_ci`` its credible interval; ``donor_weights`` are the
    posterior-mean simplex weights. The Bayesian detail -- the posterior draws,
    the per-draw ATT samples, and the pointwise counterfactual bands -- stays in
    the typed fields below.

    Parameters
    ----------
    inputs : MVBBSCInputs
    posterior : MVBBSCPosterior
    inference_detail : MVBBSCInference
    weight_means : dict
        ``donor_label -> E[w_i | y]`` posterior-mean simplex weights.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MVBBSCInputs
    posterior: MVBBSCPosterior
    inference_detail: MVBBSCInference
    weight_means: dict

    @model_validator(mode="after")
    def _populate(self) -> "MVBBSCResults":
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
            donor_weights={str(k): float(v) for k, v in self.weight_means.items()},
            summary_stats={
                "constraint": "Bayesian simplex (uniform Dirichlet), posterior mean",
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
            method_name="MVBBSC", is_recommended=True))
        return self


MVBBSCResults.model_rebuild()
