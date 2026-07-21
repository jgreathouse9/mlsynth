"""Typed containers for the BPSCS estimator (Fernandez-Morales et al. 2026)."""

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
class BPSCSInputs:
    """Standardized panel + utility inputs fed into the BPSCS sampler.

    ``Y_std`` is the pre-period-standardized outcome matrix (time x unit, treated
    column 0); ``donor_d`` is the per-donor utility score that scales the
    shrinkage prior; ``stdv0`` / ``mean_pre_treated`` rescale the counterfactual
    back to the treated unit's original outcome scale.
    """

    Y_std: np.ndarray             # (T, D) standardized, treated column 0
    y_target: np.ndarray          # (T,) treated outcome, original scale
    mean_pre_treated: float
    stdv0: float
    X_cov: np.ndarray             # (D, P) z-scored baseline covariates
    donor_d: np.ndarray           # (J,) per-donor utility score
    rho: float                    # inclusion radius (ds2)
    kappa_d: float
    T0: int
    T: int
    D: int
    prior: str
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray


@dataclass(frozen=True)
class BPSCSPosterior:
    """Posterior draws relevant to the counterfactual and diagnostics."""

    counterfactual: np.ndarray    # (n_draws, T) treated counterfactual, original scale
    beta: np.ndarray              # (n_draws, J) donor coefficients (standardized)
    sigma: np.ndarray             # (n_draws,)
    psi: np.ndarray               # (n_draws,) autoregressive coefficient
    n_draws: int
    accept_prob: float
    n_divergent: int
    max_rhat: float
    prior: str
    rho: float
    n_included: int               # donors with utility above the inclusion radius


@dataclass(frozen=True)
class BPSCSInference:
    """Posterior summaries of the counterfactual, ATT, and credible bands.

    Point summaries use the posterior *median*, which is robust to the heavy
    upper tail of the free-running counterfactual (a few draws with a large
    autoregressive coefficient explode); the mean is not a usable summary here.
    """

    counterfactual_median: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    att_median: float
    att_lower: float
    att_upper: float
    att_samples: np.ndarray
    ci_alpha: float


class BPSCSResults(BaseEstimatorResults):
    """Top-level container returned by ``BPSCS.fit`` (an ``EffectResult``)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: BPSCSInputs
    posterior: BPSCSPosterior
    inference_detail: BPSCSInference

    @model_validator(mode="after")
    def _populate(self) -> "BPSCSResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard
            return self
        inf = self.inference_detail
        labels = np.asarray(self.inputs.time_labels)
        T0, T = self.inputs.T0, self.inputs.T
        y_obs = np.asarray(self.inputs.y_target, dtype=float)
        cf = np.asarray(inf.counterfactual_median, dtype=float)
        gap = y_obs - cf
        pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2))) if T0 > 0 else float("nan")
        att_se = float(np.std(inf.att_samples)) if inf.att_samples.size else None

        beta_med = np.median(self.posterior.beta, axis=0)
        donor_weights = {str(n): float(b)
                         for n, b in zip(self.inputs.donor_names, beta_med)}

        object.__setattr__(self, "effects", EffectsResults(
            att=None if np.isnan(inf.att_median) else float(inf.att_median),
            att_std_err=att_se))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=y_obs,
            counterfactual_outcome=cf,
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights=donor_weights,   # signed SC coefficients (posterior median)
            summary_stats={
                "prior": self.posterior.prior,
                "kappa_d": float(self.inputs.kappa_d),
                "inclusion_radius": float(self.posterior.rho),
                "n_donors_included": int(self.posterior.n_included),
                "psi_post_median": float(np.median(self.posterior.psi)),
                "sigma_post_median": float(np.median(self.posterior.sigma)),
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
            method_name="BPSCS", is_recommended=True))
        return self


BPSCSResults.model_rebuild()
