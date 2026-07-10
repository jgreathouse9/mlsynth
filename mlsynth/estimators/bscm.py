"""Bayesian Synthetic Control Methods (BSCM).

Implements:

    Kim, S., Lee, C., & Gupta, S. (2020). "Bayesian Synthetic Control
    Methods." Journal of Marketing Research 57(5):831-852.

BSCM regresses the treated unit on the donor pool with no simplex constraint
(weights may be negative and need not sum to one) and regularises the weights
with a Bayesian global-local shrinkage prior -- a ``horseshoe`` (continuous)
or a ``spike_slab`` (discrete variable selection). Posterior samples are drawn
by a pure-numpy Gibbs sampler, yielding a counterfactual with credible bands
and an ATT credible interval with no MCMC-engine dependency.

See ``mlsynth.utils.bscm_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    BSCMConfig,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.bscm_helpers.inference import compute_inference
from ..utils.bscm_helpers.plotter import plot_bscm
from ..utils.bscm_helpers.sampler import gibbs_bscm
from ..utils.bscm_helpers.setup import prepare_bscm_inputs
from ..utils.bscm_helpers.structures import (
    BSCMInference,
    BSCMPosterior,
    BSCMResults,
)
from ..utils.datautils import balance


class BSCM:
    """Bayesian Synthetic Control Methods (Kim, Lee & Gupta 2020).

    Parameters
    ----------
    config : BSCMConfig or dict
        Configuration object. See :class:`mlsynth.config_models.BSCMConfig`.

    Returns
    -------
    BSCMResults
        Posterior counterfactual with credible bands, the ATT posterior mean
        + credible interval, posterior-mean donor weights (which may be
        negative), and -- for ``spike_slab`` -- per-donor inclusion
        probabilities.
    """

    def __init__(self, config: Union[BSCMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = BSCMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid BSCM configuration: {exc}"
                ) from exc

        self.config: BSCMConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat

        self.prior: str = config.prior
        self.n_iter: int = config.n_iter
        self.burn_in: int = config.burn_in
        self.chains: int = config.chains
        self.spike_scale: float = config.spike_scale
        self.ci_alpha: float = config.ci_alpha
        self.display_graphs: bool = config.display_graphs
        self.verbose: bool = config.verbose
        self.seed: Optional[int] = config.seed

    def fit(self) -> BSCMResults:
        """Run the BSCM Gibbs sampler and assemble results."""

        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_bscm_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing BSCM inputs: {exc}") from exc

        rng = np.random.default_rng(self.seed)

        try:
            samples = gibbs_bscm(
                y_pre=inputs.y_pre, X_pre=inputs.X_pre, X_all=inputs.X_all,
                prior=self.prior, chains=self.chains, n_iter=self.n_iter,
                burn_in=self.burn_in, spike_scale=self.spike_scale, rng=rng,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"BSCM estimation failed: {exc}") from exc

        posterior = BSCMPosterior(
            beta0=samples["beta0"], beta=samples["beta"], sigma2=samples["sigma2"],
            gamma=samples["gamma"], prior=self.prior, burn_in=self.burn_in,
            n_iter=self.n_iter, chains=self.chains,
        )

        inference = compute_inference(
            inputs=inputs, beta0_samples=samples["beta0"],
            beta_samples=samples["beta"], ci_alpha=self.ci_alpha,
        )

        donor_labels = [str(d) for d in inputs.donor_names]
        weight_means = {
            donor_labels[i]: float(samples["beta"][i].mean()) for i in range(inputs.N)
        }
        inclusion_probs = None
        if samples["gamma"] is not None:
            inclusion_probs = {
                donor_labels[i]: float(samples["gamma"][i].mean())
                for i in range(inputs.N)
            }

        labels = np.asarray(inputs.time_labels)
        T0, T = inputs.T0, inputs.T
        y_obs = np.asarray(inputs.y_target, dtype=float)
        cf = np.asarray(inference.counterfactual_mean, dtype=float)
        gap = y_obs - cf
        pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2))) if T0 > 0 else float("nan")

        att_samples = np.asarray(inference.att_samples, dtype=float)
        att_se = float(np.std(att_samples)) if att_samples.size else None
        std_inference = InferenceResults(
            standard_error=att_se,
            ci_lower=None if np.isnan(inference.att_ci_lower) else float(inference.att_ci_lower),
            ci_upper=None if np.isnan(inference.att_ci_upper) else float(inference.att_ci_upper),
            confidence_level=float(1.0 - inference.ci_alpha),
            method="bayesian_posterior",
            details=inference,
        )

        summary_stats = {"constraint": f"Bayesian {self.prior} (posterior mean)"}
        if inclusion_probs is not None:
            summary_stats["inclusion_probs"] = inclusion_probs

        results = BSCMResults(
            inputs=inputs,
            posterior=posterior,
            inference_detail=inference,
            inclusion_probs=inclusion_probs,
            weight_means=weight_means,
            effects=EffectsResults(
                att=None if np.isnan(inference.att_mean) else float(inference.att_mean),
                att_std_err=att_se),
            time_series=TimeSeriesResults(
                observed_outcome=y_obs,
                counterfactual_outcome=cf,
                estimated_gap=gap,
                time_periods=labels,
                intervention_time=(labels[T0] if T0 < T else None)),
            weights=WeightsResults(
                donor_weights={str(k): float(v) for k, v in weight_means.items()},
                summary_stats=summary_stats),
            fit_diagnostics=FitDiagnosticsResults(
                rmse_pre=None if np.isnan(pre_rmse) else float(pre_rmse)),
            inference=std_inference,
            method_details=MethodDetailsResults(method_name="BSCM", is_recommended=True),
        )

        if self.display_graphs:
            try:
                plot_bscm(results)
            except MlsynthPlottingError:  # pragma: no cover - defensive passthrough
                raise
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"BSCM plotting failed: {exc}") from exc

        return results
