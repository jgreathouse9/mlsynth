"""Bayesian Variable Selection with Soft Simplex constraint (BVS-SS).

Implements:

    Xu, Y., & Zhou, Q. (2025). "Bayesian Synthetic Control with a Soft
    Simplex Constraint." arXiv:2503.06454.

BVS-SS layers a spike-and-slab prior on top of a soft simplex prior on
the donor weights, with the soft-constraint variance ``tau`` learned
from the data. Posterior samples are drawn by a Metropolis-within-
Gibbs sampler that jointly updates pairs of inclusion indicators and
the corresponding weights (Lemmas S1, S2 of the paper).

See ``mlsynth.utils.bvss_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    BVSSConfig,
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
from ..utils.bvss_helpers.inference import compute_inference
from ..utils.bvss_helpers.plotter import plot_bvss
from ..utils.bvss_helpers.sampler import gibbs_BVS
from ..utils.bvss_helpers.setup import prepare_bvss_inputs
from ..utils.bvss_helpers.structures import (
    BVSSPosterior,
    BVSSResults,
)
from ..utils.datautils import balance


class BVSS:
    """Bayesian Synthetic Control with a soft simplex constraint.

    Parameters
    ----------
    config : BVSSConfig or dict
        Configuration object. See :class:`mlsynth.config_models.BVSSConfig`.

    Returns
    -------
    BVSSResults
        Posterior samples of ``\\mu, \\phi, \\tau``, the implied 0/1
        inclusion indicators, the posterior-mean counterfactual with
        credible bands, and the ATT mean + credible interval.
    """

    def __init__(self, config: Union[BVSSConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = BVSSConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid BVSS configuration: {exc}"
                ) from exc

        self.config: BVSSConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat

        self.n_iter: int = config.n_iter
        self.burn_in: int = config.burn_in
        self.kappa1: float = config.kappa1
        self.kappa2: float = config.kappa2
        self.theta: float = config.theta
        self.tau_a: float = config.tau_a
        self.tau_b: float = config.tau_b
        self.n_tau: int = config.n_tau
        self.tau_min: float = config.tau_min
        self.ci_alpha: float = config.ci_alpha

        self.init_phi: float = config.init_phi
        self.init_tau: float = config.init_tau
        self.init_mu: Optional[np.ndarray] = (
            np.asarray(config.init_mu, dtype=float)
            if config.init_mu is not None else None
        )

        self.display_graphs: bool = config.display_graphs
        self.verbose: bool = config.verbose
        self.seed: Optional[int] = config.seed

    def fit(self) -> BVSSResults:
        """Run the BVS-SS Gibbs sampler and assemble results."""

        if self.burn_in >= self.n_iter:
            raise MlsynthConfigError(
                f"burn_in={self.burn_in} must be strictly less than "
                f"n_iter={self.n_iter}."
            )

        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"Error balancing panel data: {exc}"
            ) from exc

        try:
            inputs = prepare_bvss_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"Error preparing BVSS inputs: {exc}"
            ) from exc

        rng = np.random.default_rng(self.seed) if self.seed is not None else None

        try:
            samples = gibbs_BVS(
                Y=inputs.Y_pre_demean,
                X=inputs.X_pre_demean,
                Gram=inputs.Gram,
                M=inputs.T0,
                N=inputs.N,
                size=self.n_iter,
                kappa1=self.kappa1,
                kappa2=self.kappa2,
                theta=self.theta,
                tau_min=self.tau_min,
                a=self.tau_a,
                b=self.tau_b,
                n_tau=self.n_tau,
                init_mu=self.init_mu,
                init_phi=self.init_phi,
                init_tau=self.init_tau,
                verbose=self.verbose,
                rng=rng,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"BVSS estimation failed: {exc}"
            ) from exc

        # Drop burn-in.
        mu_post = samples["musample"][:, self.burn_in:]
        phi_post = samples["phisample"][self.burn_in:]
        tau_post = samples["tausample"][self.burn_in:]
        gamma_post = samples["gammasample"][:, self.burn_in:]

        posterior = BVSSPosterior(
            mu=mu_post,
            phi=phi_post,
            tau=tau_post,
            gamma=gamma_post,
            burn_in=self.burn_in,
            n_iter=self.n_iter,
        )

        inference = compute_inference(
            inputs=inputs, mu_samples=mu_post, ci_alpha=self.ci_alpha,
        )

        donor_labels = [str(d) for d in inputs.donor_names]
        inclusion_probs = {
            donor_labels[i]: float(gamma_post[i].mean())
            for i in range(inputs.N)
        }
        weight_means = {
            donor_labels[i]: float(mu_post[i].mean())
            for i in range(inputs.N)
        }

        labels = np.asarray(inputs.time_labels)
        T0, T = inputs.T0, inputs.T
        y_obs = np.asarray(inputs.y_target, dtype=float)
        cf = np.asarray(inference.counterfactual_mean, dtype=float)
        gap = y_obs - cf
        pre_rmse = (float(np.sqrt(np.mean(gap[:T0] ** 2)))
                    if T0 > 0 else float("nan"))
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
        results = BVSSResults(
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
                summary_stats={"constraint": "Bayesian spike-and-slab (posterior mean)",
                               "inclusion_probs": inclusion_probs}),
            fit_diagnostics=FitDiagnosticsResults(
                rmse_pre=None if np.isnan(pre_rmse) else float(pre_rmse)),
            inference=std_inference,
            method_details=MethodDetailsResults(method_name="BVSS", is_recommended=True),
        )

        if self.display_graphs:
            try:
                plot_bvss(results)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"BVSS plotting failed: {exc}"
                ) from exc

        return results
