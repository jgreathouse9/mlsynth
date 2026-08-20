"""Time Varying Parameter Bayesian Lasso (BLTVP).

Implements:

    Klinenberg, D. (2023). "Synthetic Control with Time Varying Coefficients:
    A State Space Approach with Bayesian Shrinkage." Journal of Business &
    Economic Statistics 41(4):1065-1076.

Classic synthetic control treats the relationship between the treated unit and
each donor as fixed over the whole window, which is a cross-sectional reading
of a time-series problem. When that relationship drifts -- an oil shock
changing which economies resemble each other, a competitor entering one market
-- the pre-treatment fit degrades and the counterfactual with it.

BLTVP splits each donor coefficient into a time-invariant part and a
random-walk part and shrinks the two independently under a Bayesian Lasso, so
the model decides per donor whether the relationship is time-varying, static,
or absent. That is what separates it from making every coefficient dynamic,
which is the alternative on offer and buys its flexibility with intervals wide
enough to reject little.

See ``mlsynth.utils.bltvp_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    BLTVPConfig,
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
from ..utils.bltvp_helpers.inference import compute_inference
from ..utils.bltvp_helpers.plotter import plot_bltvp
from ..utils.bltvp_helpers.sampler import gibbs_bltvp
from ..utils.bltvp_helpers.setup import prepare_bltvp_inputs
from ..utils.bltvp_helpers.structures import BLTVPPosterior, BLTVPResults
from ..utils.datautils import balance


class BLTVP:
    """Time Varying Parameter Bayesian Lasso (Klinenberg 2023).

    Parameters
    ----------
    config : BLTVPConfig or dict
        Configuration object. See :class:`mlsynth.config_models.BLTVPConfig`.

    Returns
    -------
    BLTVPResults
        Posterior counterfactual with credible bands, the ATT posterior mean
        and credible interval, posterior mean time-invariant donor weights,
        and the per-donor drift scales that say which relationships move.

    Examples
    --------
    >>> from mlsynth import BLTVP                                  # doctest: +SKIP
    >>> results = BLTVP({                                          # doctest: +SKIP
    ...     "df": panel, "outcome": "cigsale", "unitid": "state",
    ...     "time": "year", "treat": "treated",
    ... }).fit()
    >>> results.att                                                # doctest: +SKIP
    """

    def __init__(self, config: Union[BLTVPConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = BLTVPConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid BLTVP configuration: {exc}"
                ) from exc

        self.config: BLTVPConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat

        self.n_iter: int = config.n_iter
        self.burn_in: int = config.burn_in
        self.chains: int = config.chains
        self.intercept: bool = config.intercept
        self.time_varying: bool = config.time_varying
        self.ci_alpha: float = config.ci_alpha
        self.display_graphs: bool = config.display_graphs
        self.verbose: bool = config.verbose
        self.seed: Optional[int] = config.seed

    def fit(self) -> BLTVPResults:
        """Run the BLTVP Gibbs sampler and assemble results."""

        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_bltvp_inputs(
                df=self.df, outcome=self.outcome, unitid=self.unitid,
                time=self.time, treat=self.treat,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthDataError(f"Error preparing BLTVP inputs: {exc}") from exc

        try:
            samples = gibbs_bltvp(
                inputs.y_target, inputs.X_all, inputs.T0,
                n_iter=self.n_iter, burn_in=self.burn_in, chains=self.chains,
                seed=self.seed, intercept=self.intercept,
                time_varying=self.time_varying,
                a1=self.config.a1, g0=self.config.g0, G0=self.config.G0,
                z1=self.config.z1, z2=self.config.z2,
                z1_dyn=self.config.z1_dyn, z2_dyn=self.config.z2_dyn,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):  # pragma: no cover - defensive passthrough
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise MlsynthEstimationError(f"BLTVP estimation failed: {exc}") from exc

        posterior = BLTVPPosterior(
            beta=samples["beta"], sqrt_theta=samples["sqrt_theta"],
            sigma2=samples["sigma2"], predictive=samples["predictive"],
            burn_in=self.burn_in, n_iter=self.n_iter, chains=self.chains,
            intercept=self.intercept, time_varying=self.time_varying,
        )

        inference = compute_inference(
            inputs=inputs, predictive=samples["predictive"],
            ci_alpha=self.ci_alpha,
        )

        donor_labels = [str(d) for d in inputs.donor_names]
        weight_means = {
            donor_labels[i]: float(samples["beta"][i].mean())
            for i in range(inputs.N)
        }
        # Only the magnitude of the drift scale is identified -- the sampler's
        # sign switch makes its sign symmetric by construction.
        dynamic_scales = {
            donor_labels[i]: float(np.abs(samples["sqrt_theta"][i]).mean())
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

        summary_stats = {
            "constraint": "Bayesian Lasso, unconstrained weights",
            "coefficients": "time-varying" if self.time_varying else "static",
            "dynamic_scales": dynamic_scales,
        }

        results = BLTVPResults(
            inputs=inputs,
            posterior=posterior,
            inference_detail=inference,
            weight_means=weight_means,
            dynamic_scales=dynamic_scales,
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
            method_details=MethodDetailsResults(method_name="BLTVP", is_recommended=True),
        )

        if self.display_graphs:
            try:
                plot_bltvp(results)
            except MlsynthPlottingError:  # pragma: no cover - defensive passthrough
                raise
            except Exception as exc:  # pragma: no cover - defensive translation
                raise MlsynthPlottingError(f"BLTVP plotting failed: {exc}") from exc

        return results
