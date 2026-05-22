"""Factor Model Approach (FMA) estimator.

Li, K. T., & Sonnier, G. P. (2023). *"Statistical Inference for the
Factor Model Approach to Estimate Causal Effects in
Quasi-Experimental Settings."* Journal of Marketing Research,
60(3), 449-472.

FMA estimates the ATT for a single treated unit by:

1. Extracting principal-component factors from the control panel
   (paper Section 3.1; factor count chosen by the modified Bai-Ng
   (MBN) criterion for stationary data or Bai (2004) IPC1 for
   non-stationary data -- see Web Appendix D).
2. Projecting the treated unit's pre-treatment outcomes onto a
   constant + the factors via OLS to recover the loading.
3. Forming the counterfactual ``hat y^0_{1, t} = F_aug_t' lambda_hat``
   for every period; the ATT is the mean post-treatment gap.

Inference is the paper's main contribution. Three procedures live
side by side:

* **asymptotic** (default) -- Theorem 3.1 (stationary) / Theorem 3.3
  (non-stationary) normal CI for the ATT, built from the variance
  decomposition Omega_hat = Omega1 + Omega2.
* **bootstrap** -- Web Appendix F residual bootstrap for per-period
  ATT_t CIs.
* **placebo** -- Web Appendix G control-as-pseudo-treated band.

Activate any combination via ``FMAConfig.inference_methods``.
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import FMAConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.fma_helpers.factors import extract_factors
from ..utils.fma_helpers.fit import estimate_loading_and_counterfactual
from ..utils.fma_helpers.inference import (
    asymptotic_inference,
    bootstrap_inference,
    placebo_inference,
)
from ..utils.fma_helpers.plotter import plot_fma
from ..utils.fma_helpers.setup import prepare_fma_inputs
from ..utils.fma_helpers.structures import (
    FMADesign,
    FMAInference,
    FMAResults,
)


class FMA:
    """Factor Model Approach (Li & Sonnier 2023) estimator.

    Parameters
    ----------
    config : FMAConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.FMAConfig`.

    Returns
    -------
    FMAResults
        Frozen container with the optimised design, asymptotic +
        bootstrap + placebo inference, counterfactual, ATT, and
        per-period gap.
    """

    def __init__(self, config: Union[FMAConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = FMAConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid FMA configuration: {exc}"
                ) from exc

        self.config: FMAConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.stationarity: str = config.stationarity
        self.preprocessing: str = config.preprocessing
        self.n_factors = config.n_factors
        self.max_factors: int = config.max_factors
        self.alpha: float = config.alpha
        self.inference_methods: List[str] = list(config.inference_methods)
        self.n_bootstrap: int = config.n_bootstrap
        self.bootstrap_seed: int = config.bootstrap_seed
        self.display_graphs: bool = getattr(config, "display_graphs", False)

    def fit(self) -> FMAResults:
        """Run the FMA pipeline end to end."""
        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_fma_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                preprocessing=self.preprocessing,
                stationarity=self.stationarity,
            )

            # ----- Factor extraction -----
            n_factors, common_component, F_hat, source = extract_factors(
                inputs.control_outcomes,
                stationarity=self.stationarity,
                preprocessing=self.preprocessing,
                n_factors=self.n_factors,
                max_factors=self.max_factors,
            )

            # ----- Loading and counterfactual -----
            lambda_hat, counterfactual, F_aug, resid_var = (
                estimate_loading_and_counterfactual(
                    treated_outcome=inputs.treated_outcome,
                    factors=F_hat,
                    T0=inputs.T0,
                )
            )
            gap = inputs.treated_outcome - counterfactual
            pre_rmse = float(np.sqrt(np.mean(gap[:inputs.T0] ** 2)))
            att = (
                float(np.mean(gap[inputs.T0:]))
                if inputs.n_post > 0 else float("nan")
            )

            design = FMADesign(
                n_factors=int(n_factors),
                n_factors_source=source,
                factors=F_hat,
                lambda_hat=lambda_hat,
                counterfactual=counterfactual,
                gap=gap,
                common_component=common_component,
                residual_variance=float(resid_var),
            )

            # ----- Inference -----
            inference = self._run_inference(
                inputs=inputs,
                F_hat=F_hat,
                F_aug=F_aug,
                counterfactual=counterfactual,
                resid_var=resid_var,
                att=att,
                n_factors=n_factors,
            )

            results = FMAResults(
                inputs=inputs,
                design=design,
                inference=inference,
                counterfactual=counterfactual,
                gap=gap,
                att=att,
                pre_rmse=pre_rmse,
                metadata={
                    "n_factors_source": source,
                    "n_factors": int(n_factors),
                    "inference_methods": list(self.inference_methods),
                },
            )

            if self.display_graphs:
                try:
                    plot_fma(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"FMA plotting failed: {exc}"
                    ) from exc

            return results

        except (
            MlsynthConfigError,
            MlsynthDataError,
            MlsynthEstimationError,
            MlsynthPlottingError,
        ):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"FMA estimation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        inputs,
        F_hat: np.ndarray,
        F_aug: np.ndarray,
        counterfactual: np.ndarray,
        resid_var: float,
        att: float,
        n_factors: int,
    ) -> FMAInference:
        """Compose the three inference outputs into a single container."""

        if not self.inference_methods or inputs.n_post <= 0:
            return FMAInference(method="none", alpha=float(self.alpha), att=att)

        kwargs = {
            "method": ";".join(self.inference_methods),
            "alpha": float(self.alpha),
            "att": float(att),
        }

        if "asymptotic" in self.inference_methods:
            se, lo, hi, p = asymptotic_inference(
                treated_outcome=inputs.treated_outcome,
                counterfactual=counterfactual,
                factors_with_const=F_aug,
                residual_variance=resid_var,
                T0=inputs.T0,
                alpha=self.alpha,
            )
            kwargs.update(
                asymptotic_att_se=se,
                asymptotic_att_lower=lo,
                asymptotic_att_upper=hi,
                asymptotic_att_p_value=p,
            )

        if "bootstrap" in self.inference_methods:
            boot = bootstrap_inference(
                treated_outcome=inputs.treated_outcome,
                factors=F_hat,
                counterfactual=counterfactual,
                T0=inputs.T0,
                alpha=self.alpha,
                n_replicates=self.n_bootstrap,
                seed=self.bootstrap_seed,
            )
            kwargs.update(
                bootstrap_att_t_lower=boot["lower"],
                bootstrap_att_t_upper=boot["upper"],
                bootstrap_replicates=boot["replicates"],
                bootstrap_n_replicates=boot["n_replicates"],
            )

        if "placebo" in self.inference_methods:
            placebo = placebo_inference(
                control_outcomes=inputs.control_outcomes,
                treated_outcome=inputs.treated_outcome,
                T0=inputs.T0,
                n_factors=n_factors,
                stationarity=self.stationarity,
                preprocessing=self.preprocessing,
                alpha=self.alpha,
                max_factors=self.max_factors,
            )
            kwargs.update(
                placebo_att_curves=placebo["curves"],
                placebo_quantile_lower=placebo["q_lower"],
                placebo_quantile_upper=placebo["q_upper"],
            )

        return FMAInference(**kwargs)
