"""Counterfactual and Synthetic Control with Instrumented PCA (CSC-IPCA).

Wang, C. (2024). *"Counterfactual and Synthetic Control Method: Causal
Inference with Instrumented Principal Component Analysis."* Job Market Paper.

CSC-IPCA imputes the treated unit's untreated potential outcome with a factor
model whose loadings are a linear projection of observed covariates,
``Lambda_it = X_it Gamma`` (instrumented principal component analysis;
Kelly-Pruitt-Su 2019/2020):

1. Estimate the latent factors ``F`` and control mapping ``Gamma_ctrl`` by
   alternating least squares on the control panel over the whole period.
2. Re-estimate the treated mapping ``Gamma_tr`` on the treated unit's
   pre-treatment periods, holding ``F`` fixed, then normalize.
3. Impute ``hat Y_t(0) = (X_t Gamma_tr) F_t`` and report the effect
   ``Y_t - hat Y_t(0)`` over the post-period.

Unlike an outcome-only factor estimator (``CFM``, ``FMA``), CSC-IPCA turns a
time-varying covariate cube into the loadings, so it needs no convex-hull /
common-support condition and extracts signal from many covariates even when the
treated unit sits outside the donor hull. Inference is the moving-block
conformal procedure (Chernozhukov et al. 2021), giving a per-period band.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    CSCIPCAConfig,
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
from ..utils.datautils import balance
from ..utils.cscipca_helpers.inference import cscipca_conformal
from ..utils.cscipca_helpers.pipeline import fit_cscipca
from ..utils.cscipca_helpers.plotter import plot_cscipca
from ..utils.cscipca_helpers.setup import prepare_cscipca_inputs
from ..utils.cscipca_helpers.structures import (
    CSCIPCADesign,
    CSCIPCAInference,
    CSCIPCAResults,
)


class CSCIPCA:
    """Counterfactual and Synthetic Control with Instrumented PCA (Wang 2024).

    Parameters
    ----------
    config : CSCIPCAConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.CSCIPCAConfig`.

    Returns
    -------
    CSCIPCAResults
        Frozen container with the estimated design (mapping ``Gamma``, factors
        ``F``, counterfactual, effect path, ATT), the moving-block conformal
        inference, and the standardized result sub-models.
    """

    def __init__(self, config: Union[CSCIPCAConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = CSCIPCAConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid CSCIPCA configuration: {exc}"
                ) from exc

        self.config: CSCIPCAConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.covariates = list(config.covariates)
        self.n_factors: int = config.n_factors
        self.max_iter: int = config.max_iter
        self.tol: float = config.tol
        self.alpha: float = config.alpha
        self.inference: bool = config.inference
        self.n_nulls: int = config.n_nulls
        self.null_grid_scale: float = config.null_grid_scale
        self.display_graphs: bool = getattr(config, "display_graphs", False)

    def fit(self) -> CSCIPCAResults:
        """Run the CSC-IPCA pipeline end to end."""
        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_cscipca_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
                covariates=self.covariates, n_factors=self.n_factors,
            )

            fit = fit_cscipca(inputs, self.n_factors, self.max_iter, self.tol)

            design = CSCIPCADesign(
                n_factors=int(self.n_factors),
                gamma=fit.gamma,
                factors=fit.factors,
                counterfactual=fit.counterfactual,
                gap=fit.gap,
                tau=fit.tau,
                att=fit.att,
                n_iter=fit.n_iter,
                converged=fit.converged,
                pre_rmse=fit.pre_rmse,
            )

            if self.inference:
                inference = cscipca_conformal(
                    inputs, design, self.n_factors, self.max_iter, self.tol,
                    self.alpha, self.n_nulls, self.null_grid_scale,
                )
            else:
                inference = CSCIPCAInference(alpha=float(self.alpha))

            labels = np.asarray(inputs.time_labels)
            T0, T = inputs.T0, inputs.T
            att_p = inference.att_p_value
            att_lo, att_hi = inference.att_lower, inference.att_upper
            std_inference = InferenceResults(
                standard_error=None,
                ci_lower=None if not np.isfinite(att_lo) else float(att_lo),
                ci_upper=None if not np.isfinite(att_hi) else float(att_hi),
                p_value=None if not np.isfinite(att_p) else float(att_p),
                confidence_level=1.0 - float(self.alpha),
                method="conformal",
                details=inference,
            )

            results = CSCIPCAResults(
                inputs=inputs,
                design=design,
                inference_detail=inference,
                metadata={
                    "n_factors": int(self.n_factors),
                    "n_covariates": inputs.L,
                    "n_iter": fit.n_iter,
                    "converged": fit.converged,
                    "inference": self.inference,
                },
                effects=EffectsResults(
                    att=None if np.isnan(fit.att) else float(fit.att),
                    additional_effects={"covariate_instrumented_loadings": True},
                ),
                time_series=TimeSeriesResults(
                    observed_outcome=np.asarray(inputs.treated_outcome, dtype=float),
                    counterfactual_outcome=np.asarray(fit.counterfactual, dtype=float),
                    estimated_gap=np.asarray(fit.gap, dtype=float),
                    time_periods=labels,
                    intervention_time=(labels[T0] if T0 < T else None),
                ),
                weights=WeightsResults(
                    donor_weights={},
                    summary_stats={
                        "constraint": "covariate-instrumented factor model "
                        "(no donor weights)"},
                ),
                fit_diagnostics=FitDiagnosticsResults(
                    rmse_pre=None if np.isnan(fit.pre_rmse) else float(fit.pre_rmse)),
                inference=std_inference,
                method_details=MethodDetailsResults(
                    method_name="CSCIPCA", is_recommended=True,
                    parameters_used={"n_factors": int(self.n_factors),
                                     "n_covariates": inputs.L}),
            )

            if self.display_graphs:
                try:
                    plot_cscipca(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"CSCIPCA plotting failed: {exc}"
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
                f"CSCIPCA estimation failed: {exc}"
            ) from exc
