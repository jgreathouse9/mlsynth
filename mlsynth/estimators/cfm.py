"""Causal Factor Model (CFM) estimator.

Bai, J., & Wang, P. (2026). *"Causal Inference Using Factor Models."*

CFM estimates the systematic causal effect for a single treated unit by
modeling *both* potential outcomes within one factor structure and letting
the treated unit's factor loadings break at the intervention date:

1. Estimate common factors ``F_hat`` from the control panel by principal
   components on the time-demeaned controls (Bai 2009); the factor count is
   chosen by the Ahn-Horenstein (2013) ER/GR criteria or Bai-Ng (2002).
2. Regress the treated unit on ``(1, f_t)`` separately over the pre- and
   post-treatment periods, recovering ``(a0, lambda0)`` and ``(a1, lambda1)``.
3. The systematic causal effect is
   ``tau*_t = (lambda1 - lambda0)' f_t + (a1 - a0)`` for ``t > T0``.

Unlike a single-equation imputation estimator (e.g. FMA), the reported
effect is the systematic component -- not ``observed - counterfactual`` --
so it is robust to the one-sided idiosyncratic error that does not vanish
for a fixed treated unit and date. Inference follows the paper's appendix
A.2 variance decomposition (treated-regression ``V_reg`` + factor-estimation
``V_f``); the intercept-shift ``kappa`` and a Chow break statistic are
reported as diagnostics.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..config_models import (
    CFMConfig,
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
from ..utils.cfm_helpers.factors import extract_cfm_factors
from ..utils.cfm_helpers.inference import cfm_inference
from ..utils.cfm_helpers.pipeline import chow_break_statistic, fit_systematic_effect
from ..utils.cfm_helpers.plotter import plot_cfm
from ..utils.cfm_helpers.setup import prepare_cfm_inputs
from ..utils.cfm_helpers.structures import CFMDesign, CFMInference, CFMResults


class CFM:
    """Causal Factor Model (Bai & Wang 2026) estimator.

    Parameters
    ----------
    config : CFMConfig or dict
        Configuration object. See :class:`mlsynth.config_models.CFMConfig`.

    Returns
    -------
    CFMResults
        Frozen container with the systematic-effect design, asymptotic
        inference, the systematic causal-effect path, and the ATT.
    """

    def __init__(self, config: Union[CFMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = CFMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid CFM configuration: {exc}"
                ) from exc

        self.config: CFMConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.factor_selection: str = config.factor_selection
        self.n_factors = config.n_factors
        self.max_factors: int = config.max_factors
        self.factor_variance: bool = config.factor_variance
        self.alpha: float = config.alpha
        self.display_graphs: bool = getattr(config, "display_graphs", False)

    def fit(self) -> CFMResults:
        """Run the CFM pipeline end to end."""
        try:
            balance(self.df, self.unitid, self.time)
            inputs = prepare_cfm_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )

            # ----- Factor extraction -----
            n_factors, F_hat, source, _ = extract_cfm_factors(
                inputs.control_outcomes,
                selection=self.factor_selection,
                n_factors=self.n_factors,
                max_factors=self.max_factors,
            )

            # ----- Systematic effect -----
            fit = fit_systematic_effect(inputs.treated_outcome, F_hat, inputs.T0)
            chow = chow_break_statistic(inputs.treated_outcome, F_hat, inputs.T0)

            # ----- Inference -----
            inf = cfm_inference(
                inputs.treated_outcome, F_hat, inputs.control_outcomes,
                inputs.T0, alpha=self.alpha,
                factor_variance=self.factor_variance,
            )
            inference = CFMInference(
                alpha=float(self.alpha),
                factor_variance=bool(self.factor_variance),
                att=float(inf["att"]),
                att_se=float(inf["att_se"]),
                att_lower=float(inf["att_lower"]),
                att_upper=float(inf["att_upper"]),
                att_p_value=float(inf["att_p_value"]),
                kappa=float(inf["kappa"]),
                kappa_se=float(inf["kappa_se"]),
                kappa_t=float(inf["kappa_t"]),
                tau=np.asarray(inf["tau"], dtype=float),
                se_t=np.asarray(inf["se_t"], dtype=float),
                ci_lower_t=np.asarray(inf["ci_lower_t"], dtype=float),
                ci_upper_t=np.asarray(inf["ci_upper_t"], dtype=float),
            )

            design = CFMDesign(
                n_factors=int(n_factors),
                n_factors_source=source,
                factors=F_hat,
                a0=fit.a0, a1=fit.a1,
                lambda0=fit.lambda0, lambda1=fit.lambda1,
                kappa=fit.kappa,
                tau=fit.tau,
                tau_full=fit.tau_full,
                counterfactual=fit.counterfactual,
                att=fit.att,
                chow_fstat=float(chow),
            )

            # Pre-treatment fit of the systematic untreated path.
            pre_resid = (inputs.treated_outcome[:inputs.T0]
                         - fit.counterfactual[:inputs.T0])
            pre_rmse = float(np.sqrt(np.mean(pre_resid ** 2)))

            labels = np.asarray(inputs.time_labels)
            T0, T = inputs.T0, inputs.T
            att_se = inf["att_se"]
            std_inference = InferenceResults(
                standard_error=None if not np.isfinite(att_se) else float(att_se),
                ci_lower=None if not np.isfinite(inf["att_lower"]) else float(inf["att_lower"]),
                ci_upper=None if not np.isfinite(inf["att_upper"]) else float(inf["att_upper"]),
                p_value=None if not np.isfinite(inf["att_p_value"]) else float(inf["att_p_value"]),
                confidence_level=1.0 - float(self.alpha),
                method="asymptotic",
                details=inference,
            )
            results = CFMResults(
                inputs=inputs,
                design=design,
                inference_detail=inference,
                metadata={
                    "n_factors": int(n_factors),
                    "n_factors_source": source,
                    "factor_selection": self.factor_selection,
                    "factor_variance": bool(self.factor_variance),
                    "kappa": float(fit.kappa),
                    "kappa_t": float(inf["kappa_t"]),
                    "chow_fstat": float(chow),
                },
                effects=EffectsResults(
                    att=None if np.isnan(fit.att) else float(fit.att),
                    att_std_err=None if not np.isfinite(att_se) else float(att_se),
                    additional_effects={"systematic_effect": True,
                                        "kappa": float(fit.kappa)},
                ),
                time_series=TimeSeriesResults(
                    observed_outcome=np.asarray(inputs.treated_outcome, dtype=float),
                    counterfactual_outcome=np.asarray(fit.counterfactual, dtype=float),
                    estimated_gap=np.asarray(fit.tau_full, dtype=float),
                    time_periods=labels,
                    intervention_time=(labels[T0] if T0 < T else None),
                ),
                weights=WeightsResults(
                    donor_weights={},
                    summary_stats={"constraint": "factor-model projection (no donor weights)"},
                ),
                fit_diagnostics=FitDiagnosticsResults(
                    rmse_pre=None if np.isnan(pre_rmse) else float(pre_rmse)),
                inference=std_inference,
                method_details=MethodDetailsResults(
                    method_name="CFM", is_recommended=True,
                    parameters_used={"n_factors": int(n_factors),
                                     "n_factors_source": source}),
            )

            if self.display_graphs:
                try:
                    plot_cfm(results)
                except Exception as exc:
                    raise MlsynthPlottingError(
                        f"CFM plotting failed: {exc}"
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
            raise MlsynthEstimationError(f"CFM estimation failed: {exc}") from exc
