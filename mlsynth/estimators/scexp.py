import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic # For ValidationError
import cvxpy as cp # For cvxpy.error types

from ..utils.datautils import balance
from ..utils.exputils import _get_per_cluster_param, SCMEXP
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import ( # Import the Pydantic model
    MAREXConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class MAREX:

    def __init__(self, config: MAREXConfig) -> None: # Changed to MAREXConfig

        if isinstance(config, dict):
            config =MAREXConfig(**config)  # convert dict to config object
        # Panel data

        # Panel data
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time

        # Optional design parameters
        self.T0: Optional[int] = config.T0
        self.clusters: Optional[np.ndarray] = config.clusters
        self.design: str = config.design

        # Penalization parameters
        self.beta: float = config.beta
        self.lambda1: float = config.lambda1
        self.lambda2: float = config.lambda2
        self.xi: float = config.xi
        self.lambda1_unit: float = config.lambda1_unit
        self.lambda2_unit: float = config.lambda2_unit

        # Additional SCMEXP options
        self.blank_periods: int = getattr(config, "blank_periods", 0)
        self.m_eq: Optional[int] = getattr(config, "m_eq", None)
        self.m_min: Optional[int] = getattr(config, "m_min", None)
        self.m_max: Optional[int] = getattr(config, "m_max", None)
        self.exclusive: bool = getattr(config, "exclusive", True)
        self.solver = getattr(config, "solver", None)
        self.verbose: bool = getattr(config, "verbose", False)

    def _create_estimator_results(
        self, raw_sdid_estimation_output: Dict[str, Any], prepared_panel_data: Dict[str, Any]
    ) -> BaseEstimatorResults:
        # Effects
        att = raw_sdid_estimation_output.get("att")
        att_se = raw_sdid_estimation_output.get("att_se")
        att_ci = raw_sdid_estimation_output.get("att_ci")
        additional_effects = {}
        if att_se is not None:
            additional_effects["att_standard_error"] = att_se
        if att_ci is not None:
            additional_effects["att_confidence_interval"] = att_ci
        
        effects_results = EffectsResults(
            att=att,
            additional_effects=additional_effects
        )

        # Fit Diagnostics - SDID doesn't typically produce standard fit metrics like RMSE/R-squared.
        # These are more common in methods that explicitly model an outcome variable (e.g., SCM).
        fit_diagnostics_results = FitDiagnosticsResults() # Intentionally left empty.

        # Time Series
        # Extract event study estimates (dynamic treatment effects over time) for time_series representation.
        # pooled_estimates: Dict[event_time, {'tau': float, 'se': float, 'ci': List[float]}]
        pooled_estimates = raw_sdid_estimation_output.get("pooled_estimates", {})
        time_periods_event_study = None
        estimated_effects_over_time = None # This will be tau from pooled_estimates
        
        if pooled_estimates:
            sorted_event_times = sorted(pooled_estimates.keys())
            time_periods_event_study = np.array(sorted_event_times)
            estimated_effects_over_time = np.array([pooled_estimates[t]['tau'] for t in sorted_event_times])

        # SDID doesn't have a single "counterfactual_outcome" series in the same way as SCM.
        # The primary time-varying output is the sequence of event-time specific ATTs (taus).
        # We store these event study taus in the 'estimated_gap' field.
        time_series_results = TimeSeriesResults(
            time_periods=time_periods_event_study, # Event times (e.g., -2, -1, 0, 1, 2)
            estimated_gap=estimated_effects_over_time, # Dynamic treatment effects (taus) for each event time
            # `observed_outcome` and `counterfactual_outcome` series are not directly populated here
            # as they are not primary outputs of `estimate_event_study_sdid` in a simple vector form.
            # Reconstructing them would involve applying internal weights to original data.
        )

        # Weights - SDID calculates unit (omega) and time (lambda) weights internally.
        # These are not directly in raw_sdid_estimation_output from estimate_event_study_sdid.
        # They are calculated within sdid_est, which is called by estimate_event_study_sdid.
        # For now, we'll leave this empty unless we modify sdidutils to return them.
        # Placeholder:
        # If weights were available, e.g., raw_sdid_estimation_output.get("omega_hat"), raw_sdid_estimation_output.get("lambda_hat")
        # omega_hat = raw_sdid_estimation_output.get("omega_hat") # Unit weights
        # lambda_hat = raw_sdid_estimation_output.get("lambda_hat") # Time weights
        # donor_weights_dict = {str(name): val for name, val in zip(prepared_panel_data.get("donor_names", []), omega_hat)} if omega_hat is not None else {}
        
        weights_results = WeightsResults(
            # donor_weights=donor_weights_dict, # If omega_hat was available
            # additional_metrics={"time_weights_lambda": lambda_hat.tolist() if lambda_hat is not None else None}
        )

        # Inference
        # p_value might be derived from placebo_taus if B > 0 (i.e., placebo tests were run).
        placebo_taus = raw_sdid_estimation_output.get("placebo_att_values") # Raw ATT estimates from placebo runs.
        p_value = None
        # Calculate p-value: proportion of placebo ATTs as or more extreme than the observed ATT.
        # Ensure observed ATT is valid and placebo results exist.
        if att is not None and not np.isnan(att) and placebo_taus is not None and len(placebo_taus) > 0:
            # Standard two-sided p-value calculation for placebo tests.
            p_value = (np.sum(np.abs(placebo_taus) >= np.abs(att)) + 1) / (len(placebo_taus) + 1)

        inference_results = InferenceResults(
            method="Synthetic Difference-in-Differences with Placebo Inference" if self.B > 0 else "Synthetic Difference-in-Differences", # Describe inference method.
            p_value=p_value,
            confidence_interval=att_ci # Store the main ATT CI here as well
        )
        
        # Method Details
        # Store cohort_estimates and other detailed outputs
        method_details_additional = {
            "cohort_estimates": raw_sdid_estimation_output.get("cohort_estimates"),
            "tau_a_ell": raw_sdid_estimation_output.get("tau_a_ell"),
            "tau_ell": raw_sdid_estimation_output.get("tau_ell"),
            "placebo_att_values": placebo_taus if placebo_taus is not None else None, # Removed .tolist()
            "Y_donors_pre_means": raw_sdid_estimation_output.get("Y_donors_pre_means"),
            "Y_treated_pre_means": raw_sdid_estimation_output.get("Y_treated_pre_means"),
            "Y_donors_post_means": raw_sdid_estimation_output.get("Y_donors_post_means"),
            "Y_treated_post_means": raw_sdid_estimation_output.get("Y_treated_post_means"),
            # If omega_hat and lambda_hat were available:
            # "omega_hat": raw_sdid_estimation_output.get("omega_hat").tolist() if raw_sdid_estimation_output.get("omega_hat") is not None else None,
            # "lambda_hat": raw_sdid_estimation_output.get("lambda_hat").tolist() if raw_sdid_estimation_output.get("lambda_hat") is not None else None,
        }

        method_details_results = MethodDetailsResults(
            name="SDID",
            parameters_used=self.config.model_dump(exclude={'df'}),
            additional_details={k: v for k, v in method_details_additional.items() if v is not None}
        )

        return BaseEstimatorResults(
            effects=effects_results,
            fit_diagnostics=fit_diagnostics_results,
            time_series=time_series_results,
            weights=weights_results,
            inference=inference_results,
            method_details=method_details_results,
            raw_results=raw_sdid_estimation_output,
        )

    def fit(self) -> BaseEstimatorResults:

        # Ensure the panel data is balanced (all units observed for all time periods).
        balance(self.df, self.unitid, self.time)

        # Check treatment specification
        if self.m_eq is not None and (self.m_min is not None or self.m_max is not None):
            raise MlsynthConfigError(
                "Cannot specify both 'm_eq' and 'm_min/m_max' at the same time. "
                "Choose either an exact number of treated units per cluster or a range."
            )
        if self.m_eq is None and self.m_min is None and self.m_max is None:
            raise MlsynthConfigError(
                "You must specify either 'm_eq' or at least one of 'm_min'/'m_max' to define treated units."
            )

        # Assign clusters if not provided
        clusters = self.clusters if self.clusters is not None else np.zeros(len(self.df[self.unitid].unique()),
                                                                            dtype=int)
        unit_labels = self.df[self.unitid].unique()

        # Reshape data: units as rows, time as columns
        Y_full = self.df.pivot(index=self.unitid, columns=self.time, values=self.outcome).reindex(unit_labels)

        # Determine pre-treatment periods
        T0 = self.T0 if self.T0 is not None else len(self.df[self.time].unique())
        blanks = Y_full.shape[1] - T0

        # Call SCMEXP with either equality or range treatment
        scm_kwargs = dict(
            Y_full=Y_full.to_numpy(),
            T0=T0,
            clusters=clusters,
            blank_periods=blanks,
            design=self.design,
            beta=self.beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            xi=self.xi,
            lambda1_unit=self.lambda1_unit,
            lambda2_unit=self.lambda2_unit,
            solver=self.solver or cp.ECOS_BB,
            verbose=self.verbose,
            exclusive=self.exclusive,
        )

        if self.m_eq is not None:
            scm_kwargs["m_eq"] = self.m_eq
        else:
            scm_kwargs["m_min"] = self.m_min
            scm_kwargs["m_max"] = self.m_max

        result = SCMEXP(**scm_kwargs)

        return result


