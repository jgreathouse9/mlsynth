import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic # For ValidationError
import cvxpy # For cvxpy.error types

from ..utils.datautils import balance, dataprep
from ..utils.sdidutils import estimate_event_study_sdid
from ..utils.resultutils import SDID_plot
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import ( # Import the Pydantic model
    SDIDConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class SDID:
    """
    Synthetic Difference-in-Differences (SDID) estimator.

    This class implements the Synthetic Difference-in-Differences (SDID) method,
    which combines synthetic control weighting for units and time periods with a
    difference-in-differences estimation framework. It is designed for panel data
    settings, typically with a single treated unit (or a single treatment adoption
    time across multiple units treated simultaneously) and multiple control units,
    observed over pre- and post-treatment periods.

    The core estimation is performed by `estimate_event_study_sdid` from
    `sdidutils`. Inference, if requested (B > 0), is conducted using placebo tests
    by re-assigning treatment status to control units. The results include an
    overall ATT estimate and event-study style estimates for dynamic effects.

    Attributes
    ----------
    config : SDIDConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    display_graphs : bool, default True
        Whether to display the event study plot of results.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.

        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "event_study_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    counterfactual_color : str, default "red"
        Color for counterfactual lines in plots. Note: SDID's default plot (`SDID_plot`)
        has its own color scheme and may not use this directly.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    treated_color : str, default "black"
        Color for treated unit lines in plots. Note: SDID's default plot (`SDID_plot`)
        has its own color scheme and may not use this directly.
        (Inherited from `BaseEstimatorConfig` via `SDIDConfig`)
    B : int, default 500
        Number of placebo iterations for inference. If 0, no placebo tests are run.
        (From `SDIDConfig`)
    """

    def __init__(self, config: SDIDConfig) -> None: # Changed to SDIDConfig
        """
        Initializes the SDID estimator with a configuration object.

        Parameters
        ----------
        config : SDIDConfig
            A Pydantic model instance containing all configuration parameters
            for the SDID estimator. This includes:

            - df (pd.DataFrame): The input DataFrame.
            - outcome (str): Name of the outcome variable column.
            - treat (str): Name of the treatment indicator column.
            - unitid (str): Name of the unit identifier column.
            - time (str): Name of the time variable column.
            - display_graphs (bool, optional): Whether to display graphs. Defaults to True.
            - save (Union[bool, str, Dict[str, str]], optional): Configuration for saving plots.
              If `False` (default), plots are not saved. If `True`, plots are saved with
              default names. If a `str`, it's used as the base filename. If a `Dict[str, str]`,
              it maps plot keys to full file paths. Defaults to False.
            - counterfactual_color (str, optional): Color for counterfactual line. Defaults to "red".
            - treated_color (str, optional): Color for treated unit line. Defaults to "black".
            - B (int, optional): Number of placebo iterations for inference. Defaults to 500.
        """
        if isinstance(config, dict):
            config =SDIDConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save # Kept Union for flexibility
        self.B: int = config.B

    def _create_estimator_results(
        self, raw_sdid_estimation_output: Dict[str, Any], prepared_panel_data: Dict[str, Any]
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object from raw SDID outputs.

        This helper function takes the raw outputs from `estimate_event_study_sdid`
        and maps them to the standardized `BaseEstimatorResults` Pydantic model.

        Parameters
        ----------
        raw_sdid_estimation_output : Dict[str, Any]
            A dictionary containing the results from `estimate_event_study_sdid`.
            Expected keys include 'att', 'att_se', 'att_ci', 'pooled_estimates',
            'placebo_att_values', and potentially others like 'cohort_estimates'.
        prepared_panel_data : Dict[str, Any]
            The dictionary of preprocessed data originally passed to or created by
            `dataprep`. This is used here for context if needed (e.g., donor names,
            though currently not directly used for weights in this helper).

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results
            for the SDID method. Key fields include:
            - effects (EffectsResults): Contains the main treatment effect estimates.
                - att (Optional[float]): The overall Average Treatment Effect on the Treated.
                - additional_effects (Optional[Dict[str, Any]]): May include 'att_standard_error'
                  and 'att_confidence_interval' if available from `fit_output`.
            - fit_diagnostics (FitDiagnosticsResults): Typically empty for SDID, as standard
              fit metrics like RMSE/R-squared are not primary outputs of this method.
            - time_series (TimeSeriesResults): Contains event study style results.
                - time_periods (Optional[List[Union[int, float]]]): Event time periods
                  (e.g., -2, -1, 0, 1, 2 relative to treatment).
                - estimated_gap (Optional[List[float]]): The estimated dynamic treatment
                  effects (taus) for each event time period.
                (Note: `observed_outcome` and `counterfactual_outcome` are generally not
                populated for SDID in the same way as for SCM, as the primary output is
                the series of ATT_t estimates.)
            - weights (WeightsResults): Currently not populated, as unit (omega) and time
              (lambda) weights are internal to `sdidutils` and not directly returned
              by `estimate_event_study_sdid` to be mapped here.
            - inference (InferenceResults): Contains statistical inference details.
                - method (Optional[str]): Description of the inference method used
                  (e.g., "Synthetic Difference-in-Differences with Placebo Inference").
                - p_value (Optional[float]): P-value for the overall ATT, derived from
                  placebo tests if `B > 0`.
                - confidence_interval (Optional[List[float]]): Confidence interval for
                  the overall ATT.
            - method_details (MethodDetailsResults): Contains detailed outputs and parameters.
                - name (Optional[str]): Name of the method ("SDID").
                - parameters_used (Optional[Dict[str, Any]]): Configuration parameters
                  used in the estimation (excluding the DataFrame).
                - additional_details (Optional[Dict[str, Any]]): Dictionary of detailed
                  outputs, including 'cohort_estimates', 'tau_a_ell', 'tau_ell',
                  'placebo_att_values', and various mean outcome values for treated
                  and donor groups in pre/post periods.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              `estimate_event_study_sdid`.
        """
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
        """
        Fits the Synthetic Difference-in-Differences (SDID) model.

        This method orchestrates the data preparation and estimation process.
        It first balances the panel data. Then, it prepares the data using
        `dataprep` and ensures it's in a "cohorts" structure suitable for
        `estimate_event_study_sdid` (if a single treatment event is implied).
        The core estimation, including placebo tests if `B > 0`, is performed by
        `estimate_event_study_sdid`. The raw results are then mapped to the
        standardized `BaseEstimatorResults` Pydantic model. An event study plot
        can be displayed if `display_graphs` is True.

        Returns
        -------
        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:

            - effects (EffectsResults)
                Contains the overall Average Treatment Effect on the Treated (ATT).
                May also include standard error and confidence interval for the ATT
                in `additional_effects`.
            - time_series (TimeSeriesResults)
                Provides event study style results, with `time_periods`
                representing event times relative to treatment and `estimated_gap`
                holding the dynamic treatment effects (taus) for these periods.
            - inference (InferenceResults)
                Includes a description of the inference method (e.g., placebo-based
                if `B > 0`), the p-value for the overall ATT if placebo tests
                were run, and potentially the confidence interval for the ATT.
            - method_details (MethodDetailsResults)
                Contains the method name ("SDID"), the configuration parameters
                used, and a dictionary of `additional_details` which includes
                detailed outputs like cohort-specific estimates, raw placebo ATT
                values, and various mean outcome values for treated and donor groups.
            - fit_diagnostics (FitDiagnosticsResults)
                Typically not populated with standard metrics like RMSE or R-squared,
                as these are not primary outputs of SDID.
            - weights (WeightsResults)
                Currently not populated. While SDID calculates unit (omega) and
                time (lambda) weights internally, these are not directly returned by
                `estimate_event_study_sdid` in a way that's mapped here.
            - raw_results (Optional[Dict[str, Any]])
                The complete raw dictionary output from the
                `estimate_event_study_sdid` function.

        Examples
        --------
        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.estimators.sdid import SDID
        >>> from mlsynth.config_models import SDIDConfig
        >>> # Load or create panel data
        >>> data = pd.DataFrame({
        ...     'state': [1,1,1,1, 2,2,2,2, 3,3,3,3],
        ...     'year': [1990,1991,1992,1993, 1990,1991,1992,1993, 1990,1991,1992,1993],
        ...     'value': [10,12,11,15, 20,22,25,23, 15,16,14,18],
        ...     'treated_state': [0,0,1,1, 0,0,0,0, 0,0,0,0] # State 1 treated from 1992
        ... })
        >>> sdid_config = SDIDConfig(
        ...     df=data, outcome="value", treat="treated_state",
        ...     unitid="state", time="year", B=50, # B=50 for a quick example, use more in practice
        ...     display_graphs=False
        ... )
        >>> sdid_estimator = SDID(config=sdid_config)
        >>> results = sdid_estimator.fit()
        >>> print(f"Estimated ATT: {results.effects.att}")
        >>> if results.inference:
        ...     print(f"P-value: {results.inference.p_value}")
        >>> if results.time_series and results.time_series.estimated_gap is not None:
        ...     print(f"Event study estimates (taus): {results.time_series.estimated_gap}")
        """
        try:
            # Ensure the panel data is balanced (all units observed for all time periods).
            balance(self.df, self.unitid, self.time)

            # Prepare the data into a structured format required by the estimation utilities.
            # This involves identifying treated/control units, pre/post periods, etc.
            prepared_panel_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # The `estimate_event_study_sdid` function expects data structured by "cohorts"
            # (groups of units treated at the same time). If `dataprep` returns data for a
            # single treatment event (common case, e.g., one treated unit or all treated units
            # adopt treatment simultaneously), it might not have the "cohorts" key.
            # This block constructs the "cohorts" structure if it's missing.
            if "cohorts" not in prepared_panel_data:
                pre_periods_val = prepared_panel_data.get("pre_periods")
                post_periods_val = prepared_panel_data.get("post_periods")
                total_periods_val = prepared_panel_data.get("total_periods")

                if pre_periods_val is None or post_periods_val is None:
                    raise MlsynthDataError(
                        "dataprep output missing pre_periods or post_periods for single unit case"
                    )

                if total_periods_val is None:
                    warnings.warn(
                        "'total_periods' missing from dataprep single-unit output. "
                        "Calculating from pre_periods + post_periods.",
                        UserWarning,
                    )
                    total_periods_val = pre_periods_val + post_periods_val # Calculate if missing.
                
                # For a single treatment event, the "cohort time" is effectively the number of pre-periods.
                # This is used as the key in the "cohorts" dictionary.
                cohort_time_key: int = pre_periods_val
                prepared_panel_data["cohorts"] = {
                    cohort_time_key: { # Structure for this single cohort.
                        "y": prepared_panel_data["y"].reshape(-1, 1), # Outcome data.
                        "donor_matrix": prepared_panel_data["donor_matrix"],
                        "treated_indices": [prepared_panel_data["treated_unit_name"]], # Use actual treated unit name
                        "pre_periods": pre_periods_val,
                        "post_periods": post_periods_val,
                        "total_periods": total_periods_val, # Total number of time periods.
                    }
                }
            
            # Perform the core SDID estimation using the prepared (and potentially restructured) data.
            # This includes calculating unit and time weights, estimating effects, and running placebo tests if B > 0.
            raw_sdid_estimation_output: Dict[str, Any] = estimate_event_study_sdid(
                prepared_panel_data, placebo_iterations=self.B
            )

            # Map the raw estimation output to the standardized BaseEstimatorResults Pydantic model.
            results_obj = self._create_estimator_results(raw_sdid_estimation_output, prepared_panel_data)

        except MlsynthDataError: # Re-raise specific data-related errors from utilities.
            raise
        except MlsynthConfigError: # Re-raise specific configuration-related errors.
            raise
        except MlsynthEstimationError: # Errors from estimate_event_study_sdid
            raise
        except pydantic.ValidationError as e_val:
            raise MlsynthEstimationError(f"Error validating SDID results: {e_val}") from e_val
        except (cvxpy.error.SolverError, cvxpy.error.DCPError) as e_cvx:
             raise MlsynthEstimationError(f"CVXPY solver error in SDID: {e_cvx}") from e_cvx
        except KeyError as e_key:
            raise MlsynthDataError(f"Missing expected key during SDID data processing: {e_key}") from e_key
        except ValueError as e_val_general: # Catch other ValueErrors not already MlsynthDataError
            raise MlsynthDataError(f"ValueError during SDID processing: {e_val_general}") from e_val_general
        except TypeError as e_type:
            raise MlsynthEstimationError(f"Type error during SDID estimation: {e_type}") from e_type
        except IndexError as e_idx:
            raise MlsynthDataError(f"Index out of bounds during SDID data processing: {e_idx}") from e_idx
        except AttributeError as e_attr: # Should be caught by Pydantic or earlier, but as a safeguard
            raise MlsynthEstimationError(f"Attribute error during SDID estimation: {e_attr}") from e_attr
        except np.linalg.LinAlgError as e_linalg:
            raise MlsynthEstimationError(f"Linear algebra error in SDID: {e_linalg}") from e_linalg
        except Exception as e: # Catch-all for unexpected errors
            raise MlsynthEstimationError(f"An unexpected error occurred in SDID fit: {e}") from e

        # If display_graphs is enabled, attempt to generate and show the event study plot.
        if self.display_graphs:
            try:
                # SDID_plot is a utility function specifically designed for visualizing SDID results.
                SDID_plot(raw_sdid_estimation_output) 
            except MlsynthPlottingError as e_plot: # Catch specific plotting errors defined in the library.
                warnings.warn(f"SDID plotting failed: {e_plot}", UserWarning)
            except MlsynthDataError as e_plot_data: # Catch data errors that might occur during plotting.
                 warnings.warn(f"SDID plotting failed due to data issue: {e_plot_data}", UserWarning)
            except Exception as e_plot_general: # Catch any other unexpected errors during plotting.
                warnings.warn(f"An unexpected error occurred during SDID plotting: {e_plot_general}", UserWarning)
            
        return results_obj
