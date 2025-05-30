import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic

from ..utils.datautils import balance, dataprep, proxy_dataprep, clean_surrogates2
from ..utils.resultutils import effects, plot_estimates
from ..utils.estutils import pi, pi_surrogate, pi_surrogate_post
from ..config_models import (
    PROXIMALConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)


class PROXIMAL:
    """
    Implements the Proximal Inference framework for causal effect estimation.

    This class allows for the estimation of treatment effects using several
    proximal inference approaches, including:
    1.  **Proximal Inference (PI):** Uses donor units and their proxies.
    2.  **Proximal Inference with Surrogates (PIS):** Incorporates surrogate
        outcomes and their proxies, in addition to donor units and their proxies.
    3.  **Proximal Inference Post-Surrogates (PIPost):** A variation of PIS.

    The methods are based on the work by Shi et al. (2023) and Liu et al. (2023),
    adapting synthetic control ideas within a proximal causal inference framework.
    The `fit` method can return results for one or all of these approaches,
    depending on whether surrogate information is provided.

    Attributes
    ----------
    config : PROXIMALConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
    outcome : str
        Name of the outcome variable column in `df`.
    treat : str
        Name of the treatment indicator column in `df`.
    unitid : str
        Name of the unit identifier column in `df`.
    time : str
        Name of the time variable column in `df`.
    display_graphs : bool, default True
        Whether to display graphs of results.
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
    counterfactual_color : Union[str, List[str]], default_factory=lambda: ["grey", "red", "blue"]
        Color(s) for the counterfactual line(s) in plots.
    treated_color : str, default "black"
        Color for the treated unit line in plots.
    donors : List[Union[str, int]]
        List of donor unit identifiers used to construct the counterfactual.
    surrogates : List[Union[str, int]], default_factory=list
        List of surrogate unit identifiers.
    vars : Dict[str, List[str]], default_factory=dict
        Dictionary specifying proxy variables.

    References
    ----------
    Xu Shi, Kendrick Li, Wang Miao, Mengtong Hu, and Eric Tchetgen Tchetgen.
    "Theory for identification and Inference with Synthetic Controls: A Proximal Causal Inference Framework."
    arXiv preprint arXiv:2108.13935, 2023. URL: https://arxiv.org/abs/2108.13935.

    Jizhou Liu, Eric J. Tchetgen Tchetgen, and Carlos Varjão.
    "Proximal Causal Inference for Synthetic Control with Surrogates."
    arXiv preprint arXiv:2308.09527, 2023. URL: https://arxiv.org/abs/2308.09527.
    """

    def __init__(self, config: PROXIMALConfig) -> None:
        if isinstance(config, dict):
            config =PROXIMALConfig(**config)  # convert dict to config object
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save
        self.surrogates: List[Union[str, int]] = config.surrogates
        self.donors: List[Union[str, int]] = config.donors
        self.vars: Dict[str, List[str]] = config.vars

    def _create_proximal_estimator_results( # Add try-except for robustness
        self,
        method_name: str, # Name of the proximal method (e.g., "PI", "PIS", "PIPost")
        raw_effects_dict: Dict[str, Any], # Raw effects dictionary from effects.calculate
        raw_fit_dict: Dict[str, Any],     # Raw fit diagnostics dictionary from effects.calculate
        counterfactual_outcome_array: np.ndarray, # Estimated counterfactual series
        att_standard_error_float: Optional[float], # Standard error of the ATT
        alpha_weights: Optional[np.ndarray],       # Estimated alpha weights from proximal method
        prepared_data_dict: Dict[str, Any],        # Output from dataprep utility
    ) -> BaseEstimatorResults:
        try:
            # --- Effects ---
            effects = EffectsResults(
                att=raw_effects_dict.get("ATT"),
                att_percent=raw_effects_dict.get("Percent ATT"),
                att_std_err=att_standard_error_float if att_standard_error_float is not None else None,
            )

            # --- Fit Diagnostics ---
            fit_diagnostics = FitDiagnosticsResults(
                rmse_pre=raw_fit_dict.get("T0 RMSE"), # Pre-treatment RMSE
                rmse_post=raw_fit_dict.get("T1 RMSE"),# Post-treatment RMSE
                r_squared_pre=raw_fit_dict.get("R-Squared"), # Pre-treatment R-squared
            )
            
            # --- Time Series Data ---
            # Extract and process observed outcome series
            observed_outcome_series = prepared_data_dict.get("y") # 'y' from dataprep is the treated unit's outcome
            observed_outcome_arr: Optional[np.ndarray] = None
            if observed_outcome_series is not None:
                if isinstance(observed_outcome_series, pd.Series):
                    observed_outcome_arr = observed_outcome_series.to_numpy().flatten()
                elif isinstance(observed_outcome_series, np.ndarray):
                    observed_outcome_arr = observed_outcome_series.flatten()
                else: # Handle list or other iterable cases
                    observed_outcome_arr = np.array(observed_outcome_series).flatten()

            # Ensure counterfactual outcome is a 1D NumPy array
            counterfactual_outcome_1d_array = counterfactual_outcome_array.flatten() if isinstance(counterfactual_outcome_array, np.ndarray) else np.array(counterfactual_outcome_array).flatten()
            
            # Calculate the estimated gap (treatment effect over time)
            gap_arr: Optional[np.ndarray] = None
            if observed_outcome_arr is not None and counterfactual_outcome_1d_array is not None and len(observed_outcome_arr) == len(counterfactual_outcome_1d_array):
                gap_arr = observed_outcome_arr - counterfactual_outcome_1d_array
                
            # Extract time periods from 'Ywide' in prepared_data_dict
            all_time_periods_arr: Optional[np.ndarray] = None
            ywide_data = prepared_data_dict.get("Ywide") # 'Ywide' from dataprep has time periods as index
            if ywide_data is not None and isinstance(ywide_data, pd.DataFrame):
                all_time_periods_arr = ywide_data.index.to_numpy()
            
            time_series = TimeSeriesResults(
                observed_outcome=observed_outcome_arr,
                counterfactual_outcome=counterfactual_outcome_1d_array,
                estimated_gap=gap_arr, 
                time_periods=all_time_periods_arr, # Pass the NumPy array of time periods
            )

            # --- Inference ---
            # Store the ATT standard error if available
            inference = InferenceResults(
                standard_error=att_standard_error_float if att_standard_error_float is not None else None,
            )

            # --- Method Details ---
            # Store method name and alpha weights (if available)
            method_details = MethodDetailsResults(
                method_name=method_name,
                parameters_used={"alpha_weights": alpha_weights.tolist() if alpha_weights is not None else None}
            )
            
            # --- Raw Results Packaging ---
            # Reconstruct a dictionary of vectors similar to what other estimators might produce for raw_results
            raw_vectors: Dict[str, Optional[np.ndarray]] = {
                 "Observed Unit": observed_outcome_arr.reshape(-1,1) if observed_outcome_arr is not None else None,
                 "Counterfactual": counterfactual_outcome_1d_array.reshape(-1,1) if counterfactual_outcome_1d_array is not None else None,
                 "Gap": gap_arr.reshape(-1,1) if gap_arr is not None else None,
            }

            # Assemble the method-specific raw output dictionary
            method_specific_raw_output = {
                "Effects": raw_effects_dict,
                "Fit": raw_fit_dict,
                "Vectors": raw_vectors, # Store the reconstructed vectors
                "se_tau": att_standard_error_float if att_standard_error_float is not None else None, # Standard error of ATT
                "alpha_weights": alpha_weights.tolist() if alpha_weights is not None else None, # Alpha weights
            }

            # Create the final BaseEstimatorResults object
            return BaseEstimatorResults(
                effects=effects,
                fit_diagnostics=fit_diagnostics,
                time_series=time_series,
                inference=inference,
                method_details=method_details,
                raw_results=method_specific_raw_output, # Store the assembled raw output
            )
        except Exception as e:
            # Catch any error during results object creation and wrap it.
            raise MlsynthEstimationError(f"Error creating results object for {method_name}: {e}")


    def fit(self) -> List[BaseEstimatorResults]:
        try:
            # Step 1: Balance the input panel data.
            balance(self.df, self.unitid, self.time)

            # Step 2: Prepare main data (outcomes for treated unit, outcomes for donor units).
            # `dataprep` returns a dictionary with 'y', 'Ywide', 'donor_names', 'pre_periods', etc.
            prepared_data = dataprep(self.df, self.unitid, self.time, self.outcome, self.treat)
            
            # Filter configured donors to only include those present in the prepared data.
            valid_donors = [
                donor for donor in self.donors if donor in prepared_data["Ywide"].columns
            ]
            # Extract outcome matrix for valid donor units.
            donor_outcome_matrix = prepared_data["Ywide"][valid_donors].to_numpy()

            # Step 3: Prepare donor proxy data.
            # Pivot the DataFrame to get proxy variables for donor units over time.
            # Assumes `self.vars["donorproxies"][0]` contains the column name of the donor proxy variable.
            donor_proxy_pivot_df = self.df.pivot(
                index=self.time, columns=self.unitid, values=self.vars["donorproxies"][0]
            )
            donor_proxy_matrix = donor_proxy_pivot_df[valid_donors].to_numpy()

            # Step 4: Prepare surrogate data if surrogate units are specified in the config.
            surrogate_main_outcome_matrix: Optional[np.ndarray] = None
            surrogate_proxy_matrix: Optional[np.ndarray] = None
            if self.surrogates: # If surrogate units are provided
                # `proxy_dataprep` prepares surrogate outcomes (X_temp) and surrogate proxies (Z1_temp).
                X_temp, Z1_temp = proxy_dataprep(
                    df=self.df,
                    surrogate_units=self.surrogates,
                    proxy_variable_column_names_map=self.vars, # Contains mappings for surrogate proxies
                    unit_id_column_name=self.unitid,
                    time_period_column_name=self.time,
                    num_total_periods=prepared_data["total_periods"],
                )
                surrogate_main_outcome_matrix = X_temp
                surrogate_proxy_matrix = Z1_temp

            # Step 5: Calculate bandwidth parameter 'h', used in proximal inference methods.
            # This is a rule-of-thumb bandwidth, often used in kernel-based methods.
            bandwidth_parameter_h = int(np.floor(4 * (prepared_data["post_periods"] / 100) ** (2 / 9)))

            # --- Proximal Inference (PI) ---
            # Estimate counterfactual, alpha weights, and ATT standard error using the PI method.
            pi_counterfactual_outcome, pi_alpha_weights, pi_att_std_err = pi(
                prepared_data["y"], # Treated unit outcomes
                donor_outcome_matrix,   # Donor unit outcomes
                donor_proxy_matrix,     # Donor unit proxies
                prepared_data["pre_periods"],
                prepared_data["post_periods"],
                prepared_data["total_periods"],
                bandwidth_parameter_h,
            )
            # Calculate effects (ATT, %ATT) and fit diagnostics (RMSE, R^2) for PI.
            pi_raw_effects, pi_raw_fit_diagnostics, _ = effects.calculate(
                prepared_data["y"], pi_counterfactual_outcome, prepared_data["pre_periods"], prepared_data["post_periods"]
            )
            results_list: List[BaseEstimatorResults] = []
            # Package PI results into the standardized Pydantic model.
            pi_results_object = self._create_proximal_estimator_results(
                method_name="PI",
                raw_effects_dict=pi_raw_effects,
                raw_fit_dict=pi_raw_fit_diagnostics,
                counterfactual_outcome_array=pi_counterfactual_outcome,
                att_standard_error_float=pi_att_std_err,
                alpha_weights=pi_alpha_weights,
                prepared_data_dict=prepared_data,
            )
            results_list.append(pi_results_object)
            # Initialize lists for plotting multiple counterfactuals if surrogates are used.
            plot_counterfactuals = [pi_counterfactual_outcome]
            plot_counterfactual_names = ["Proximal Inference"]

            # --- Proximal Inference with Surrogates (PIS & PIPost) ---
            # These methods are applied only if surrogate data is available.
            if self.surrogates and surrogate_main_outcome_matrix is not None and surrogate_proxy_matrix is not None:
                # Clean/prepare surrogate data relative to donor proxies and outcomes.
                # `clean_surrogates2` likely orthogonalizes or adjusts surrogate outcomes.
                cleaned_surrogate_data = clean_surrogates2(
                    surrogate_main_outcome_matrix, donor_proxy_matrix, donor_outcome_matrix, prepared_data["pre_periods"]
                )

                # --- PIS (Proximal Inference with Surrogates) ---
                # Estimate treatment effect series, alpha weights, and ATT SE using PIS.
                _, pis_estimated_treatment_effects_timeseries, pis_alpha_weights, pis_att_std_err = pi_surrogate(
                    prepared_data["y"], donor_outcome_matrix, donor_proxy_matrix, surrogate_proxy_matrix,
                    cleaned_surrogate_data, # Use the cleaned surrogate data
                    prepared_data["pre_periods"], prepared_data["post_periods"], prepared_data["total_periods"], bandwidth_parameter_h,
                )
                # Derive PIS counterfactual: Observed Outcome - Estimated Treatment Effect.
                pis_counterfactual_outcome = prepared_data["y"] - pis_estimated_treatment_effects_timeseries # type: ignore
                pis_raw_effects, pis_raw_fit_diagnostics, _ = effects.calculate(
                    prepared_data["y"], pis_counterfactual_outcome, prepared_data["pre_periods"], prepared_data["post_periods"]
                )
                pis_results_object = self._create_proximal_estimator_results(
                    method_name="PIS", raw_effects_dict=pis_raw_effects, raw_fit_dict=pis_raw_fit_diagnostics,
                    counterfactual_outcome_array=pis_counterfactual_outcome, att_standard_error_float=pis_att_std_err, alpha_weights=pis_alpha_weights,
                    prepared_data_dict=prepared_data,
                )
                results_list.append(pis_results_object)
                plot_counterfactuals.append(pis_counterfactual_outcome)
                plot_counterfactual_names.append("Proximal Surrogates")

                # --- PIPost (Proximal Inference Post-Surrogates) ---
                # Estimate treatment effect series, alpha weights, and ATT SE using PIPost.
                _, pipost_estimated_treatment_effects_timeseries, pipost_alpha_weights, pipost_att_std_err = pi_surrogate_post(
                    prepared_data["y"], donor_outcome_matrix, donor_proxy_matrix, surrogate_proxy_matrix,
                    cleaned_surrogate_data, # Use the cleaned surrogate data
                    prepared_data["pre_periods"], prepared_data["post_periods"], prepared_data["total_periods"], bandwidth_parameter_h,
                )
                # Derive PIPost counterfactual.
                pipost_counterfactual_outcome = prepared_data["y"] - pipost_estimated_treatment_effects_timeseries # type: ignore
                pipost_raw_effects, pipost_raw_fit_diagnostics, _ = effects.calculate(
                    prepared_data["y"], pipost_counterfactual_outcome, prepared_data["pre_periods"], prepared_data["post_periods"]
                )
                pipost_results_object = self._create_proximal_estimator_results(
                    method_name="PIPost", raw_effects_dict=pipost_raw_effects, raw_fit_dict=pipost_raw_fit_diagnostics,
                    counterfactual_outcome_array=pipost_counterfactual_outcome, att_standard_error_float=pipost_att_std_err, alpha_weights=pipost_alpha_weights,
                    prepared_data_dict=prepared_data,
                )
                results_list.append(pipost_results_object)
                plot_counterfactuals.append(pipost_counterfactual_outcome)
                plot_counterfactual_names.append("Proximal Post")

        except MlsynthDataError: # Re-raise MlsynthDataError as it's already specific
            # Custom data-related errors from utilities like dataprep.
            raise
        except MlsynthConfigError: # Re-raise MlsynthConfigError
            # Custom configuration-related errors.
            raise
        except (ValueError, np.linalg.LinAlgError, KeyError, IndexError) as e:
            # Catch common errors during numerical computation or data manipulation.
            raise MlsynthEstimationError(f"Proximal estimation failed: {type(e).__name__}: {e}") from e
        except pydantic.ValidationError as e:
            # Catch errors during Pydantic model validation when creating results.
            raise MlsynthEstimationError(f"Failed to create Pydantic results model in Proximal: {e}") from e
        except Exception as e: # Catch any other unexpected errors
            # General catch-all for other unforeseen issues.
            raise MlsynthEstimationError(f"An unexpected error occurred during Proximal fit: {type(e).__name__}: {e}") from e

        # Step 6: Optionally display or save plots of the results.
        if self.display_graphs:
            try:
                plot_estimates(
                    processed_data_dict=prepared_data, # Pass the prepared data dictionary
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treat,
                    treated_unit_name=prepared_data["treated_unit_name"], # Name of the treated unit
                    observed_outcome_series=prepared_data["y"], # Observed outcomes of the treated unit
                    counterfactual_series_list=plot_counterfactuals, # List of counterfactual series (PI, PIS, PIPost)
                    estimation_method_name="PI", # Base method name for plot title (can be generic for multiple series)
                    counterfactual_names=plot_counterfactual_names, # Names for legend
                    treated_series_color=self.treated_color,
                    save_plot_config=self.save if isinstance(self.save, str) else None, # Path if save is a string
                    counterfactual_series_colors=[self.counterfactual_color, "green", "red"], # Colors for counterfactual lines
                )
            except (MlsynthPlottingError, MlsynthDataError) as e:
                # Warn if plotting fails due to known plotting or data issues.
                warnings.warn(f"Plotting failed in Proximal estimator: {e}", UserWarning)
            except Exception as e: # Catch any other unexpected plotting errors
                # Warn for any other unexpected plotting errors.
                warnings.warn(f"An unexpected error occurred during Proximal plotting: {type(e).__name__}: {e}", UserWarning)
        
        # Step 7: Return the list of results objects (one for each method: PI, PIS, PIPost).
        return results_list
