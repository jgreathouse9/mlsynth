import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic
from mlsynth.utils.inferutils import ag_conformal
from ..utils.datautils import dataprep, build_donor_segments
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, MlsynthPlottingError
from ..utils.resultutils import effects, plot_estimates
from ..utils.selector_helpers import stepwise_donor_selection
from ..utils.estutils import smooth, loocv_bandwidth, tune_lambda_ashc, _solve_SHC_QP
from ..config_models import ( # Import Pydantic models
    SHCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class SHC:
    """
    Implements the Synthetic Historical Control (SRC) method for estimating treatment effects.

    This method....

    Attributes
    ----------
    config : SHCConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `SHCConfig`)

    Methods
    -------
    fit()
        Fits the SHC model and returns standardized estimation results.

    References
    ----------

    Zhu, Rong J. B. "Synthetic Regressing Control Method." arXiv preprint arXiv:2306.02584 (2023).
    https://arxiv.org/abs/2306.02584
    """

    def __init__(self, config: SHCConfig) -> None: # Changed to SHCConfig
        """
        Initializes the SHC estimator with a configuration object.

        Parameters
        ----------

        config : SHCConfig
            A Pydantic model instance containing all configuration parameters
            for the ShC estimator. Since `SHCConfig` inherits directly from
            `BaseEstimatorConfig` without adding new fields, this includes:
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
        """
        if isinstance(config, dict):
            config = SHCConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save # Align with BaseEstimatorConfig
        self.m: int = config.m
        self.use_augmented: bool = config.use_augmented

    def _create_estimator_results(
        self,
        combined_raw_estimation_output: Dict[str, Any],
        prepared_panel_data: Dict[str, Any],
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object from raw SHC outputs.

        This helper function takes the raw outputs from the SHC estimation process
        (which includes results from `SRCest` and `effects.calculate`) and maps
        them to the standardized `BaseEstimatorResults` Pydantic model structure.

        Parameters
        ----------

        combined_raw_estimation_output : Dict[str, Any]
            A dictionary containing the combined raw results. This should include
            keys like 'Counterfactual', 'Weights' (from `SRCest`), and 'ATT',
            'Fit', 'Vectors' (from `effects.calculate`).
        prepared_panel_data : Dict[str, Any]
            The dictionary of preprocessed data from `dataprep`, containing elements
            like 'y' (treated unit outcomes), 'time_labels', and 'donor_names'.
        theta_hat_val : np.ndarray
            The estimated theta_hat parameters from the `SRCest` function.
            Shape: (number_of_donors + 1,).

        Returns
        -------

        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results
            for the SHC method. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates like ATT
              and percentage ATT.
            - fit_diagnostics (FitDiagnosticsResults): Includes goodness-of-fit metrics
              such as pre-treatment RMSE, post-treatment RMSE, and pre-treatment R-squared.
            - time_series (TimeSeriesResults): Provides time-series data including the
              observed outcome for the treated unit, its estimated counterfactual outcome,
              the estimated treatment effect (gap) over time, and the corresponding
              time periods.
            - weights (WeightsResults): Contains an array of donor weights and a list of
              corresponding donor names.
            - inference (InferenceResults): Typically not populated by SRC's core logic,
              as it primarily provides point estimates.
            - method_details (MethodDetailsResults): Details about the estimation method,
              including its name ("SRC") and the estimated `theta_hat` parameters in
              `custom_params`.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the estimation process, combining results from `SRCest` and
              `effects.calculate`.
        """
        try:
            att_dict_raw = combined_raw_estimation_output.get("Effects", {})
            fit_dict_raw = combined_raw_estimation_output.get("Fit", {})
            vectors_dict_raw = combined_raw_estimation_output.get("Vectors", {})
            counterfactual_y_arr = combined_raw_estimation_output.get("Counterfactual")
            donor_weights_dict_raw = combined_raw_estimation_output.get("Weights", {})

            effects = EffectsResults(
                att=att_dict_raw.get("ATT"),
                att_percent=att_dict_raw.get("Percent ATT"),
            )

            fit_diagnostics = FitDiagnosticsResults(
                rmse_pre=fit_dict_raw.get("T0 RMSE"),
                rmse_post=fit_dict_raw.get("T1 RMSE"),
                r_squared_pre=fit_dict_raw.get("R-Squared"),
            )

            observed_outcome_arr: Optional[np.ndarray] = None
            # Attempt to get the observed outcome series.
            # Primary source is 'Vectors' from effects.calculate, fallback to 'y' from dataprep.
            if vectors_dict_raw.get("Observed Unit") is not None:
                 observed_outcome_arr = np.array(vectors_dict_raw["Observed Unit"]).flatten()
            elif prepared_panel_data.get("y") is not None: # Fallback if not in 'Vectors'.
                y_series = prepared_panel_data.get("y")
                # Ensure 'y' (which might be a pd.Series or np.ndarray from dataprep) is a flat NumPy array.
                observed_outcome_arr = y_series.to_numpy().flatten() if isinstance(y_series, pd.Series) else np.array(y_series).flatten()

            # Ensure the counterfactual outcome series is a flat NumPy array.
            cf_outcome_arr_flat: Optional[np.ndarray] = None
            if counterfactual_y_arr is not None:
                cf_outcome_arr_flat = np.array(counterfactual_y_arr).flatten()

            # Calculate the gap (treatment effect) series.
            # Primary calculation is observed - counterfactual. Fallback to 'Gap' from 'Vectors'.
            gap_arr: Optional[np.ndarray] = None
            if observed_outcome_arr is not None and cf_outcome_arr_flat is not None and \
               len(observed_outcome_arr) == len(cf_outcome_arr_flat): # Ensure compatible shapes for subtraction.
                gap_arr = observed_outcome_arr - cf_outcome_arr_flat
            elif vectors_dict_raw.get("Gap") is not None: # Fallback if direct calculation is not possible.
                gap_arr = np.array(vectors_dict_raw["Gap"]).flatten()
                
            # Get time periods from dataprep output.
            time_periods_arr: Optional[np.ndarray] = None
            time_labels = prepared_panel_data.get("time_labels") # These are the actual time values (e.g., years).
            if time_labels is not None:
                time_periods_arr = np.array(time_labels)

            time_series = TimeSeriesResults(
                observed_outcome=observed_outcome_arr, # The observed outcome of the treated unit.
                counterfactual_outcome=cf_outcome_arr_flat,
                gap=gap_arr,
                time_periods=time_periods_arr, # Use the ndarray
            )
            
            donor_names_list = list(donor_weights_dict_raw.keys())
            # donor_weights_values_list = list(donor_weights_dict_raw.values()) # Not used

            #weights = WeightsResults(
              #  donor_weights=donor_weights_dict_raw if donor_weights_dict_raw else None,
                #donor_names=donor_names_list if donor_names_list else None,
            #)
            
            inference = InferenceResults(**inference_kwargs)

            method_details = MethodDetailsResults(
                method_name="SHC" # Name of the estimation method.
                # theta_hat represents the estimated regression coefficients from the SHC model.
                # It includes coefficients for donor outcomes and potentially an intercept.
            )

            return BaseEstimatorResults(
                effects=effects,
                fit_diagnostics=fit_diagnostics,
                time_series=time_series, #weights=weights,
                inference=inference,
                method_details=method_details,
                raw_results=combined_raw_estimation_output,
            )
        except pydantic.ValidationError as e:
            raise MlsynthEstimationError(f"Error creating Pydantic results model for SRC: {e}") from e
        except Exception as e:
            raise MlsynthEstimationError(f"Unexpected error in _create_estimator_results for SRC: {type(e).__name__}: {e}") from e

    def fit(self) -> BaseEstimatorResults:
        """
        Fits the Synthetic Historical Control (SHC) model to the provided data.

        This method performs the following steps:
        1. Prepares the data using `dataprep` to structure outcomes for the
           treated unit (`y1`)
        2. Calls the `_SHC_est` utility function, which implements the core SHC
           algorithm to estimate the counterfactual outcome (`y_SHC`), donor
           weights (`w_hat`), and regression coefficients (`theta_hat`).
        3. Calculates treatment effects (ATT) and goodness-of-fit statistics
           using the observed and counterfactual outcomes.
        4. Optionally plots the observed and counterfactual outcomes.
        5. Returns a `BaseEstimatorResults` object containing all relevant outputs.

        Returns
        -------

        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates such as
              Average Treatment Effect on the Treated (ATT) and percentage ATT.
            - fit_diagnostics (FitDiagnosticsResults): Includes goodness-of-fit metrics
              like pre-treatment RMSE, post-treatment RMSE, and pre-treatment R-squared.
            - time_series (TimeSeriesResults): Provides time-series data including the
              observed outcome for the treated unit, its estimated counterfactual outcome,
              the estimated treatment effect (gap) over time, and the corresponding
              time periods.
            - weights (WeightsResults): Contains an array of weights assigned to donor
              units and a list of the corresponding donor names.
            - inference (InferenceResults): Typically not populated with detailed statistical
              inference (like p-values or CIs) by the core `SRCest` logic, as it
              focuses on point estimation.
            - method_details (MethodDetailsResults): Details about the estimation method,
              including its name ("SRC") and the estimated `theta_hat` regression
              coefficients stored in `custom_params`.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the estimation process, which combines results from `SRCest` (like
              'Counterfactual' and 'Weights') and `effects.calculate` (like 'ATT',
              'Fit', 'Vectors').

        Examples
        --------

        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.estimators.src import SRC
        >>> from mlsynth.config_models import SHCConfig
        >>> # Load or create panel data
        >>> data = pd.DataFrame({
        ...     'unit': [1,1,1,1, 2,2,2,2, 3,3,3,3],
        ...     'time_val': [2010,2011,2012,2013, 2010,2011,2012,2013, 2010,2011,2012,2013],
        ...     'value': [10,11,9,8, 20,22,21,23, 15,16,17,14],
        ...     'is_treated_unit': [0,0,1,1, 0,0,0,0, 0,0,0,0] # Unit 1 treated from 2012
        ... })
        >>> src_config = SHCConfig(
        ...     df=data, outcome="value", treat="is_treated_unit",
        ...     unitid="unit", time="time_val", display_graphs=False
        ... )
        >>> src_estimator = SRC(config=src_config)
        >>> results = src_estimator.fit()
        >>> print(f"Estimated ATT: {results.effects.att}")
        >>> if results.weights and results.weights.donor_names:
        ...     print(f"Donor weights for {results.weights.donor_names}: {results.weights.donor_weights}")
        >>> if results.method_details and results.method_details.custom_params:
        ...     print(f"Theta_hat: {results.method_details.custom_params.get('theta_hat')}")
        """

        try:
            # Prepare the data using the standard dataprep utility.
            # This structures the data into treated unit outcomes, donor outcomes, period info, etc.


            # Extract the number of pre-treatment and post-treatment periods.
            # These are crucial for the SRCest algorithm and subsequent effect calculations.

            prepared_panel_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat, allow_no_donors=True
            )
            
            T0 = num_pre_periods = prepared_panel_data.get("pre_periods")
            num_post_periods = prepared_panel_data.get("post_periods")

            # Validate that period information is available from dataprep.
            if num_pre_periods is None:
                raise MlsynthDataError(
                    "Critical: 'pre_periods' key missing from dataprep output. "
                    "Ensure dataprep correctly identifies treated unit and pre-treatment periods."
                )
            if num_post_periods is None:
                raise MlsynthDataError(
                    "Critical: 'post_periods' key missing from dataprep output. "
                    "Ensure dataprep correctly identifies treated unit and post-treatment periods."
                )
                
            # Extract the outcome vector for the treated unit and the outcome matrix for donor units.
            treated_outcome_vector: np.ndarray = prepared_panel_data["y"]

            # Call the core SHC estimation function from estutils.
            # This function returns the estimated counterfactual, donor weights, and theta_hat coefficients.
            # SRCest itself can raise MlsynthDataError or MlsynthEstimationError.

            y = treated_outcome_vector
            T = len(y)
            n = T - T0
            N = T0 - self.m - n + 1
            if N <= 0:
                raise ValueError("Insufficient pre-treatment data to construct donor pool.")

            y_pre = y[:T0]

            bandwidth_grid = np.linspace(0.05, 1.0, 50)
            bandwidth, _ = loocv_bandwidth(y_pre, bandwidth_grid)
            # Step 1: Estimate latent trend
            ell_hat = smooth(y_pre, bandwidth)

            # Step 2: Build donor segments
            L_full, L_post, ell_eval = build_donor_segments(ell_hat, self.m, T0, n)

            # Step 3: SHC stepwise selection
            selection_results = stepwise_donor_selection(L_full, L_post, ell_eval, self.m)
            selected_indices = selection_results["best_donors"]
            shc_weights = selection_results["best_weights"]

            # Step 4: Expand SHC weights to full length vector
            w_shc_full = np.zeros(L_full.shape[1])
            w_shc_full[selected_indices] = shc_weights

            # Step 5: Conditional ASHC refinement
            if self.use_augmented:
                best_lambda, lambda_errors = tune_lambda_ashc(
                    L=L_full,
                    ell_eval=ell_eval,
                    w_shc=w_shc_full,
                    lambda_grid=np.linspace(0.0005, 0.5, 10)
                )

                final_weights, _ = _solve_SHC_QP(
                    L=L_full,
                    ell_eval=ell_eval,
                    use_augmented=True,
                    w_shc=w_shc_full,
                    lam=best_lambda
                )
            else:
                # If SHC, just use the convex hull solution directly
                final_weights = w_shc_full

            # Step 6: Estimate full counterfactual outcome
            estimated_counterfactual_outcome = np.vstack([L_full, L_post]) @ final_weights
            # Calculate treatment effects (ATT, etc.) and goodness-of-fit statistics.
            # This uses the observed outcome of the treated unit and the estimated counterfactual.
            # effects.calculate can raise MlsynthDataError.
            # Extract estimated counterfactual and match length with y segment
            y_aligned = treated_outcome_vector[num_pre_periods - self.m:num_pre_periods + num_post_periods]

            # Sanity check
            if len(estimated_counterfactual_outcome) != len(y_aligned):
                raise MlsynthEstimationError(
                    f"Mismatch between estimated counterfactual length ({len(estimated_counterfactual_outcome)}) "
                    f"and aligned observed outcome ({len(y_aligned)}). Check 'm' and period slicing."
                )

            # Now calculate ATT and fit diagnostics using only these aligned windows
            attdict, fitdict, Vectors = effects.calculate(
                y_aligned, estimated_counterfactual_outcome,
                self.m, num_post_periods
            )

            # Combine all raw results into a single dictionary for storage and potential inspection.
            combined_raw_estimation_output = {
                "Counterfactual": estimated_counterfactual_outcome, # The estimated synthetic outcome series.
                "Effects": attdict, # Dictionary of ATT and related effect measures.
                "Fit": fitdict, # Dictionary of fit statistics (RMSE, R-squared).
                "Vectors": Vectors, # Dictionary of observed, counterfactual, and gap series.
            }
            


            # Extract arrays
            observed = Vectors['Observed Unit'].flatten()
            counterfactual = Vectors['Counterfactual'].flatten()
            gap = Vectors['Gap']
            time = gap[:, 1]

            # Extract pre/post data
            pre_mask = time < 0
            post_mask = time >= 0

            actual_pre = observed[pre_mask]
            pred_pre = counterfactual[pre_mask]
            pred_post = counterfactual[post_mask]

            # Generate conformal prediction intervals
            lower_ag, upper_ag = ag_conformal(
                actual_outcomes_pre_treatment=actual_pre,
                predicted_outcomes_pre_treatment=pred_pre,
                predicted_outcomes_post_treatment=pred_post,
                miscoverage_rate=0.1,  # 90% interval
                pad_value=np.nan
            )

            # Extract only the post-treatment intervals.
            lower_ci_post = lower_ag[len(pre_mask):]
            upper_ci_post = upper_ag[len(pre_mask):]
            prediction_intervals_post_only: np.ndarray = np.vstack([lower_ci_post, upper_ci_post]).T
            # Store full intervals (pre and post) for potential later use or if plotting needs them.
            final_prediction_intervals = np.vstack([lower_ag, upper_ag]).T

            inference_kwargs = {
                "method": "conformal",
                "confidence_level": 0.90,
                "details": {
                    "lower_bound": lower_ci_post,
                    "upper_bound": upper_ci_post,
                    "full_interval": final_prediction_intervals,  # Optional if you want pre + post
                    "interval_type": "AG"  # If you're using Andrews–Genton conformal
                }
            }

            # Convert the combined raw results into the standardized Pydantic BaseEstimatorResults model.
            # This helper function handles the mapping and can raise MlsynthEstimationError.
            pydantic_results = self._create_estimator_results(
                combined_raw_estimation_output=combined_raw_estimation_output,
                prepared_panel_data=prepared_panel_data,
                inference_kwargs=inference_kwargs
            )

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e:
            # Re-raise known specific errors from utilities or earlier checks.
            raise
        except (KeyError, ValueError, TypeError, AttributeError, np.linalg.LinAlgError) as e:
            # Wrap common Python errors that might occur during the estimation process
            # into a MlsynthEstimationError for consistent error reporting.
            raise MlsynthEstimationError(f"SRC estimation failed due to an unexpected error: {type(e).__name__}: {e}") from e
        except Exception as e:
            # Catch any other truly unexpected errors and wrap them.
            raise MlsynthEstimationError(f"An unexpected critical error occurred during SHC fit: {type(e).__name__}: {e}") from e

        # --- Plotting Phase ---
        if self.display_graphs: # If plotting is enabled in the configuration.
            try:
                # Attempt to get the counterfactual outcome series from the Pydantic results object.
                cf_to_plot = pydantic_results.time_series.counterfactual_outcome if pydantic_results.time_series else None
                if cf_to_plot is None: # Fallback to the raw estimated counterfactual if not found in Pydantic results.
                    cf_to_plot = estimated_counterfactual_outcome
                
                # Validate that necessary data for plotting is available.
                if cf_to_plot is None or not isinstance(cf_to_plot, np.ndarray):
                     warnings.warn("Cannot plot SHC results: counterfactual outcome is not available or not an array.", UserWarning)
                elif prepared_panel_data.get("y") is None: # Check for observed outcome.
                     warnings.warn("Cannot plot SHC results: observed outcome 'y' is not available in prepared_panel_data.", UserWarning)
                elif prepared_panel_data.get("treated_unit_name") is None: # Check for treated unit name.
                     warnings.warn("Cannot plot SHC results: 'treated_unit_name' is not available in prepared_panel_data.", UserWarning)
                else:
                    # Truncate Ywide to last m + post periods
                    m_plus_post = self.m + num_post_periods
                    prepared_panel_data["Ywide"] = prepared_panel_data["Ywide"].iloc[-m_plus_post:]
                    prepared_panel_data["pre_periods"] = self.m
                    # Call the generic plotting utility.
                    plot_estimates(
                        processed_data_dict=prepared_panel_data,
                        time_axis_label=self.time,
                        unit_identifier_column_name=self.unitid,
                        outcome_variable_label=self.outcome,
                        treatment_name_label=self.treat,
                        treated_unit_name=prepared_panel_data["treated_unit_name"],
                        observed_outcome_series=y_aligned,  # Observed outcome vector.
                        counterfactual_series_list=[cf_to_plot.flatten()],
                        # List of counterfactual vectors.
                        estimation_method_name="SynthHistControl",
                        counterfactual_names=["Synthetic Historical Control"],  # Names for legend.
                        treated_series_color=self.treated_color,
                        counterfactual_series_colors=self.counterfactual_color,
                        save_plot_config=self.save, uncertainty_intervals_array=final_prediction_intervals)
            except (MlsynthPlottingError, MlsynthDataError) as e: # Catch known plotting or data errors.
                warnings.warn(f"Plotting failed for SHC due to data or plotting issue: {e}", UserWarning)
            except Exception as e: # Catch any other unexpected errors during plotting.
                warnings.warn(f"An unexpected error occurred during SHC plotting: {type(e).__name__}: {e}", UserWarning)

        return pydantic_results
