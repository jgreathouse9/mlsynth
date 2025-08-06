import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic
from mlsynth.utils.inferutils import ag_conformal
from ..utils.datautils import dataprep
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, MlsynthPlottingError
from ..utils.resultutils import effects, plot_estimates
from ..utils.estutils import standardize_data, back_transform_predictions, fit_l2_relaxation, generate_taus, rescmcross_validate_tau
from ..config_models import ( # Import Pydantic models
    RESCMConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class RESCM:
    """
    Implements the Relaxed Balanced SCM

    Attributes
    ----------
    config : RESCMConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `RESCMConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCMConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCMConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `RESCM`)

    Methods
    -------
    fit()
        Fits the SHC model and returns standardized estimation results.

    References
    ----------

    Zhu, Rong J. B. "Synthetic Regressing Control Method." arXiv preprint arXiv:2306.02584 (2023).
    https://arxiv.org/abs/2306.02584
    """

    def __init__(self, config: RESCMConfig) -> None: # Changed to RESCM
        """
        Initializes the SHC estimator with a configuration object.

        Parameters
        ----------

        config : RESCM
            A Pydantic model instance containing all configuration parameters
            for the ShC estimator. Since `RESCM` inherits directly from
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
            config = RESCMConfig(**config)  # convert dict to config object
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
            donor_weights_values_list = list(donor_weights_dict_raw.values()) # Not used

            #weights = WeightsResults(
              #  donor_weights=donor_weights_dict_raw if donor_weights_dict_raw else None,
                #donor_names=donor_names_list if donor_names_list else None,
            #)

            method_details = MethodDetailsResults(
                method_name="SHC" # Name of the estimation method.
                # theta_hat represents the estimated regression coefficients from the SHC model.
                # It includes coefficients for donor outcomes and potentially an intercept.
            )

            return BaseEstimatorResults(
                effects=effects,
                fit_diagnostics=fit_diagnostics,
                time_series=time_series, #weights=weights,
                method_details=method_details,
                raw_results=combined_raw_estimation_output,
            )
        except pydantic.ValidationError as e:
            raise MlsynthEstimationError(f"Error creating Pydantic results model for RESCM: {e}") from e
        except Exception as e:
            raise MlsynthEstimationError(f"Unexpected error in _create_estimator_results for RESCM: {type(e).__name__}: {e}") from e

    def fit(self) -> BaseEstimatorResults:
        """
        Fits the Relaxed Balanced SCM (RESCM) model to the provided data.

        This method performs the following steps:
        1. Prepares the data using `dataprep` to structure outcomes for the
           treated unit (`y1`)
        2. Calls the `_SHC_est` utility function, which implements the core SHC
           algorithm to estimate the counterfactual outcome (`y_RESCM`), donor
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
        >>> from mlsynth.config_models import RESCM
        >>> # Load or create panel data
        >>> data = pd.DataFrame({
        ...     'unit': [1,1,1,1, 2,2,2,2, 3,3,3,3],
        ...     'time_val': [2010,2011,2012,2013, 2010,2011,2012,2013, 2010,2011,2012,2013],
        ...     'value': [10,11,9,8, 20,22,21,23, 15,16,17,14],
        ...     'is_treated_unit': [0,0,1,1, 0,0,0,0, 0,0,0,0] # Unit 1 treated from 2012
        ... })
        >>> src_config = RESCM(
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
                self.df, self.unitid, self.time, self.outcome, self.treat
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
            y: np.ndarray = prepared_panel_data["y"]

            Y0 = prepared_panel_data["donor_matrix"]

            y_pre_std, X_pre_std, y_post_std, X_post_std, stats = standardize_data(y[:T0], Y0[:T0], y[T0:], Y0[T0:])
            best_tau, taus, cv_errors = rescmcross_validate_tau(X_pre_std, y_pre_std, n_splits=5, n_taus=50)
            weights_std = fit_l2_relaxation(X_pre_std, y_pre_std, best_tau)
            donor_weights = {
                donor: round(float(w), 3)
                for donor, w in zip(prepared_panel_data['donor_names'], weights_std)
            }

            # Back-transform predictions
            y_RESCM = back_transform_predictions(weights_std, Y0, stats)


            # Now calculate ATT and fit diagnostics using only these aligned windows
            attdict, fitdict, Vectors = effects.calculate(
                y, y_RESCM,
                T0, num_post_periods
            )

            # Combine all raw results into a single dictionary for storage and potential inspection.
            combined_raw_estimation_output = {
                "Effects": attdict, # Dictionary of ATT and related effect measures.
                "Fit": fitdict, # Dictionary of fit statistics (RMSE, R-squared).
                "Vectors": Vectors, # Dictionary of observed, counterfactual, and gap series.
            }


            # Convert the combined raw results into the standardized Pydantic BaseEstimatorResults model.
            # This helper function handles the mapping and can raise MlsynthEstimationError.
            pydantic_results = self._create_estimator_results(
                combined_raw_estimation_output=combined_raw_estimation_output,
                prepared_panel_data=prepared_panel_data
            )

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e:
            # Re-raise known specific errors from utilities or earlier checks.
            raise
        except (KeyError, ValueError, TypeError, AttributeError, np.linalg.LinAlgError) as e:
            # Wrap common Python errors that might occur during the estimation process
            # into a MlsynthEstimationError for consistent error reporting.
            raise MlsynthEstimationError(f"RESCM estimation failed due to an unexpected error: {type(e).__name__}: {e}") from e
        except Exception as e:
            # Catch any other truly unexpected errors and wrap them.
            raise MlsynthEstimationError(f"An unexpected critical error occurred during RESCM fit: {type(e).__name__}: {e}") from e

        # --- Plotting Phase ---
        if self.display_graphs: # If plotting is enabled in the configuration.
            try:
                # Attempt to get the counterfactual outcome series from the Pydantic results object.
                cf_to_plot = pydantic_results.time_series.counterfactual_outcome if pydantic_results.time_series else None
                if cf_to_plot is None: # Fallback to the raw estimated counterfactual if not found in Pydantic results.
                    cf_to_plot = y_RESCM

                # Validate that necessary data for plotting is available.
                if cf_to_plot is None or not isinstance(cf_to_plot, np.ndarray):
                     warnings.warn("Cannot plot RESCM results: counterfactual outcome is not available or not an array.", UserWarning)
                elif prepared_panel_data.get("y") is None: # Check for observed outcome.
                     warnings.warn("Cannot plot RESCM results: observed outcome 'y' is not available in prepared_panel_data.", UserWarning)
                elif prepared_panel_data.get("treated_unit_name") is None: # Check for treated unit name.
                     warnings.warn("Cannot plot RESCM results: 'treated_unit_name' is not available in prepared_panel_data.", UserWarning)
                else:
                    # Call the generic plotting utility.
                    plot_estimates(
                        processed_data_dict=prepared_panel_data,
                        time_axis_label=self.time,
                        unit_identifier_column_name=self.unitid,
                        outcome_variable_label=self.outcome,
                        treatment_name_label=self.treat,
                        treated_unit_name=prepared_panel_data["treated_unit_name"],
                        observed_outcome_series=y,  # Observed outcome vector.
                        counterfactual_series_list=[y_RESCM],
                        # List of counterfactual vectors.
                        estimation_method_name="RESCM",
                        counterfactual_names=["Relaxed SCM"],  # Names for legend.
                        treated_series_color=self.treated_color,
                        counterfactual_series_colors=self.counterfactual_color,
                        save_plot_config=self.save)
            except (MlsynthPlottingError, MlsynthDataError) as e: # Catch known plotting or data errors.
                warnings.warn(f"Plotting failed for RESCM due to data or plotting issue: {e}", UserWarning)
            except Exception as e: # Catch any other unexpected errors during plotting.
                warnings.warn(f"An unexpected error occurred during RESCM plotting: {type(e).__name__}: {e}", UserWarning)

        return pydantic_results
