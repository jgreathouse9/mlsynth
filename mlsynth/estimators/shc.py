import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings
import pydantic # For ValidationError

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import (
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
    Nonlinear Synthetic Control (NSC) model.

    Estimates the treatment effect for a single treated unit using an affine
    combination of control units. The method involves a regularization term
    that encourages sparsity and stability in the weights. Hyperparameters
    (denoted `a` and `b` in the underlying optimization) for the weight
    estimation are selected via cross-validation (`NSCcv`) to minimize
    pre-treatment Mean Squared Prediction Error (MSPE).

    The `fit` method orchestrates data preparation, hyperparameter tuning via
    cross-validation, final weight estimation using the optimal hyperparameters,
    and computation of the counterfactual and treatment effects.

    Attributes
    ----------
    config : NSCConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`).
    display_graphs : bool, default=True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`).
    save : Union[bool, str], default=False
        If False, plots are not saved. If True, plots are saved with default names.
        If a string, it's used as a prefix for saved plot filenames.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`).
    counterfactual_color : str, default="red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`).
    treated_color : str, default="black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `NSCConfig`).

    Examples
    --------
    >>> from mlsynth import SHC
    >>> from mlsynth.config_models import NSCConfig
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data for demonstration
    >>> data = pd.DataFrame({
    ...     'unit': np.repeat(np.arange(1, 4), 10), # 3 units
    ...     'time': np.tile(np.arange(1, 11), 3),   # 10 time periods
    ...     'outcome': np.random.rand(30) + np.repeat(np.arange(0,3),10)*0.5,
    ...     'treated_unit_1': ((np.repeat(np.arange(1, 4), 10) == 1) & \
    ...                        (np.tile(np.arange(1, 11), 3) >= 6)).astype(int)
    ... })
    >>> shc_config = SHCConfig(
    ...     df=data,
    ...     outcome='outcome',
    ...     treat='treated_unit_1',
    ...     unitid='unit',
    ...     time='time',
    ...     display_graphs=False # Typically True, False for non-interactive examples
    ... )
    >>> estimator = SHC(config=nsc_config)
    >>> # Results can be obtained by calling estimator.fit()
    >>> # results = estimator.fit() # doctest: +SKIP
    """

    def __init__(self, config: SHCConfig) -> None:
        """
        Initializes the SHC estimator with a configuration object.

        Parameters
        ----------
        config : SHCConfig
            A Pydantic model instance containing all configuration parameters
            for the NSC estimator. `NSCConfig` inherits from `BaseEstimatorConfig`.
            The fields include:

                df : pd.DataFrame
                    The input panel data. Must contain columns for outcome, treatment
                    indicator, unit identifier, and time identifier.
                outcome : str
                    Name of the outcome variable column in `df`.
                treat : str
                    Name of the binary treatment indicator column in `df`.
                unitid : str
                    Name of the unit identifier (e.g., country, individual ID) column in `df`.
                time : str
                    Name of the time period column in `df`.
                display_graphs : bool, default=True
                    Whether to display plots of the results after fitting.
                save : Union[bool, str], default=False
                    If False, plots are not saved. If True, plots are saved with default names.
                    If a string, it's used as a prefix for saved plot filenames.
                counterfactual_color : str, default="red"
                    Color for the counterfactual line(s) in plots.

                treated_color : str, default="black"
                    Color for the treated unit line in plots.

        """
        if isinstance(config, dict):
            config =SHCConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save

    def _create_estimator_results(
        self, aggregated_fit_results: Dict[str, Any]
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object from raw SHC outputs.

        Parameters
        ----------
        aggregated_fit_results : Dict[str, Any]
            A dictionary containing the comprehensive results from the NSC fitting
            process. Expected to include keys like 'Effects', 'Fit', 'Vectors',
            'Weights', '_prepped' (for preprocessed data which includes 'Ywide'
            for time periods), 'best_a', and 'best_b'.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results.
            Key populated fields include:
            - `effects`: From `aggregated_fit_results['Effects']`.
            - `fit_diagnostics`: From `aggregated_fit_results['Fit']`.
            - `time_series`: From `aggregated_fit_results['Vectors']` and `aggregated_fit_results['_prepped']`.
            - `weights`: From `aggregated_fit_results['Weights']`.
            - `inference`: Basic method description, as NSC core doesn't provide SE/p-values.
            - `method_details`: Includes estimator name, config, and optimal hyperparameters.
            - `raw_results`: The input `aggregated_fit_results` dictionary.
        """
        prepped_data = aggregated_fit_results.get("_prepped", {})
        
        # --- Effects ---
        effects_data = aggregated_fit_results.get("Effects", {})
        effects_results = EffectsResults(
            att=effects_data.get("ATT"),
            att_percent=effects_data.get("Percent ATT"),
            additional_effects={
                k: v for k, v in effects_data.items() if k not in ["ATT", "Percent ATT"]
            },
        )

        # --- Fit Diagnostics ---
        fit_data = aggregated_fit_results.get("Fit", {})
        fit_diagnostics_results = FitDiagnosticsResults(
            pre_treatment_rmse=fit_data.get("T0 RMSE"),
            pre_treatment_r_squared=fit_data.get("R-Squared"),
            # Store any other fit metrics in additional_metrics, converting keys to snake_case
            additional_metrics={
                k.lower().replace(" ", "_"): v for k, v in fit_data.items() if k not in ["T0 RMSE", "R-Squared"]
            } if fit_data else None,
        )

        # --- Time Series Data ---
        # Extract time periods from the 'Ywide' DataFrame in prepped_data, if available
        time_periods_arr: Optional[np.ndarray] = None
        if "Ywide" in prepped_data: # NSC's dataprep should always have Ywide
            Ywide_df = prepped_data["Ywide"]
            if isinstance(Ywide_df, pd.DataFrame) and isinstance(Ywide_df.index, pd.Index):
                time_values = Ywide_df.index.to_numpy()
                if time_values.size > 0:
                    time_periods_arr = time_values # Ywide_df.index is typically the time periods
            
        vectors_data = aggregated_fit_results.get("Vectors", {})
        time_series_results = TimeSeriesResults(
            observed_outcome=vectors_data.get("Observed Unit"),
            counterfactual_outcome=vectors_data.get("Counterfactual"),
            estimated_gap=vectors_data.get("Gap"), # 'Gap' from effects.calculate matches Pydantic model's 'estimated_gap'
            time_periods=time_periods_arr,
        )

        # --- Weights ---
        # Weights are expected as a list: [donor_weights_dict, cardinality_dict]
        weights_data = aggregated_fit_results.get("Weights", [{}, {}]) # Default to list of empty dicts for safety
        donor_weights_dict = weights_data[0] if len(weights_data) > 0 and isinstance(weights_data[0], dict) else {}
        
        # Process cardinality dictionary (e.g., "Cardinality of Positive Donors")
        cardinality_dict_raw = weights_data[1] if len(weights_data) > 1 and isinstance(weights_data[1], dict) else None
        cardinality_dict_processed = {k.lower().replace(" ", "_"): v for k,v in cardinality_dict_raw.items()} if cardinality_dict_raw else None

        weights_results = WeightsResults(
            donor_weights=donor_weights_dict if donor_weights_dict else None, # Ensure None if empty
            summary_stats=cardinality_dict_processed # Store processed cardinality info
        )
        
        # --- Inference ---
        # NSC's core method provides point estimates; formal SE/p-values are not standard output.
        inference_results = InferenceResults(method="Point estimate from weighted donors (NSCcv)") # Other fields default to None

        # --- Method Details ---
        method_details_results = MethodDetailsResults(
            name="SHC", # Estimator name
            parameters_used=self.config.model_dump(exclude={'df'}, exclude_none=True), # Store config, excluding DataFrame
            additional_outputs={ # Store optimal hyperparameters found during CV
                "best_a": aggregated_fit_results.get("best_a"),
                "best_b": aggregated_fit_results.get("best_b"),
            } if aggregated_fit_results.get("best_a") is not None or aggregated_fit_results.get("best_b") is not None else None
        )

        return BaseEstimatorResults(
            effects=effects_results,
            fit_diagnostics=fit_diagnostics_results,
            time_series=time_series_results,
            weights=weights_results,
            inference=inference_results,
            method_details=method_details_results,
            raw_results=aggregated_fit_results,
        )

    def fit(self) -> BaseEstimatorResults:
        """
        Fits the Synthetic Historical Control (SHC) model to the provided data.

        The method performs the following steps:
        1. Balances the input panel data to ensure consistent time periods across units.
        2. Prepares the data into matrices for the treated unit's outcomes (`y_treated`)
           and control units' outcomes (`Y0_donors`).
        3. Uses cross-validation (`NSCcv`) on the pre-treatment data to find the
           optimal hyperparameters (`best_a`, `best_b`) for the NSC weight estimation.
        4. Estimates the optimal donor weights (`weights_estimated`) using `NSC_opt`
           with the selected hyperparameters.
        5. Constructs the counterfactual outcome series for the treated unit by applying
           the estimated weights to the control units' outcomes over the entire period.
        6. Calculates treatment effects (ATT) and goodness-of-fit statistics.
        7. Optionally plots the observed and counterfactual outcomes.
        8. Returns a `BaseEstimatorResults` object containing all relevant outputs.

        Returns
        -------
        BaseEstimatorResults
            An object containing the standardized estimation results:

            - `effects` (EffectsResults)
                Contains `att` (Average Treatment Effect on the Treated) and
                `att_percent` (Percentage ATT).
            - `fit_diagnostics` (FitDiagnosticsResults)
                Contains `pre_treatment_rmse` and `pre_treatment_r_squared`.
            - `time_series` (TimeSeriesResults)
                Contains `observed_outcome` (for the treated unit),
                `counterfactual_outcome`, `estimated_gap` (effect over time),
                and `time_periods` (actual time values or event time indices).
            - `weights` (WeightsResults)
                Contains `donor_weights` (a dictionary mapping donor unit names
                to their optimal weights) and `summary_stats` (e.g., donor
                cardinality).
            - `method_details` (MethodDetailsResults)
                Contains the method `name` ("NSC"), `parameters_used` (the
                estimator config), and `additional_outputs` (optimal
                hyperparameters `best_a`, `best_b`).
            - `inference` (InferenceResults)
                Basic information indicating the estimation method ("Point
                estimate from weighted donors (NSCcv)"), as NSC's core logic
                doesn't provide standard errors or p-values.
            - `raw_results` (Dict[str, Any])
                The comprehensive dictionary returned by the internal fitting
                process, including the `_prepped` data.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from mlsynth.estimators.nsc import NSC
        >>> from mlsynth.config_models import NSCConfig
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'unit_id': np.repeat(np.arange(1, 5), 10), # 4 units, 10 time periods
        ...     'time_id': np.tile(np.arange(2000, 2010), 4),
        ...     'gdp': np.random.rand(40) * 100 + np.repeat(np.arange(1,5),10)*10,
        ...     'intervention': ((np.repeat(np.arange(1, 5), 10) == 1) & \
        ...                      (np.tile(np.arange(2000, 2010), 4) >= 2005)).astype(int)
        ... }) # Unit 1 treated from 2005 onwards
        >>> nsc_config = SHCConfig(
        ...     df=data,
        ...     outcome="gdp",
        ...     treat="intervention",
        ...     unitid="unit_id",
        ...     time="time_id",
        ...     display_graphs=False # Disable plots for example
        ... )
        >>> nsc_estimator = NSC(config=nsc_config)
        >>> results = nsc_estimator.fit() # doctest: +SKIP
        >>> # Example: Accessing results (actual values will vary due to random data)
        >>> print(f"Estimated ATT: {results.effects.att}") # doctest: +SKIP
        >>> if results.method_details and results.method_details.additional_outputs: # doctest: +SKIP
        ...     print(f"Optimal a: {results.method_details.additional_outputs.get('best_a')}") # doctest: +SKIP
        ...     print(f"Optimal b: {results.method_details.additional_outputs.get('best_b')}") # doctest: +SKIP
        >>> if results.weights and results.weights.donor_weights: # doctest: +SKIP
        ...     print(f"Donor weights: {results.weights.donor_weights}") # doctest: +SKIP
        """
        try:
            # Step 1: Balance the panel data to ensure consistent time periods across units.
            balance(self.df, self.unitid, self.time)

            # Step 2: Prepare data into matrices for treated unit and control units.
            # `dataprep` returns a dictionary with 'y' (treated outcomes), 'donor_matrix' (control outcomes),
            # 'pre_periods', 'post_periods', 'donor_names', 'treated_unit_name', 'Ywide', etc.
            prepared_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # Extract key matrices and pre-treatment period length
            treated_unit_outcomes_all_periods: np.ndarray = prepared_data["y"]
            donor_outcomes_all_periods: np.ndarray = prepared_data["donor_matrix"]
            pre_periods: int = prepared_data["pre_periods"]

            # Step 3: Perform cross-validation on pre-treatment data to find optimal hyperparameters (a, b).
            # `NSCcv` iterates through combinations of 'a' and 'b' values (or default ranges if not provided)
            # to minimize the Mean Squared Prediction Error (MSPE) on the pre-treatment outcomes.
            best_a, best_b = NSCcv(
                treated_unit_outcomes_all_periods[:pre_periods], # Use only pre-treatment data for CV
                donor_outcomes_all_periods[:pre_periods],      # Use only pre-treatment data for CV
                a_vals=self.a_search_space, # Pass configured search space for 'a'
                b_vals=self.b_search_space  # Pass configured search space for 'b'
            )

            # Step 4: Estimate optimal donor weights using the best hyperparameters found by NSCcv.
            # `NSC_opt` solves the NSC optimization problem with the chosen 'a' and 'b'.
            optimal_donor_weights_array: np.ndarray = NSC_opt(
                treated_unit_outcomes_all_periods[:pre_periods], # Weights based on pre-treatment fit
                donor_outcomes_all_periods[:pre_periods],      # Weights based on pre-treatment fit
                best_a,
                best_b
            )

            # Step 5: Construct the counterfactual outcome series for the treated unit.
            # This is done by applying the estimated weights to the control units' outcomes over the entire period.
            counterfactual_outcome_series: np.ndarray = np.dot(donor_outcomes_all_periods, optimal_donor_weights_array)

            # Create a dictionary of donor weights, ensuring donor names are strings for JSON compatibility.
            donor_weights_dictionary: Dict[str, float] = {
                str(prepared_data["donor_names"][i]): round(optimal_donor_weights_array[i], 3) # Ensure keys are strings
                for i in range(len(prepared_data["donor_names"]))
            }
            # The print statement was removed as it's a side effect not typical for a library function.
            # print(donor_weights_dictionary)

            # Step 6: Calculate treatment effects (ATT) and goodness-of-fit statistics.
            # `effects.calculate` compares observed treated outcomes with the counterfactual.
            calculated_effects_dict, fit_diagnostics_dict, time_series_data_dict = effects.calculate(
                prepared_data["y"], # Observed outcomes of the treated unit
                counterfactual_outcome_series, # Estimated counterfactual outcomes
                pre_periods, # Number of pre-treatment periods
                prepared_data["post_periods"], # Number of post-treatment periods
            )

            # Step 7: Aggregate all results into a dictionary for packaging.
            aggregated_fit_results = {
                "Effects": calculated_effects_dict, # ATT, Percent ATT
                "Fit": fit_diagnostics_dict,       # Pre-treatment RMSE, R-squared
                "Vectors": time_series_data_dict,  # Observed, Counterfactual, Gap series
                "Weights": [ # List containing weights dict and cardinality dict
                    donor_weights_dictionary,
                    { # Summary statistics for weights, e.g., number of non-zero donors
                        "Cardinality of Positive Donors": np.sum(
                            np.abs(optimal_donor_weights_array) > 0.001 # Count donors with weight > 0.001
                        )
                    },
                ],
                "_prepped": prepared_data, # Store the preprocessed data for potential reuse/inspection
                "best_a": best_a, # Store optimal hyperparameter 'a'
                "best_b": best_b, # Store optimal hyperparameter 'b'
            }
            # Convert the aggregated dictionary into a structured Pydantic model.
            results_obj = self._create_estimator_results(aggregated_fit_results)

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError):
            # Re-raise custom errors that might originate from utility functions or config validation.
            raise # Re-raise specific custom errors
        except pydantic.ValidationError as ve:
            raise MlsynthEstimationError(f"Error creating results model: {ve}") from ve
        except (KeyError, TypeError, ValueError) as e:
            # Catch common Python errors during data processing or calculation
            raise MlsynthEstimationError(f"NSC estimation failed due to an unexpected error: {e}") from e
        except Exception as e:
            # Catch-all for any other unexpected errors not caught by more specific handlers.
            raise MlsynthEstimationError(f"An unexpected error occurred during NSC fitting: {e}") from e

        # Step 8: Optionally display or save plots of the results.
        if self.display_graphs:
            try:
                plot_estimates(
                    processed_data_dict=prepared_data,
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treat,
                    treated_unit_name=prepared_data["treated_unit_name"],
                    observed_outcome_series=prepared_data["y"],  # Observed outcome vector.
                    counterfactual_series_list=[counterfactual_outcome_series.flatten()],
                    # List of counterfactual vectors.
                    estimation_method_name="NSC",
                    counterfactual_names=["Nonlinear Synthetic Control"],  # Names for legend.
                    treated_series_color=self.treated_color,
                    counterfactual_series_colors=self.counterfactual_color,
                    save_plot_config=self.save)
            except (MlsynthPlottingError, MlsynthDataError) as plot_err:
                # Warn if plotting fails due to known plotting or data issues.
                warnings.warn(f"NSC: Plotting failed with error: {plot_err}", UserWarning)
            except Exception as plot_err: # Catch any other unexpected error during plotting
                # Warn for any other unexpected plotting errors.
                warnings.warn(f"NSC: An unexpected error occurred during plotting: {plot_err}", UserWarning)
        
        # Step 9: Return the structured results object.
        return results_obj # type: ignore # results_obj is defined in the try block
