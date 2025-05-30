import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Union, Optional
import cvxpy as cp # For catching solver errors
from pydantic import ValidationError # For catching Pydantic errors if models are created internally

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates
from ..utils.estutils import Opt
from ..exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import (
    FSCMConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class FSCM:
    """
    Estimates Average Treatment Effect on the Treated (ATT) using the Forward Selected Synthetic Control Method (FSCM).

    This approach, based on Cerulli (2024), optimally selects a subset of donor
    units using a forward selection algorithm. It begins by identifying the
    single best-fitting donor unit. Then, it iteratively adds other donor units
    to the pool if their inclusion improves the predictive fit (minimizes MSE)
    for the treated unit's pre-treatment outcomes. The final weights for the
    selected donor pool and the counterfactual outcome series are derived using
    a constrained optimization procedure (SIMPLEX).

    Attributes
    ----------
    config : FSCMConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    save : Union[bool, str], default False
        If False, plots are not saved. If True, plots are saved with default names.
        If a string, it's used as the directory path to save plots.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `FSCMConfig`)

    Methods
    -------
    fit()
        Fits the FSCM model and returns the standardized results.
    evaluate_donor(donor_index, donor_columns, y_pre, T0)
        Evaluates the Mean Squared Error (MSE) for a single potential donor.
    fSCM(y_pre, Y0, T0)
        Performs the core Forward Selected Synthetic Control Method optimization.

    References
    ----------
    Cerulli, Giovanni. "Optimal initial donor selection for the synthetic control method."
    *Economics Letters*, 244 (2024): 111976. https://doi.org/10.1016/j.econlet.2024.111976

    Examples
    --------
    >>> from mlsynth import FSCM
    >>> from mlsynth.config_models import FSCMConfig
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
    >>> fscm_config = FSCMConfig(
    ...     df=data,
    ...     outcome='outcome',
    ...     treat='treated_unit_1',
    ...     unitid='unit',
    ...     time='time',
    ...     display_graphs=False # Typically True, False for non-interactive examples
    ... )
    >>> estimator = FSCM(config=fscm_config)
    >>> # Results can be obtained by calling estimator.fit()
    >>> # results = estimator.fit() # doctest: +SKIP
    """

    def __init__(self, config: FSCMConfig) -> None: # Changed to FSCMConfig
        """
        Initializes the FSCM estimator with a configuration object.

        Parameters
        ----------
        config : FSCMConfig
            A Pydantic model instance containing all configuration parameters
            for the FSCM estimator. `FSCMConfig` inherits from `BaseEstimatorConfig`.
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
                (Note: The internal `plot_estimates` function might handle a list of colors,
                but `FSCMConfig` defines it as `str`).
            treated_color : str, default="black"
                Color for the treated unit line in plots.
        """
        if isinstance(config, dict):
            config = FSCMConfig(**config)  # convert dict to config object
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

    def _create_estimator_results( # Helper method to package results into the standard Pydantic model
        self,
        raw_estimation_output: Dict[str, Any],
        prepared_data_dict: Dict[str, Any],
        selected_optimal_donor_indices: List[int], 
        final_pre_treatment_rmse: float,   
        optimal_donor_weights_array: np.ndarray,   
        names_of_selected_donors: List[str] 
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object from raw FSCM outputs.

        Parameters
        ----------
        raw_estimation_output : Dict[str, Any]
            Dictionary containing the raw results from the FSCM fitting process,
            typically including 'Effects', 'Fit', 'Vectors', and 'Weights' keys.
        prepared_data_dict : Dict[str, Any]
            Dictionary of preprocessed data from `dataprep`, containing elements
            like 'y' (treated unit outcomes) and 'time_labels'.
        selected_optimal_donor_indices : List[int]
            List of integer indices for the selected optimal donor units.
        final_pre_treatment_rmse : float
            The final Root Mean Squared Error for the pre-treatment fit.
        optimal_donor_weights_array : np.ndarray
            Array of weights corresponding to the `names_of_selected_donors`.
            Shape: (number_of_selected_donors,).
        names_of_selected_donors : List[str]
            List of names for the selected optimal donor units.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results.
        """
        # Extract raw data components from the estimation output dictionary
        raw_effects_data = raw_estimation_output.get("Effects", {})
        raw_fit_diagnostics_data = raw_estimation_output.get("Fit", {})
        raw_time_series_data = raw_estimation_output.get("Vectors", {})
        # Weights package might contain a dictionary of weights and a summary dictionary
        raw_weights_package = raw_estimation_output.get("Weights", [{}, {}]) 
        
        # Create EffectsResults Pydantic model
        effects_results_obj = EffectsResults(
            att=raw_effects_data.get("ATT"),
            att_percent=raw_effects_data.get("Percent ATT"),
        )

        # Create FitDiagnosticsResults Pydantic model
        fit_diagnostics_results_obj = FitDiagnosticsResults(
            pre_treatment_rmse=final_pre_treatment_rmse, # Use the explicitly passed final RMSE
            pre_treatment_r_squared=raw_fit_diagnostics_data.get("R-Squared"), 
            additional_metrics={"post_treatment_gap_std": raw_fit_diagnostics_data.get("T1 RMSE")} # Store T1 RMSE as post_treatment_gap_std
        )

        # Prepare observed outcome array
        treated_outcome_series_from_prepared_data = prepared_data_dict.get("y") 
        treated_outcome_array: Optional[np.ndarray] = None
        if treated_outcome_series_from_prepared_data is not None:
            # Ensure it's a flattened NumPy array
            treated_outcome_array = treated_outcome_series_from_prepared_data.to_numpy().flatten() if isinstance(treated_outcome_series_from_prepared_data, pd.Series) else np.array(treated_outcome_series_from_prepared_data).flatten()

        # Prepare counterfactual outcome array
        counterfactual_outcome_array: Optional[np.ndarray] = None
        if raw_time_series_data.get("Synthetic") is not None:
            counterfactual_outcome_array = np.array(raw_time_series_data["Synthetic"]).flatten()

        # Prepare estimated gap array
        estimated_gap_array: Optional[np.ndarray] = None
        if raw_time_series_data.get("Effect") is not None:
            estimated_gap_array = np.array(raw_time_series_data["Effect"]).flatten()
        elif treated_outcome_array is not None and counterfactual_outcome_array is not None and len(treated_outcome_array) == len(counterfactual_outcome_array):
            # Calculate gap if not directly provided but observed and counterfactual are available
            estimated_gap_array = treated_outcome_array - counterfactual_outcome_array
            
        # Prepare time periods array
        time_periods_array_for_results: Optional[np.ndarray] = None 
        time_labels_from_prepared_data = prepared_data_dict.get("time_labels") 
        if time_labels_from_prepared_data is not None:
            time_periods_array_for_results = np.array(time_labels_from_prepared_data) 

        # Create TimeSeriesResults Pydantic model
        time_series_results_obj = TimeSeriesResults(
            observed_outcome=treated_outcome_array,
            counterfactual_outcome=counterfactual_outcome_array,
            estimated_gap=estimated_gap_array, 
            time_periods=time_periods_array_for_results, 
        )
        
        # Prepare donor weights map
        final_donor_weights_map: Optional[Dict[str, float]] = None
        if optimal_donor_weights_array is not None and names_of_selected_donors is not None:
            # Create a dictionary mapping donor names to their weights
            final_donor_weights_map = {name: weight for name, weight in zip(names_of_selected_donors, optimal_donor_weights_array.flatten())}

        # Create WeightsResults Pydantic model
        weights_results_obj = WeightsResults(
            donor_weights=final_donor_weights_map, 
        )
        
        # Create InferenceResults Pydantic model (FSCM typically doesn't provide standard inference)
        inference_results_obj = InferenceResults() # Defaults to None for all fields

        # Prepare method details
        raw_weights_summary_data = raw_weights_package[1] if len(raw_weights_package) > 1 else {}
        method_details_results_obj = MethodDetailsResults(
            name="FSCM", 
            parameters_used={ # Store key parameters and outcomes of the FSCM process
                "optimal_donor_indices": selected_optimal_donor_indices,
                "optimal_pre_rmse": final_pre_treatment_rmse,
                "cardinality_positive_donors": raw_weights_summary_data.get("Cardinality of Positive Donors"),
                "cardinality_selected_donors": raw_weights_summary_data.get("Cardinality of Selected Donor Pool"),
            }
        )

        # Assemble the final BaseEstimatorResults object
        return BaseEstimatorResults(
            effects=effects_results_obj,
            fit_diagnostics=fit_diagnostics_results_obj,
            time_series=time_series_results_obj,
            weights=weights_results_obj,
            inference=inference_results_obj,
            method_details=method_details_results_obj,
            raw_results=raw_estimation_output, 
        )

    def evaluate_donor(
        self,
        donor_index: int,
        list_of_all_donor_outcome_vectors_pre_treatment: List[np.ndarray],
        treated_outcome_pre_treatment_vector: np.ndarray,
        num_pre_treatment_periods: int,
    ) -> Tuple[int, float]:
        """
        Evaluate the MSE for a given donor index using the SCM optimization.

        This method calculates the Mean Squared Error (MSE) that results from
        using a single specified donor unit to construct a synthetic control
        for the treated unit's pre-treatment outcomes. The optimization is
        performed using a SIMPLEX constraint (weights sum to 1 and are non-negative,
        though for a single donor, the weight will be 1).

        Parameters
        ----------
        donor_index : int
            Index of the donor unit in the `list_of_all_donor_outcome_vectors_pre_treatment` list.
        list_of_all_donor_outcome_vectors_pre_treatment : List[np.ndarray]
            A list where each element is a NumPy array representing the
            pre-treatment outcome time series for a potential donor unit.
            Each array should have shape (T0, 1).
        treated_outcome_pre_treatment_vector : np.ndarray
            NumPy array of the treated unit's pre-treatment outcomes.
            Shape: (T0,).
        num_pre_treatment_periods : int
            The number of pre-treatment time periods.

        Returns
        -------
        Tuple[int, float]
            A tuple containing:
            - donor_index (int): The original index of the evaluated donor.
            - mean_squared_error_for_donor (float): The Mean Squared Error from fitting this donor.
        """
        current_donor_outcome_vector_pre_treatment: np.ndarray = list_of_all_donor_outcome_vectors_pre_treatment[donor_index]
        optimization_problem_result = Opt.SCopt(1, treated_outcome_pre_treatment_vector, num_pre_treatment_periods, current_donor_outcome_vector_pre_treatment, scm_model_type="SIMPLEX")
        mean_squared_error_for_donor: float = optimization_problem_result.solution.opt_val
        return donor_index, mean_squared_error_for_donor

    def fSCM( # Core Forward Selected Synthetic Control Method logic
        self, treated_outcome_pre_treatment_vector: np.ndarray, all_donors_outcomes_matrix_pre_treatment: np.ndarray, num_pre_treatment_periods: int
    ) -> Tuple[List[int], np.ndarray, float, np.ndarray, np.ndarray]: # Renamed from fSCM to _perform_forward_selection
        """
        Perform Forward Selected Synthetic Control Method (SCM) optimization.

        This method implements the core forward selection algorithm. It starts by
        finding the single best donor unit (lowest MSE). Then, it iteratively
        adds remaining donor units to the current best set if the addition
        improves the overall MSE of the synthetic control fit. The weights for
        each candidate set are determined using SIMPLEX optimization.

        Parameters
        ----------
        treated_outcome_pre_treatment_vector : np.ndarray
            Outcome vector for the treated unit in the pre-treatment period.
            Shape: (T0,).
        all_donors_outcomes_matrix_pre_treatment : np.ndarray
            Matrix of outcome vectors for all potential donor units in the
            pre-treatment period. Each column represents a donor.
            Shape: (T0, N_donors).
        num_pre_treatment_periods : int
            Number of pre-treatment time periods.

        Returns
        -------
        Tuple[List[int], np.ndarray, float, np.ndarray, np.ndarray]
            A tuple containing:
            - final_optimal_donor_indices (List[int]): List of integer indices (referring to
              columns in the original `all_donors_outcomes_matrix_pre_treatment`) of the selected optimal donor units.
            - final_optimal_donor_weights_array (np.ndarray): Optimal weights for the selected donor units.
              Shape: (number_of_selected_donors,).
            - final_optimal_rmse (float): The final Root Mean Squared Error of the
              synthetic control fit using the `final_optimal_donor_indices` and `final_optimal_donor_weights_array`.
            - outcomes_matrix_for_final_optimal_donors (np.ndarray): Matrix of pre-treatment outcomes for only
              the selected `final_optimal_donor_indices` donor units. Shape: (T0, number_of_selected_donors).
            - treated_outcome_pre_treatment_vector_out (np.ndarray): The original pre-treatment outcome vector for
              the treated unit (passed as input). Shape: (T0,).
        """
        # Initialize variables for tracking the best donor set and its MSE
        current_best_mse: float = float("inf") # Stores the MSE (sum of squared errors) of the best donor set found so far
        current_best_donor_index_set: Optional[List[int]] = None # Stores indices of donors in the best set

        # Convert columns of the donor matrix to a list of individual donor outcome vectors.
        # This format is suitable for the `evaluate_donor` method used in parallel processing.
        list_of_all_donor_outcome_vectors_pre_treatment: List[np.ndarray] = [
            all_donors_outcomes_matrix_pre_treatment[:, i].reshape(-1, 1) for i in range(all_donors_outcomes_matrix_pre_treatment.shape[1])
        ] 

        # Step 1: Initial Donor Evaluation - Find the single best donor.
        # This step evaluates each potential donor unit individually to find the one
        # that best predicts the treated unit's pre-treatment outcomes on its own.
        # A ThreadPoolExecutor is used to perform these evaluations in parallel for efficiency.
        with ThreadPoolExecutor(max_workers=10) as executor: # max_workers can be tuned based on system capabilities
            # Map the `evaluate_donor` method to each donor index.
            # `evaluate_donor` calculates the MSE for a single donor using SCM optimization.
            mse_results_for_single_donors: List[Tuple[int, float]] = list(
                executor.map( 
                    lambda i: self.evaluate_donor(i, list_of_all_donor_outcome_vectors_pre_treatment, treated_outcome_pre_treatment_vector, num_pre_treatment_periods),
                    range(all_donors_outcomes_matrix_pre_treatment.shape[1]), # Iterate over all donor indices
                )
            )

        # Sort the results by MSE (the second element of the tuple) in ascending order.
        mse_results_for_single_donors.sort(key=lambda x: x[1]) 
        
        # Handle the case where no donors were successfully evaluated (e.g., if the donor matrix was empty).
        if not mse_results_for_single_donors:
            raise MlsynthEstimationError("No donor units were successfully evaluated in the initial parallel step.")

        # Initialize the best donor set with the single best donor found.
        best_single_donor_index, best_initial_mse = mse_results_for_single_donors[0] # Get index and MSE of the best single donor
        current_best_donor_index_set = [best_single_donor_index] # The initial best set contains only this donor
        current_best_mse = best_initial_mse # This MSE is the sum of squared errors (V'V from SCopt), not yet RMSE.

        # Step 2: Iterative Forward Selection - Add more donors to the pool if they improve the fit.
        # Prepare sets of all donor indices and the remaining donors to try adding to the current best set.
        set_of_all_donor_indices: set[int] = set(range(all_donors_outcomes_matrix_pre_treatment.shape[1]))
        set_of_remaining_donor_indices_to_try: set[int] = set_of_all_donor_indices - set(current_best_donor_index_set)

        current_best_donor_index_combination: List[int] = current_best_donor_index_set # Track the current optimal combination of donor indices
        mse_for_current_best_combination: float = current_best_mse # Track the MSE for this combination

        # Iterate through the remaining donor units, trying to add each one to the current best set.
        for candidate_donor_index_to_add in set_of_remaining_donor_indices_to_try:
            # Create a candidate donor set by adding the new donor to the current best combination.
            candidate_donor_index_set_for_evaluation: List[int] = current_best_donor_index_combination + [candidate_donor_index_to_add]
            # Get the outcome matrix for this candidate set of donors.
            outcomes_matrix_for_candidate_donor_set: np.ndarray = all_donors_outcomes_matrix_pre_treatment[:, candidate_donor_index_set_for_evaluation]
            
            try:
                # Perform SCM optimization (SIMPLEX constraint) for the candidate donor set.
                optimization_problem_result = Opt.SCopt(
                    len(candidate_donor_index_set_for_evaluation), # Number of donors in the candidate set
                    treated_outcome_pre_treatment_vector, 
                    num_pre_treatment_periods, 
                    outcomes_matrix_for_candidate_donor_set, 
                    scm_model_type="SIMPLEX" # Use SIMPLEX for constrained optimization (weights sum to 1, non-negative)
                )
                # Check if the optimization was successful and yielded an optimal value (MSE).
                if optimization_problem_result.solution.opt_val is None:
                    # If optimization failed or didn't converge, treat MSE as infinite for this candidate.
                    current_mse_for_candidate_set = float('inf') 
                else:
                    # `opt_val` from SCopt is typically the sum of squared errors (V'V).
                    current_mse_for_candidate_set = optimization_problem_result.solution.opt_val 
            except cp.error.SolverError as e: # Handle CVXPY solver errors specifically.
                current_mse_for_candidate_set = float('inf') # Treat MSE as infinite if solver fails.
                print(f"Warning: CVXPY solver error during forward selection for donor set {candidate_donor_index_set_for_evaluation}: {e}") # Log a warning.
            except Exception as e: # Handle other unexpected errors during optimization.
                raise MlsynthEstimationError(f"Unexpected error during SCM optimization in forward selection for donor set {candidate_donor_index_set_for_evaluation}: {e}") from e

            # If the MSE for the current candidate set is better (lower) than the MSE of the current best combination,
            # update the best combination and its MSE.
            if current_mse_for_candidate_set < mse_for_current_best_combination: # Comparing sum of squared errors directly
                mse_for_current_best_combination = current_mse_for_candidate_set
                current_best_donor_index_combination = candidate_donor_index_set_for_evaluation
        
        # Ensure that an optimal combination of donors was found (should always be true if at least one donor exists).
        if not current_best_donor_index_combination:
            raise MlsynthEstimationError("Optimal combination of donors could not be determined after forward selection.")

        # Step 3: Final Optimization with the Selected Optimal Donor Set.
        # The `current_best_donor_index_combination` now holds the indices of the final selected donors.
        final_optimal_donor_indices: List[int] = current_best_donor_index_combination
        # Get the outcome matrix for this final optimal set of donors.
        outcomes_matrix_for_final_optimal_donors: np.ndarray = all_donors_outcomes_matrix_pre_treatment[:, final_optimal_donor_indices]
        
        try:
            # Perform the final SCM optimization using the selected optimal donor set to get the final weights and MSE.
            final_optimization_result = Opt.SCopt(
                len(final_optimal_donor_indices), 
                treated_outcome_pre_treatment_vector, 
                num_pre_treatment_periods, 
                outcomes_matrix_for_final_optimal_donors, 
                scm_model_type="SIMPLEX"
            )
            # Check if donor weights were successfully obtained from the optimization.
            if final_optimization_result.solution.primal_vars is None or not final_optimization_result.solution.primal_vars:
                raise MlsynthEstimationError("Final SCM optimization did not yield donor weights.")
            
            # Extract the optimal donor weights.
            # `primal_vars` is a dictionary; the weights are typically the value associated with the first (and likely only) key.
            first_key = list(final_optimization_result.solution.primal_vars.keys())[0]
            final_optimal_donor_weights_array: np.ndarray = final_optimization_result.solution.primal_vars[first_key]

            # Check if an optimal MSE value was obtained and calculate the Root Mean Squared Error (RMSE).
            if final_optimization_result.solution.opt_val is None:
                 raise MlsynthEstimationError("Final SCM optimization did not yield an optimal value (MSE).")
            final_mse = final_optimization_result.solution.opt_val # This is the Sum of Squared Errors (V'V).
            # Calculate RMSE from the final MSE.
            final_optimal_rmse = np.sqrt(final_mse / num_pre_treatment_periods) if num_pre_treatment_periods > 0 else float('inf')

        except cp.error.SolverError as e: # Handle CVXPY solver errors in the final optimization.
            raise MlsynthEstimationError(f"CVXPY solver error during final SCM optimization: {e}") from e
        except Exception as e: # Handle other unexpected errors in the final optimization.
            raise MlsynthEstimationError(f"Unexpected error during final SCM optimization: {e}") from e

        # Return the selected donor indices, their weights, the final RMSE, the outcome matrix for these donors,
        # and the original treated outcome vector (for consistency, though it's an input).
        return final_optimal_donor_indices, final_optimal_donor_weights_array, final_optimal_rmse, outcomes_matrix_for_final_optimal_donors, treated_outcome_pre_treatment_vector

    def fit(self) -> BaseEstimatorResults: # Main method to fit the FSCM estimator
        """
        Fits the Forward Selected Synthetic Control Method (FSCM) model.

        This method prepares the data using `dataprep`, then applies the `fSCM`
        optimization algorithm to select an optimal set of donor units and their
        weights. It constructs the counterfactual outcome for the treated unit,
        calculates treatment effects, and formats the results into a
        `BaseEstimatorResults` object. Optionally, it can display and/or save
        plots of the outcomes and effects.

        Returns
        -------
        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:
            - `effects` (EffectsResults): Contains `att` (Average Treatment Effect
              on the Treated) and `att_percent` (Percentage ATT).
            - `fit_diagnostics` (FitDiagnosticsResults): Contains `pre_treatment_rmse`
              (the optimal RMSE from the forward selection), `pre_treatment_r_squared`
              (if calculable from `effects.calculate`), and `additional_metrics`
              (e.g., standard deviation of the post-treatment gap).
            - `time_series` (TimeSeriesResults): Contains `observed_outcome`
              (for the treated unit), `counterfactual_outcome`, `estimated_gap`
              (effect over time), and `time_periods` (actual time values or
              event time indices).
            - `weights` (WeightsResults): Contains `donor_weights`, a dictionary mapping
              selected donor unit names to their optimal weights.
            - `method_details` (MethodDetailsResults): Contains the method `name` ("FSCM")
              and `parameters_used` (including optimal donor indices, optimal
              pre-treatment RMSE, and donor cardinality statistics).
            - `inference` (InferenceResults): Typically not populated by FSCM's core
              logic (fields default to `None`), as standard errors and p-values are
              not directly computed by this algorithm.
            - `raw_results` (Dict[str, Any]): The original dictionary of results
              from the internal estimation functions, including "Effects", "Fit",
              "Vectors", and "Weights" sub-dictionaries.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from mlsynth.estimators.fscm import FSCM
        >>> from mlsynth.config_models import FSCMConfig
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'unit': np.tile(np.arange(1, 5), 10), # 4 units, 10 time periods each
        ...     'time_period': np.repeat(np.arange(1, 11), 4),
        ...     'outcome_val': np.random.rand(40) + \
        ...                    np.tile(np.arange(1, 5), 10)*0.3 + \
        ...                    np.repeat(np.arange(1,11),4)*0.05,
        ...     'treated_indicator': ((np.tile(np.arange(1, 5), 10) == 1) & \
        ...                           (np.repeat(np.arange(1, 11), 4) >= 6)).astype(int)
        ... }) # Unit 1 treated from period 6 onwards
        >>> fscm_config = FSCMConfig(
        ...     df=data,
        ...     outcome="outcome_val",
        ...     treat="treated_indicator",
        ...     unitid="unit",
        ...     time="time_period",
        ...     display_graphs=False # Disable plots for example
        ... )
        >>> fscm_estimator = FSCM(config=fscm_config)
        >>> results = fscm_estimator.fit() # doctest: +SKIP
        >>> # Example: Accessing results (actual values will vary due to random data)
        >>> print(f"Estimated ATT: {results.effects.att}") # doctest: +SKIP
        >>> if results.weights and results.weights.donor_weights: # doctest: +SKIP
        ...     print(f"Selected donor weights: {results.weights.donor_weights}") # doctest: +SKIP
        """
        try:
            # Step 1: Validate data balance (ensures each unit has the same time periods)
            balance(self.df, self.unitid, self.time) # This can raise MlsynthDataError

            # Step 2: Prepare data using the dataprep utility
            # This separates data into treated/control, pre/post periods, and formats it.
            prepared_data_dict: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            ) # This can raise MlsynthDataError or MlsynthConfigError

            # Step 3: Perform essential checks on the output of dataprep
            # Ensure required keys are present and pre_periods is valid.
            required_keys = ["donor_matrix", "pre_periods", "y", "donor_names", "treated_unit_name"]
            for key in required_keys:
                if key not in prepared_data_dict or prepared_data_dict[key] is None:
                    raise MlsynthEstimationError(f"Essential key '{key}' missing or None in dataprep output.")
            if not isinstance(prepared_data_dict["pre_periods"], int) or prepared_data_dict["pre_periods"] <= 0:
                raise MlsynthEstimationError(f"Invalid 'pre_periods' ({prepared_data_dict['pre_periods']}) from dataprep.")

            # Extract pre-treatment outcome data for donors and the treated unit
            all_donors_outcomes_matrix_pre_treatment: np.ndarray = prepared_data_dict["donor_matrix"][
                : prepared_data_dict["pre_periods"] # Slice donor matrix for pre-treatment period
            ]
            treated_outcome_pre_treatment_vector: np.ndarray = prepared_data_dict["y"][
                : prepared_data_dict["pre_periods"] # Slice treated unit's outcome for pre-treatment period
            ]
            num_pre_treatment_periods: int = prepared_data_dict["pre_periods"]

            # Validate shapes of pre-treatment data
            if all_donors_outcomes_matrix_pre_treatment.shape[0] != num_pre_treatment_periods or \
               treated_outcome_pre_treatment_vector.shape[0] != num_pre_treatment_periods:
                raise MlsynthEstimationError("Mismatch in pre-treatment period lengths between donor matrix and treated vector.")
            if all_donors_outcomes_matrix_pre_treatment.shape[1] == 0: # Check if any donor units are available
                raise MlsynthEstimationError("No donor units available after data preparation.")

            # Step 4: Perform the Forward Selected Synthetic Control Method (FSCM) optimization
            # This calls the core fSCM (now _perform_forward_selection) method to get optimal donors, weights, and RMSE.
            (
                selected_optimal_donor_indices, # List of indices of selected donors
                optimal_donor_weights_array,    # NumPy array of weights for selected donors
                final_pre_treatment_rmse,       # Final RMSE of the fit in the pre-treatment period
                outcomes_matrix_for_optimal_donors_pre_treatment, # Pre-treatment outcomes of selected donors
                _, # The fifth element (treated_outcome_pre_treatment_vector) is returned by fSCM but already available here
            ) = self.fSCM(treated_outcome_pre_treatment_vector, all_donors_outcomes_matrix_pre_treatment, num_pre_treatment_periods)

            # Step 5: Construct the full counterfactual outcome series
            # Get the outcome data for the selected optimal donors across all time periods.
            all_periods_outcomes_for_selected_donors: np.ndarray = prepared_data_dict["donor_matrix"][:, selected_optimal_donor_indices]

            # Validate consistency between selected donors and their weights
            if all_periods_outcomes_for_selected_donors.shape[1] != len(optimal_donor_weights_array):
                raise MlsynthEstimationError("Mismatch between number of selected donors and weights array length.")
            
            # Calculate the synthetic counterfactual by applying weights to selected donor outcomes.
            full_counterfactual_outcome_fscm: np.ndarray = np.dot(all_periods_outcomes_for_selected_donors, optimal_donor_weights_array)
            
            # Step 6: Prepare donor names and weights for results
            all_donor_names_from_prepared_data: List[str] = prepared_data_dict["donor_names"]
            # Ensure all donor names are strings
            if not all(isinstance(name, str) for name in all_donor_names_from_prepared_data):
                 raise MlsynthEstimationError("Donor names from dataprep are not all strings.")

            # Get the names of the selected optimal donors using their indices.
            names_of_selected_donors: List[str] = [
                all_donor_names_from_prepared_data[i] for i in selected_optimal_donor_indices
            ]

            # Create a dictionary mapping selected donor names to their weights (rounded for display).
            final_donor_weights_map_for_results: Dict[str, float] = {
                names_of_selected_donors[i]: round(optimal_donor_weights_array[i], 3)
                for i in range(len(selected_optimal_donor_indices))
            }

            # Step 7: Calculate overall effects, fit diagnostics, and time series components
            # Uses the `effects.calculate` utility.
            calculated_effects_fscm, calculated_fit_diagnostics_fscm, calculated_time_series_components_fscm = effects.calculate(
                prepared_data_dict["y"], # Observed outcome for the treated unit
                full_counterfactual_outcome_fscm, # Calculated synthetic counterfactual
                prepared_data_dict["pre_periods"], # Number of pre-treatment periods
                prepared_data_dict["post_periods"], # Number of post-treatment periods
            )
            
            # Step 8: Package raw results into a dictionary for potential inclusion in the final results object.
            raw_estimation_output_dict = {
                "Effects": calculated_effects_fscm,
                "Fit": calculated_fit_diagnostics_fscm,
                "Vectors": calculated_time_series_components_fscm,
                "Weights": [ # Package weights and summary statistics
                    final_donor_weights_map_for_results, # Dictionary of donor weights
                    { # Summary statistics about the donor pool
                        "Cardinality of Positive Donors": int(
                            np.sum(np.abs(optimal_donor_weights_array) > 0.001) # Count donors with non-negligible weights
                        ),
                        "Cardinality of Selected Donor Pool": int(
                            np.shape(outcomes_matrix_for_optimal_donors_pre_treatment)[1] # Total number of selected donors
                        ),
                    },
                ],
            }

            # Step 9: Create the standardized BaseEstimatorResults object using the helper method.
            fscm_estimator_results_obj = self._create_estimator_results(
                raw_estimation_output=raw_estimation_output_dict,
                prepared_data_dict=prepared_data_dict,
                selected_optimal_donor_indices=selected_optimal_donor_indices,
                final_pre_treatment_rmse=final_pre_treatment_rmse,
                optimal_donor_weights_array=optimal_donor_weights_array,
                names_of_selected_donors=names_of_selected_donors
            )

        # Step 10: Handle specific and general exceptions during the fitting process.
        except (MlsynthDataError, MlsynthConfigError) as e: # Propagate custom Mlsynth errors directly.
            raise e
        except KeyError as e: # Handle errors due to missing keys in data structures.
            raise MlsynthEstimationError(f"Missing expected key in data structures: {e}") from e
        except IndexError as e: # Handle errors due to invalid indexing.
            raise MlsynthEstimationError(f"Index out of bounds, likely during donor selection or data processing: {e}") from e
        except ValueError as e: # Catch other ValueErrors (e.g., from np.dot if shapes mismatch).
            raise MlsynthEstimationError(f"ValueError during FSCM estimation: {e}") from e
        except Exception as e: # Catch-all for any other unexpected errors.
            raise MlsynthEstimationError(f"An unexpected error occurred during FSCM fitting: {e}") from e

        # Step 11: Display graphs if requested by the user configuration.
        try:
            if self.display_graphs:
                counterfactual_vector_for_plot = full_counterfactual_outcome_fscm

                plot_estimates(
                    processed_data_dict=prepared_data_dict,
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treat,
                    treated_unit_name=prepared_data_dict["treated_unit_name"],
                    observed_outcome_series=prepared_data_dict["y"],  # Observed outcome vector.
                    counterfactual_series_list=[counterfactual_vector_for_plot.flatten()],  # List of counterfactual vectors.
                    estimation_method_name="FSCM",
                    counterfactual_names=["Forward SCM"],  # Names for legend.
                    treated_series_color=self.treated_color,
                    counterfactual_series_colors=self.counterfactual_color,
                    save_plot_config=self.save)
                
        except MlsynthPlottingError as e: # Handle specific plotting errors defined in Mlsynth.
            print(f"Warning: Plotting failed with MlsynthPlottingError: {e}")
        except MlsynthDataError as e: # Handle data-related errors that might occur during plotting.
            print(f"Warning: Plotting failed due to data issues: {e}")
        except Exception as e: # Catch-all for any other unexpected errors during plotting.
            print(f"Warning: An unexpected error occurred during plotting: {e}")
            
        return fscm_estimator_results_obj
