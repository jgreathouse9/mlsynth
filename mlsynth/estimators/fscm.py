import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Union, Optional
import cvxpy as cp  # For catching solver errors
from pydantic import ValidationError  # For catching Pydantic errors if models are created internally

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates
from ..utils.estutils import Opt, fit_affine_hull_scm, fSCM
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

    def __init__(self, config: FSCMConfig) -> None:  # Changed to FSCMConfig
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
        self.config = config  # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color  # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save  # Align with BaseEstimatorConfig
        self.use_augmented = config.use_augmented

    def _create_estimator_results(  # Helper method to package results into the standard Pydantic model
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
            pre_treatment_rmse=final_pre_treatment_rmse,  # Use the explicitly passed final RMSE
            pre_treatment_r_squared=raw_fit_diagnostics_data.get("R-Squared"),
            additional_metrics={"post_treatment_gap_std": raw_fit_diagnostics_data.get("T1 RMSE")}
            # Store T1 RMSE as post_treatment_gap_std
        )

        # Prepare observed outcome array
        treated_outcome_series_from_prepared_data = prepared_data_dict.get("y")
        treated_outcome_array: Optional[np.ndarray] = None
        if treated_outcome_series_from_prepared_data is not None:
            # Ensure it's a flattened NumPy array
            treated_outcome_array = treated_outcome_series_from_prepared_data.to_numpy().flatten() if isinstance(
                treated_outcome_series_from_prepared_data, pd.Series) else np.array(
                treated_outcome_series_from_prepared_data).flatten()

        # Prepare counterfactual outcome array
        counterfactual_outcome_array: Optional[np.ndarray] = raw_estimation_output['Vectors']['Counterfactual']
        if raw_time_series_data.get("Synthetic") is not None:
            counterfactual_outcome_array = np.array(raw_time_series_data["Synthetic"]).flatten()

        # Prepare estimated gap array
        estimated_gap_array: Optional[np.ndarray] = raw_estimation_output['Vectors']['Gap']

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
            final_donor_weights_map = {name: weight for name, weight in zip(prepared_data_dict["donor_names"].tolist(),
                                                                            optimal_donor_weights_array.flatten())}

        # Create WeightsResults Pydantic model
        weights_results_obj = WeightsResults(
            donor_weights=final_donor_weights_map,
        )

        # Create InferenceResults Pydantic model (FSCM typically doesn't provide standard inference)
        inference_results_obj = InferenceResults()  # Defaults to None for all fields

        # Prepare method details
        raw_weights_summary_data = raw_weights_package if len(raw_weights_package) > 1 else {}
        method_details_results_obj = MethodDetailsResults(
            name="FSCM",
            parameters_used={  # Store key parameters and outcomes of the FSCM process
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
            time_series=time_series_results_obj,  # weights=weights_results_obj,
            inference=inference_results_obj,
            method_details=method_details_results_obj,
            raw_results=raw_estimation_output,
            weights=weights_results_obj
        )


    def fit(self) -> BaseEstimatorResults:  # Main method to fit the FSCM estimator
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
            balance(self.df, self.unitid, self.time)  # This can raise MlsynthDataError

            # Step 2: Prepare data using the dataprep utility
            # This separates data into treated/control, pre/post periods, and formats it.
            prepared_data_dict: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )  # This can raise MlsynthDataError or MlsynthConfigError
            # Step 3: Perform essential checks on the output of dataprep
            required_keys = ["donor_matrix", "pre_periods", "y", "donor_names", "treated_unit_name"]
            for key in required_keys:
                if key not in prepared_data_dict or prepared_data_dict[key] is None:
                    raise MlsynthEstimationError(f"Essential key '{key}' missing or None in dataprep output.")

            if not isinstance(prepared_data_dict["pre_periods"], int) or prepared_data_dict["pre_periods"] <= 0:
                raise MlsynthEstimationError(
                    f"Invalid 'pre_periods' ({prepared_data_dict['pre_periods']}) from dataprep.")

            # Extract pre-treatment outcome data
            all_donors_outcomes_matrix_pre_treatment = prepared_data_dict["donor_matrix"][
                                                       : prepared_data_dict["pre_periods"]]
            treated_outcome_pre_treatment_vector = prepared_data_dict["y"][: prepared_data_dict["pre_periods"]]
            num_pre_treatment_periods = prepared_data_dict["pre_periods"]

            if all_donors_outcomes_matrix_pre_treatment.shape[0] != num_pre_treatment_periods or \
                    treated_outcome_pre_treatment_vector.shape[0] != num_pre_treatment_periods:
                raise MlsynthEstimationError(
                    "Mismatch in pre-treatment period lengths between donor matrix and treated vector.")

            if all_donors_outcomes_matrix_pre_treatment.shape[1] == 0:
                raise MlsynthEstimationError("No donor units available after data preparation.")

            if self.use_augmented:
                selected_optimal_donor_indices, w_augmented, w_original = fSCM(
                    treated_outcome_pre_treatment_vector,
                    all_donors_outcomes_matrix_pre_treatment,
                    num_pre_treatment_periods,
                    augmented=True
                )
                # Compute both counterfactuals
                y_hat_fscm = np.dot(prepared_data_dict["donor_matrix"], w_original)
                y_hat_aug = np.dot(prepared_data_dict["donor_matrix"], w_augmented)

                # Calculate effects for both
                effects_fscm, fit_fscm, vectors_fscm = effects.calculate(
                    prepared_data_dict["y"],
                    y_hat_fscm,
                    prepared_data_dict["pre_periods"],
                    prepared_data_dict["post_periods"],
                )

                effects_aug, fit_aug, vectors_aug = effects.calculate(
                    prepared_data_dict["y"],
                    y_hat_aug,
                    prepared_data_dict["pre_periods"],
                    prepared_data_dict["post_periods"],
                )

                # Package both
                raw_estimation_output_dict = {
                    "Effects (FSCM)": effects_fscm,
                    "Fit (FSCM)": fit_fscm,
                    "Vectors (FSCM)": vectors_fscm,
                    "Effects (Augmented)": effects_aug,
                    "Fit (Augmented)": fit_aug,
                    "Vectors (Augmented)": vectors_aug,
                    "Weights": [
                        {
                            "Donor Weights (FSCM)": {
                                name: round(weight, 3)
                                for name, weight in zip(prepared_data_dict["donor_names"], w_original)
                            },
                            "Donor Weights (Augmented)": {
                                name: round(weight, 3)
                                for name, weight in zip(prepared_data_dict["donor_names"], w_augmented)
                            },
                        },
                        {
                            "Cardinality (FSCM)": int(np.sum(np.abs(w_original) > 0.001)),
                            "Cardinality (Augmented)": int(np.sum(np.abs(w_augmented) > 0.001)),
                        },
                    ],
                }

                # Use w_augmented as the primary weights for downstream summary
                final_weight_vector = w_augmented

            else:
                selected_optimal_donor_indices, w0 = fSCM(
                    treated_outcome_pre_treatment_vector,
                    all_donors_outcomes_matrix_pre_treatment,
                    num_pre_treatment_periods,
                    augmented=False
                )

                y_hat_fscm = np.dot(prepared_data_dict["donor_matrix"], w0)

                effects_fscm, fit_fscm, vectors_fscm = effects.calculate(
                    prepared_data_dict["y"],
                    y_hat_fscm,
                    prepared_data_dict["pre_periods"],
                    prepared_data_dict["post_periods"],
                )

                raw_estimation_output_dict = {
                    "Effects": effects_fscm,
                    "Fit": fit_fscm,
                    "Vectors": vectors_fscm,
                    "Weights": [
                        {
                            name: round(weight, 3)
                            for name, weight in zip(prepared_data_dict["donor_names"], w0)
                        },
                        {
                            "Cardinality": int(np.sum(np.abs(w0) > 0.001)),
                        },
                    ],
                }

            # Create FSCM results

            if self.use_augmented:

                raw_estimation_outputfscm = {
                    "Effects": raw_estimation_output_dict["Effects (FSCM)"],
                    "Fit": raw_estimation_output_dict["Fit (FSCM)"],
                    "Vectors": raw_estimation_output_dict["Vectors (FSCM)"],
                    "Weights": raw_estimation_output_dict['Weights'][0]['Donor Weights (FSCM)'],
                }
            else:

                raw_estimation_outputfscm = {
                    "Effects": raw_estimation_output_dict["Effects"],
                    "Fit": raw_estimation_output_dict["Fit"],
                    "Vectors": raw_estimation_output_dict["Vectors"],
                    "Weights": raw_estimation_output_dict['Weights'][0],
                }

            if self.use_augmented:
                w0 = w_original

            fscm_results = self._create_estimator_results(
                raw_estimation_output=raw_estimation_outputfscm,
                prepared_data_dict=prepared_data_dict,
                selected_optimal_donor_indices=selected_optimal_donor_indices,
                final_pre_treatment_rmse=raw_estimation_outputfscm['Fit']['T0 RMSE'],
                optimal_donor_weights_array=w0,
                names_of_selected_donors=[
                    name for name, weight in zip(prepared_data_dict['donor_names'], w0)
                    if abs(weight) > 0.001
                ],
            )

            if self.use_augmented:

                raw_estimation_outputasc = {
                    "Effects": raw_estimation_output_dict["Effects (Augmented)"],
                    "Fit": raw_estimation_output_dict["Fit (Augmented)"],
                    "Vectors": raw_estimation_output_dict["Vectors (Augmented)"],
                    "Weights": raw_estimation_output_dict['Weights'][0]['Donor Weights (Augmented)'],
                }

                # Create Augmented results
                augmented_results = self._create_estimator_results(
                    raw_estimation_output=raw_estimation_outputasc,
                    prepared_data_dict=prepared_data_dict,
                    selected_optimal_donor_indices=selected_optimal_donor_indices,
                    final_pre_treatment_rmse=raw_estimation_output_dict['Fit (Augmented)']['T0 RMSE'],
                    optimal_donor_weights_array=w_augmented,
                    names_of_selected_donors=[
                        name for name, weight in zip(prepared_data_dict['donor_names'], w_augmented)
                        if abs(weight) > 0.001
                    ],
                )

                # Return a parent results object with sub-method results keyed by method name
                fscm_estimator_results_obj = BaseEstimatorResults(
                    method_details=MethodDetailsResults(
                        name="FSCM",
                        parameters_used={
                            "note": "This is a composite result with FSCM and Augmented sub-methods."}
                    ),
                    sub_method_results={
                        "FSCM": fscm_results,
                        "Augmented": augmented_results,
                    },
                    raw_results=raw_estimation_output_dict,  # raw dict includes both methods
                )

            else:

                # Return a parent results object with sub-method results keyed by method name
                fscm_estimator_results_obj = BaseEstimatorResults(
                    method_details=MethodDetailsResults(
                        name="FSCM",
                        parameters_used={
                            "note": "This is a composite result with FSCM and Augmented sub-methods."}
                    ),
                    sub_method_results={
                        "FSCM": fscm_results
                    },
                    raw_results=raw_estimation_output_dict,  # raw dict includes both methods
                )

        # Step 10: Handle specific and general exceptions during the fitting process.
        except (MlsynthDataError, MlsynthConfigError) as e:  # Propagate custom Mlsynth errors directly.
            raise e
        except KeyError as e:  # Handle errors due to missing keys in data structures.
            raise MlsynthEstimationError(f"Missing expected key in data structures: {e}") from e
        except IndexError as e:  # Handle errors due to invalid indexing.
            raise MlsynthEstimationError(
                f"Index out of bounds, likely during donor selection or data processing: {e}") from e
        except ValueError as e:  # Catch other ValueErrors (e.g., from np.dot if shapes mismatch).
            raise MlsynthEstimationError(f"ValueError during FSCM estimation: {e}") from e
        except Exception as e:  # Catch-all for any other unexpected errors.
            raise MlsynthEstimationError(f"An unexpected error occurred during FSCM fitting: {e}") from e

        # Step 11: Display graphs if requested by the user configuration.
        try:

            if self.display_graphs:

                if self.use_augmented:
                    counterfactual_vector_for_plot = [
                        fscm_estimator_results_obj.sub_method_results['FSCM'].time_series.counterfactual_outcome.flatten(),
                        fscm_estimator_results_obj.sub_method_results[
                            'Augmented'].time_series.counterfactual_outcome.flatten()]
                    cflist = ["FSCM", "FASC"]
                else:
                    counterfactual_vector_for_plot = [
                        fscm_estimator_results_obj.sub_method_results['FSCM'].time_series.counterfactual_outcome.flatten()]
                    cflist = ["FSCM"]

                plot_estimates(
                    processed_data_dict=prepared_data_dict,
                    time_axis_label=self.time,
                    unit_identifier_column_name=self.unitid,
                    outcome_variable_label=self.outcome,
                    treatment_name_label=self.treat,
                    treated_unit_name=prepared_data_dict["treated_unit_name"],
                    observed_outcome_series=prepared_data_dict["y"],  # Observed outcome vector.
                    counterfactual_series_list=counterfactual_vector_for_plot,  # List of counterfactual vectors.
                    estimation_method_name="FSCM",
                    counterfactual_names=cflist,  # Names for legend.
                    treated_series_color=self.treated_color,
                    counterfactual_series_colors=self.counterfactual_color,
                    save_plot_config=self.save)

        except MlsynthPlottingError as e:  # Handle specific plotting errors defined in Mlsynth.
            print(f"Warning: Plotting failed with MlsynthPlottingError: {e}")
        except MlsynthDataError as e:  # Handle data-related errors that might occur during plotting.
            print(f"Warning: Plotting failed due to data issues: {e}")
        except Exception as e:  # Catch-all for any other unexpected errors during plotting.
            print(f"Warning: An unexpected error occurred during plotting: {e}")

        return fscm_estimator_results_obj
