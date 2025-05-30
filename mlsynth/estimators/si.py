import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import warnings # Added for UserWarning
import pydantic # For pydantic.ValidationError

from ..utils.datautils import balance, dataprep
from ..utils.estutils import pcr
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, MlsynthPlottingError
from ..utils.resultutils import effects, plot_estimates
from ..config_models import ( # Import the Pydantic models
    SIConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults, # Though SI might not populate this much
    MethodDetailsResults,
)


class SI:
    """
    SI: Synthetic Interventions.

    This class implements the Synthetic Interventions method, which estimates
    counterfactual outcomes for a focal treated unit under various alternative
    treatment scenarios. For each alternative intervention specified in the
    `inters` list (which are columns in the dataframe indicating units that
    received that alternative treatment), the method uses Principal Component
    Regression (PCR) to construct a synthetic control for the focal unit.
    The donors for this synthetic control are the units that received the
    specific alternative intervention being considered.

    The `fit` method iterates through each alternative intervention, performs
    the PCR-based estimation, and returns a dictionary of results, with each
    key corresponding to an alternative intervention.

    Attributes
    ----------
    config : SIConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    treat : str
        Name of the treatment indicator column for the focal unit whose
        counterfactuals under alternative interventions are being estimated.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    counterfactual_color : Union[str, List[str]], default "red"
        Default color or list of colors for counterfactual lines in plots. If a single
        string, it's used for all counterfactuals. If a list, colors are cycled.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    treated_color : str, default "black"
        Color for the focal treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `SIConfig`)
    inters : List[str]
        A required list of column names in `df`. Each column must be a binary
        indicator (0 or 1) specifying which units received a particular
        alternative intervention.
        (From `SIConfig`)

    Methods
    -------
    fit()
        Fits the SI model and returns estimation results for each alternative intervention.
    """

    def __init__(self, config: SIConfig) -> None: # Changed to SIConfig
        """
        Initializes the SI (Synthetic Interventions) estimator.

        Parameters
        ----------

        config : SIConfig
            A Pydantic model instance containing all configuration parameters
            for the SI estimator. This includes:
            - df (pd.DataFrame): The input DataFrame.
            - outcome (str): Name of the outcome variable column.
            - treat (str): Name of the treatment indicator column for the focal unit.
            - unitid (str): Name of the unit identifier column.
            - time (str): Name of the time variable column.
            - inters (List[str]): Required list of column names for alternative interventions.
            - display_graphs (bool, optional): Whether to display graphs. Defaults to True.
            - save (Union[bool, str, Dict[str, str]], optional): Configuration for saving plots.
              If `False` (default), plots are not saved. If `True`, plots are saved with
              default names. If a `str`, it's used as the base filename. If a `Dict[str, str]`,
              it maps plot keys to full file paths. Defaults to False.
            - counterfactual_color (Union[str, List[str]], optional): Default color or list of
              colors for counterfactual lines. Defaults to "red".
            - treated_color (str, optional): Color for treated unit line. Defaults to "black".

        Raises
        ------

        ValueError
            If `inters` is an empty list or if any specified intervention column
            is not found in the DataFrame.
        """
        if isinstance(config, dict):
            config =SIConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.inters: List[str] = config.inters
        self.treat: str = config.treat

        # Pydantic ensures 'inters' is List[str] and is provided (as it's a required field in SIConfig
        # with min_length=1). The explicit check `if not self.inters:` is therefore redundant.
        
        # df is guaranteed by BaseEstimatorConfig to be a DataFrame.
        # The check `self.df is None` is redundant.
        # BaseEstimatorConfig also validates that outcome, treat, unitid, time columns exist in df.
        for inter_col in self.inters:
            if inter_col not in self.df.columns:
                raise MlsynthConfigError( # Changed ValueError to MlsynthConfigError
                    f"Intervention column '{inter_col}' specified in 'inters' not found in dataframe."
                )

        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save # Kept Union for flexibility
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color

    def _create_single_intervention_results(
        self,
        pcr_estimation_output: Dict[str, Any],
        focal_unit_prepared_data: Dict[str, Any],
        intervention_name: str,
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object for a single alternative intervention.

        This helper function takes the raw outputs from the PCR estimation for one
        alternative intervention and maps them to the standardized
        `BaseEstimatorResults` Pydantic model structure.

        Parameters
        ----------

        pcr_estimation_output : Dict[str, Any]
            A dictionary containing the results from the PCR estimation for the
            specified alternative intervention. Expected keys include 'Effects',
            'Fit', 'Vectors', and 'Weights'. May also include an 'Error' key.
        focal_unit_prepared_data : Dict[str, Any]
            The dictionary of preprocessed data for the focal treated unit,
            returned by `dataprep`. Used here to extract time period information.
        intervention_name : str
            The name of the alternative intervention being processed.

        Returns
        -------

        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results
            for the specified alternative intervention. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates like ATT
              and percentage ATT for the focal unit under this alternative intervention.
            - fit_diagnostics (FitDiagnosticsResults): Includes goodness-of-fit metrics
              such as pre-treatment RMSE and R-squared.
            - time_series (TimeSeriesResults): Provides time-series data including the
              observed outcome of the focal unit, its estimated counterfactual outcome
              under this alternative intervention, the estimated gap (treatment effect)
              over time, and the corresponding time periods.
            - weights (WeightsResults): Contains the weights assigned to donor units
              (those that received this specific alternative intervention) used to
              construct the counterfactual.
            - inference (InferenceResults): Basic information indicating the estimation
              method ("PCR-based point estimate"), as SI via PCR doesn't inherently
              produce detailed statistical inference like p-values or CIs.
            - method_details (MethodDetailsResults): Details about the specific synthetic
              intervention, including its name (e.g., "SI_for_intervention_X") and the
              configuration parameters used.
            - execution_summary (Optional[Dict[str, Any]]): If an error occurred during
              estimation for this intervention, this field will contain error details.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the PCR estimation for this intervention.
        """
        try:
            if pcr_estimation_output.get("Error"):
                return BaseEstimatorResults(
                    method_details=MethodDetailsResults(name=f"SI_for_{intervention_name}_FAILED"),
                    execution_summary={"error": pcr_estimation_output["Error"]},
                    raw_results=pcr_estimation_output
                )

            effects_data = pcr_estimation_output.get("Effects", {})
            effects_results = EffectsResults(
                att=effects_data.get("ATT"),
                att_percent=effects_data.get("Percent ATT"),
                additional_effects={
                    k: v for k, v in effects_data.items() if k not in ["ATT", "Percent ATT"]
                },
            )

            fit_data = pcr_estimation_output.get("Fit", {})
            fit_diagnostics_results = FitDiagnosticsResults(
                pre_treatment_rmse=fit_data.get("T0 RMSE"),
                pre_treatment_r_squared=fit_data.get("R-Squared"),
                additional_metrics={
                    k: v for k, v in fit_data.items() if k not in ["T0 RMSE", "R-Squared"]
                },
            )
            
            # Attempt to get sorted time periods for the focal unit from the original dataframe.
            # This provides the time axis for the observed and counterfactual series.
            focal_treated_unit_id = focal_unit_prepared_data.get("treated_unit_name")
            time_periods_sorted = None
            if focal_treated_unit_id is not None: # Ensure we have a focal unit ID.
                 focal_unit_df = self.df[self.df[self.unitid] == focal_treated_unit_id] # Filter original df for the focal unit.
                 if not focal_unit_df.empty: # Ensure the focal unit exists in the df.
                    time_periods_sorted = np.sort(focal_unit_df[self.time].unique()) # Get unique, sorted time periods.


            vectors_data = pcr_estimation_output.get("Vectors", {})
            time_series_results = TimeSeriesResults(
                observed_outcome=vectors_data.get("Observed Unit"),
                counterfactual_outcome=vectors_data.get("Counterfactual"),
                estimated_gap=vectors_data.get("Gap"),
                time_periods=time_periods_sorted,
            )

            weights_results = WeightsResults(
                donor_weights=pcr_estimation_output.get("Weights")
            )
            
            inference_results = InferenceResults(method="PCR-based point estimate")

            method_details_results = MethodDetailsResults(
                name=f"SI_for_{intervention_name}",
                parameters_used=self.config.model_dump(exclude={'df'}),
            )

            return BaseEstimatorResults(
                effects=effects_results,
                fit_diagnostics=fit_diagnostics_results,
                time_series=time_series_results,
                weights=weights_results,
                inference=inference_results,
                method_details=method_details_results,
                raw_results=pcr_estimation_output,
            )
        except pydantic.ValidationError as e:
            error_detail = f"Pydantic validation failed constructing results for {intervention_name}: {e}"
            warnings.warn(f"{error_detail} from _create_single_intervention_results", UserWarning) # Added warning
            return BaseEstimatorResults(
                method_details=MethodDetailsResults(name=f"SI_for_{intervention_name}_RESULT_ERROR"),
                execution_summary={"error": error_detail},
                raw_results=pcr_estimation_output # Include raw PCR output if available.
            )
        except Exception as e: # Catch any other unexpected error during the construction of Pydantic result models.
            error_detail = f"Unexpected error constructing results for {intervention_name}: {type(e).__name__}: {e}"
            warnings.warn(f"{error_detail} from _create_single_intervention_results", UserWarning) # Log the error.
            return BaseEstimatorResults( # Return an error-specific result object.
                method_details=MethodDetailsResults(name=f"SI_for_{intervention_name}_RESULT_ERROR"),
                execution_summary={"error": error_detail},
                raw_results=pcr_estimation_output # Include raw PCR output if available, might be None or partial.
            )

    def fit(self) -> Dict[str, BaseEstimatorResults]: # Changed return type
        """
        Fits the Synthetic Interventions (SI) model.

        This method iterates through each alternative intervention specified in the
        `inters` list (defined in the configuration). For each alternative
        intervention:
        1. It identifies the set of donor units that received that specific
           alternative intervention.
        2. It uses Principal Component Regression (`pcr` from `estutils`) to estimate
           a counterfactual outcome for the focal treated unit, using the identified
           donors.
        3. The results (effects, fit diagnostics, time series, weights) for this
           specific intervention are packaged into a `BaseEstimatorResults` object.

        The method returns a dictionary where each key is an alternative intervention's
        name (column name from `inters`), and the value is the corresponding
        `BaseEstimatorResults` object.

        Returns
        -------

        Dict[str, BaseEstimatorResults]
            A dictionary mapping each alternative intervention name (str) to its
            `BaseEstimatorResults` object. Each `BaseEstimatorResults` object,
            as detailed in `_create_single_intervention_results`, contains fields for
            effects, fit diagnostics, time series, weights, method details, and raw results
            pertaining to the focal unit's outcome under that specific alternative
            intervention. If an error occurred for a particular intervention, the
            corresponding `BaseEstimatorResults` object will primarily contain error
            information in `execution_summary` and `method_details`.

        Raises
        ------

        MlsynthDataError
            If essential data (e.g., period information) is missing or malformed
            after initial data preparation.
        MlsynthConfigError
            If there's an issue with the provided configuration that prevents setup.
        MlsynthEstimationError
            If an unexpected error occurs during the core estimation setup that
            is not a data or config issue.

        Examples
        --------

        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.estimators.si import SI
        >>> from mlsynth.config_models import SIConfig
        >>> # Panel data with a focal treatment and two alternative interventions
        >>> data = pd.DataFrame({
        ...     'unit': [1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5],
        ...     'time': [1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3],
        ...     'outcome': [10,12,15, 20,21,22, 15,16,18, 11,13,14, 25,26,28],
        ...     'focal_treat': [0,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0], # Unit 1 is focal treated
        ...     'alt_treat_A': [0,0,0, 0,1,1, 0,1,1, 0,0,0, 0,0,0], # Units 2,3 got alt_treat_A
        ...     'alt_treat_B': [0,0,0, 0,0,0, 0,0,0, 0,1,1, 0,1,1]  # Units 4,5 got alt_treat_B
        ... })
        >>> si_config = SIConfig(
        ...     df=data, outcome="outcome", treat="focal_treat", unitid="unit", time="time",
        ...     inters=["alt_treat_A", "alt_treat_B"], display_graphs=False
        ... )
        >>> si_estimator = SI(config=si_config)
        >>> results_dict = si_estimator.fit()
        >>> # Results for alternative treatment A
        >>> results_A = results_dict.get("alt_treat_A")
        >>> if results_A and results_A.effects:
        ...     print(f"ATT for alt_treat_A: {results_A.effects.att}")
        >>> # Results for alternative treatment B
        >>> results_B = results_dict.get("alt_treat_B")
        >>> if results_B and results_B.effects:
        ...     print(f"ATT for alt_treat_B: {results_B.effects.att}")
        """
        all_intervention_results: Dict[str, BaseEstimatorResults] = {}
        counterfactuals_list: List[np.ndarray] = []
        counterfactual_names_list: List[str] = []
        focal_unit_prepared_data: Optional[Dict[str, Any]] = None # To store dataprep output for the focal unit.
        pre_periods_focal: Optional[int] = None # Number of pre-treatment periods for the focal unit.
        post_periods_focal: Optional[int] = None # Number of post-treatment periods for the focal unit.
        intervention_sets: Dict[str, set] = {} # Maps alternative intervention names to sets of units that received them.

        # --- Setup Phase: Prepare data for the focal unit and identify donor sets for alternative interventions ---
        try:
            # Ensure the panel data is balanced (all units observed for all time periods).
            balance(self.df, self.unitid, self.time) # Can raise MlsynthDataError.

            # Prepare data for the focal unit based on its own treatment status (self.treat).
            # This provides the observed outcome series for the focal unit and its pre/post periods.
            focal_unit_prepared_data = dataprep( # Can raise MlsynthDataError.
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # Handle cases where dataprep might return cohortized data for the focal unit.
            # This is unusual for SI's focal unit (which typically has a single treatment timing)
            # but dataprep might produce it if self.treat implies multiple cohorts.
            # We need to extract pre/post periods, typically from the first cohort if multiple exist.
            if "cohorts" in focal_unit_prepared_data and not all(
                k in focal_unit_prepared_data for k in ["pre_periods", "post_periods"] # Check if top-level period keys are missing.
            ): # This condition implies 'cohorts' exists, but top-level period keys do not.
                if focal_unit_prepared_data.get("cohorts"): # Ensure 'cohorts' is not empty.
                    example_cohort_key = next(iter(focal_unit_prepared_data["cohorts"])) # Get the key of the first cohort.
                    cohort_data = focal_unit_prepared_data["cohorts"][example_cohort_key]
                    pre_periods_focal = cohort_data.get("pre_periods")
                    post_periods_focal = cohort_data.get("post_periods")
                    warnings.warn( # Warn user if using cohort data for focal unit's periods.
                        f"dataprep returned cohortized data for the focal unit based on '{self.treat}'. "
                        f"Using pre_periods={pre_periods_focal} and post_periods={post_periods_focal} "
                        f"from the first cohort ({example_cohort_key}). This may not be the intended behavior for SI's focal unit.",
                        UserWarning,
                    )
                # If, after trying to get from cohorts, period info is still missing, it's an error.
                if pre_periods_focal is None or post_periods_focal is None:
                    raise MlsynthDataError(
                        "Essential period keys ('pre_periods', 'post_periods') missing from cohortized focal unit data "
                        "after dataprep. Cannot proceed."
                    )
            else: # If not cohortized, or if top-level period keys exist, use them.
                pre_periods_focal = focal_unit_prepared_data.get("pre_periods")
                post_periods_focal = focal_unit_prepared_data.get("post_periods")

            # Validate that pre_periods_focal and post_periods_focal were successfully determined.
            if pre_periods_focal is None:
                raise MlsynthDataError(
                    f"Key 'pre_periods' is missing from focal_unit_prepared_data for focal treatment '{self.treat}'. "
                    f"Dataprep output keys: {list(focal_unit_prepared_data.keys())}"
                )
            if post_periods_focal is None:
                raise MlsynthDataError(
                    f"Key 'post_periods' is missing from focal_unit_prepared_data for focal treatment '{self.treat}'. "
                    f"Dataprep output keys: {list(focal_unit_prepared_data.keys())}"
                )

            # For each alternative intervention, identify the set of units that received it.
            # These units will serve as potential donors for that specific intervention scenario.
            intervention_sets = {
                col: set(self.df.loc[self.df[col] == 1, self.unitid]) # Get unit IDs where intervention col is 1.
                for col in self.inters
            }

        # --- Error Handling for Setup Phase ---
        except (MlsynthDataError, MlsynthConfigError) as e: # Catch known data or config errors.
            error_msg = f"SI fit failed during initial data/config setup: {type(e).__name__}: {e}"
            warnings.warn(error_msg, UserWarning)
            # If setup fails, populate error results for all planned interventions and return.
            for alt_treat_name_iter in self.inters:
                all_intervention_results[alt_treat_name_iter] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name_iter
                )
            return all_intervention_results
        except KeyError as e: # Should be caught by MlsynthDataError from period checks, but as a fallback
            error_msg = f"SI fit failed due to missing key during setup: {e}. This likely indicates an issue with dataprep output."
            warnings.warn(error_msg, UserWarning)
            for alt_treat_name_iter in self.inters:
                all_intervention_results[alt_treat_name_iter] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name_iter
                )
            return all_intervention_results
        except Exception as e: # Catch any other unexpected errors during setup
            # This is a more critical failure if it happens during setup.
            # We might choose to re-raise it as MlsynthEstimationError after logging.
            error_msg = f"Unexpected critical error during SI setup: {type(e).__name__}: {e}"
            warnings.warn(error_msg, UserWarning) # Log it
            # Decide whether to return partial error dict or re-raise
            # For now, let's populate error dicts and return, to be consistent with per-intervention errors.
            for alt_treat_name_iter in self.inters:
                 all_intervention_results[alt_treat_name_iter] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name_iter
                )
            return all_intervention_results
        except KeyError as e: # Fallback for unexpected missing keys during setup.
            error_msg = f"SI fit failed due to missing key during setup: {e}. This likely indicates an issue with dataprep output."
            warnings.warn(error_msg, UserWarning)
            for alt_treat_name_iter in self.inters: # Populate error results for all interventions.
                all_intervention_results[alt_treat_name_iter] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name_iter
                )
            return all_intervention_results
        except Exception as e: # Catch any other unexpected critical errors during setup.
            error_msg = f"Unexpected critical error during SI setup: {type(e).__name__}: {e}"
            warnings.warn(error_msg, UserWarning) # Log the critical error.
            for alt_treat_name_iter in self.inters: # Populate error results for all interventions.
                 all_intervention_results[alt_treat_name_iter] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name_iter
                )
            return all_intervention_results
            # For a hard stop, one might re-raise: raise MlsynthEstimationError(error_msg) from e

        # --- Per-Intervention Loop: Estimate counterfactual for each alternative intervention ---
        for alt_treat_name, donor_units_set in intervention_sets.items(): # Iterate through each defined alternative intervention.
            try:
                # Defensive check, though setup phase should ensure focal_unit_prepared_data is populated.
                if focal_unit_prepared_data is None:
                    raise MlsynthEstimationError("Critical error: focal_unit_prepared_data is None before intervention loop.")
                
                # Ensure necessary data structures from dataprep are available.
                if "Ywide" not in focal_unit_prepared_data or "y" not in focal_unit_prepared_data:
                    raise MlsynthDataError(f"Essential keys 'Ywide' or 'y' missing from focal_unit_prepared_data for intervention '{alt_treat_name}'.")

                # Identify donor units for the current alternative intervention from the wide-format outcome data.
                # These are units that received `alt_treat_name` and are present in `focal_unit_prepared_data["Ywide"]`.
                donor_cols_pd = focal_unit_prepared_data["Ywide"].columns.intersection(list(donor_units_set))
                donor_outcome_matrix_for_pcr: np.ndarray = focal_unit_prepared_data["Ywide"][donor_cols_pd].to_numpy()

                # If no donors are found for this alternative intervention, skip estimation for it.
                if donor_outcome_matrix_for_pcr.shape[1] == 0:
                    warnings.warn(f"No common donors found for intervention '{alt_treat_name}'. Skipping.", UserWarning)
                    error_output = {"Error": "No common donors found"}
                    all_intervention_results[alt_treat_name] = self._create_single_intervention_results(
                        error_output, focal_unit_prepared_data, alt_treat_name
                    )
                    continue # Move to the next alternative intervention.

                # Defensive check for pre_periods_focal.
                if pre_periods_focal is None:
                    raise MlsynthEstimationError("pre_periods_focal is None before calling pcr.")

                # Perform Principal Component Regression (PCR) to estimate the counterfactual.
                # Uses donors who received `alt_treat_name` to synthesize the focal unit.
                pcr_result: Dict[str, Any] = pcr(
                    donor_outcomes_matrix=donor_outcome_matrix_for_pcr, # Outcomes of donors for this intervention.
                    treated_unit_outcome_vector=focal_unit_prepared_data["y"], # Observed outcome of the focal unit.
                    scm_objective_model_type="OLS", # Standard SCM objective for PCR.
                    all_donor_names=[str(name) for name in donor_cols_pd], # Names of the donor units.
                    num_pre_treatment_periods=pre_periods_focal, # Number of pre-periods for the focal unit.
                    enable_clustering=False, # Clustering not typically used in standard SI via PCR.
                    use_frequentist_scm=True # Use standard SCM optimization.
                )

                # Validate PCR result structure.
                if "cf_mean" not in pcr_result or "weights" not in pcr_result:
                    raise MlsynthEstimationError(f"PCR result for '{alt_treat_name}' missing 'cf_mean' or 'weights'.")
                
                cf_pcr: np.ndarray = pcr_result["cf_mean"] # Estimated counterfactual outcome series.
                weight_dict_current: Dict[str, float] = pcr_result["weights"] # Weights assigned to donors.

                # Defensive check for post_periods_focal.
                if post_periods_focal is None:
                    raise MlsynthEstimationError("post_periods_focal is None before calling effects.calculate.")

                # Calculate treatment effects (ATT, etc.) and fit diagnostics.
                attdict, fitdict, vectors = effects.calculate(
                    focal_unit_prepared_data["y"], cf_pcr, pre_periods_focal, post_periods_focal
                )
                
                # Package all raw results for this intervention.
                current_intervention_raw_output = {
                    "Effects": attdict, "Fit": fitdict, "Vectors": vectors, "Weights": weight_dict_current,
                }
                # Convert raw results to the standardized Pydantic model.
                all_intervention_results[alt_treat_name] = self._create_single_intervention_results(
                    current_intervention_raw_output, focal_unit_prepared_data, alt_treat_name
                )

                # If estimation was successful, add the counterfactual series for potential plotting.
                if "Error" not in all_intervention_results[alt_treat_name].execution_summary if all_intervention_results[alt_treat_name].execution_summary else True:
                    counterfactuals_list.append(cf_pcr)
                    counterfactual_names_list.append(
                        f"{alt_treat_name} (Synthetic for {focal_unit_prepared_data.get('treated_unit_name', 'Unknown Unit')})"
                    )
            
            # --- Error Handling for a Single Intervention's Estimation ---
            except (MlsynthDataError, MlsynthEstimationError, MlsynthConfigError) as e: # Catch known errors.
                error_msg = f"Estimation failed for {alt_treat_name}: {type(e).__name__}: {e}"
                all_intervention_results[alt_treat_name] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name
                )
            except pydantic.ValidationError as e: # Catch Pydantic errors during result packaging.
                error_msg = f"Result processing failed for {alt_treat_name} due to Pydantic error: {e}"
                all_intervention_results[alt_treat_name] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name
                )
            except KeyError as e: # Catch missing keys, often indicating data issues.
                error_msg = f"Missing data key during processing for {alt_treat_name}: {e}"
                all_intervention_results[alt_treat_name] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name
                )
            except Exception as e: # Catch-all for any other unexpected error for this specific intervention.
                error_msg = f"Unexpected error processing {alt_treat_name}: {type(e).__name__}: {e}"
                warnings.warn(f"Unexpected error during SI for intervention '{alt_treat_name}': {e}", UserWarning)
                all_intervention_results[alt_treat_name] = self._create_single_intervention_results(
                    {"Error": error_msg}, focal_unit_prepared_data if focal_unit_prepared_data else {}, alt_treat_name
                )
        
        # --- Plotting Phase: Visualize results if requested and data is available ---
        if self.display_graphs and counterfactuals_list and focal_unit_prepared_data:
            # Ensure all necessary data for plotting is valid and present.
            try:
                # Defensive checks for data required by plot_estimates.
                if focal_unit_prepared_data is None:
                    raise MlsynthPlottingError("Cannot plot: focal_unit_prepared_data is None.")
                if "y" not in focal_unit_prepared_data or "treated_unit_name" not in focal_unit_prepared_data :
                     raise MlsynthDataError("Cannot plot: 'y' or 'treated_unit_name' missing from focal_unit_prepared_data.")

                # Call the generic plotting utility.
                plot_estimates(
                    processed_data_dict=focal_unit_prepared_data, # Pass the prepared data for context if needed by plotter.
                    time_axis_label=self.time, # Name of the time column.
                    unit_identifier_column_name=self.unitid, # Name of the unit ID column.
                    outcome_variable_label=self.outcome, # Name of the outcome column.
                    treatment_name_label="Focal Treatment Start", # Label for the treatment start time.
                    treated_unit_name=str(focal_unit_prepared_data["treated_unit_name"]), # Name of the focal treated unit.
                    observed_outcome_series=focal_unit_prepared_data["y"], # Observed outcome series of the focal unit.
                    counterfactual_series_list=counterfactuals_list, # List of estimated counterfactual series.
                    counterfactual_names=counterfactual_names_list, # Names for each counterfactual series.
                    estimation_method_name="SI", # Method name for plot titles/legends.
                    treated_series_color=self.treated_color, # Color for the observed outcome line.
                    counterfactual_series_colors=( # Colors for the counterfactual lines.
                        [self.counterfactual_color] # Use single color if string.
                        if isinstance(self.counterfactual_color, str)
                        else self.counterfactual_color # Use list of colors if provided.
                    ),
                    save_plot_config=self.save, # Plot saving configuration.
                )
            except (MlsynthPlottingError, MlsynthDataError) as e: # Catch known plotting or data errors.
                warnings.warn(f"Plotting failed for SI due to data or plotting issue: {e}", UserWarning)
            except Exception as e: # Catch any other unexpected errors during plotting.
                warnings.warn(f"An unexpected error occurred during SI plotting: {type(e).__name__}: {e}", UserWarning)

        return all_intervention_results
