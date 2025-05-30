import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
import warnings
import pydantic # For ValidationError

from ..utils.datautils import dataprep
from ..utils.helperutils import prenorm
from ..utils.estutils import Opt
from ..utils.resultutils import effects, plot_estimates
from ..utils.inferutils import ag_conformal
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..config_models import ( # Import the Pydantic models
    SCMOConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)


class SCMO:
    """
    SCMO: Synthetic Control with Multiple Outcomes.

    Implements synthetic control estimators for settings with a single treated unit
    and potentially multiple auxiliary outcomes. This class supports:
    - **TLP (Tian, Lee, and Panchenko):** A synthetic control method that can
      incorporate auxiliary outcomes by stacking them.
    - **SBMF (Sun et al.):** Another synthetic control variant, potentially
      differing in its optimization or weighting scheme when multiple outcomes
      are considered.
    - **BOTH (Model Averaging):** Averages the TLP and SBMF models based on their
      pre-treatment fit.

    The estimator uses `dataprep` for initial data processing and `Opt.SCopt`
    for the core weight optimization. Conformal prediction intervals can be
    generated for the estimated counterfactual using `ag_conformal`.

    Attributes
    ----------
    config : SCMOConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    outcome : str
        Name of the primary outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    display_graphs : bool, default True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    save : Union[bool, str, Dict[str, str]], default False
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    counterfactual_color : str, default "red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    treated_color : str, default "black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `SCMOConfig`)
    addout : Union[str, List[str]], default_factory=list
        List of names of auxiliary outcome variable(s) to be used in outcome
        stacking. Can be a single string or a list of strings.
        (From `SCMOConfig`)
    method : str, default "TLP"
        The estimation method to use: 'TLP', 'SBMF', or 'BOTH' (for model averaging).
        (From `SCMOConfig`)

    Methods
    -------
    fit()
        Fits the SCMO model and returns standardized estimation results.
    """

    def __init__(self, config: SCMOConfig) -> None: # Changed to SCMOConfig
        """
        Initializes the SCMO estimator with a configuration object.

        Parameters
        ----------

        config : SCMOConfig
            A Pydantic model instance containing all configuration parameters
            for the SCMO estimator. This includes:
            - df (pd.DataFrame): The input DataFrame.
            - outcome (str): Name of the primary outcome variable column.
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
            - addout (Union[str, List[str]], optional): Auxiliary outcome(s). Defaults to empty list.
            - method (str, optional): Estimation method ('TLP', 'SBMF', 'BOTH'). Defaults to "TLP".
        """
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
        self.addout: Union[str, List[str]] = config.addout
        self.method: str = config.method # Pydantic model ensures it's one of "TLP", "SBMF", "BOTH" (uppercase)
        self.conformal_alpha: float = config.conformal_alpha # Added for conformal prediction

        # Pydantic handles pattern validation for method.

    def _create_estimator_results(
        self, raw_method_fit_output: Dict[str, Any], primary_outcome_prepared_data: Dict[str, Any]
    ) -> BaseEstimatorResults:
        """
        Constructs a BaseEstimatorResults object from raw SCMO outputs.

        This helper function takes the raw outputs from one of the SCMO
        fitting procedures (TLP, SBMF, or model-averaged) and maps them to the
        standardized `BaseEstimatorResults` Pydantic model structure.

        Parameters
        ----------

        raw_method_fit_output : Dict[str, Any]
            A dictionary containing the comprehensive results from the SCMO fitting
            process for a specific method (TLP, SBMF, or averaged). This typically
            includes keys like 'Effects', 'Fit', 'Vectors', 'Weights',
            'Conformal Prediction' (for intervals), and 'Lambdas' (for model averaging).
        primary_outcome_prepared_data : Dict[str, Any]
            The dictionary of preprocessed data for the primary outcome,
            returned by `dataprep`. Used here to extract time period information.

        Returns
        -------

        BaseEstimatorResults
            A Pydantic model instance containing the standardized estimation results
            for the specified SCMO method. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates.
                - att (Optional[float]): Average Treatment Effect on the Treated.
                - att_percent (Optional[float]): ATT as a percentage of the
                  pre-treatment mean of the outcome for the treated unit.
                - additional_effects (Optional[Dict[str, Any]]): Dictionary of other
                  named effects.
            - fit_diagnostics (FitDiagnosticsResults): Contains goodness-of-fit metrics.
                - pre_treatment_rmse (Optional[float]): Root Mean Squared Error for the
                  pre-treatment period.
                - pre_treatment_r_squared (Optional[float]): R-squared value for the
                  pre-treatment period.
                - additional_metrics (Optional[Dict[str, Any]]): Dictionary of other
                  diagnostic metrics.
            - time_series (TimeSeriesResults): Contains time-series data for the primary outcome.
                - observed_outcome (Optional[List[float]]): Observed outcome for the
                  treated unit.
                - counterfactual_outcome (Optional[List[float]]): Estimated counterfactual
                  outcome for the treated unit.
                - estimated_gap (Optional[List[float]]): Estimated treatment effect (gap)
                  over time.
                - time_periods (Optional[List[Any]]): Time periods corresponding to
                  the series data.
            - weights (WeightsResults): Contains weights assigned by the model.
                - donor_weights (Optional[Dict[str, float]]): Weights assigned to
                  donor (control) units.
                - summary_stats (Optional[Dict[str, Any]]): If `method="BOTH"`, may
                  contain a summary of positive weights.
            - inference (InferenceResults): Contains statistical inference results.
                - details (Optional[Any]): Typically an `np.ndarray` of conformal
                  prediction intervals (lower and upper bounds for each post-treatment period).
                - method (Optional[str]): Name of the inference method (e.g., "conformal").
                - confidence_level (Optional[float]): The confidence level used for
                  the intervals (e.g., 0.90 for 90% CI, derived from `self.conformal_alpha`).
            - method_details (MethodDetailsResults): Details about the estimation method.
                - name (Optional[str]): Name of the SCMO method used (e.g., "TLP", "SCMO_MA").
                - parameters_used (Optional[Dict[str, Any]]): Configuration parameters
                  used in the estimation (excluding the DataFrame).
                - additional_details (Optional[Dict[str, Any]]): If `method="BOTH"`,
                  contains the model averaging lambdas.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the underlying estimation function for the specific method.
        """
        # --- Effects ---
        # Extract ATT (Average Treatment Effect on the Treated) and related metrics.
        effects_data = raw_method_fit_output.get("Effects", {})
        effects_results = EffectsResults(
            att=effects_data.get("ATT"),
            att_percent=effects_data.get("Percent ATT"),
            additional_effects={ # Store any other effect metrics
                k: v for k, v in effects_data.items() if k not in ["ATT", "Percent ATT"]
            },
        )

        # --- Fit Diagnostics ---
        # Extract pre-treatment fit diagnostics like RMSE and R-squared.
        fit_data = raw_method_fit_output.get("Fit", {})
        fit_diagnostics_results = FitDiagnosticsResults(
            pre_treatment_rmse=fit_data.get("T0 RMSE"),
            pre_treatment_r_squared=fit_data.get("R-Squared"),
            additional_metrics={ # Store any other fit metrics
                k: v for k, v in fit_data.items() if k not in ["T0 RMSE", "R-Squared"]
            },
        )

        # --- Time Series Data ---
        # Extract observed, counterfactual, and gap series for the primary outcome.
        vectors_data = raw_method_fit_output.get("Vectors", {})
        # Determine the correct time periods by looking at the unique sorted time values for the treated unit.
        treated_unit_df = self.df[self.df[self.unitid] == primary_outcome_prepared_data["treated_unit_name"]]
        time_periods_sorted = np.sort(treated_unit_df[self.time].unique())

        time_series_results = TimeSeriesResults(
            observed_outcome=vectors_data.get("Observed Unit"),
            counterfactual_outcome=vectors_data.get("Counterfactual"),
            estimated_gap=vectors_data.get("Gap"),
            time_periods=time_periods_sorted, # Use the sorted unique time periods for consistency
        )
        
        # --- Weights ---
        # Extract donor weights. For "BOTH" method, weights_data might be a list [donor_weights, positive_weights_summary].
        # For TLP/SBMF, it's expected to be a dictionary of donor weights.
        weights_data = raw_method_fit_output.get("weights", raw_method_fit_output.get("Weights")) # Accommodate both key names
        weights_results = WeightsResults()
        if self.method == "BOTH" and isinstance(weights_data, list) and len(weights_data) >= 1:
            weights_results.donor_weights = weights_data[0] # First element is the averaged donor weights
            if len(weights_data) >= 2: # Second element (if present) is a summary of positive weights
                 weights_results.summary_stats = {"positive_weights": weights_data[1]}
        elif isinstance(weights_data, dict): # For single methods (TLP or SBMF)
            weights_results.donor_weights = weights_data
        
        # --- Inference ---
        # Store conformal prediction intervals and the confidence level used.
        inference_results = InferenceResults(
            details=raw_method_fit_output.get("Conformal Prediction"), # This should be the np.ndarray of [lower, upper] bounds
            method="conformal", # Indicate the inference method used
            confidence_level=round(1.0 - self.conformal_alpha, 2) # Calculate and store the confidence level (e.g., 0.90 for 90%)
        )

        # --- Method Details ---
        # Store the SCMO method name and configuration parameters used.
        method_details_results = MethodDetailsResults(
            name=self.method if self.method != "BOTH" else "SCMO_MA", # Use "SCMO_MA" for model averaging
            parameters_used=self.config.model_dump(exclude={'df'}), # Exclude the potentially large DataFrame
        )
        if self.method == "BOTH": # If model averaging, also store the lambdas (weights for TLP/SBMF)
            method_details_results.additional_details = {"lambdas": raw_method_fit_output.get("Lambdas")}

        # Assemble the final BaseEstimatorResults object.
        return BaseEstimatorResults(
            effects=effects_results,
            fit_diagnostics=fit_diagnostics_results,
            time_series=time_series_results,
            weights=weights_results,
            inference=inference_results,
            method_details=method_details_results,
            raw_results=raw_method_fit_output,
        )

    def fit(self) -> BaseEstimatorResults:
        """
        Fits the Synthetic Control with Multiple Outcomes (SCMO) model.

        This method prepares the data for the primary outcome and any specified
        auxiliary outcomes (`addout`). It then applies the chosen estimation
        method ('TLP', 'SBMF', or 'BOTH' for model averaging).
        - For 'TLP' and 'SBMF', it stacks the outcomes (after pre-normalization for TLP),
          estimates donor weights using `Opt.SCopt`, calculates the counterfactual
          for the primary outcome, and generates conformal prediction intervals.
        - For 'BOTH', it performs the above for TLP and SBMF individually, then
          averages these models using `Opt.SCopt` with `model="MA"`.

        The results, including effects, diagnostics, time series, weights, and
        conformal prediction intervals, are packaged into a `BaseEstimatorResults` object.

        Returns
        -------

        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates like ATT
              and percentage ATT.
            - fit_diagnostics (FitDiagnosticsResults): Includes goodness-of-fit metrics
              such as pre-treatment RMSE and R-squared.
            - time_series (TimeSeriesResults): Provides time-series data for the primary
              outcome, including observed outcome, estimated counterfactual, the gap
              (treatment effect) over time, and corresponding time periods.
            - weights (WeightsResults): Contains the weights assigned to donor units.
              If `method="BOTH"`, may also include a summary of positive weights.
            - inference (InferenceResults): Details of statistical inference, primarily
              conformal prediction intervals for the counterfactual outcome in the
              post-treatment periods, along with the confidence level used.
            - method_details (MethodDetailsResults): Information about the SCMO method
              employed (e.g., "TLP", "SBMF", "SCMO_MA"), the configuration parameters
              used (excluding the DataFrame), and, if `method="BOTH"`, the model
              averaging lambdas.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              the underlying estimation function, for detailed inspection.

        Raises
        ------

        MlsynthDataError
            If input data is unsuitable (e.g., multiple treated units for primary outcome,
            non-numeric values in stacked data, issues found by `dataprep`).
        MlsynthConfigError
            If configuration issues arise that are not caught by Pydantic model validation
            but are identified during the fitting process.
        MlsynthEstimationError
            If an error occurs during the core estimation or optimization process
            (e.g., solver failures, issues in conformal prediction).
        pydantic.ValidationError
            If the results data cannot be validated against the `BaseEstimatorResults` model.

        Examples
        --------

        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.estimators.scmo import SCMO
        >>> from mlsynth.config_models import SCMOConfig
        >>> # Load or create panel data
        >>> data = pd.DataFrame({
        ...     'unit': [1,1,1,1, 2,2,2,2, 3,3,3,3],
        ...     'time': [2000,2001,2002,2003, 2000,2001,2002,2003, 2000,2001,2002,2003],
        ...     'Y1':   [10,11,15,16, 20,22,23,24, 12,13,14,15], # Main outcome
        ...     'Y2':   [5,6,7,8, 10,11,12,13, 6,7,8,9],       # Auxiliary outcome
        ...     'D':    [0,0,1,1, 0,0,0,0, 0,0,0,0]             # Treatment for unit 1 from 2002
        ... })
        >>> # Using TLP method with an auxiliary outcome
        >>> scmo_tlp_config = SCMOConfig(
        ...     df=data, outcome="Y1", treat="D", unitid="unit", time="time",
        ...     addout="Y2", method="TLP", display_graphs=False
        ... )
        >>> scmo_tlp_estimator = SCMO(config=scmo_tlp_config)
        >>> results_tlp = scmo_tlp_estimator.fit()
        >>> print(f"TLP ATT: {results_tlp.effects.att}")
        >>> # Using model averaging (BOTH)
        >>> scmo_both_config = SCMOConfig(
        ...     df=data, outcome="Y1", treat="D", unitid="unit", time="time",
        ...     addout="Y2", method="BOTH", display_graphs=False
        ... )
        >>> scmo_both_estimator = SCMO(config=scmo_both_config)
        >>> results_both = scmo_both_estimator.fit()
        >>> print(f"Model Averaged ATT: {results_both.effects.att}")
        >>> if results_both.method_details and results_both.method_details.additional_details:
        ...     print(f"Model Lambdas: {results_both.method_details.additional_details.get('lambdas')}")
        """
        try:
            # Step 1: Prepare data for the primary outcome.
            # `dataprep` handles structuring the data for synthetic control methods.
            primary_outcome_prepared_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # SCMO is designed for a single treated unit for its primary analysis path.
            # If `dataprep` returns a 'cohorts' key, it implies multiple treated units or time periods were found.
            if "cohorts" in primary_outcome_prepared_data:
                raise MlsynthDataError(
                    "SCMO.fit encountered multiple treated units or cohort structure for the primary outcome. "
                    "Please ensure the treatment indicator defines a single treated unit for SCMO's main path."
                )
            
            # Extract key information from the prepared primary outcome data.
            T0: int = primary_outcome_prepared_data['pre_periods'] # Number of pre-treatment periods
            post_periods: int = primary_outcome_prepared_data['post_periods'] # Number of post-treatment periods
            primary_outcome_donor_matrix_all_periods: np.ndarray = primary_outcome_prepared_data["donor_matrix"] # Donor outcomes (all periods)
            primary_outcome_treated_vector_all_periods: np.ndarray = primary_outcome_prepared_data["y"] # Treated unit outcomes (all periods)

            # Step 2: Prepare data for auxiliary outcomes, if specified.
            # Each auxiliary outcome is processed by `dataprep` and stored.
            prepared_data_all_outcomes: List[Dict[str, Any]] = [primary_outcome_prepared_data]
            
            aux_outcomes_to_process: List[str] = [] # Ensure it's always a list
            if isinstance(self.addout, str) and self.addout: # Handle single string case
                aux_outcomes_to_process = [self.addout]
            elif isinstance(self.addout, list): # Handle list case, filtering out empty strings
                aux_outcomes_to_process = [aux for aux in self.addout if aux] 

            for aux_out_name in aux_outcomes_to_process:
                aux_prepared_data = dataprep(self.df, self.unitid, self.time, aux_out_name, self.treat)
                # Ensure auxiliary outcomes also conform to the single treated unit structure.
                if "cohorts" in aux_prepared_data:
                    raise MlsynthDataError(
                        f"Auxiliary outcome '{aux_out_name}' resulted in cohort structure from dataprep. "
                        "SCMO expects consistent single-unit treatment definition across primary and auxiliary outcomes."
                    )
                prepared_data_all_outcomes.append(aux_prepared_data)

            # Dictionary to store results from individual methods (TLP, SBMF) if 'BOTH' is chosen.
            estimators: Dict[str, Dict[str, Any]] = {}
            final_prediction_intervals: Optional[np.ndarray] = None # To store conformal intervals for the final model

            # Step 3: Loop through specified methods (TLP, SBMF, or both if method="BOTH").
            for method_key in (["TLP", "SBMF"] if self.method == "BOTH" else [self.method]):
                # --- Outcome Stacking for TLP/SBMF ---
                # Prepare stacked outcome vectors and donor matrices for the pre-treatment period.
                # TLP uses pre-normalized outcomes, SBMF uses raw outcomes.
                stacked_treated_outcome_vectors_pre_treatment: List[np.ndarray] = []
                stacked_donor_outcome_matrices_pre_treatment: List[np.ndarray] = []

                if method_key == "TLP":
                    # For TLP, pre-normalize each outcome series (treated and donors) before stacking.
                    stacked_treated_outcome_vectors_pre_treatment = [prenorm(r["y"][:T0]) for r in prepared_data_all_outcomes]
                    stacked_donor_outcome_matrices_pre_treatment = [prenorm(r["donor_matrix"][:T0]) for r in prepared_data_all_outcomes]
                elif method_key == "SBMF":
                    # For SBMF, stack the raw outcome series.
                    stacked_treated_outcome_vectors_pre_treatment = [r["y"][:T0] for r in prepared_data_all_outcomes]
                    stacked_donor_outcome_matrices_pre_treatment = [r["donor_matrix"][:T0] for r in prepared_data_all_outcomes]
                
                # Concatenate the lists of arrays into single stacked arrays.
                y_stacked: np.ndarray = np.concatenate(stacked_treated_outcome_vectors_pre_treatment, axis=0)
                Y0_stacked: np.ndarray = np.concatenate(stacked_donor_outcome_matrices_pre_treatment, axis=0)

                # Ensure stacked data is numeric.
                if not (np.all(np.isfinite(y_stacked)) and np.all(np.isfinite(Y0_stacked))):
                    raise MlsynthDataError("Stacked data contains non-numeric values (NaN or inf).")

                # --- Weight Optimization for TLP/SBMF ---
                # Determine the SCM optimization model type based on the method.
                model_type_opt: str = "MSCa" if method_key == "SBMF" else "SIMPLEX" # SBMF uses MSCa, TLP uses SIMPLEX
                # Call Opt.SCopt to find optimal donor weights using the stacked pre-treatment data.
                sc_optimization_result = Opt.SCopt(
                    Y0_stacked.shape[1], y_stacked, T0 * len(prepared_data_all_outcomes), Y0_stacked, scm_model_type=model_type_opt
                )
                
                # Extract the optimized weights.
                first_primal_key = list(sc_optimization_result.solution.primal_vars.keys())[0]
                weights_opt: np.ndarray = sc_optimization_result.solution.primal_vars[first_primal_key]
                
                # --- Counterfactual Calculation for Primary Outcome ---
                # Apply the optimized weights to the *primary outcome's* donor matrix for all periods.
                donor_weights_dict: Dict[str, float]
                estimated_counterfactual_primary_outcome: np.ndarray

                if method_key == "SBMF": # SBMF includes an intercept term in weights_opt[0]
                    donor_weights_dict = {
                        str(primary_outcome_prepared_data["donor_names"][i]): round(weights_opt[i + 1], 3)
                        for i in range(len(primary_outcome_prepared_data["donor_names"]))
                    }
                    estimated_counterfactual_primary_outcome = np.dot(primary_outcome_donor_matrix_all_periods, weights_opt[1:]) + weights_opt[0]
                else: # TLP weights do not include an intercept in this way
                    donor_weights_dict = {
                        str(primary_outcome_prepared_data["donor_names"][i]): round(weights_opt[i], 3)
                        for i in range(len(primary_outcome_prepared_data["donor_names"]))
                    }
                    estimated_counterfactual_primary_outcome = np.dot(primary_outcome_donor_matrix_all_periods, weights_opt)

                # Calculate effects (ATT, %ATT) and fit diagnostics (RMSE, R^2) for the primary outcome.
                attdict, fitdict, vectors_dict = effects.calculate(primary_outcome_treated_vector_all_periods, estimated_counterfactual_primary_outcome, T0, post_periods)
                
                # --- Conformal Prediction for Primary Outcome ---
                # Generate conformal prediction intervals for the post-treatment counterfactual.
                lower_ci_full, upper_ci_full = ag_conformal(
                    primary_outcome_treated_vector_all_periods[:T0], # Pre-treatment observed
                    estimated_counterfactual_primary_outcome[:T0],   # Pre-treatment counterfactual
                    estimated_counterfactual_primary_outcome[T0:],   # Post-treatment counterfactual (to predict intervals for)
                    miscoverage_rate=self.conformal_alpha # Significance level (e.g., 0.1 for 90% CI)
                )
                # Extract only the post-treatment intervals.
                lower_ci_post = lower_ci_full[T0:]
                upper_ci_post = upper_ci_full[T0:]
                prediction_intervals_post_only: np.ndarray = np.vstack([lower_ci_post, upper_ci_post]).T
                # Store full intervals (pre and post) for potential later use or if plotting needs them.
                final_prediction_intervals = np.vstack([lower_ci_full, upper_ci_full]).T 

                # Store results for the current method (TLP or SBMF).
                estimators[method_key] = {
                    "weights": donor_weights_dict, 
                    "Effects": attdict,
                    "Fit": fitdict,
                    "_internal_raw_weights_opt": weights_opt, # Store raw weights for model averaging
                    "Vectors": {
                        **vectors_dict, # Contains Observed, Counterfactual, Gap
                        "Agnostic Prediction Intervals": prediction_intervals_post_only, # Store post-treatment CIs
                    },
                }

            # Step 4: Handle Model Averaging if method is "BOTH".
            output_estimators: Dict[str, Any] # This will hold the final results to be packaged
            cf_vectors_plot: List[np.ndarray] # For plotting
            cf_names_plot: List[str]          # For plotting legend

            if self.method == "BOTH":
                # --- Model Averaging (MA) ---
                # Prepare inputs for model averaging: counterfactuals and full weights from TLP & SBMF.
                cf_vectors_avg: Dict[str, Dict[str, Any]] = {}
                for method_name_avg, info_avg in estimators.items():
                    cf_avg: np.ndarray = info_avg["Vectors"]["Counterfactual"]
                    raw_weights_for_ma: np.ndarray = info_avg["_internal_raw_weights_opt"]
                    w_full_for_ma: np.ndarray # Weights including intercept if SBMF
                    if method_name_avg == "SBMF":
                        w_full_for_ma = raw_weights_for_ma # SBMF weights array includes intercept at index 0
                    elif method_name_avg == "TLP":
                        w_full_for_ma = np.insert(raw_weights_for_ma, 0, 0.0) # TLP needs explicit 0 for intercept for MA
                    else: # Should not happen given the loop structure
                        raise MlsynthEstimationError(f"Unexpected method '{method_name_avg}' encountered during model averaging preparation.")
                    cf_vectors_avg[method_name_avg] = {"cf": cf_avg, "weights": w_full_for_ma}

                # Perform model averaging using Opt.SCopt with model_type="MA".
                # This finds optimal lambdas (weights) for averaging TLP and SBMF.
                model_averaging_optimization_result = Opt.SCopt(
                    len(estimators), # Number of models to average (typically 2: TLP, SBMF)
                    primary_outcome_treated_vector_all_periods[:T0], # Pre-treatment observed primary outcome
                    T0, # Number of pre-treatment periods
                    primary_outcome_donor_matrix_all_periods, # Donor outcomes for primary (used if MA needs to re-evaluate fit)
                    scm_model_type="MA",
                    base_model_results_for_averaging=cf_vectors_avg, # Dict of {method_name: {"cf": series, "weights": array}}
                    donor_names=primary_outcome_prepared_data["donor_names"]
                )
                
                # Calculate the model-averaged counterfactual for the primary outcome.
                # model_averaging_optimization_result["w_MA"] contains [intercept_MA, donor_weights_MA...]
                MA_cf: np.ndarray = (primary_outcome_donor_matrix_all_periods @ model_averaging_optimization_result["w_MA"][1:]) + model_averaging_optimization_result["w_MA"][0]
                
                # Generate conformal prediction intervals for the model-averaged counterfactual.
                lower_ma_ci_full, upper_ma_ci_full = ag_conformal(primary_outcome_treated_vector_all_periods[:T0], MA_cf[:T0], MA_cf[T0:], miscoverage_rate=self.conformal_alpha)
                lower_ma_ci_post = lower_ma_ci_full[T0:]
                upper_ma_ci_post = upper_ma_ci_full[T0:]
                conformal_prediction_post_only = np.vstack([lower_ma_ci_post, upper_ma_ci_post]).T
                final_prediction_intervals = np.vstack([lower_ma_ci_full, upper_ma_ci_full]).T # Update for plotting

                # Calculate effects and fit diagnostics for the model-averaged results.
                MAattdict, MAfitdict, MAVectors = effects.calculate(primary_outcome_treated_vector_all_periods, MA_cf, T0, post_periods)
                
                # Prepare donor weights dictionary for the averaged model.
                ma_donor_weights_only = model_averaging_optimization_result["w_MA"][1:]
                ma_weights_dict: Dict[str, float] = {
                    str(primary_outcome_prepared_data["donor_names"][i]): round(ma_donor_weights_only[i], 3)
                    for i in range(len(primary_outcome_prepared_data["donor_names"]))
                }
                
                # Assemble the output dictionary for the "BOTH" (model-averaged) method.
                output_estimators = {
                    "Effects": MAattdict, "Fit": MAfitdict, "Vectors": MAVectors,
                    "Conformal Prediction": conformal_prediction_post_only,
                    # For "Weights", store averaged donor weights and a summary of positive ones.
                    "Weights": [ma_weights_dict, {k: v for k, v in ma_weights_dict.items() if v > 0.001}],
                    "Lambdas": model_averaging_optimization_result["Lambdas"] # Store the model averaging weights (lambdas)
                }
                cf_vectors_plot = [MA_cf] # For plotting, use the averaged counterfactual
                cf_names_plot = [f"MA {primary_outcome_prepared_data['treated_unit_name']}"]
            else: # Single method (TLP or SBMF) was chosen
                method_result_key = list(estimators.keys())[0] # Should be only one key: "TLP" or "SBMF"
                output_estimators = estimators[method_result_key]
                # Move "Agnostic Prediction Intervals" to "Conformal Prediction" key for consistency.
                output_estimators["Conformal Prediction"] = output_estimators["Vectors"].pop("Agnostic Prediction Intervals")
                cf_vectors_plot = [output_estimators["Vectors"]["Counterfactual"]]
                cf_names_plot = [f"{method_result_key} {primary_outcome_prepared_data['treated_unit_name']}"]

            # Step 5: Package the final results into the standardized Pydantic model.
            results_obj = self._create_estimator_results(output_estimators, primary_outcome_prepared_data)

        except MlsynthDataError: # Re-raise specific errors from utilities or own checks
            # Custom data-related errors from utilities like dataprep or internal checks.
            raise
        except MlsynthConfigError:
            # Custom configuration-related errors.
            raise
        except MlsynthEstimationError: # Errors from Opt.SCopt, ag_conformal, effects.calculate, etc.
            # Custom errors related to the estimation process itself.
            raise
        except pydantic.ValidationError as e_val:
            # Errors during Pydantic model validation when creating the results object.
            raise MlsynthEstimationError(f"Error validating SCMO results: {e_val}") from e_val
        except KeyError as e_key:
            # Missing expected keys in dictionaries (e.g., from dataprep or Opt.SCopt outputs).
            raise MlsynthDataError(f"Missing expected key during SCMO data processing: {e_key}") from e_key
        except IndexError as e_idx:
            # Indexing errors, often related to array manipulation.
            raise MlsynthDataError(f"Index out of bounds during SCMO data processing: {e_idx}") from e_idx
        except TypeError as e_type:
            # Type mismatches during operations.
            raise MlsynthEstimationError(f"Type error during SCMO estimation: {e_type}") from e_type
        except AttributeError as e_attr:
            # Accessing attributes that don't exist.
            raise MlsynthEstimationError(f"Attribute error during SCMO estimation: {e_attr}") from e_attr
        except np.linalg.LinAlgError as e_linalg:
            # Errors from NumPy linear algebra operations (e.g., singular matrix).
            raise MlsynthEstimationError(f"Linear algebra error in SCMO: {e_linalg}") from e_linalg
        except Exception as e: # Catch-all for any other unexpected errors
            # General catch-all for unforeseen issues during the fitting process.
            raise MlsynthEstimationError(f"An unexpected error occurred in SCMO fit: {e}") from e

        # Step 6: Optionally display or save plots of the results.
        if self.display_graphs:
            try:
                plot_estimates(
                    df=primary_outcome_prepared_data, # Pass the prepared data for the primary outcome
                    time=self.time,
                    unitid=self.unitid,
                    outcome=self.outcome, # Primary outcome name
                    treatmentname=self.treat,
                    treated_unit_name=primary_outcome_prepared_data["treated_unit_name"],
                    y=primary_outcome_treated_vector_all_periods, # Observed primary outcome series
                    cf_list=cf_vectors_plot, # List of counterfactual series to plot
                    counterfactual_names=cf_names_plot, # Names for the legend
                    method=self.method if self.method != "BOTH" else "SCMO_MA", # Method name for title/saving
                    treatedcolor=self.treated_color,
                    counterfactualcolors=( # Ensure counterfactualcolors is a list
                        [self.counterfactual_color]
                        if isinstance(self.counterfactual_color, str)
                        else self.counterfactual_color
                    ),
                    save=self.save, # Save behavior based on config
                    uncvectors=final_prediction_intervals # Pass the full conformal prediction intervals (pre and post)
                )
            except MlsynthPlottingError as e_plot: # Specific plotting error
                warnings.warn(f"SCMO plotting failed: {e_plot}", UserWarning)
            except MlsynthDataError as e_plot_data: # Data error encountered during plotting
                 warnings.warn(f"SCMO plotting failed due to data issue: {e_plot_data}", UserWarning)
            except Exception as e_plot_general: # General error during plotting
                warnings.warn(f"An unexpected error occurred during SCMO plotting: {e_plot_general}", UserWarning)
        
        # Step 7: Return the final standardized results object.
        return results_obj
