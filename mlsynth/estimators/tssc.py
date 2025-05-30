import numpy as np
import pandas as pd # Added for type hinting
from typing import Dict, Any, List, Union, Optional
import scipy.stats # For Z-score calculation
import warnings # For issuing warnings
import pydantic # For ValidationError

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import plot_estimates
from ..utils.estutils import TSEST
# --- Temporary direct import for debugging inferutils (Reinstated) ---
import importlib.util
import sys
from pathlib import Path
inferutils_path = Path(__file__).resolve().parent.parent / "utils" / "inferutils.py"
spec = importlib.util.spec_from_file_location("mlsynth.utils.inferutils_debug", str(inferutils_path))
inferutils_debug_module = importlib.util.module_from_spec(spec)
sys.modules["mlsynth.utils.inferutils_debug"] = inferutils_debug_module
spec.loader.exec_module(inferutils_debug_module)
step2 = inferutils_debug_module.step2
# from ..utils.inferutils import step2 # Original import commented out
# --- End temporary import (Reinstated) ---
from ..config_models import ( # Import Pydantic models
    TSSCConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)


# Constants for keys from TSEST output
_KEY_EFFECTS = "Effects"
_KEY_FIT = "Fit"
_KEY_VECTORS = "Vectors"
_KEY_ATT = "ATT"
_KEY_ATT_CI = "ATT CI"  # Assuming TSEST might provide this for ATT directly
_KEY_PERCENT_ATT = "Percent ATT"
_KEY_T0_RMSE = "T0 RMSE"
_KEY_T1_RMSE = "T1 RMSE"
_KEY_R_SQUARED = "R-Squared"
_KEY_COUNTERFACTUAL = "Counterfactual"
_KEY_GAP = "Gap"
_KEY_WEIGHT_V = "WeightV"
_KEY_DEFAULT_CI_FROM_TSEST = "95% CI" # Key for the main CI from TSEST
_KEY_P_VALUE_MANUAL = "p_value_manual"

# SC Method Names (used internally and with TSEST/step2)
_SC_METHOD_SIMPLEX = "SIMPLEX"
_SC_METHOD_MSCA = "MSCa"
_SC_METHOD_MSCB = "MSCb"
_SC_METHOD_MSCC = "MSCc"

# Plotting related constants
_PLOT_METHOD_NAME = "TSSC" # For plot_estimates method arg
_PLOT_RECOMMENDED_LABEL_SUFFIX = " (Recommended)"

# Warning Messages
_WARN_MSCC_WEIGHTS_NOT_FOUND = (
    "Warning: MSCc weights not found for step2 validation. Plotting may be affected."
)
_WARN_NO_COUNTERFACTUAL_FOR_PLOT = (
    "Warning: Could not find counterfactual data for recommended model '{}' to plot."
)


class TSSC:
    """Two-Step Synthetic Control (TSSC) estimator.

    This estimator implements the Two-Step Synthetic Control method, which
    evaluates several underlying synthetic control (SC) model specifications
    (SIMPLEX, MSCa, MSCb, MSCc) and recommends one based on a second-step
    validation procedure. It provides estimates and inference for each SC
    method considered.

    The first step involves fitting each of the SC model variants. The second
    step uses a validation procedure (detailed in `utils.inferutils.step2`)
    to select a "recommended" model from these variants, typically based on
    pre-treatment fit or other criteria.

    Parameters
    ----------
    config : TSSCConfig
        Configuration object containing all necessary parameters. See
        `TSSCConfig` and `BaseEstimatorConfig` for details on attributes
        like `df`, `outcome`, `treat`, `unitid`, `time`, `draws`, etc.

    Attributes
    ----------
    config : TSSCConfig
        The configuration object passed during instantiation.
    df : pd.DataFrame
        The input DataFrame containing panel data.
    outcome : str
        Name of the outcome variable column in `df`.
    treat : str
        Name of the treatment indicator column in `df`.
    unitid : str
        Name of the unit identifier (ID) column in `df`.
    time : str
        Name of the time variable column in `df`.
    counterfactual_color : Union[str, List[str]]
        Color(s) to use for the counterfactual line(s) in plots. If a single string,
        it's used for the recommended model's counterfactual. If a list, colors
        could be cycled if multiple counterfactuals were plotted simultaneously (though
        TSSC typically plots only the recommended one). Default is "red".
        (Inherited from `BaseEstimatorConfig` via `TSSCConfig`)
    treated_color : str
        Color to use for the treated unit line in plots. Default is "black".
        (Inherited from `BaseEstimatorConfig` via `TSSCConfig`)
    display_graphs : bool
        If True, graphs of the results for the recommended model will be
        displayed. Default is True.
        (Inherited from `BaseEstimatorConfig` via `TSSCConfig`)
    save : Union[bool, str, Dict[str, str]]
        Configuration for saving plots.
        - If `False` (default), plots are not saved.
        - If `True`, plots are saved with default names in the current directory.
        - If a `str`, it's used as the base filename for saved plots.
        - If a `Dict[str, str]`, it maps specific plot keys (e.g., "estimates_plot")
          to full file paths.
        (Inherited from `BaseEstimatorConfig` via `TSSCConfig`)
    draws : int
        Number of subsample replications or draws used for inference procedures
        within the underlying SC methods and the second-step validation.
        Default is 500.
    ci : float
        Confidence interval level (e.g., 0.95 for 95% CI). Note: This is part
        of the config but its direct usage for CI calculation might be embedded
        within `utils.estutils.TSEST` or `utils.inferutils.step2`.
        Default is 0.95.
    parallel : bool
        Whether to use parallel processing for computationally intensive parts
        like draws. Default is False.
    cores : Optional[int]
        Number of CPU cores to use for parallel processing. If None (default)
        and `parallel` is True, it may default to all available cores.
    scm_weights_args : Optional[Dict[str, Any]]
        Additional arguments that can be passed to the SCM weight optimization
        routine within `utils.estutils.TSEST`. Default is None.

    Methods
    -------
    fit()
        Fits the TSSC model, evaluates multiple SC variants, recommends one,
        and returns a list of results for each variant.
    """

    def __init__(self, config: TSSCConfig) -> None: # Changed to TSSCConfig
        """Initialize the TSSC estimator.

        Parameters
        ----------
        config : TSSCConfig
            Configuration object for the TSSC estimator. This object holds all
            necessary parameters for the estimator, inheriting common ones from
            `BaseEstimatorConfig` and adding TSSC-specific ones.
            Key attributes include:

            df : pd.DataFrame
                The input DataFrame containing panel data. Must include columns
                for outcome, treatment indicator, unit identifier, and time.
            outcome : str
                Name of the outcome variable column in `df`.
            treat : str
                Name of the treatment indicator column in `df`.
            unitid : str
                Name of the unit identifier (ID) column in `df`.
            time : str
                Name of the time variable column in `df`.
            draws : int, default 500
                Number of subsample replications or draws for inference procedures.
            ci : float, default 0.95
                Confidence interval level (e.g., 0.95 for 95% CI).
            parallel : bool, default False
                Whether to use parallel processing for draws.
            cores : Optional[int], default None
                Number of CPU cores for parallel processing. Defaults to all
                available if None and `parallel` is True.
            scm_weights_args : Optional[Dict[str, Any]], default None
                Additional arguments for SCM weight optimization.
            display_graphs : bool, default True
                If True, graphs of the results for the recommended model will be displayed.
            save : Union[bool, str, Dict[str, str]], default False
                Configuration for saving plots.
                If `False` (default), plots are not saved. If `True`, plots are saved with
                default names. If a `str`, it's used as the base filename. If a `Dict[str, str]`,
                it maps plot keys to full file paths.
            counterfactual_color : Union[str, List[str]], default "red"
                Default color or list of colors for counterfactual lines in plots.
            treated_color : str, default "black"
                Default color for the treated unit's line in plots.

            For authoritative definitions, defaults, and validation rules,
            refer to `mlsynth.config_models.TSSCConfig` and
            `mlsynth.config_models.BaseEstimatorConfig`.
        """
        if isinstance(config, dict):
            config =TSSCConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility if Base changes
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str] = config.save # Align with BaseEstimatorConfig
        self.draws: int = config.draws
        # The following are in TSSCConfig but not explicitly used by TSSC methods directly yet:
        # self.ci: float = config.ci # Available via self.config.ci
        # self.parallel: bool = config.parallel
        # self.cores: Optional[int] = config.cores
        # self.scm_weights_args: Optional[Dict[str, Any]] = config.scm_weights_args


    def _create_single_estimator_results(
        self,
        sc_method_variant_name: str,
        raw_sc_variant_output_dict: Dict[str, Any],
        prepared_panel_data: Dict[str, Any],
        is_recommended_variant: bool,
    ) -> BaseEstimatorResults:
        # sourcery: skip: extract-method
        """Map raw results for a single TSSC sub-method to BaseEstimatorResults.

        This internal helper takes the raw dictionary output for one of the
        SC methods (e.g., SIMPLEX, MSCa) from `TSEST` and transforms it into
        the standardized `BaseEstimatorResults` Pydantic model.

        Parameters
        ----------
        sc_method_variant_name : str
            The name of the SC sub-method (e.g., "SIMPLEX", "MSCa").
        raw_sc_variant_output_dict : Dict[str, Any]
            A dictionary containing the raw results for this specific sub-method
            as produced by `utils.estutils.TSEST`. Expected keys include
            constants like _KEY_EFFECTS, _KEY_FIT, _KEY_VECTORS, _KEY_WEIGHT_V,
            _KEY_DEFAULT_CI_FROM_TSEST, _KEY_P_VALUE_MANUAL.
        prepared_panel_data : Dict[str, Any]
            The dictionary output from `dataprep`, containing processed data like
            normalized outcomes (`y`), donor matrix, time labels, and donor names.
        is_recommended_variant : bool
            True if this `sc_method_variant_name` is the one recommended by the TSSC's
            second-step validation procedure.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model instance containing structured results for the
            specified SC sub-method. Key fields include:
            - effects (EffectsResults): Contains treatment effect estimates like ATT,
              percentage ATT, and an estimated standard error for ATT (approximated
              from the CI if available).
            - fit_diagnostics (FitDiagnosticsResults): Includes goodness-of-fit metrics
              such as pre-treatment RMSE, post-treatment RMSE (std of post-treatment gap),
              and pre-treatment R-squared.
            - time_series (TimeSeriesResults): Provides time-series data including the
              observed outcome for the treated unit, its estimated counterfactual outcome,
              the estimated treatment effect (gap) over time, and the corresponding
              time periods.
            - weights (WeightsResults): Contains an array of weights assigned to donor
              units and a list of the corresponding donor names.
            - inference (InferenceResults): Includes the p-value (if _KEY_P_VALUE_MANUAL
              is provided by `TSEST`) and the confidence interval (lower and upper bounds,
              typically from _KEY_DEFAULT_CI_FROM_TSEST).
            - method_details (MethodDetailsResults): Details about the specific SC
              sub-method, including its name (e.g., _SC_METHOD_SIMPLEX) and a boolean flag
              `is_recommended_variant` indicating if it was chosen by the second-step validation.
            - raw_results (Optional[Dict[str, Any]]): The raw dictionary output from
              `TSEST` for this specific sub-method.
        """
        # Extract raw data subsections from the main dictionary for clarity.
        raw_effects_data_for_variant = raw_sc_variant_output_dict.get(_KEY_EFFECTS, {})
        raw_fit_diagnostics_data_for_variant = raw_sc_variant_output_dict.get(_KEY_FIT, {})
        raw_time_series_vectors_for_variant = raw_sc_variant_output_dict.get(
            _KEY_VECTORS, {}
        )
        # Inference data like _KEY_DEFAULT_CI_FROM_TSEST is directly in raw_sc_variant_output_dict for TSEST output
        raw_inference_data_for_variant = (
            raw_sc_variant_output_dict  # For _KEY_DEFAULT_CI_FROM_TSEST, _KEY_P_VALUE_MANUAL etc.
        )

        # --- Effects Results ---
        average_treatment_effect_on_treated = raw_effects_data_for_variant.get(_KEY_ATT)
        # TSEST 'Effects' dictionary might contain a pre-calculated confidence interval for ATT.
        raw_att_confidence_interval = raw_effects_data_for_variant.get(_KEY_ATT_CI)
        calculated_att_standard_error: Optional[float] = None
        
        # If a CI for ATT is provided, approximate the standard error.
        # This assumes a symmetric CI based on a Z-distribution (common for large samples or known variance).
        # The formula used is: SE_approx = (Upper_Bound - Lower_Bound) / (2 * Z_score_for_CI_level).
        if (
            raw_att_confidence_interval
            and isinstance(raw_att_confidence_interval, list)
            and len(raw_att_confidence_interval) == 2
        ):
            # Ensure CI bounds are numeric before calculation.
            if isinstance(raw_att_confidence_interval[0], (int, float)) and \
               isinstance(raw_att_confidence_interval[1], (int, float)):
                # Calculate the Z-score corresponding to the configured confidence level (self.config.ci).
                # e.g., for 95% CI (self.config.ci=0.95), Z_score = norm.ppf((1+0.95)/2) = norm.ppf(0.975) approx 1.96.
                z_score = scipy.stats.norm.ppf((1 + self.config.ci) / 2)
                if z_score > 0: # Avoid division by zero if CI is 0% or invalid.
                    # SE approx = (Upper Bound - Lower Bound) / (2 * Z_score)
                    calculated_att_standard_error = (
                        raw_att_confidence_interval[1] - raw_att_confidence_interval[0]
                    ) / (2 * z_score)

        structured_effects_results = EffectsResults(
            att=average_treatment_effect_on_treated, # Average Treatment Effect on the Treated.
            att_percent=raw_effects_data_for_variant.get(_KEY_PERCENT_ATT), # If available
            att_std_err=calculated_att_standard_error, # Use the calculated SE
        )

        # --- Fit Diagnostics Results ---
        structured_fit_diagnostics_results = FitDiagnosticsResults(
            rmse_pre=raw_fit_diagnostics_data_for_variant.get(_KEY_T0_RMSE), # Pre-treatment Root Mean Squared Error.
            rmse_post=raw_fit_diagnostics_data_for_variant.get(_KEY_T1_RMSE), # Post-treatment RMSE (often std of post-treatment gap).
            r_squared_pre=raw_fit_diagnostics_data_for_variant.get(_KEY_R_SQUARED), # Pre-treatment R-squared.
        )

        # --- Time Series Results ---
        # Get the observed outcome series for the treated unit from the prepared_panel_data.
        treated_outcome_series_all_periods = prepared_panel_data.get("y")
        treated_outcome_array_all_periods: Optional[np.ndarray] = None
        if treated_outcome_series_all_periods is not None:
            # Ensure it's a flat NumPy array for consistency.
            treated_outcome_array_all_periods = (
                treated_outcome_series_all_periods.to_numpy().flatten()
                if isinstance(treated_outcome_series_all_periods, pd.Series)
                else np.array(treated_outcome_series_all_periods).flatten()
            )

        # Get the estimated counterfactual outcome series from the raw TSEST output.
        counterfactual_outcome_array_all_periods: Optional[np.ndarray] = None
        if raw_time_series_vectors_for_variant.get(_KEY_COUNTERFACTUAL) is not None:
            counterfactual_outcome_array_all_periods = np.array(
                raw_time_series_vectors_for_variant[_KEY_COUNTERFACTUAL]
            ).flatten()

        # Get or calculate the estimated gap (treatment effect) series.
        estimated_gap_array_all_periods: Optional[np.ndarray] = None
        if raw_time_series_vectors_for_variant.get(_KEY_GAP) is not None:  # TSEST provides 'Gap' directly.
            estimated_gap_array_all_periods = np.array(
                raw_time_series_vectors_for_variant[_KEY_GAP]
            ).flatten()
        elif ( # Fallback: calculate gap if observed and counterfactual are available and compatible.
               # This ensures 'estimated_gap' is populated even if not directly provided by TSEST.
            treated_outcome_array_all_periods is not None
            and counterfactual_outcome_array_all_periods is not None
            and len(treated_outcome_array_all_periods)
            == len(counterfactual_outcome_array_all_periods)
        ):
            estimated_gap_array_all_periods = (
                treated_outcome_array_all_periods
                - counterfactual_outcome_array_all_periods
            )

        # Get the time period labels (e.g., years) from prepared_panel_data.
        time_period_labels_from_prep = prepared_panel_data.get("time_labels")
        time_periods_array: Optional[np.ndarray] = None
        if time_period_labels_from_prep is not None:
            # Ensure it's a NumPy array for consistency in the results model.
            if isinstance(time_period_labels_from_prep, list):
                time_periods_array = np.array(time_period_labels_from_prep)
            elif isinstance(time_period_labels_from_prep, pd.Series):
                 time_periods_array = time_period_labels_from_prep.to_numpy()
            elif isinstance(time_period_labels_from_prep, np.ndarray):
                time_periods_array = time_period_labels_from_prep
            # else: time_periods_array remains None if the type is unexpected.

        try:
            structured_time_series_results = TimeSeriesResults(
                observed_outcome=treated_outcome_array_all_periods,
                counterfactual_outcome=counterfactual_outcome_array_all_periods,
                estimated_gap=estimated_gap_array_all_periods, 
                time_periods=time_periods_array,
            )

            # --- Weights Results ---
            # Extract donor weights (WeightV from TSEST) and match with donor names.
            donor_weights_values_array: Optional[np.ndarray] = raw_sc_variant_output_dict.get(_KEY_WEIGHT_V)
            donor_names_list: Optional[List[str]] = prepared_panel_data.get("donor_names")
            
            final_donor_weights_dict: Optional[Dict[str, float]] = None
            if donor_weights_values_array is not None and donor_names_list is not None:
                # Ensure donor names are strings for dictionary keys, as Pydantic models might expect this.
                string_donor_names_list = [str(name) for name in donor_names_list]
                # Check for consistency between the number of weights and names.
                # This is crucial for correctly mapping weights to their respective donors.
                if len(donor_weights_values_array.flatten()) == len(string_donor_names_list):
                    # Create a dictionary mapping donor names to their weights.
                    final_donor_weights_dict = dict(zip(string_donor_names_list, donor_weights_values_array.flatten()))
                else:
                    # If there's a mismatch, log a warning. The weights dictionary will remain None.
                    # This prevents errors from trying to zip lists of different lengths.
                    warnings.warn(
                        f"Warning: Mismatch between number of donor weights ({len(donor_weights_values_array.flatten())}) "
                        f"and names ({len(string_donor_names_list)}) for method {sc_method_variant_name}. "
                        "Donor weights will not be populated for this method.",
                        UserWarning
                    )

            structured_weights_results = WeightsResults(
                donor_weights=final_donor_weights_dict # Store the donor_name: weight dictionary.
                # summary_stats can be added if available/calculated
            )

            # --- Inference Results ---
            # Extract confidence interval and p-value if provided by TSEST.
            # TSEST might provide a default CI (e.g., "95% CI") directly in its output.
            raw_confidence_interval_from_tsest = raw_inference_data_for_variant.get(_KEY_DEFAULT_CI_FROM_TSEST)
            parsed_ci_lower: Optional[float] = None 
            parsed_ci_upper: Optional[float] = None 
            if (
                raw_confidence_interval_from_tsest
                and isinstance(raw_confidence_interval_from_tsest, list)
                and len(raw_confidence_interval_from_tsest) == 2
            ):
                # Ensure CI bounds are numeric.
                if isinstance(raw_confidence_interval_from_tsest[0], (int, float)) and \
                   isinstance(raw_confidence_interval_from_tsest[1], (int, float)):
                    parsed_ci_lower, parsed_ci_upper = (
                        raw_confidence_interval_from_tsest[0],
                        raw_confidence_interval_from_tsest[1],
                    )

            # TSEST might also provide a manually calculated p-value.
            p_value_for_att = raw_inference_data_for_variant.get(_KEY_P_VALUE_MANUAL)

            structured_inference_results = InferenceResults(
                p_value=p_value_for_att,
                ci_lower=parsed_ci_lower, 
                ci_upper=parsed_ci_upper, 
                # Store the confidence level used if a CI is successfully parsed.
                confidence_level=self.config.ci if (parsed_ci_lower is not None and parsed_ci_upper is not None) else None,
            )

            # --- Method Details Results ---
            structured_method_details_results = MethodDetailsResults(
                method_name=sc_method_variant_name, # Name of the specific SC variant (e.g., SIMPLEX).
                is_recommended=is_recommended_variant, # Flag if this variant was chosen by step2.
            )

            # --- Assemble Final BaseEstimatorResults ---
            return BaseEstimatorResults(
                effects=structured_effects_results,
                fit_diagnostics=structured_fit_diagnostics_results,
                time_series=structured_time_series_results,
                weights=structured_weights_results,
                inference=structured_inference_results,
                method_details=structured_method_details_results,
                raw_results=raw_sc_variant_output_dict, # Store the original raw output for this variant.
            )
        except pydantic.ValidationError as e: # Catch Pydantic validation errors during model creation.
            # This helps in debugging issues related to data types or missing fields in Pydantic models.
            raise MlsynthEstimationError(
                f"Error creating results model for {sc_method_variant_name} due to Pydantic validation: {e}"
            ) from e
        except Exception as e: # Catch any other unexpected error during model creation.
            raise MlsynthEstimationError(
                f"Unexpected error creating results model for {sc_method_variant_name}: {e}"
            ) from e

    def fit(self) -> List[BaseEstimatorResults]:
        """Fit the Two-Step Synthetic Control (TSSC) model.

        This method executes the TSSC estimation pipeline:
        1. Balances the input panel data using `utils.datautils.balance`.
        2. Prepares data using `utils.datautils.dataprep`, which separates
           the treated unit, donor units, and other relevant matrices.
        3. Calls `utils.estutils.TSEST` to obtain estimates from four
           underlying synthetic control methods: SIMPLEX, MSCa, MSCb, and MSCc.
           This step involves generating donor weights and counterfactuals for
           each method.
        4. Performs a second-step validation using `utils.inferutils.step2` to
           determine a "recommended" model among the four variants. This
           typically involves assessing pre-treatment fit or other criteria.
        5. If `display_graphs` is True in the configuration, plots the observed
           outcome versus the counterfactual from the recommended model.
        6. Transforms the raw results from each of the four SC methods into
           standardized `BaseEstimatorResults` objects using the internal
           `_create_single_estimator_results` helper.

        Returns
        -------
        List[BaseEstimatorResults]
            A list of `BaseEstimatorResults` objects. Each object in the list
            corresponds to one of the SC methods evaluated (SIMPLEX, MSCa, MSCb,
            MSCc). Each `BaseEstimatorResults` object, as detailed in
            `_create_single_estimator_results`, contains fields for effects, fit
            diagnostics, time series, weights, inference details, method details
            (including an `is_recommended_variant` flag), and raw results for that
            specific SC variant.

        Examples
        --------
        # doctest: +SKIP
        >>> import pandas as pd
        >>> from mlsynth.config_models import TSSCConfig
        >>> from mlsynth.estimators.tssc import TSSC
        >>> # Assume `data` is a pandas DataFrame with columns:
        >>> # 'unit_id', 'time_period', 'outcome_var', 'treatment_status'
        >>> data = pd.DataFrame({
        ...     'unit_id': [1]*10 + [2]*10 + [3]*10 + [4]*10,
        ...     'time_period': list(range(1,11))*4,
        ...     'outcome_var': ([1,2,3,4,5,6,7,8,9,10] +
        ...                     [1,2,3,4,4,3,2,1,1,1] +
        ...                     [2,3,4,5,6,5,4,3,2,2] +
        ...                     [0,1,2,3,4,5,6,7,8,8]),
        ...     'treatment_status': ([0]*5 + [1]*5) + [0]*30
        ... })
        >>> config = TSSCConfig(
        ...     df=data,
        ...     unitid='unit_id',
        ...     time='time_period',
        ...     outcome='outcome_var',
        ...     treat='treatment_status',
        ...     draws=100, # Fewer draws for quick example
        ...     display_graphs=False
        ... )
        >>> estimator = TSSC(config=config)
        >>> results_list = estimator.fit()
        >>> recommended_method_results = None
        >>> for res in results_list:
        ...     print(f"Method: {res.method_details.method_name}, Recommended: {res.method_details.is_recommended}")
        ...     if res.method_details.is_recommended:
        ...         recommended_method_results = res
        >>> if recommended_method_results:
        ...     print(f"Recommended ATT: {recommended_method_results.effects.att}")
        """
        try:
            # Ensure panel data is balanced (i.e., all units have observations for all time periods).
            # This is a common prerequisite for many panel data methods.
            balance(self.df, self.unitid, self.time)

            num_inference_draws: int = self.draws # Number of draws for inference procedures.
            # Prepare data into structured format (treated unit outcomes, donor matrix, period info, etc.).
            # `dataprep` handles the separation of treated/control units and pre/post-intervention periods.
            prepared_panel_data: Dict[str, Any] = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )

            # --- Prepare data for Step 2 Validation ---
            # The `step2` validation function in `inferutils` requires the donor matrix 
            # to include an intercept term for its internal regression-based calculations.
            # Defensive checks for keys from dataprep output.
            if "total_periods" not in prepared_panel_data or "donor_matrix" not in prepared_panel_data:
                raise MlsynthDataError("dataprep output missing 'total_periods' or 'donor_matrix', needed for step2 validation.")
            
            # Concatenate a column of ones (for intercept) to the donor matrix.
            # This augmented matrix is specifically for the `step2` validation.
            donor_matrix_with_intercept_for_validation: np.ndarray = np.concatenate(
                (
                    np.ones((prepared_panel_data["total_periods"], 1)), # Intercept column (all ones).
                    prepared_panel_data["donor_matrix"], # Original donor outcomes from dataprep.
                ),
                axis=1, # Concatenate column-wise.
            )

            # --- Step 1: Obtain results from all underlying SC methods using TSEST ---
            # `TSEST` (Two-Step Estimator) fits multiple SC variants (SIMPLEX, MSCa, MSCb, MSCc)
            # and returns their raw results. It handles the core SCM optimization for each variant.
            raw_results_all_sc_variants: List[Dict[str, Any]] = TSEST(
                prepared_panel_data["donor_matrix"], # Original donor matrix (without intercept for TSEST).
                prepared_panel_data["y"], # Outcome series for the treated unit.
                prepared_panel_data["pre_periods"], # Number of pre-treatment periods.
                num_inference_draws, # Number of draws for inference.
                prepared_panel_data["donor_names"], # List of donor names.
                prepared_panel_data["post_periods"], # Number of post-treatment periods.
            )

            # Number of donors plus the intercept term, used for defining restriction matrices for step2.
            num_donors_plus_intercept_for_validation: int = (
                donor_matrix_with_intercept_for_validation.shape[1]
            )

            # Extract weights for step2 validation
            # Helper to safely extract weights for a specific SC variant from the TSEST output.
            def extract_donor_weights_for_variant(
                sc_method_variant_name_to_extract: str,
            ) -> Optional[np.ndarray]:
                # Iterate through the list of raw results (one dict per SC variant).
                for sc_variant_result_dict in raw_results_all_sc_variants:
                    # Each dict has one key: the SC method name (e.g., "SIMPLEX").
                    if sc_method_variant_name_to_extract in sc_variant_result_dict:
                        # _KEY_WEIGHT_V corresponds to the donor weights array within that method's results.
                        return sc_variant_result_dict[
                            sc_method_variant_name_to_extract
                        ][_KEY_WEIGHT_V] 
                return None # Return None if the variant or its weights are not found.

            # Extract weights for the MSCc variant. These are specifically used as input
            # to the `step2` validation procedure.
            msc_c_variant_donor_weights: Optional[
                np.ndarray
            ] = extract_donor_weights_for_variant(_SC_METHOD_MSCC)
            # Other variants' weights (SIMPLEX, MSCa, MSCb) are not directly used as input to step2 in this setup.
            # b_sc_weights: Optional[np.ndarray] = extract_donor_weights_for_variant(_SC_METHOD_SIMPLEX)
            # b_msc_a_weights: Optional[np.ndarray] = extract_donor_weights_for_variant(_SC_METHOD_MSCA)
            # b_msc_b_weights: Optional[np.ndarray] = extract_donor_weights_for_variant(_SC_METHOD_MSCB)

            # --- Step 2: Perform validation to recommend a model ---
            # Ensure MSCc weights (b_MSC_c for step2) are available, as they are a key input for step2.
            if msc_c_variant_donor_weights is None:
                # If MSCc weights couldn't be extracted (e.g., MSCc method failed in TSEST),
                # issue a warning and default the recommended model to SIMPLEX.
                # This is a fallback to ensure a recommendation is always made.
                warnings.warn(_WARN_MSCC_WEIGHTS_NOT_FOUND, UserWarning)
                recommended_sc_variant_name: str = _SC_METHOD_SIMPLEX  # Default recommendation.
            else:
                # Prepare specific matrices (R1t, R2t, Rt) and scalars (q1t, q2t, qt) required by the `step2` function.
                # These define restrictions or hypotheses being tested in the second-step validation.
                # Their exact form depends on the specific validation procedure implemented in `step2`.
                # For example, they might relate to testing the stability of weights or pre-treatment fit.
                # R1t_val: Tests if sum of donor weights equals 1. First element (intercept) is 0, rest are 1.
                R1t_val = np.array( 
                    [[0] + list(np.ones(num_donors_plus_intercept_for_validation - 1))]
                )
                # R2t_val: Tests if intercept weight equals 0. First element is 1, rest are 0.
                R2t_val = np.array( 
                    [[1] + list(np.zeros(num_donors_plus_intercept_for_validation - 1))]
                )
                # Rt_val: Combines R1t and R2t for a joint test.
                Rt_val = np.array([ 
                    [0] + list(np.ones(num_donors_plus_intercept_for_validation - 1)),
                    [1] + list(np.zeros(num_donors_plus_intercept_for_validation - 1)),
                ])
                q1t_val = 1 # Target value for the sum of donor weights restriction (R1t * weights = q1t).
                q2t_val = 0 # Target value for the intercept weight restriction (R2t * weights = q2t).
                qt_val = np.array([[1], [0]]).flatten()  # Target vector for the combined restriction (Rt * weights = qt).

                # Call the `step2` validation function from inferutils.
                # This function compares the SC variants based on the provided restrictions and data,
                # and returns the name of the recommended one (e.g., "SIMPLEX", "MSCa").
                recommended_sc_variant_name: str = step2(
                    R1t_val, # Restriction matrix 1.
                    R2t_val, # Restriction matrix 2.
                    Rt_val,  # Combined restriction matrix.
                    msc_c_variant_donor_weights, # Weights from the MSCc model (used as a reference in step2).
                    q1t_val, # Scalar q1t.
                    q2t_val, # Scalar q2t.
                    qt_val,  # Vector qt.
                    prepared_panel_data["pre_periods"], # Number of pre-treatment periods.
                    donor_matrix_with_intercept_for_validation[ # Donor matrix (with intercept) for pre-treatment period.
                        : prepared_panel_data["pre_periods"], :
                    ],
                    prepared_panel_data["y"][: prepared_panel_data["pre_periods"]], # Treated unit outcome for pre-treatment period.
                    num_inference_draws, # Number of draws for inference in step2.
                    donor_matrix_with_intercept_for_validation.shape[1],  # n (number of coefficients, including intercept).
                    np.zeros( # Placeholder for bm_MSC_c (bootstrap/permutation matrix for MSCc weights), if needed by step2.
                              # This might be used if step2 involves resampling MSCc weights.
                        (num_donors_plus_intercept_for_validation, num_inference_draws)
                    ),
                )

            # Extract counterfactual for the recommended model for plotting
            # This is done after step2 determines the recommended_sc_variant_name.
            recommended_counterfactual_outcome_series: Optional[np.ndarray] = None
            for sc_variant_output_dict in raw_results_all_sc_variants:
                if recommended_sc_variant_name in sc_variant_output_dict:
                    recommended_counterfactual_outcome_series = sc_variant_output_dict[
                        recommended_sc_variant_name
                    ][_KEY_VECTORS][_KEY_COUNTERFACTUAL] # Use constants for robust key access.
                    break # Found the recommended model's counterfactual.

            # Convert raw results to list of Pydantic objects
            # This loop processes the raw output from TSEST for each SC variant (SIMPLEX, MSCa, etc.)
            # and transforms it into the standardized BaseEstimatorResults Pydantic model.
            structured_results_all_sc_variants: List[BaseEstimatorResults] = []
            for sc_variant_output_dict_loop in raw_results_all_sc_variants:
                # Each dict in raw_results_all_sc_variants has one key: the method name (e.g., "SIMPLEX").
                current_sc_variant_name = next(iter(sc_variant_output_dict_loop))
                current_raw_sc_variant_results = sc_variant_output_dict_loop[
                    current_sc_variant_name
                ]

                # Check if the current SC variant being processed is the one recommended by step2.
                is_current_variant_recommended = (
                    current_sc_variant_name == recommended_sc_variant_name
                )

                # Use the internal helper to map raw data to the Pydantic model.
                current_structured_sc_variant_result = (
                    self._create_single_estimator_results(
                        sc_method_variant_name=current_sc_variant_name,
                        raw_sc_variant_output_dict=current_raw_sc_variant_results,
                        prepared_panel_data=prepared_panel_data,
                        is_recommended_variant=is_current_variant_recommended,
                    )
                )
                structured_results_all_sc_variants.append(
                    current_structured_sc_variant_result
                )

        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError):
            # Re-raise specific Mlsynth errors if they occur during the process.
            # This allows for targeted error handling upstream.
            raise 
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            # Catch common data access or type-related errors that might occur if
            # `dataprep` or `TSEST` outputs are not as expected.
            raise MlsynthDataError(f"Data access or type error during TSSC estimation: {e}") from e
        except ValueError as e: # Catch generic ValueErrors that might not be Mlsynth specific.
            # These could arise from various numerical operations or function calls.
            raise MlsynthEstimationError(f"ValueError during TSSC estimation: {e}") from e
        except np.linalg.LinAlgError as e:
            # Catch errors from linear algebra operations (e.g., matrix inversion failures).
            raise MlsynthEstimationError(f"Linear algebra error during TSSC estimation: {e}") from e
        except Exception as e: # Catch-all for other unexpected errors.
            # This ensures that any unforeseen issues are still caught and reported.
            raise MlsynthEstimationError(f"Unexpected error during TSSC estimation: {e}") from e


        # Plotting logic using the Pydantic results list
        # This section handles the visualization of results if `display_graphs` is enabled.
        if self.display_graphs:
            try:
                # Find the structured results for the SC variant that was recommended by `step2`.
                recommended_structured_sc_variant_result: Optional[
                    BaseEstimatorResults
                ] = None
                for sc_variant_result_object in structured_results_all_sc_variants:
                    if (
                        sc_variant_result_object.method_details # Ensure method_details exists
                        and sc_variant_result_object.method_details.is_recommended # Check the flag
                    ):
                        recommended_structured_sc_variant_result = (
                            sc_variant_result_object
                        )
                        break # Found the recommended model's results.

                # Proceed with plotting if the recommended model's results and its counterfactual are available.
                if (
                    recommended_structured_sc_variant_result
                    and recommended_structured_sc_variant_result.time_series # Ensure time_series exists
                    and recommended_structured_sc_variant_result.time_series.counterfactual_outcome # Ensure counterfactual exists
                    is not None
                ):
                    counterfactual_series_for_plotting = (
                        recommended_structured_sc_variant_result.time_series.counterfactual_outcome
                    )
                    # The counterfactual series is typically stored as a 1D NumPy array (T,) in TimeSeriesResults.
                    # `plot_estimates` should be able to handle this format.

                    # Call the generic plotting utility to display the results.
                    # This utility plots the observed outcome of the treated unit against its estimated counterfactual.
                    plot_estimates(
                        processed_data_dict=prepared_panel_data, # Pass dataprep output for context if needed by plotter.
                        time_axis_label=self.time, # Name of the time column.
                        unit_identifier_column_name=self.unitid, # Name of the unit ID column.
                        outcome_variable_label=self.outcome, # Name of the outcome column.
                        treatment_name_label=self.treat, # Name of the treatment column (for plot title/label).
                        treated_unit_name=prepared_panel_data["treated_unit_name"], # Name of the treated unit.
                        observed_outcome_series=prepared_panel_data["y"],  # Observed outcome series (pd.Series from dataprep).
                        counterfactual_series_list=[counterfactual_series_for_plotting],  # List containing the counterfactual (np.ndarray).
                        counterfactual_names=[ # Label for the counterfactual line in the plot.
                            f"{recommended_sc_variant_name}{_PLOT_RECOMMENDED_LABEL_SUFFIX}"
                        ],
                        estimation_method_name=_PLOT_METHOD_NAME, # Overall method name for the plot (e.g., "TSSC").
                        treated_series_color=self.treated_color, # Color for the observed outcome line.
                        counterfactual_series_colors=( # Color(s) for the counterfactual line(s).
                            [self.counterfactual_color] # Use single color if string.
                            if isinstance(self.counterfactual_color, str)
                            else self.counterfactual_color # Use list of colors if provided.
                        ),
                        save_plot_config=self.save if isinstance(self.save, str) else None, # Path for saving the plot.
                    )
                else:
                    # If the counterfactual for the recommended model is not available, issue a warning.
                    # This might happen if the recommended model failed or produced no counterfactual.
                    warnings.warn(
                        _WARN_NO_COUNTERFACTUAL_FOR_PLOT.format(recommended_sc_variant_name),
                        UserWarning
                    )
            except (MlsynthPlottingError, MlsynthDataError) as e: # Catch known plotting or data-related errors.
                warnings.warn(f"Plotting failed for TSSC: {e}", UserWarning)
            except Exception as e: # Catch any other unexpected errors during the plotting phase.
                warnings.warn(f"Unexpected error during TSSC plotting: {e}", UserWarning)


        return structured_results_all_sc_variants
