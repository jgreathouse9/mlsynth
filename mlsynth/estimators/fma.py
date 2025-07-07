import numpy as np
import pandas as pd # Added for type hinting
from numpy.linalg import inv
from scipy.stats import norm
from typing import Dict, Any, List, Union, Optional, Tuple

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import effects, plot_estimates
from ..utils.denoiseutils import nbpiid, demean_matrix, standardize
from ..config_models import (
    FMAConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    InferenceResults,
    MethodDetailsResults
)
from ..exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, MlsynthPlottingError
from pydantic import ValidationError


class FMA:
    """
    Factor Model Approach (FMA) for estimating Average Treatment Effect on the Treated (ATT).

    This method uses a factor model to construct a counterfactual for the treated
    unit and provides statistical inference for the estimated treatment effects.
    It is based on the work by Li & Sonnier (2023).

    The estimator takes panel data and a configuration object specifying the
    outcome, treatment, unit, and time identifiers, along with method-specific
    parameters like the stationarity criterion and demeaning method. The `fit`
    method performs factor estimation, counterfactual construction, and
    statistical inference, returning a standardized results object.

    Attributes
    ----------
    config : FMAConfig
        The configuration object holding all parameters for the estimator.
    df : pd.DataFrame
        The input DataFrame containing panel data.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    outcome : str
        Name of the outcome variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    treat : str
        Name of the treatment indicator column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    unitid : str
        Name of the unit identifier column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    time : str
        Name of the time variable column in `df`.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    display_graphs : bool, default=True
        Whether to display graphs of results.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    save : Union[bool, str], default=False
        If False, plots are not saved. If True, plots are saved with default names.
        If a string, it's used as the prefix for saved plot filenames.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    counterfactual_color : str, default="red"
        Color for the counterfactual line in plots.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`).
    treated_color : str, default="black"
        Color for the treated unit line in plots.
        (Inherited from `BaseEstimatorConfig` via `FMAConfig`)
    criti : int, default 11
        Criterion for determining stationarity: 11 for nonstationary, 10 for stationary.
        (From `FMAConfig`)
    DEMEAN : int, default 1
        Data processing method: 1 for demean, 2 for standardize.
        (From `FMAConfig`)

    References
    ----------
    Li, K. T. & Sonnier, G. P. (2023). "Statistical Inference for the Factor Model
    Approach to Estimate Causal Effects in Quasi-Experimental Settings."
    *Journal of Marketing Research*, Volume 60, Issue 3.
    """

    def __init__(self, config: FMAConfig) -> None: # Changed to FMAConfig
        """
        Initializes the FMA estimator with a configuration object.

        Parameters
        ----------
        config : FMAConfig
            A Pydantic model instance containing all configuration parameters
            for the FMA estimator. This includes:
            - df (pd.DataFrame): The input DataFrame.
            - outcome (str): Name of the outcome variable column.
            - treat (str): Name of the treatment indicator column.
            - unitid (str): Name of the unit identifier column.
            - time (str): Name of the time variable column.
            - display_graphs (bool, optional): Whether to display graphs. Defaults to True.
            - save (Union[bool, str], optional): If False (default), plots are not saved.
              If True, plots are saved with default names. If a string, it's used as a
              prefix for saved plot filenames. (Note: The internal `plot_estimates`
              function might handle more complex types like Dict for `save`, but
              `FMAConfig` defines it as `Union[bool, str]`).
            - counterfactual_color (str, optional): Color for counterfactual line.
              Defaults to "red". (Note: The internal `plot_estimates` function might
              handle a list of colors, but `FMAConfig` defines it as `str`).
            - treated_color (str, optional): Color for treated unit line. Defaults to "black".
            - criti (int, optional): Criterion for stationarity (11 for nonstationary,
              10 for stationary). Defaults to 11.
            - DEMEAN (int, optional): Data processing method (1 for demean,
              2 for standardize). Defaults to 1.
        """
        if isinstance(config, dict):
            config =FMAConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save # Kept Union for flexibility, though FMAConfig.save is Union[bool, str]
        self.criti: int = config.criti
        self.DEMEAN: int = config.DEMEAN

    def _estimate_factors_and_loadings( # Helper method to encapsulate factor estimation logic
        self,
        control_outcomes_matrix: np.ndarray,
        treated_outcome_pre_treatment: np.ndarray,
        num_pre_treatment_periods: int,
        total_periods: int,
        criti: int,
        DEMEAN: int,
        max_factors_to_consider_cv: int,
        m_N_penalty_adjustment: int,
        m_T_penalty_adjustment: int,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates factors, loadings, and selects the number of factors.

        Helper method for the FMA.fit() process. Encapsulates the logic for: # Renamed from _get_factors_and_loadings
        1. Processing control outcomes (demeaning).
        2. Eigen decomposition to get initial unrestricted factors.
        3. Cross-validation to determine optimal number of factors.
        4. Applying Bai & Ng (2002) criteria if specified.
        5. Finalizing factor selection and estimating loadings.

        Parameters
        ----------
        control_outcomes_matrix : np.ndarray
            Matrix of control unit outcomes (T x N_co).
        treated_outcome_pre_treatment : np.ndarray
            Vector of treated unit outcomes in pre-treatment periods (T0 x 1).
        num_pre_treatment_periods : int
            Number of pre-treatment periods (T0).
        total_periods : int
            Total number of time periods (T).
        criti : int
            Criterion for determining stationarity (11 for nonstationary, 10 for stationary).
        DEMEAN : int
            Data processing method (1 for demean, 2 for standardize).
        max_factors_to_consider_cv : int
            Maximum number of factors to consider during cross-validation.
        m_N_penalty_adjustment : int
            Penalty adjustment based on number of control units.
        m_T_penalty_adjustment : int
            Penalty adjustment based on total time periods.

        Returns
        -------
        Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - final_num_factors_selected (int): The selected number of factors.
            - estimated_factors_unrestricted (np.ndarray): All estimated factors (T x max_eigen_components).
            - estimated_loadings_final (np.ndarray): Final estimated loadings ((k+1) x 1).
            - final_factors_with_intercept_pre_treatment (np.ndarray): Selected factors with intercept for pre-treatment (T0 x (k+1)).
            - final_factors_with_intercept_post_treatment (np.ndarray): Selected factors with intercept for post-treatment (T1 x (k+1)).
        """
        # Step 1: Process control outcomes (demean or standardize)
        if DEMEAN == 1:
            processed_control_outcomes_matrix: np.ndarray = demean_matrix(control_outcomes_matrix)
        elif DEMEAN == 2:
            processed_control_outcomes_matrix: np.ndarray = standardize(control_outcomes_matrix)
        else:
            # Default to demean if DEMEAN is not 1 or 2.
            # This case should ideally be caught by Pydantic validation in FMAConfig.
            processed_control_outcomes_matrix: np.ndarray = demean_matrix(control_outcomes_matrix)
        
        # Step 2: Eigen decomposition to get initial unrestricted factors
        # XX' (T x T) matrix from processed control outcomes
        processed_controls_outer_product_matrix: np.ndarray = np.dot(processed_control_outcomes_matrix, processed_control_outcomes_matrix.T) 
        eigenvalues, estimated_factors_unrestricted = np.linalg.eigh(processed_controls_outer_product_matrix)
        # Sort eigenvectors (factors) by eigenvalue in descending order
        estimated_factors_unrestricted = estimated_factors_unrestricted[:, ::-1] 
        
        # Prepare intercept vectors for pre and post treatment periods
        intercept_vector_pre_treatment: np.ndarray = np.ones((num_pre_treatment_periods, 1))
        num_post_treatment_periods = total_periods - num_pre_treatment_periods
        intercept_vector_post_treatment: np.ndarray = np.ones((num_post_treatment_periods, 1))
        
        # Step 3: Cross-validation to determine optimal number of factors
        cross_validation_mse_by_num_factors: np.ndarray = np.zeros(max_factors_to_consider_cv)

        # Iterate through possible numbers of factors for cross-validation
        for current_num_factors_cv in range(1, max_factors_to_consider_cv + 1):
            # Ensure we don't try to select more factors than available from eigen decomposition
            if current_num_factors_cv > estimated_factors_unrestricted.shape[1]:
                cross_validation_mse_by_num_factors[current_num_factors_cv - 1] = np.inf # Assign infinite MSE if not enough factors
                continue

            # Construct factors matrix with intercept for the current number of CV factors
            factors_with_intercept_pre_treatment_cv: np.ndarray = np.hstack((intercept_vector_pre_treatment, estimated_factors_unrestricted[:num_pre_treatment_periods, :current_num_factors_cv]))
            leave_one_out_residuals_cv: np.ndarray = np.zeros(num_pre_treatment_periods) 
            
            # Perform leave-one-out cross-validation
            for leave_one_out_index_cv in range(num_pre_treatment_periods): 
                # Create LOO datasets by removing one pre-treatment observation
                factors_with_intercept_pre_treatment_loo_cv: np.ndarray = np.delete(factors_with_intercept_pre_treatment_cv, leave_one_out_index_cv, axis=0)
                treated_outcome_pre_treatment_loo_cv: np.ndarray = np.delete(treated_outcome_pre_treatment, leave_one_out_index_cv)
                
                try:
                    # Estimate loadings on the LOO dataset
                    loadings_matrix_inv_cv = inv(factors_with_intercept_pre_treatment_loo_cv.T @ factors_with_intercept_pre_treatment_loo_cv)
                    estimated_loadings_loo_cv: np.ndarray = loadings_matrix_inv_cv @ factors_with_intercept_pre_treatment_loo_cv.T @ treated_outcome_pre_treatment_loo_cv
                    # Calculate residual for the left-out observation
                    leave_one_out_residuals_cv[leave_one_out_index_cv] = treated_outcome_pre_treatment[leave_one_out_index_cv] - factors_with_intercept_pre_treatment_cv[leave_one_out_index_cv, :] @ estimated_loadings_loo_cv
                except np.linalg.LinAlgError: # Handle cases where matrix inversion fails
                    leave_one_out_residuals_cv[leave_one_out_index_cv] = np.inf # Assign infinite residual

            # Calculate Mean Squared Error for the current number of factors
            cross_validation_mse_by_num_factors[current_num_factors_cv - 1] = np.mean(leave_one_out_residuals_cv**2)
        
        # Select number of factors that minimizes MSE in CV
        num_factors_selected_by_cv: int = np.argmin(cross_validation_mse_by_num_factors) + 1
        
        # Step 4: Apply Bai & Ng (2002) criteria if specified by `criti`
        final_num_factors_selected: int
        if criti == 11: # Nonstationary criterion
            num_factors_selected_by_bai_ng_nonstationary, _, _ = nbpiid(processed_control_outcomes_matrix, max_factors_to_consider_cv, criti, DEMEAN, m_N_penalty_adjustment, m_T_penalty_adjustment)
            final_num_factors_selected = min(num_factors_selected_by_bai_ng_nonstationary, num_factors_selected_by_cv)
        elif criti == 10: # Stationary criterion
            num_factors_selected_by_bai_ng_stationary, _, _ = nbpiid(processed_control_outcomes_matrix, max_factors_to_consider_cv, criti, DEMEAN, m_N_penalty_adjustment, m_T_penalty_adjustment)
            final_num_factors_selected = min(num_factors_selected_by_bai_ng_stationary, num_factors_selected_by_cv)
        else: # If `criti` is not 10 or 11, use only CV result
            final_num_factors_selected = num_factors_selected_by_cv
        
        # Ensure the selected number of factors does not exceed available factors
        final_num_factors_selected = min(final_num_factors_selected, estimated_factors_unrestricted.shape[1])

        # Step 5: Finalize factor selection and estimate loadings
        # Construct final factor matrices (with intercept) for pre and post-treatment periods
        final_factors_with_intercept_pre_treatment: np.ndarray = np.hstack((intercept_vector_pre_treatment, estimated_factors_unrestricted[:num_pre_treatment_periods, :final_num_factors_selected]))
        final_factors_with_intercept_post_treatment: np.ndarray = np.hstack((intercept_vector_post_treatment, estimated_factors_unrestricted[num_pre_treatment_periods:total_periods, :final_num_factors_selected]))

        # Estimate final loadings using all pre-treatment data and selected factors
        try:
            loadings_matrix_inv_final = inv(final_factors_with_intercept_pre_treatment.T @ final_factors_with_intercept_pre_treatment)
            estimated_loadings_final: np.ndarray = loadings_matrix_inv_final @ final_factors_with_intercept_pre_treatment.T @ treated_outcome_pre_treatment
        except np.linalg.LinAlgError as e: # Handle potential singularity
            raise MlsynthEstimationError("Singular matrix encountered in final loading estimation. Check data or factor selection.") from e

        return (
            final_num_factors_selected,
            estimated_factors_unrestricted,
            estimated_loadings_final,
            final_factors_with_intercept_pre_treatment,
            final_factors_with_intercept_post_treatment,
        )

    def _calculate_inference(
        self,
        residuals_pre_treatment_fma: np.ndarray,
        final_factors_with_intercept_pre_treatment: np.ndarray,
        final_factors_with_intercept_post_treatment: np.ndarray,
        num_pre_treatment_periods: int,
        num_post_treatment_periods: int,
        att_value_fma: float,
    ) -> Tuple[float, float, float, Tuple[float, float]]:
        """
        Calculates statistical inference for the ATT.

        Helper method for the FMA.fit() process. Encapsulates the logic for: # Renamed from _get_inference
        1. Calculating variance of residuals.
        2. Estimating variance components for ATT.
        3. Calculating standard error, t-statistic, p-value, and confidence interval.

        Parameters
        ----------
        residuals_pre_treatment_fma : np.ndarray
            Residuals from the pre-treatment fit (T0 x 1).
        final_factors_with_intercept_pre_treatment : np.ndarray
            Selected factors with intercept for pre-treatment (T0 x (k+1)).
        final_factors_with_intercept_post_treatment : np.ndarray
            Selected factors with intercept for post-treatment (T1 x (k+1)).
        num_pre_treatment_periods : int
            Number of pre-treatment periods (T0).
        num_post_treatment_periods : int
            Number of post-treatment periods (T1).
        att_value_fma : float
            The estimated Average Treatment Effect on the Treated.

        Returns
        -------
        Tuple[float, float, float, Tuple[float, float]]
            A tuple containing:
            - standard_error_att_fma (float): Standard error of the ATT.
            - t_statistic_fma (float): t-statistic for the ATT.
            - p_value_fma (float): p-value for the ATT.
            - final_confidence_interval_tuple_fma (Tuple[float, float]): 95% CI (lower, upper).
        """
        # Step 1: Calculate variance of residuals from pre-treatment fit
        variance_residuals_pre_treatment_fma: float = np.mean(residuals_pre_treatment_fma**2)
        
        # Calculate mean of factors (with intercept) in the post-treatment period
        mean_factors_post_treatment_with_intercept: np.ndarray = np.mean(final_factors_with_intercept_post_treatment, axis=0)[:, np.newaxis]
        
        # Squared residuals for variance calculation
        squared_residuals_pre_treatment_fma: np.ndarray = residuals_pre_treatment_fma**2
        
        # Handle case with no pre-treatment periods to avoid division by zero
        if num_pre_treatment_periods == 0: 
            standard_error_att_fma = np.nan
            t_statistic_fma = np.nan
            p_value_fma = np.nan
            final_confidence_interval_tuple_fma = (np.nan, np.nan)
            return standard_error_att_fma, t_statistic_fma, p_value_fma, final_confidence_interval_tuple_fma

        # Step 2: Estimate variance components for ATT
        # Calculate E(F'F)/T0 for pre-treatment factors
        expected_factors_outer_product_pre_treatment: np.ndarray = (final_factors_with_intercept_pre_treatment.T @ final_factors_with_intercept_pre_treatment) / num_pre_treatment_periods
        
        try:
            # Calculate Psi_hat matrix, a key component for variance estimation
            inv_expected_factors_outer_product = inv(expected_factors_outer_product_pre_treatment)
            psi_hat_matrix_for_variance: np.ndarray = inv_expected_factors_outer_product @ (final_factors_with_intercept_pre_treatment.T @ np.diag(squared_residuals_pre_treatment_fma) @ final_factors_with_intercept_pre_treatment / num_pre_treatment_periods) @ inv_expected_factors_outer_product
        except np.linalg.LinAlgError: # Handle potential singularity
            standard_error_att_fma = np.nan
            t_statistic_fma = np.nan
            p_value_fma = np.nan
            final_confidence_interval_tuple_fma = (np.nan, np.nan)
            return standard_error_att_fma, t_statistic_fma, p_value_fma, final_confidence_interval_tuple_fma

        # Calculate Omega1_hat and Omega2_hat variance components
        omega1_hat_variance_component: float = (num_post_treatment_periods / num_pre_treatment_periods) * (mean_factors_post_treatment_with_intercept.T @ psi_hat_matrix_for_variance @ mean_factors_post_treatment_with_intercept).item()
        omega2_hat_variance_component: float = variance_residuals_pre_treatment_fma
        
        # Total variance Omega_hat
        total_variance_omega_hat: float = omega1_hat_variance_component + omega2_hat_variance_component
        
        # Step 3: Calculate standard error, t-statistic, p-value, and confidence interval
        # Standard error of the ATT
        standard_error_att_fma: float = np.sqrt(total_variance_omega_hat) / np.sqrt(num_post_treatment_periods) if num_post_treatment_periods > 0 else np.nan

        # Helper function to calculate confidence interval bounds
        def calculate_ci_bound(percentile: float, att_val: float, std_err_val: float) -> float:
            if np.isnan(att_val) or np.isnan(std_err_val): # Handle NaN inputs
                return np.nan
            return att_val - norm.ppf(percentile) * std_err_val # Uses normal distribution critical value

        # Calculate 95% confidence interval bounds
        ci_95_lower_bound_fma = calculate_ci_bound(0.025, att_value_fma, standard_error_att_fma) # Lower bound (2.5th percentile)
        ci_95_upper_bound_fma = calculate_ci_bound(0.975, att_value_fma, standard_error_att_fma) # Upper bound (97.5th percentile)
        
        final_confidence_interval_tuple_fma: Tuple[float, float] = (ci_95_lower_bound_fma, ci_95_upper_bound_fma)
        
        # Calculate t-statistic for the ATT
        t_statistic_fma: float = att_value_fma / standard_error_att_fma if standard_error_att_fma != 0 and not np.isnan(standard_error_att_fma) else np.nan
        
        # Calculate p-value (two-tailed) for the ATT
        p_value_fma: float = 2 * (1 - norm.cdf(abs(t_statistic_fma))) if not np.isnan(t_statistic_fma) else np.nan
        
        return standard_error_att_fma, t_statistic_fma, p_value_fma, final_confidence_interval_tuple_fma

    def fit(self) -> BaseEstimatorResults:
        """
        Fits the Factor Model Approach model to the provided data.

        This method performs data preparation, factor estimation using principal
        components, construction of the counterfactual for the treated unit,
        estimation of the Average Treatment Effect on the Treated (ATT), and
        statistical inference for the ATT.

        Returns
        -------
        BaseEstimatorResults
            An object containing the standardized estimation results. Key fields include:
            - `effects` (EffectsResults): Contains `att` (Average Treatment Effect
              on the Treated), `att_percent` (Percentage ATT), and
              `additional_effects` for other method-specific effect measures.
            - `fit_diagnostics` (FitDiagnosticsResults): Contains `pre_treatment_rmse`,
              `pre_treatment_r_squared`, and `additional_metrics` for other
              goodness-of-fit statistics.
            - `time_series` (TimeSeriesResults): Contains `observed_outcome`
              (for the treated unit), `counterfactual_outcome`, `estimated_gap`
              (effect over time), and `time_periods` (actual time values or
              event time indices).
            - `inference` (InferenceResults): Contains `p_value`, `ci_lower_bound`,
              `ci_upper_bound` (for a 95% confidence interval by default),
              `standard_error` of the ATT, `confidence_level` (e.g., 0.95),
              `method` (e.g., "asymptotic"), and `details` (e.g., t-statistic).
            - `method_details` (MethodDetailsResults): Contains the method `name` ("FMA")
              and `parameters_used` (the estimator config).
            - `weights` (WeightsResults): Typically `None` for FMA as it does not
              produce explicit donor weights in the same way as SCM.
            - `raw_results`: Not populated by this implementation.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from mlsynth.estimators.fma import FMA
        >>> from mlsynth.config_models import FMAConfig
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'unit': np.tile(np.arange(1, 6), 20), # 5 units, 20 time periods each
        ...     'time_period': np.repeat(np.arange(1, 21), 5),
        ...     'outcome_var': np.random.rand(100) + \
        ...                    np.tile(np.arange(1, 6), 20)*0.2 + \
        ...                    np.repeat(np.arange(1,21),5)*0.1,
        ...     'treatment_status': ((np.tile(np.arange(1, 6), 20) == 1) & \
        ...                          (np.repeat(np.arange(1, 21), 5) >= 15)).astype(int)
        ... }) # Unit 1 treated from period 15 onwards
        >>> fma_config = FMAConfig(
        ...     df=data,
        ...     outcome="outcome_var",
        ...     treat="treatment_status",
        ...     unitid="unit",
        ...     time="time_period",
        ...     display_graphs=False, # Disable graph display for example
        ...     criti=11, # Example: nonstationary assumption
        ...     DEMEAN=1  # Example: demeaning
        ... )
        >>> fma_estimator = FMA(config=fma_config)
        >>> results = fma_estimator.fit() # doctest: +SKIP
        >>> # Example: Accessing results (actual values will vary due to random data)
        >>> print(f"Estimated ATT: {results.effects.att}") # doctest: +SKIP
        >>> if results.inference and results.inference.p_value is not None: # doctest: +SKIP
        ...     print(f"P-value: {results.inference.p_value:.4f}") # doctest: +SKIP
        """
        # Initialize variables to store intermediate and final results
        prepared_data: Dict[str, Any] = {} # Stores output from dataprep utility
        full_counterfactual_outcome_fma: Optional[np.ndarray] = None # Stores the full counterfactual series
        treated_outcome_all_periods: Optional[np.ndarray] = None # Stores the observed outcome for the treated unit
        results_to_return: Optional[BaseEstimatorResults] = None # Stores the final Pydantic results object

        try:
            # Step 1: Validate data balance (ensures each unit has same time periods)
            balance(self.df, self.unitid, self.time)

            # Step 2: Prepare data using the dataprep utility
            # This separates data into treated/control, pre/post periods, etc.
            prepared_data = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treat
            )
            
            # Extract key dimensions and data matrices from prepared_data
            total_periods: int = prepared_data["total_periods"]
            num_pre_treatment_periods: int = prepared_data["pre_periods"]
            num_post_treatment_periods: int = prepared_data["post_periods"]
            
            control_outcomes_matrix: np.ndarray = prepared_data["donor_matrix"] # Outcomes of control units (T x N_co)
            treated_outcome_pre_treatment: np.ndarray = prepared_data["y"][:num_pre_treatment_periods] # Outcome of treated unit in pre-period (T0 x 1)
            treated_outcome_all_periods = prepared_data["y"] # Full outcome series for treated unit (T x 1)

            if np.isnan(control_outcomes_matrix).any() or np.isnan(treated_outcome_all_periods).any():
                raise MlsynthEstimationError(
                    "Outcome matrix contains missing values. "
                    "FMA does not currently support missing data. Please impute or drop missing values."
                )
            
            # Determine penalty adjustments for Bai & Ng factor selection criteria
            num_control_units: int = control_outcomes_matrix.shape[1]
            sample_size_cutoff_for_m: int = 70 # Threshold for penalty adjustment
            
            m_N_penalty_adjustment: int = max(0, sample_size_cutoff_for_m - num_control_units) # Penalty based on number of control units
            m_T_penalty_adjustment: int = max(0, sample_size_cutoff_for_m - total_periods) # Penalty based on total time periods
            
            max_factors_to_consider_cv: int = 10 # Maximum number of factors for cross-validation

            # Step 3: Estimate factors, loadings, and select the number of factors
            # This is done by calling the helper method `_estimate_factors_and_loadings`
            (
                final_num_factors_selected, # k, the selected number of factors
                estimated_factors_unrestricted, # F_hat, all estimated factors (T x max_eigen_components)
                estimated_loadings_final, # Lambda_hat, final estimated loadings ((k+1) x 1)
                final_factors_with_intercept_pre_treatment, # F_hat_pre_intercept, selected factors with intercept for pre-treatment (T0 x (k+1))
                final_factors_with_intercept_post_treatment, # F_hat_post_intercept, selected factors with intercept for post-treatment (T1 x (k+1))
            ) = self._estimate_factors_and_loadings(
                control_outcomes_matrix=control_outcomes_matrix,
                treated_outcome_pre_treatment=treated_outcome_pre_treatment,
                num_pre_treatment_periods=num_pre_treatment_periods,
                total_periods=total_periods,
                criti=self.criti, # Stationarity criterion from config
                DEMEAN=self.DEMEAN, # Demeaning/standardization method from config
                max_factors_to_consider_cv=max_factors_to_consider_cv,
                m_N_penalty_adjustment=m_N_penalty_adjustment,
                m_T_penalty_adjustment=m_T_penalty_adjustment,
            )
            
            # Step 4: Construct counterfactual outcomes
            # Fitted values for treated unit in pre-treatment period
            fitted_treated_outcome_pre_treatment: np.ndarray = final_factors_with_intercept_pre_treatment @ estimated_loadings_final
            # Predicted counterfactual for treated unit in post-treatment period
            predicted_counterfactual_post_treatment: np.ndarray = final_factors_with_intercept_post_treatment @ estimated_loadings_final
            # Combine pre-treatment fitted values and post-treatment predictions for the full counterfactual series
            full_counterfactual_outcome_fma = np.concatenate((fitted_treated_outcome_pre_treatment, predicted_counterfactual_post_treatment))
            
            # Step 5: Calculate treatment effects and fit diagnostics
            # Uses the `effects.calculate` utility
            calculated_effects_fma, calculated_fit_diagnostics_fma, calculated_time_series_components_fma = effects.calculate(
                treated_outcome_all_periods, # Observed outcome
                full_counterfactual_outcome_fma, # Synthetic counterfactual
                num_pre_treatment_periods,
                num_post_treatment_periods, 
            )
            # Calculate residuals from the pre-treatment fit
            residuals_pre_treatment_fma: np.ndarray = treated_outcome_pre_treatment - fitted_treated_outcome_pre_treatment
            # Extract the estimated ATT
            att_value_fma: float = calculated_effects_fma["ATT"]

            # Step 6: Perform statistical inference for the ATT
            # This is done by calling the helper method `_calculate_inference`
            (
                standard_error_att_fma, # Standard error of ATT
                t_statistic_fma, # t-statistic for ATT
                p_value_fma, # p-value for ATT
                final_confidence_interval_tuple_fma, # 95% CI for ATT
            ) = self._calculate_inference(
                residuals_pre_treatment_fma=residuals_pre_treatment_fma,
                final_factors_with_intercept_pre_treatment=final_factors_with_intercept_pre_treatment,
                final_factors_with_intercept_post_treatment=final_factors_with_intercept_post_treatment,
                num_pre_treatment_periods=num_pre_treatment_periods,
                num_post_treatment_periods=num_post_treatment_periods,
                att_value_fma=att_value_fma,
            )

            # Step 7: Package results into Pydantic models
            # Store inference results in a dictionary for easier access
            inference_results_map_fma: Dict[str, Any] = {
                "SE": standard_error_att_fma,
                "t_stat": t_statistic_fma,
                "95% CI": final_confidence_interval_tuple_fma,
                "p_value": p_value_fma,
            }
            
            # Create EffectsResults object
            effects_results_obj = EffectsResults(
                att=calculated_effects_fma.get("ATT"),
                att_percent=calculated_effects_fma.get("Percent ATT"),
                additional_effects={k: v for k, v in calculated_effects_fma.items() if k not in ["ATT", "Percent ATT"]}
            )

            # Create FitDiagnosticsResults object
            fit_diagnostics_results_obj = FitDiagnosticsResults(
                pre_treatment_rmse=calculated_fit_diagnostics_fma.get("T0 RMSE"),
                pre_treatment_r_squared=calculated_fit_diagnostics_fma.get("R-Squared"),
                additional_metrics={k: v for k, v in calculated_fit_diagnostics_fma.items() if k not in ["T0 RMSE", "R-Squared"]}
            )
            
            # Prepare time periods array for TimeSeriesResults
            ywide_df = prepared_data.get("Ywide") # Get wide format data from dataprep
            time_periods_array_for_results: Optional[np.ndarray] = None
            if ywide_df is not None and isinstance(ywide_df, pd.DataFrame):
                time_periods_array_for_results = ywide_df.index.to_numpy() # Use index as time periods
            
            # Ensure time_periods_array_for_results is a NumPy array if not None
            if time_periods_array_for_results is not None and not isinstance(time_periods_array_for_results, np.ndarray):
                time_periods_array_for_results = np.array(time_periods_array_for_results)

            # Create TimeSeriesResults object
            time_series_results_obj = TimeSeriesResults(
                observed_outcome=calculated_time_series_components_fma.get("Observed Unit"),
                counterfactual_outcome=calculated_time_series_components_fma.get("Counterfactual"),
                estimated_gap=calculated_time_series_components_fma.get("Gap"),
                time_periods=time_periods_array_for_results
            )

            # Create InferenceResults object
            raw_confidence_interval_tuple = inference_results_map_fma.get("95% CI")
            inference_results_obj = InferenceResults(
                p_value=inference_results_map_fma.get("p_value"),
                ci_lower_bound=raw_confidence_interval_tuple[0] if raw_confidence_interval_tuple and len(raw_confidence_interval_tuple) == 2 else None,
                ci_upper_bound=raw_confidence_interval_tuple[1] if raw_confidence_interval_tuple and len(raw_confidence_interval_tuple) == 2 else None,
                standard_error=inference_results_map_fma.get("SE"),
                confidence_level=0.95 if raw_confidence_interval_tuple else None, # Default 95% CI
                method="asymptotic", # Inference method used
                details={"t_statistic": inference_results_map_fma.get("t_stat")} # Additional details like t-statistic
            )
            
            # Create MethodDetailsResults object
            method_details_results_obj = MethodDetailsResults(name="FMA") # Store method name

            # Assemble the final BaseEstimatorResults object
            results_to_return = BaseEstimatorResults(
                effects=effects_results_obj,
                fit_diagnostics=fit_diagnostics_results_obj,
                time_series=time_series_results_obj,
                inference=inference_results_obj,
                method_details=method_details_results_obj
                # Note: `weights` field is typically None for FMA as it doesn't produce explicit donor weights like SCM.
                # `raw_results` is not populated by this implementation.
            )

        # Step 8: Handle exceptions
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e: # Catch specific custom exceptions
            raise e # Re-raise them
        except ValidationError as e: # Catch Pydantic validation errors during results creation
            raise MlsynthEstimationError(f"Error creating results structure for FMA: {str(e)}") from e
        except Exception as e: # Catch any other unexpected errors
            raise MlsynthEstimationError(f"An unexpected error occurred during FMA estimation: {str(e)}") from e

        # Step 9: Display graphs if requested
        if self.display_graphs and prepared_data and full_counterfactual_outcome_fma is not None and treated_outcome_all_periods is not None:
            try:
                # Call the plot_estimates utility
                plot_estimates(
                    processed_data_dict=prepared_data, # Data prepared by dataprep
                    time_axis_label=self.time, # Time column name
                    unit_identifier_column_name=self.unitid, # Unit ID column name
                    outcome_variable_label=self.outcome, # Outcome column name
                    treatment_name_label=self.treat, # Treatment indicator column name
                    treated_unit_name=prepared_data["treated_unit_name"], # Name of the treated unit
                    observed_outcome_series=treated_outcome_all_periods, # Observed outcome series
                    counterfactual_series_list=[full_counterfactual_outcome_fma], # List of counterfactual series (only one for FMA)
                    estimation_method_name="FMA", # Method name for plot title
                    treated_series_color=self.treated_color, # Color for treated line
                    counterfactual_series_colors=( # Color(s) for counterfactual line(s)
                        [self.counterfactual_color]
                        if isinstance(self.counterfactual_color, str)
                        else self.counterfactual_color
                    ), 
                    counterfactual_names=[f"FMA {prepared_data['treated_unit_name']}"], # Names for counterfactual series
                    save_plot_config=self.save, # Save option from config
                )
            except (MlsynthPlottingError, MlsynthDataError) as e: # Catch specific plotting or data errors
                print(f"Warning: Plotting failed for FMA - {str(e)}") # Print a warning
            except Exception as e: # Catch any other unexpected plotting errors
                print(f"Warning: An unexpected error occurred during FMA plotting - {str(e)}") # Print a warning
        
        # Ensure results_to_return is not None before returning
        if results_to_return is None:
            # This case should ideally not be reached if exceptions are properly raised.
            # As a fallback, raise an error indicating failure to produce results.
            raise MlsynthEstimationError("FMA estimation failed to produce results due to an unhandled state.")
            
        return results_to_return
