import numpy as np
import pandas as pd # Added for type hinting
from scipy.stats import norm
from typing import Dict, Any, List, Union, Optional, Tuple 

from ..utils.datautils import balance, dataprep
from ..utils.resultutils import plot_estimates
from ..config_models import ( # Import the Pydantic models
    FDIDConfig,
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    TimeSeriesResults,
    WeightsResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError, MlsynthPlottingError
from pydantic import ValidationError


class FDID:
    """
    Implements the Forward Difference-in-Differences (FDID) estimator.

    This class computes estimates for three related methods:
    1.  Forward Difference-in-Differences (FDID): Uses a forward selection
        procedure to choose a subset of control units to construct the
        counterfactual for the treated unit.
    2.  Standard Difference-in-Differences (DID): Uses all available control
        units.
    3.  Augmented Difference-in-Differences (ADID): Also uses all available
        control units but incorporates them differently in the regression.

    The `fit` method returns results for all three estimators. Configuration is
    managed via a Pydantic model. Key instance attributes are derived from this
    configuration object. Refer to the `__init__` method for detailed parameter
    descriptions.

    References
    ----------
    Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences Method."
    *Marketing Science* 43(2):267-279.

    Li, K. T. & Van den Bulte, C. (2023). "Augmented Difference-in-Differences."
    *Marketing Science* 42:4, 746-767.

    Examples
    --------
    >>> from mlsynth import FDID
    >>> from mlsynth.config_models import FDIDConfig
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
    >>> fdid_config = FDIDConfig(
    ...     df=data,
    ...     outcome='outcome',
    ...     treat='treated_unit_1',
    ...     unitid='unit',
    ...     time='time',
    ...     display_graphs=False # Typically True, False for non-interactive examples
    ... )
    >>> estimator = FDID(config=fdid_config)
    >>> # Results can be obtained by calling estimator.fit()
    >>> # results_list = estimator.fit() # doctest: +SKIP
    """

    def __init__(self, config: FDIDConfig) -> None:
        """
        Initializes the FDID estimator with a configuration object.

        The configuration object `config` bundles all parameters for the estimator.
        `FDIDConfig` inherits from `BaseEstimatorConfig`.

        Parameters
        ----------
        config : FDIDConfig
            A Pydantic model instance containing the configuration parameters.
            The fields are inherited from `BaseEstimatorConfig`:
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
                (Note: The internal `plot_estimates` function might handle more complex
                types like Dict for `save`, but `FDIDConfig` defines it as `Union[bool, str]`).
            counterfactual_color : str, default="red"
                Color for the counterfactual line(s) in plots.
                (Note: The internal `plot_estimates` function might handle a list of colors,
                but `FDIDConfig` defines it as `str`).
            treated_color : str, default="black"
                Color for the treated unit line in plots.
        """
        if isinstance(config, dict):
            config = FDIDConfig(**config)  # convert dict to config object
        self.config = config # Store the config object
        self.df: pd.DataFrame = config.df
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.outcome: str = config.outcome
        self.treated: str = config.treat # Maps to 'treat' from config
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color # Kept Union for flexibility
        self.treated_color: str = config.treated_color
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, Dict[str, str]] = config.save # Kept Union for flexibility

    def DID(
        self, treated_outcome_all_periods: np.ndarray, control_outcomes_all_periods: np.ndarray, num_pre_treatment_periods: int, placebo_iteration_num: int = 0
    ) -> Dict[str, Any]:
        """
        Computes standard Difference-in-Differences (DID) estimates.

        This method calculates the DID effect by comparing the change in outcomes
        for the treated unit to the change in outcomes for the (average of) control units.

        Parameters
        ----------
        treated_outcome_all_periods : np.ndarray
            Outcome vector for the treated unit, shape (T, 1), where T is the total
            number of time periods.
        control_outcomes_all_periods : np.ndarray
            Outcome matrix for control units, shape (T, N_controls), where N_controls
            is the number of control units.
        num_pre_treatment_periods : int
            Number of pre-treatment periods (T0).
        placebo_iteration_num : int, optional
            Placeholder for placebo iteration, not actively used in this method.
            Default is 0.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing raw estimation results for the DID method.
            The dictionary includes the following keys and sub-keys:
            - "Effects" (Dict[str, float]):
                - "ATT": Average Treatment Effect on the Treated.
                - "Percent ATT": Percentage ATT.
                - "SATT": Standardized ATT.
            - "Vectors" (Dict[str, np.ndarray]):
                - "Observed Unit": Outcome vector for the treated unit, shape (T, 1).
                - "Counterfactual": Estimated counterfactual outcome vector, shape (T, 1).
                - "Gap": Treatment effect over time, shape (T, 2), where columns are
                  [gap_value, event_time_index].
            - "Fit" (Dict[str, Union[float, int]]):
                - "T0 RMSE": Root Mean Squared Error in the pre-treatment period.
                - "R-Squared": R-squared value in the pre-treatment period.
                - "Pre-Periods": Number of pre-treatment periods (t1).
            - "Inference" (Dict[str, float]):
                - "P-Value": P-value for the ATT.
                - "95 LB": Lower bound of the 95% confidence interval for ATT.
                - "95 UB": Upper bound of the 95% confidence interval for ATT.
                - "Width": Width of the 95% confidence interval.
            - "SE": Standard error of the ATT.
            - "Intercept": Estimated DID intercept.
        """
        total_periods: int = len(treated_outcome_all_periods)
        # Ensure control_outcomes_all_periods is a 2D array (T x N_controls) even if only one control unit.
        if control_outcomes_all_periods.ndim == 1: 
            control_outcomes_all_periods = control_outcomes_all_periods.reshape(-1,1)

        # Calculate the average outcome of control units for pre and post-treatment periods.
        mean_control_outcome_pre_treatment: np.ndarray = np.mean(control_outcomes_all_periods[:num_pre_treatment_periods], axis=1).reshape(-1, 1)
        mean_control_outcome_post_treatment: np.ndarray = np.mean(control_outcomes_all_periods[num_pre_treatment_periods:total_periods], axis=1).reshape(-1, 1)
        
        # The DID intercept is the average difference between the treated unit's outcome and the mean control outcome during the pre-treatment period.
        # This captures the baseline difference.
        did_intercept: float = np.mean(treated_outcome_all_periods[:num_pre_treatment_periods] - mean_control_outcome_pre_treatment).item()

        # Construct the counterfactual for the treated unit.
        # Pre-treatment: fitted values based on the intercept and mean control pre-treatment outcomes.
        fitted_treated_outcome_pre_treatment: np.ndarray = did_intercept + mean_control_outcome_pre_treatment
        # Post-treatment: predicted values based on the intercept and mean control post-treatment outcomes (assuming parallel trends).
        predicted_treated_outcome_post_treatment: np.ndarray = did_intercept + mean_control_outcome_post_treatment
        counterfactual_outcome_did: np.ndarray = np.vstack((fitted_treated_outcome_pre_treatment, predicted_treated_outcome_post_treatment))

        # Calculate the Average Treatment Effect on the Treated (ATT).
        # This is the average difference between the observed outcome and the counterfactual outcome in the post-treatment period.
        att_did: float = np.mean(treated_outcome_all_periods[num_pre_treatment_periods:total_periods] - counterfactual_outcome_did[num_pre_treatment_periods:total_periods]).item()
        mean_counterfactual_outcome_post_treatment_did: float = np.mean(counterfactual_outcome_did[num_pre_treatment_periods:total_periods]).item()
        # Calculate ATT as a percentage of the mean counterfactual outcome in the post-treatment period.
        att_percentage_did: float = (
            100 * att_did / mean_counterfactual_outcome_post_treatment_did if mean_counterfactual_outcome_post_treatment_did != 0 else np.nan
        )

        # Calculate R-squared for the pre-treatment period to assess goodness of fit.
        # Handle cases where variance of treated_outcome_all_periods[:num_pre_treatment_periods] is zero (e.g., num_pre_treatment_periods <= 1) to avoid division by zero.
        if num_pre_treatment_periods == 0: # No pre-periods to calculate R-squared.
            r_squared_pre_treatment_did = np.nan
        else:
            variance_treated_outcome_pre_treatment = np.var(treated_outcome_all_periods[:num_pre_treatment_periods]) # Total sum of squares (proportional).
            if num_pre_treatment_periods <= 1 or np.isclose(variance_treated_outcome_pre_treatment, 0): # If constant or single point, R2 is undefined.
                r_squared_pre_treatment_did = np.nan
            else:
                mse_model_pre_treatment_did = np.mean((treated_outcome_all_periods[:num_pre_treatment_periods] - counterfactual_outcome_did[:num_pre_treatment_periods]) ** 2) # Residual sum of squares (proportional).
                r_squared_pre_treatment_did = 1 - (mse_model_pre_treatment_did / variance_treated_outcome_pre_treatment)
        
        # Calculate residuals and variance components for statistical inference.
        residuals_pre_treatment_did: np.ndarray = treated_outcome_all_periods[:num_pre_treatment_periods] - counterfactual_outcome_did[:num_pre_treatment_periods]
        num_post_treatment_periods: int = total_periods - num_pre_treatment_periods
        
        # Estimate variance components omega1 and omega2 based on pre-treatment residuals.
        # Handle num_pre_treatment_periods=0 case for omega1_hat_did to avoid division by zero.
        omega1_hat_did: float = (
            (num_post_treatment_periods / num_pre_treatment_periods) * np.mean(residuals_pre_treatment_did**2) if num_pre_treatment_periods > 0 else np.nan
        )
        omega2_hat_did: float = np.mean(residuals_pre_treatment_did**2) if num_pre_treatment_periods > 0 else np.nan # Ensure omega2_hat is also nan if no pre-periods
        std_omega_hat_did: float = np.sqrt(omega1_hat_did + omega2_hat_did) if not (np.isnan(omega1_hat_did) or np.isnan(omega2_hat_did)) else np.nan


        # Calculate Standardized ATT (SATT) and p-value.
        standardized_att_did: float = (
            np.sqrt(num_post_treatment_periods) * att_did / std_omega_hat_did
            if std_omega_hat_did != 0 and not np.isnan(std_omega_hat_did) and num_post_treatment_periods > 0 else np.nan
        )
        p_value_did: float = 2 * (1 - norm.cdf(np.abs(standardized_att_did))) if not np.isnan(standardized_att_did) else np.nan

        # Calculate 95% Confidence Interval for ATT.
        z_critical: float = norm.ppf(0.975) # Z-score for 95% CI.
        standard_error_att_did: float = std_omega_hat_did / np.sqrt(num_post_treatment_periods) if num_post_treatment_periods > 0 and not np.isnan(std_omega_hat_did) else np.nan
        
        ci_95_lower_bound_did: float = att_did - z_critical * standard_error_att_did if not np.isnan(standard_error_att_did) else np.nan
        ci_95_upper_bound_did: float = att_did + z_critical * standard_error_att_did if not np.isnan(standard_error_att_did) else np.nan
        
        # Compile fit diagnostics results.
        fit_diagnostics_results_did: Dict[str, Any] = {
            "T0 RMSE": round(float(np.std(residuals_pre_treatment_did)), 3) if num_pre_treatment_periods > 0 else np.nan,
            "R-Squared": round(r_squared_pre_treatment_did, 3) if not np.isnan(r_squared_pre_treatment_did) else np.nan,
            "Pre-Periods": num_pre_treatment_periods,
        }
        # Compile effects results.
        effects_results_did: Dict[str, float] = {
            "ATT": round(att_did, 3),
            "Percent ATT": round(att_percentage_did, 3) if not np.isnan(att_percentage_did) else np.nan,
            "SATT": round(standardized_att_did, 3) if not np.isnan(standardized_att_did) else np.nan,
        }
        # Compile inference results.
        inference_results_did: Dict[str, Any] = {
            "P-Value": round(p_value_did, 3) if not np.isnan(p_value_did) else np.nan,
            "95 LB": round(ci_95_lower_bound_did, 3) if not np.isnan(ci_95_lower_bound_did) else np.nan,
            "95 UB": round(ci_95_upper_bound_did, 3) if not np.isnan(ci_95_upper_bound_did) else np.nan,
            "Width": ci_95_upper_bound_did - ci_95_lower_bound_did if not (np.isnan(ci_95_lower_bound_did) or np.isnan(ci_95_upper_bound_did)) else np.nan,
            "SE": round(standard_error_att_did, 4) if not np.isnan(standard_error_att_did) else np.nan,
            "Intercept": round(did_intercept, 3),
        }
        # Calculate the treatment effect (gap) over time.
        gap_over_time_did: np.ndarray = treated_outcome_all_periods - counterfactual_outcome_did
        # Create a matrix with gap values and corresponding event time indices.
        gap_matrix_did: np.ndarray = np.column_stack((gap_over_time_did, np.arange(gap_over_time_did.shape[0]) - num_pre_treatment_periods + 1))
        # Compile time series results.
        time_series_results_did: Dict[str, np.ndarray] = {
            "Observed Unit": np.round(treated_outcome_all_periods, 3),
            "Counterfactual": np.round(counterfactual_outcome_did, 3),
            "Gap": np.round(gap_matrix_did, 3),
        }
        # Return all compiled results.
        return {
            "Effects": effects_results_did, "Vectors": time_series_results_did,
            "Fit": fit_diagnostics_results_did, "Inference": inference_results_did,
        }

    def AUGDID(
        self, control_outcomes_all_periods: np.ndarray, total_periods: int, num_pre_treatment_periods: int, num_post_treatment_periods: int, 
        treated_outcome_all_periods: np.ndarray, treated_outcome_pre_treatment: np.ndarray, treated_outcome_post_treatment: np.ndarray
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Computes Augmented Difference-in-Differences (ADID) estimates.

        ADID extends the standard DID by regressing the pre-treatment outcomes of
        the treated unit on an intercept and the average pre-treatment outcomes of
        the control units. These regression coefficients are then used to predict
        the counterfactual outcomes in the post-treatment period.

        Parameters
        ----------
        control_outcomes_all_periods : np.ndarray
            Outcome matrix for control units, shape (T, N_controls).
        total_periods : int
            Total number of time periods (T).
        num_pre_treatment_periods : int
            Number of pre-treatment periods (T0).
        num_post_treatment_periods : int
            Number of post-treatment periods (T_post).
        treated_outcome_all_periods : np.ndarray
            Full outcome vector for the treated unit, shape (T,).
        treated_outcome_pre_treatment : np.ndarray
            Pre-treatment outcome vector for the treated unit, shape (T0,).
        treated_outcome_post_treatment : np.ndarray
            Post-treatment outcome vector for the treated unit, shape (T_post,).

        Returns
        -------
        Tuple[Dict[str, Any], np.ndarray]
            A tuple containing:
            - raw_results_adid (Dict[str, Any]): Dictionary with raw estimation results for ADID.
              The dictionary includes the following keys and sub-keys:
                - "Effects" (Dict[str, float]):
                    - "ATT": Average Treatment Effect on the Treated.
                    - "Percent ATT": Percentage ATT.
                    - "SATT": Standardized ATT.
                - "Vectors" (Dict[str, np.ndarray]):
                    - "Observed Unit": Outcome vector for the treated unit, shape (T, 1).
                    - "Counterfactual": Estimated counterfactual outcome vector, shape (T, 1).
                    - "Gap": Treatment effect over time, shape (T, 2), where columns are
                      [gap_value, event_time_index].
                - "Fit" (Dict[str, Union[float, int]]):
                    - "T0 RMSE": Root Mean Squared Error in the pre-treatment period.
                    - "R-Squared": R-squared value in the pre-treatment period.
                    - "T0": Number of pre-treatment periods (num_pre_treatment_periods).
                - "Inference" (Dict[str, float]):
                    - "P-Value": P-value for the ATT.
                    - "95 LB": Lower bound of the 95% confidence interval for ATT.
                    - "95 UB": Upper bound of the 95% confidence interval for ATT.
                    - "Width": Width of the 95% confidence interval.
            - counterfactual_outcome_adid (np.ndarray): The counterfactual outcome vector for ADID, shape (T, 1).
        """
        # Create a design matrix for controls including an intercept term.
        intercept_column_vector: np.ndarray = np.ones(total_periods)
        control_outcomes_with_intercept: np.ndarray = np.column_stack([intercept_column_vector, control_outcomes_all_periods])
        # Split this design matrix into pre-treatment and (implicitly) post-treatment periods.
        control_outcomes_with_intercept_pre_treatment: np.ndarray = control_outcomes_with_intercept[:num_pre_treatment_periods, :]
        # control_outcomes_with_intercept_post_treatment: np.ndarray = control_outcomes_with_intercept[num_pre_treatment_periods:, :] # Not directly used in this form for regression parameters

        # Prepare regressors for ADID: an intercept and the mean of control outcomes for each period.
        # Pre-treatment regressors:
        control_outcomes_pre_treatment_no_intercept: np.ndarray = control_outcomes_all_periods[:num_pre_treatment_periods, :]
        regressors_pre_treatment_adid: np.ndarray = np.column_stack([np.ones(control_outcomes_pre_treatment_no_intercept.shape[0]), np.mean(control_outcomes_pre_treatment_no_intercept, axis=1)])
        # Post-treatment regressors:
        control_outcomes_post_treatment_no_intercept: np.ndarray = control_outcomes_all_periods[num_pre_treatment_periods:, :]
        regressors_post_treatment_adid: np.ndarray = np.column_stack([np.ones(control_outcomes_post_treatment_no_intercept.shape[0]), np.mean(control_outcomes_post_treatment_no_intercept, axis=1)])

        # Estimate ADID parameters (beta_0, beta_1) using OLS on pre-treatment data.
        # treated_y_pre = beta_0 + beta_1 * mean_control_y_pre + error_pre
        estimated_parameters_adid: np.ndarray = np.linalg.inv(regressors_pre_treatment_adid.T @ regressors_pre_treatment_adid) @ (regressors_pre_treatment_adid.T @ treated_outcome_pre_treatment)
        
        # Construct the counterfactual for the treated unit using estimated parameters.
        # Pre-treatment fitted values:
        fitted_treated_outcome_pre_treatment_adid: np.ndarray = regressors_pre_treatment_adid @ estimated_parameters_adid
        # Post-treatment predicted (counterfactual) values:
        predicted_treated_outcome_post_treatment_adid: np.ndarray = regressors_post_treatment_adid @ estimated_parameters_adid
        counterfactual_outcome_adid: np.ndarray = np.concatenate([fitted_treated_outcome_pre_treatment_adid, predicted_treated_outcome_post_treatment_adid]).reshape(-1, 1)

        # Calculate ATT for ADID.
        att_adid: float = np.mean(treated_outcome_post_treatment - predicted_treated_outcome_post_treatment_adid).item()
        mean_counterfactual_outcome_post_treatment_adid: float = np.mean(predicted_treated_outcome_post_treatment_adid).item()
        # Calculate ATT as a percentage.
        att_percentage_adid: float = 100 * att_adid / mean_counterfactual_outcome_post_treatment_adid if mean_counterfactual_outcome_post_treatment_adid != 0 else np.nan
        
        # Calculate residuals and variance components for inference.
        residuals_pre_treatment_adid: np.ndarray = treated_outcome_pre_treatment - fitted_treated_outcome_pre_treatment_adid
        variance_residuals_pre_treatment_adid: float = np.mean(residuals_pre_treatment_adid**2) if num_pre_treatment_periods > 0 else np.nan

        # eta_adid: mean of (intercept, control_outcomes) in post-treatment period.
        eta_adid: np.ndarray = np.mean(control_outcomes_with_intercept[num_pre_treatment_periods:, :], axis=0).reshape(-1, 1) 
        # psi_adid: (X_pre' X_pre) / T0, where X_pre includes intercept and control outcomes.
        psi_adid: np.ndarray = control_outcomes_with_intercept_pre_treatment.T @ control_outcomes_with_intercept_pre_treatment / num_pre_treatment_periods if num_pre_treatment_periods > 0 else np.empty((control_outcomes_with_intercept.shape[1], control_outcomes_with_intercept.shape[1])) * np.nan
        
        # Calculate omega1_hat component for variance, handling potential singularity in psi_adid using pseudo-inverse.
        try:
            # omega1_hat = var_resid_pre * eta_post' * pinv(psi_pre) * eta_post
            omega1_hat_matrix_adid: np.ndarray = (variance_residuals_pre_treatment_adid * eta_adid.T) @ np.linalg.pinv(psi_adid) @ eta_adid if num_pre_treatment_periods > 0 and not np.any(np.isnan(psi_adid)) else np.array([[np.nan]])
            omega1_hat_adid: float = omega1_hat_matrix_adid.item() if omega1_hat_matrix_adid.size == 1 else np.nan
        except np.linalg.LinAlgError: # Catch errors during matrix operations.
            omega1_hat_adid = np.nan

        # omega2_hat is the variance of pre-treatment residuals.
        omega2_hat_adid: float = variance_residuals_pre_treatment_adid
        # Total variance for ATT estimator.
        total_variance_omega_adid: float = (num_post_treatment_periods / num_pre_treatment_periods) * omega1_hat_adid + omega2_hat_adid if num_pre_treatment_periods > 0 and not np.isnan(omega1_hat_adid) else np.nan
        
        # Standardized ATT (SATT).
        standardized_att_adid: float = (
            np.sqrt(num_post_treatment_periods) * att_adid / np.sqrt(total_variance_omega_adid)
            if total_variance_omega_adid > 0 and not np.isnan(total_variance_omega_adid) and num_post_treatment_periods > 0 else np.nan
        )
        
        # Calculate 95% Confidence Interval (using a simplified SE for CI as in original paper's code).
        z_critical: float = norm.ppf(0.975)
        standard_error_att_adid: float = np.sqrt(variance_residuals_pre_treatment_adid) / np.sqrt(num_post_treatment_periods) if num_post_treatment_periods > 0 and not np.isnan(variance_residuals_pre_treatment_adid) else np.nan
        
        ci_95_lower_bound_adid: float = att_adid - z_critical * standard_error_att_adid if not np.isnan(standard_error_att_adid) else np.nan
        ci_95_upper_bound_adid: float = att_adid + z_critical * standard_error_att_adid if not np.isnan(standard_error_att_adid) else np.nan

        # Calculate R-squared for the pre-treatment period.
        r_squared_pre_treatment_adid: float = 1 - (np.mean(residuals_pre_treatment_adid**2)) / np.mean((treated_outcome_pre_treatment - np.mean(treated_outcome_pre_treatment))**2) if num_pre_treatment_periods > 1 and not np.isclose(np.var(treated_outcome_pre_treatment),0) else np.nan
        # Calculate p-value for ATT.
        p_value_adid: float = 2 * (1 - norm.cdf(np.abs(standardized_att_adid))) if not np.isnan(standardized_att_adid) else np.nan

        # Compile results into dictionaries.
        fit_diagnostics_results_adid: Dict[str, Any] = {
            "T0 RMSE": round(float(np.std(residuals_pre_treatment_adid)), 3) if num_pre_treatment_periods > 0 else np.nan,
            "R-Squared": round(r_squared_pre_treatment_adid, 3) if not np.isnan(r_squared_pre_treatment_adid) else np.nan,
            "T0": num_pre_treatment_periods, 
        }
        effects_results_adid: Dict[str, float] = {
            "ATT": round(att_adid, 3),
            "Percent ATT": round(att_percentage_adid, 3) if not np.isnan(att_percentage_adid) else np.nan,
            "SATT": round(standardized_att_adid, 3) if not np.isnan(standardized_att_adid) else np.nan,
        }
        inference_results_adid: Dict[str, Any] = {
            "P-Value": round(p_value_adid, 3) if not np.isnan(p_value_adid) else np.nan,
            "95 LB": round(ci_95_lower_bound_adid, 3) if not np.isnan(ci_95_lower_bound_adid) else np.nan,
            "95 UB": round(ci_95_upper_bound_adid, 3) if not np.isnan(ci_95_upper_bound_adid) else np.nan,
            "Width": ci_95_upper_bound_adid - ci_95_lower_bound_adid if not (np.isnan(ci_95_lower_bound_adid) or np.isnan(ci_95_upper_bound_adid)) else np.nan,
        }
        gap_over_time_adid: np.ndarray = treated_outcome_all_periods.reshape(-1,1) - counterfactual_outcome_adid 
        gap_matrix_adid: np.ndarray = np.column_stack((gap_over_time_adid, np.arange(gap_over_time_adid.shape[0]) - num_pre_treatment_periods + 1))
        time_series_results_adid: Dict[str, np.ndarray] = {
            "Observed Unit": np.round(treated_outcome_all_periods.reshape(-1,1), 3),
            "Counterfactual": np.round(counterfactual_outcome_adid, 3),
            "Gap": np.round(gap_matrix_adid, 3),
        }
        raw_results_adid: Dict[str, Any] = {
            "Effects": effects_results_adid, "Vectors": time_series_results_adid,
            "Fit": fit_diagnostics_results_adid, "Inference": inference_results_adid,
        }
        return raw_results_adid, counterfactual_outcome_adid

    def est(
        self, selected_control_outcomes: np.ndarray, total_periods: int, num_pre_treatment_periods: int, num_post_treatment_periods: int,
        treated_outcome_all_periods: np.ndarray, treated_outcome_pre_treatment: np.ndarray, treated_outcome_post_treatment: np.ndarray, all_control_outcomes: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]:
        """
        Estimates FDID, DID, and ADID models based on provided data components.

        This method is called internally by `fit` after control selection (for FDID)
        and data preparation. It orchestrates calls to `DID` and `AUGDID`.

        Parameters
        ----------
        selected_control_outcomes : np.ndarray
            Outcome matrix for control units selected by the FDID procedure,
            shape (T, N_selected_controls).
        total_periods : int
            Total number of time periods (T).
        num_pre_treatment_periods : int
            Number of pre-treatment periods (T0).
        num_post_treatment_periods : int
            Number of post-treatment periods (T_post).
        treated_outcome_all_periods : np.ndarray
            Full outcome vector for the treated unit, shape (T,).
        treated_outcome_pre_treatment : np.ndarray
            Pre-treatment outcome vector for the treated unit, shape (T0,).
        treated_outcome_post_treatment : np.ndarray
            Post-treatment outcome vector for the treated unit, shape (T_post,).
        all_control_outcomes : np.ndarray
            Full outcome matrix for all available control units, shape (T, N_all_controls).
            Used for standard DID and ADID.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]
            A tuple containing:
            - raw_results_fdid (Dict[str, Any]): Raw results dictionary for the Forward DID method,
              structured as per the return of the `DID` method.
            - raw_results_did (Dict[str, Any]): Raw results dictionary for the standard DID method,
              structured as per the return of the `DID` method.
            - raw_results_augdid (Dict[str, Any]): Raw results dictionary for the Augmented DID method,
              structured as per the first element of the tuple returned by the `AUGDID` method.
            - counterfactual_outcome_fdid (np.ndarray): Counterfactual outcome vector from the FDID method,
              shape (T, 1).
        """
        # Estimate Forward DID using selected controls.
        raw_results_fdid: Dict[str, Any] = self.DID(treated_outcome_all_periods.reshape(-1, 1), selected_control_outcomes, num_pre_treatment_periods)
        counterfactual_outcome_fdid: np.ndarray = raw_results_fdid["Vectors"]["Counterfactual"]

        # Estimate standard DID using all available controls.
        raw_results_did: Dict[str, Any] = self.DID(treated_outcome_all_periods.reshape(-1, 1), all_control_outcomes, num_pre_treatment_periods)
        # Estimate Augmented DID using all available controls. The second returned element (counterfactual_outcome_adid) is not directly used here.
        raw_results_augdid, _ = self.AUGDID(all_control_outcomes, total_periods, num_pre_treatment_periods, num_post_treatment_periods, treated_outcome_all_periods, treated_outcome_pre_treatment, treated_outcome_post_treatment)

        # Calculate the width of DID and ADID confidence intervals relative to FDID's CI width.
        # This provides a measure of precision relative to FDID.
        if raw_results_fdid["Inference"].get("Width") is not None and not np.isnan(raw_results_fdid["Inference"]["Width"]) and raw_results_fdid["Inference"]["Width"] != 0:
            if raw_results_did["Inference"].get("Width") is not None and not np.isnan(raw_results_did["Inference"]["Width"]):
                width_ratio_did_to_fdid: float = raw_results_did["Inference"]["Width"] / raw_results_fdid["Inference"]["Width"]
                raw_results_did["Inference"]["WidthRFDID"] = round(width_ratio_did_to_fdid, 3) # Store relative width in DID results.
            
            if raw_results_augdid["Inference"].get("Width") is not None and not np.isnan(raw_results_augdid["Inference"]["Width"]):
                width_ratio_augdid_to_fdid: float = raw_results_augdid["Inference"]["Width"] / raw_results_fdid["Inference"]["Width"]
                raw_results_augdid["Inference"]["WidthRFDID"] = round(width_ratio_augdid_to_fdid, 3) # Store relative width in ADID results.
        
        return raw_results_fdid, raw_results_did, raw_results_augdid, counterfactual_outcome_fdid

    def selector(
        self, total_available_controls: int, num_pre_treatment_periods: int, total_periods: int,
        treated_outcome_all_periods: np.ndarray, all_control_outcomes_matrix: np.ndarray, available_control_column_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects control units for FDID using a forward selection procedure.

        This method iteratively adds control units one by one. At each step, it
        selects the control unit that, when added to the current set of selected
        controls, maximizes the pre-treatment R-squared of a DID model. The process
        continues until all control units have been considered. The final set of
        controls is determined by the iteration that achieved the highest R-squared.

        Parameters
        ----------
        total_available_controls : int
            Total number of available control units (N_all_controls). This corresponds
            to the number of columns in `all_control_outcomes_matrix`.
        num_pre_treatment_periods : int
            Number of pre-treatment periods (T0).
        total_periods : int
            Total number of time periods (T).
        treated_outcome_all_periods : np.ndarray
            Full outcome vector for the treated unit, shape (T,).
        all_control_outcomes_matrix : np.ndarray
            Full outcome matrix of all available control units, shape (T, N_all_controls).
        available_control_column_indices : np.ndarray
            Array of 0-based indices representing the columns in `all_control_outcomes_matrix` that are
            available for selection. Typically `np.arange(N_all_controls)`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - final_selected_control_column_indices (np.ndarray): A 1D array of 0-based column indices
              (relative to the input `all_control_outcomes_matrix`) of the optimally selected control units,
              shape (N_selected_controls,).
            - r_squared_at_each_step (np.ndarray): A 1D array of R-squared values achieved at each
              step of the forward selection (i.e., after adding 1st control, 2nd, etc.),
              shape (N_all_controls,).
            - final_selected_controls_outcomes_matrix (np.ndarray): The outcome matrix for the finally
              selected control units, shape (T, N_selected_controls).
        """
        # Initialize arrays to store R-squared values at each step and the indices of selected controls.
        r_squared_at_each_step: np.ndarray = np.zeros(total_available_controls)
        # Ensure available_control_column_indices is a NumPy array for set operations.
        zero_based_control_column_indices: np.ndarray = np.array(available_control_column_indices) 

        # Stores the 0-based column index (from all_control_outcomes_matrix) of the control selected at each step.
        selected_control_indices_in_step: np.ndarray = np.zeros(total_available_controls, dtype=int)
        # Matrix to accumulate outcomes of selected controls, starts empty.
        current_selected_controls_outcomes_matrix: np.ndarray = np.empty((total_periods, 0))
        # Ensure treated outcome is a column vector for consistency in DID calls.
        treated_outcome_column_vector: np.ndarray = treated_outcome_all_periods.reshape(-1, 1)

        # Iterate through each selection step, up to the total number of available controls.
        for current_selection_step in range(total_available_controls):
            # Determine which control units are still available for selection.
            # These are original column indices from all_control_outcomes_matrix.
            remaining_control_indices_to_consider: np.ndarray = np.setdiff1d(zero_based_control_column_indices, selected_control_indices_in_step[:current_selection_step])
            if not remaining_control_indices_to_consider.size: # If no controls left, stop.
                break 

            # Array to store R-squared values for each candidate control in the current step.
            r_squared_for_current_candidates: np.ndarray = np.zeros(len(remaining_control_indices_to_consider))

            # Evaluate each remaining control unit as a candidate to be added.
            for candidate_control_loop_index, candidate_control_column_index in enumerate(remaining_control_indices_to_consider):
                # Basic check for index validity, though this should be guaranteed by `available_control_column_indices`.
                if candidate_control_column_index < 0 or candidate_control_column_index >= all_control_outcomes_matrix.shape[1]:
                    r_squared_for_current_candidates[candidate_control_loop_index] = -np.inf # Mark as invalid.
                    continue

                # Get the outcome vector for the current candidate control.
                outcome_vector_for_candidate_control: np.ndarray = all_control_outcomes_matrix[:, candidate_control_column_index].reshape(-1, 1)
                # Temporarily combine outcomes of already selected controls with the current candidate's outcomes.
                temp_combined_controls_for_r2_calc: np.ndarray = np.hstack((current_selected_controls_outcomes_matrix, outcome_vector_for_candidate_control))
                
                # Calculate DID using this temporary set of controls to get pre-treatment R-squared.
                temp_did_results_for_r2_calc: Dict[str, Any] = self.DID(treated_outcome_column_vector, temp_combined_controls_for_r2_calc, num_pre_treatment_periods)
                r_squared_for_current_candidates[candidate_control_loop_index] = temp_did_results_for_r2_calc["Fit"]["R-Squared"]
            
            # If no valid R-squared values were calculated (e.g., all candidates led to errors or NaN R2), stop.
            if not r_squared_for_current_candidates.size or np.all(np.isneginf(r_squared_for_current_candidates)) or np.all(np.isnan(r_squared_for_current_candidates)):
                 break

            # Select the candidate control that yields the highest R-squared in this step.
            # np.nanargmax handles NaNs by ignoring them, which is desired here.
            best_candidate_index_in_candidates_array = np.nanargmax(r_squared_for_current_candidates)
            r_squared_at_each_step[current_selection_step] = r_squared_for_current_candidates[best_candidate_index_in_candidates_array]
            
            best_candidate_control_column_index_this_step: int = remaining_control_indices_to_consider[best_candidate_index_in_candidates_array]
            # Record the index of the selected control for this step.
            selected_control_indices_in_step[current_selection_step] = best_candidate_control_column_index_this_step
            # Add the selected control's outcomes to the matrix of currently selected controls.
            current_selected_controls_outcomes_matrix = np.hstack((current_selected_controls_outcomes_matrix, all_control_outcomes_matrix[:, best_candidate_control_column_index_this_step].reshape(-1, 1)))
        
        # Determine the optimal number of controls: the step that maximized R-squared.
        # Filter out any initial zero R-squared values if they occurred before valid selections.
        valid_r_squared_values = r_squared_at_each_step[r_squared_at_each_step != 0] if np.any(r_squared_at_each_step != 0) else r_squared_at_each_step[:1] 
        if not valid_r_squared_values.size or np.all(np.isnan(valid_r_squared_values)): # If no valid R2 values found
            optimal_number_of_selected_controls = 0
        else:
            # Add 1 because argmax returns 0-based index, and we want the count of controls.
            optimal_number_of_selected_controls = np.nanargmax(valid_r_squared_values) + 1 
            
        # Get the column indices (from original all_control_outcomes_matrix) of the final set of selected controls.
        final_selected_control_column_indices: np.ndarray = selected_control_indices_in_step[:optimal_number_of_selected_controls]
        # Get the outcome matrix for this final set of selected controls.
        final_selected_controls_outcomes_matrix: np.ndarray = all_control_outcomes_matrix[:, final_selected_control_column_indices]

        return final_selected_control_column_indices, r_squared_at_each_step, final_selected_controls_outcomes_matrix

    def _create_estimator_results(
        self, method_name: str, results_dict: Dict[str, Any], prepared_data: Dict[str, Any]
    ) -> BaseEstimatorResults:
        """
        Converts a raw results dictionary from an estimation method (FDID, DID, ADID)
        into a standardized BaseEstimatorResults Pydantic model.

        Parameters
        ----------
        method_name : str
            The name of the estimation method (e.g., "FDID", "DID", "AUGDID").
        results_dict : Dict[str, Any]
            The raw dictionary of results produced by the internal DID or AUGDID methods.
            Expected to contain keys like "Effects", "Fit", "Vectors", "Inference".
        prepared_data : Dict[str, Any]
            The preprocessed data dictionary from `dataprep`. Used for context if needed.

        Returns
        -------
        BaseEstimatorResults
            A Pydantic model populated with the standardized results.
        """
        # Extract core effect metrics (ATT, Percent ATT) and store other effects in `additional_effects`.
        raw_effects_data = results_dict.get("Effects", {})
        additional_effects_map = {
            k.lower().replace(" ", "_"): v for k, v in raw_effects_data.items() if k not in ["ATT", "Percent ATT"]
        }

        effects_results_obj = EffectsResults(
            att=raw_effects_data.get("ATT"),
            att_percent=raw_effects_data.get("Percent ATT"),
            additional_effects=additional_effects_map if additional_effects_map else None
        )

        # Extract core fit diagnostics (RMSE, R-squared) and store others (like pre-period count) in `additional_metrics`.
        raw_fit_diagnostics_data = results_dict.get("Fit", {})
        # Key for pre-period count differs slightly between DID/FDID ("Pre-Periods") and AUGDID ("T0").
        key_for_pre_period_count = "T0" if method_name == "AUGDID" else "Pre-Periods"
        additional_fit_metrics_map = {
            "pre_periods_count": raw_fit_diagnostics_data.get(key_for_pre_period_count)
        }
        # Store any other fit metrics not explicitly mapped.
        for k,v in raw_fit_diagnostics_data.items():
            if k not in ["T0 RMSE", "R-Squared", key_for_pre_period_count]:
                additional_fit_metrics_map[k.lower().replace("-", "_")] = v
        
        fit_diagnostics_results_obj = FitDiagnosticsResults(
            pre_treatment_rmse=raw_fit_diagnostics_data.get("T0 RMSE"),
            pre_treatment_r_squared=raw_fit_diagnostics_data.get("R-Squared"),
            additional_metrics=additional_fit_metrics_map if any(v is not None for v in additional_fit_metrics_map.values()) else None
        )

        # Extract time series data: observed, counterfactual, gap, and time periods.
        raw_time_series_data = results_dict.get("Vectors", {})
        raw_gap_series_data = raw_time_series_data.get("Gap") # Expected to be a (T, 2) array: [gap_value, event_time_index]
        final_estimated_gap_series = None
        final_time_periods_for_series = None 

        # Attempt to get actual time period values (e.g., years) from the `prepared_data` (from `dataprep`'s `Ywide` index).
        outcome_wide_df_from_prepared_data = prepared_data.get("Ywide")
        if isinstance(outcome_wide_df_from_prepared_data, pd.DataFrame):
            final_time_periods_for_series = outcome_wide_df_from_prepared_data.index.to_numpy()
        
        # Process the raw gap series.
        if isinstance(raw_gap_series_data, np.ndarray) and raw_gap_series_data.ndim == 2 and raw_gap_series_data.shape[1] == 2:
            final_estimated_gap_series = raw_gap_series_data[:, 0] # First column contains the gap values.
            # If actual time periods weren't found from `Ywide`, use event time indices from the second column of `raw_gap_series_data`.
            if final_time_periods_for_series is None:
                 final_time_periods_for_series = raw_gap_series_data[:, 1]
            # If lengths mismatch (should not happen if data is consistent), fallback to event time indices.
            elif len(final_time_periods_for_series) != len(final_estimated_gap_series):
                 final_time_periods_for_series = raw_gap_series_data[:, 1]


        time_series_results_obj = TimeSeriesResults(
            observed_outcome=raw_time_series_data.get("Observed Unit"),
            counterfactual_outcome=raw_time_series_data.get("Counterfactual"),
            estimated_gap=final_estimated_gap_series,
            time_periods=final_time_periods_for_series
        )

        # Extract donor weights (relevant primarily for FDID, which explicitly selects donors).
        raw_weights_data = results_dict.get("Weights") 
        weights_results_obj = WeightsResults(donor_weights=raw_weights_data) if raw_weights_data is not None else None
        
        # Extract core inference metrics (p-value, CI bounds, SE) and store others in `details`.
        raw_inference_data = results_dict.get("Inference", {})
        additional_inference_details_map = {
            "intercept": raw_inference_data.get("Intercept"), # Relevant for DID
            "ci_width": raw_inference_data.get("Width"),
            "width_relative_to_fdid": raw_inference_data.get("WidthRFDID") # Custom metric comparing CI width to FDID's
        }
        # Store any other inference metrics not explicitly mapped.
        for k,v in raw_inference_data.items():
            if k not in ["P-Value", "95 LB", "95 UB", "Width", "SE", "Intercept", "WidthRFDID"]:
                 additional_inference_details_map[k.lower().replace("-", "_")] = v
        
        inference_results_obj = InferenceResults(
            p_value=raw_inference_data.get("P-Value"),
            ci_lower_bound=raw_inference_data.get("95 LB"),
            ci_upper_bound=raw_inference_data.get("95 UB"),
            standard_error=raw_inference_data.get("SE"),
            details={k:v for k,v in additional_inference_details_map.items() if v is not None} if any(v is not None for v in additional_inference_details_map.values()) else None
        )
        
        # Store method details, including the estimator name and parameters used (from the config).
        method_details_results_obj = MethodDetailsResults(
            method_name=method_name, # "FDID", "DID", or "AUGDID"
            parameters_used=self.config.model_dump(exclude={'df'}, exclude_none=True) # Exclude DataFrame itself.
        )

        # Assemble and return the final standardized results object.
        return BaseEstimatorResults(
            effects=effects_results_obj,
            fit_diagnostics=fit_diagnostics_results_obj,
            time_series=time_series_results_obj,
            weights=weights_results_obj,
            inference=inference_results_obj,
            method_details=method_details_results_obj,
            raw_results=results_dict, # Include the original raw dictionary for full transparency/debugging.
        )

    def fit(self) -> List[BaseEstimatorResults]:
        """
        Fits the Forward Difference-in-Differences (FDID), standard DID, and Augmented DID (ADID) models.

        The process involves:
        1. Balancing the input panel data.
        2. Preprocessing data using `dataprep`.
        3. Performing forward selection of control units based on pre-treatment R-squared
           to determine the optimal set of controls for the FDID method.
        4. Estimating effects using the selected controls (for FDID) and all controls
           (for standard DID and ADID).
        5. Constructing standardized results objects for each of the three methods.
        6. Optionally displaying plots comparing observed outcomes to counterfactuals
           from FDID and standard DID.

        Returns
        -------
        List[BaseEstimatorResults]
            A list containing three `BaseEstimatorResults` objects, corresponding to
            the FDID, DID, and ADID methods, in that order. Each object details:
            - `effects`: `EffectsResults` model with `att` (Average Treatment Effect
              on the Treated) and `att_percent`. Other method-specific effects like
              `satt` (Standardized ATT) are in `additional_effects`.
            - `fit_diagnostics`: `FitDiagnosticsResults` model with
              `pre_treatment_rmse` and `pre_treatment_r_squared`.
              `additional_metrics` contains `pre_periods_count`.
            - `time_series`: `TimeSeriesResults` model with `observed_outcome`,
              `counterfactual_outcome`, `estimated_gap` (effect over time),
              and `time_periods` (actual time values or event time indices).
            - `weights`: `WeightsResults` model. For FDID, `donor_weights` contains
              the weights for selected controls (equally weighted). For DID/ADID,
              this is typically `None` as they use all controls implicitly.
            - `inference`: `InferenceResults` model with `p_value`, `ci_lower_bound`,
              `ci_upper_bound`, and `standard_error`. `details` may contain
              `intercept`, `ci_width`, and `width_relative_to_fdid`.
            - `method_details`: `MethodDetailsResults` model with the method `name`
              ("FDID", "DID", or "AUGDID") and `parameters_used` (the estimator config).
            - `raw_results`: The original dictionary of results from the internal
              estimation functions.

        Examples
        --------
        >>> from mlsynth import FDID
        >>> from mlsynth.config_models import FDIDConfig
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'unit': np.repeat(np.arange(1, 11), 20),
        ...     'time': np.tile(np.arange(1, 21), 10),
        ...     'outcome': np.random.rand(200) + np.repeat(np.arange(0,10),20)*0.1,
        ...     'treated': ((np.repeat(np.arange(1, 11), 20) == 1) & (np.tile(np.arange(1, 21), 10) >= 15)).astype(int)
        ... })
        >>> config = FDIDConfig(
        ...     df=data, outcome='outcome', treat='treated', unitid='unit', time='time',
        ...     display_graphs=False
        ... )
        >>> estimator = FDID(config=config)
        >>> results_list = estimator.fit() # doctest: +SKIP
        >>> fdid_results = results_list[0] # doctest: +SKIP
        >>> did_results = results_list[1] # doctest: +SKIP
        >>> augdid_results = results_list[2] # doctest: +SKIP
        >>> # Example of accessing results (actual values will vary due to random data)
        >>> print(f"FDID ATT: {fdid_results.effects.att}") # doctest: +SKIP
        >>> print(f"DID ATT: {did_results.effects.att}") # doctest: +SKIP
        >>> print(f"AUGDID ATT: {augdid_results.effects.att}") # doctest: +SKIP
        """
        # Initialize containers for results and intermediate data.
        list_of_estimator_results_objects: List[BaseEstimatorResults] = []
        counterfactual_outcome_fdid_vector: Optional[np.ndarray] = None # To store FDID counterfactual for plotting.
        counterfactual_outcome_did_vector: Optional[np.ndarray] = None  # To store DID counterfactual for plotting.
        prepared_data: Dict[str, Any] = {} # To store output from dataprep.

        try:
            # Step 1: Balance the panel data.
            balance(self.df, self.unitid, self.time)

            # Step 2: Prepare data into matrix format.
            prepared_data = dataprep(
                self.df, self.unitid, self.time, self.outcome, self.treated
            )

            # Extract pre- and post-treatment outcomes for the treated unit.
            treated_outcome_pre_treatment_vector: np.ndarray = np.ravel(prepared_data["y"][: prepared_data["pre_periods"]])
            treated_outcome_post_treatment_vector: np.ndarray = np.ravel(prepared_data["y"][-prepared_data["post_periods"] :])
            
            # Get information about available donor units for the selector.
            number_of_available_donors = prepared_data["donor_matrix"].shape[1]
            donor_column_indices_for_selector = np.arange(number_of_available_donors) # 0-based indices.

            # Step 3: Perform forward selection of control units for FDID.
            # `selector` returns indices of selected donors, R2 at each step, and outcomes of selected donors.
            selected_donor_column_indices, r_squared_values_from_selector, selected_donors_outcomes_matrix = self.selector(
                number_of_available_donors, 
                prepared_data["pre_periods"],
                prepared_data["total_periods"],
                prepared_data["y"], # Full outcome vector for the treated unit.
                prepared_data["donor_matrix"], # Outcome matrix for all potential donor units.
                donor_column_indices_for_selector, 
            )
            
            # Get names of the selected donors using their column indices.
            names_of_selected_donors: List[str] = [prepared_data["donor_names"][i] for i in selected_donor_column_indices]

            # Step 4: Estimate FDID, standard DID, and ADID models.
            # `est` orchestrates calls to `DID` (for FDID and standard DID) and `AUGDID`.
            raw_results_fdid, raw_results_did, raw_results_augdid, counterfactual_outcome_fdid_vector = self.est(
                selected_donors_outcomes_matrix, # Outcomes for FDID-selected controls.
                prepared_data["total_periods"],
                prepared_data["pre_periods"],
                prepared_data["post_periods"],
                prepared_data["y"], # Full treated outcome.
                treated_outcome_pre_treatment_vector, # Pre-treatment treated outcome.
                treated_outcome_post_treatment_vector, # Post-treatment treated outcome.
                prepared_data["donor_matrix"], # Outcomes for all controls (for standard DID & ADID).
            )

            # For FDID, selected donors are typically equally weighted. Calculate and store these weights.
            if names_of_selected_donors: 
                individual_donor_weight_fdid: float = 1 / len(names_of_selected_donors) if names_of_selected_donors else 0
                donor_weights_map_fdid: Dict[str, float] = {unit: individual_donor_weight_fdid for unit in names_of_selected_donors}
                raw_results_fdid["Weights"] = donor_weights_map_fdid
            else: # Handle case where no donors are selected (should be rare if selector works).
                raw_results_fdid["Weights"] = {}
            
            # Step 5: Convert raw results for each method into standardized Pydantic result objects.
            fdid_estimator_results_obj = self._create_estimator_results("FDID", raw_results_fdid, prepared_data)
            did_estimator_results_obj = self._create_estimator_results("DID", raw_results_did, prepared_data)
            augdid_estimator_results_obj = self._create_estimator_results("AUGDID", raw_results_augdid, prepared_data)

            # Compile a list of all estimator results.
            list_of_estimator_results_objects = [
                fdid_estimator_results_obj,
                did_estimator_results_obj,
                augdid_estimator_results_obj,
            ]
            
            # Store the DID counterfactual vector for potential plotting.
            if did_estimator_results_obj.time_series:
                counterfactual_outcome_did_vector = did_estimator_results_obj.time_series.counterfactual_outcome

        # --- Error Handling for the entire fit process ---
        except (MlsynthDataError, MlsynthConfigError, MlsynthEstimationError) as e:
            # Re-raise known custom errors.
            raise e
        except ValidationError as e: # Catch Pydantic validation errors during results model creation.
            raise MlsynthEstimationError(f"Error creating results structure for FDID: {str(e)}") from e
        except Exception as e: # Catch any other unexpected errors.
            raise MlsynthEstimationError(f"An unexpected error occurred during FDID estimation: {str(e)}") from e

        # Step 6: Plotting (if enabled and data is available).
        if self.display_graphs and prepared_data: 
            try:
                # Collect counterfactual vectors for plotting (FDID and standard DID).
                list_of_counterfactual_vectors_for_plot = []
                if counterfactual_outcome_fdid_vector is not None:
                    list_of_counterfactual_vectors_for_plot.append(counterfactual_outcome_fdid_vector)
                if counterfactual_outcome_did_vector is not None:
                    list_of_counterfactual_vectors_for_plot.append(counterfactual_outcome_did_vector)
                
                # Proceed only if there are counterfactuals to plot.
                if list_of_counterfactual_vectors_for_plot:
                    plot_estimates(
                        processed_data_dict=prepared_data, # Pass prepared_data which contains necessary context like treated_unit_name.
                        time_axis_label=self.time,
                        unit_identifier_column_name=self.unitid,
                        outcome_variable_label=self.outcome,
                        treatment_name_label=self.treated, 
                        treated_unit_name=prepared_data["treated_unit_name"],
                        observed_outcome_series=prepared_data["y"], # Observed outcome for the treated unit.
                        counterfactual_series_list=[cf.flatten() for cf in list_of_counterfactual_vectors_for_plot], # List of CF vectors (FDID, DID).
                        estimation_method_name="FDID", # Main method name for plot title/saving.
                        counterfactual_names=[ # Legend names for the CF lines.
                            f"FDID {prepared_data['treated_unit_name']}",
                            f"DID {prepared_data['treated_unit_name']}",
                        ][:len(list_of_counterfactual_vectors_for_plot)], # Slice to match number of CFs.
                        treated_series_color=self.treated_color,
                        save_plot_config=self.save, # Plot saving configuration.
                        counterfactual_series_colors=( # Colors for CF lines.
                             [self.counterfactual_color] # Ensure it's a list even if one color string.
                             if isinstance(self.counterfactual_color, str)
                             else self.counterfactual_color
                        ), 
                    )
            except (MlsynthPlottingError, MlsynthDataError) as e: # Catch known plotting errors.
                print(f"Warning: Plotting failed for FDID - {str(e)}")
            except Exception as e: # Catch any other unexpected errors during plotting.
                print(f"Warning: An unexpected error occurred during FDID plotting - {str(e)}")

        return list_of_estimator_results_objects
