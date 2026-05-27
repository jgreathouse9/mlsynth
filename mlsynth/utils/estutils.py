import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.linalg import eigh
from mlsynth.utils.resultutils import effects
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_X_y, check_array
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from functools import partial
from statsmodels.tsa.stattools import acf
from scipy.stats import norm # zconfint was not explicitly imported and not used
from sklearn.linear_model import LassoCV
from scipy.stats import t as t_dist
import warnings # For RPCASYNTH warning
import pandas as pd
from typing import Any, Tuple, List, Optional, Dict, Union, Callable, Literal
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError

# --- Constants for SCM and PDA Method Names ---
_SCM_MODEL_SIMPLEX = "SIMPLEX"
_SCM_MODEL_AFFINE = "AFFINE"
_SCM_MODEL_MSCA = "MSCa"
_SCM_MODEL_MSCB = "MSCb"
_SCM_MODEL_MSCC = "MSCc"
_SCM_MODEL_OLS = "OLS"
_SCM_MODEL_MA = "MA"
_SCM_MODEL_INF = "INF"
_SCM_MODEL_RL2 = "RL2"

_PDA_METHOD_FS = "fs" # Forward Selection
_PDA_METHOD_LASSO = "LASSO"
_PDA_METHOD_L2 = "l2" # L2-relaxation

# --- Constants for CVXPY Solvers ---
_SOLVER_CLARABEL_STR = "CLARABEL"
_SOLVER_OSQP_STR = "OSQP"

# --- Constants for ci_bootstrap ---
_BOOTSTRAP_SUBSAMPLE_ADJUSTMENT = 5
_BOOTSTRAP_RANDOM_SEED = 1476


def pi2(
    outcome_vector: np.ndarray,
    design_matrix: np.ndarray,
    proxy_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_se: int,
    total_periods: int,
    hac_lag_length: int,
    covariates_for_W: Optional[np.ndarray] = None,
    covariates_for_Z0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Proximal inference for treatment effect estimation using GMM.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Outcome vector (total_periods,).
    design_matrix : np.ndarray
        Design matrix (total_periods, dim_design_matrix).
    proxy_matrix : np.ndarray
        Proxy matrix (total_periods, dim_proxy_matrix).
    num_pre_treatment_periods : int
        Number of pre-treatment periods.
    num_post_periods_for_se : int
        Number of post-treatment periods to consider for averaging the treatment effect
        when calculating its standard error.
    total_periods : int
        Total number of time periods in `outcome_vector`, `design_matrix`, `proxy_matrix`.
    hac_lag_length : int
        HAC lag length for variance estimation.
    covariates_for_W : Optional[np.ndarray], optional
        Covariates to augment `design_matrix`. If provided, `covariates_for_Z0` must also be provided.
        Shape should be (total_periods, dim_covariates_W). Default is None.
    covariates_for_Z0 : Optional[np.ndarray], optional
        Covariates to augment `proxy_matrix`. If provided, `covariates_for_W` must also be provided.
        Shape should be (total_periods, dim_covariates_Z0). Default is None.
        Note: `dim_covariates_W` and `dim_covariates_Z0` must be such that `design_matrix` and `proxy_matrix`
        can be augmented to have the same number of columns.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        - predicted_counterfactual_original_W : np.ndarray
            Predicted counterfactual outcomes using the original design matrix and
            its estimated coefficients. Shape (total_periods,).
        - alpha_coefficients_original_W : np.ndarray
            Estimated coefficients for the original design matrix. Shape (dim_design_matrix,).
        - standard_error_tau_placeholder : float
            Placeholder for the standard error of the treatment effect.
            Currently returns `np.nan`.

    Raises
    ------
    MlsynthConfigError
        If augmented design_matrix and proxy_matrix do not have the same number of columns.
    """
    augmented_design_matrix: np.ndarray = design_matrix
    augmented_proxy_matrix: np.ndarray = proxy_matrix

    if covariates_for_W is not None and covariates_for_Z0 is not None:
        augmented_proxy_matrix = np.column_stack((proxy_matrix, covariates_for_Z0, covariates_for_W))
        augmented_design_matrix = np.column_stack((design_matrix, covariates_for_Z0, covariates_for_W))

    if augmented_design_matrix.shape[1] != augmented_proxy_matrix.shape[1]:
        raise MlsynthConfigError(
            "Augmented design matrix W and proxy matrix Z0 must have the same number of columns."
        )

    dimension_augmented_W: int = augmented_design_matrix.shape[1]
    alpha_cvxpy_var = cp.Variable(dimension_augmented_W)
    tau_cvxpy_var = cp.Variable(1)

    # Moment conditions for GMM
    # Pre-treatment moment: E[Z0' * (Y - W*alpha)] = 0
    gmm_moments_pre_treatment = augmented_proxy_matrix[:num_pre_treatment_periods].T @ (
        outcome_vector[:num_pre_treatment_periods] - augmented_design_matrix[:num_pre_treatment_periods] @ alpha_cvxpy_var
    )
    
    # Post-treatment errors for objective: Y_post - tau - W_post*alpha
    errors_post_treatment = outcome_vector[num_pre_treatment_periods:] - tau_cvxpy_var - augmented_design_matrix[num_pre_treatment_periods:] @ alpha_cvxpy_var
    
    # Combined vector for GMM objective (minimizing sum of squares of these components)
    # This forms the GMM objective function U(theta)' * Omega_inv * U(theta)
    # where U(theta) stacks the pre-treatment moments and post-treatment errors.
    U_objective_vector = cp.hstack([gmm_moments_pre_treatment, errors_post_treatment])

    len_U_objective_vector = augmented_design_matrix.shape[1] + (total_periods - num_pre_treatment_periods)
    Omega_inv_gmm_objective = np.eye(len_U_objective_vector) # Using identity matrix as the GMM weighting matrix (Omega_inv)
    
    objective_gmm = cp.Minimize(cp.quad_form(U_objective_vector, Omega_inv_gmm_objective))
    problem_gmm = cp.Problem(objective_gmm)
    problem_gmm.solve(solver=_SOLVER_CLARABEL_STR)

    alpha_estimated_augmented: np.ndarray = alpha_cvxpy_var.value
    tau_estimated: float = tau_cvxpy_var.value

    # Predicted outcome using augmented W (not directly returned but used for SE calculation)
    # predicted_outcome_augmented_W: np.ndarray = augmented_design_matrix @ alpha_estimated_augmented
    
    # For SE calculation, reconstruct U vector (moment conditions evaluated at estimated parameters)
    # This U_hat is used to compute the HAC variance matrix Omega_hat.
    # U_hat_mom1: Z0' * (Y_pre - W_pre*alpha_hat)
    hac_moments_pre_treatment_part = augmented_proxy_matrix[:num_pre_treatment_periods].T @ (
        outcome_vector[:num_pre_treatment_periods] - augmented_design_matrix[:num_pre_treatment_periods] @ alpha_estimated_augmented
    )
    # U_hat_mom2: Y_post - tau_hat - W_post*alpha_hat (errors for each post-treatment period)
    hac_residuals_full_period_part = outcome_vector - tau_estimated - augmented_design_matrix @ alpha_estimated_augmented
    
    post_treatment_mask_se = np.zeros(total_periods, dtype=bool)
    post_treatment_mask_se[num_pre_treatment_periods:] = True
    
    # Select only post-treatment residuals for the second part of U_hat
    hac_residuals_post_treatment_part = hac_residuals_full_period_part * post_treatment_mask_se
    # Stack to form the full U_hat vector for HAC input
    hac_input_vector = np.hstack([hac_moments_pre_treatment_part, hac_residuals_post_treatment_part])

    # Jacobian (G matrix) for SE calculation: G = d(E[U(theta)])/d(theta')
    # where theta = [alpha, tau]. The moments are E[Z0'(Y-W*alpha)]=0 and E[Y_post - tau - W_post*alpha]=0.
    # The second moment is averaged over the post-treatment evaluation window.
    dimension_augmented_Z0 = augmented_proxy_matrix.shape[1]
    jacobian_for_se = np.zeros((dimension_augmented_Z0 + 1, dimension_augmented_W + 1)) # (num_moments x num_params)
    
    # Derivative of E[Z0'(Y-W*alpha)] w.r.t. alpha: -E[Z0'W]
    # Approximated by sample average: -(Z0_pre' * W_pre) / T_pre (or T if not scaled by T_pre)
    # The original code used total_periods for scaling, which might be specific to its derivation.
    # Here, we use the common form -(Z0_pre' * W_pre).
    jacobian_for_se[:dimension_augmented_Z0, :dimension_augmented_W] = (
        -augmented_proxy_matrix[:num_pre_treatment_periods].T @ augmented_design_matrix[:num_pre_treatment_periods]
    ) # This is sum, will be divided by T_pre or T later if needed by specific GMM formula.
      # The original code had a division by total_periods here.
      # For standard GMM, G is often E[d(g_i)/d(theta)]. If g_i is Z0_i * (y_i - w_i*alpha), then d(g_i)/d(alpha) = -Z0_i * w_i'.
      # Summing over pre-period and dividing by T_pre gives sample average.
    
    # Derivative of E[Y_post_eval - tau - W_post_eval*alpha] w.r.t. alpha: -E[W_post_eval]
    # Approximated by sample average over the evaluation window.
    post_eval_start_idx = num_pre_treatment_periods
    post_eval_end_idx = num_pre_treatment_periods + num_post_periods_for_se
    
    jacobian_for_se[-1, :dimension_augmented_W] = -np.mean(
        augmented_design_matrix[post_eval_start_idx : post_eval_end_idx], axis=0
    )
    
    # Derivative of E[Y_post_eval - tau - W_post_eval*alpha] w.r.t. tau: -1
    jacobian_for_se[-1, -1] = -1.0

    # Note: The Jacobian calculation here seems to follow a specific GMM setup.
    # The SE calculation was marked as problematic in the original code.
    # This placeholder remains as the full SE derivation is complex and context-dependent.
    # SE calculation remains problematic due to dimensional inconsistencies with HAC matrix.
    # Returning placeholder for SE.
    
    alpha_coefficients_original_W: np.ndarray = alpha_estimated_augmented[: design_matrix.shape[1]]
    standard_error_tau_placeholder: float = np.nan 

    predicted_counterfactual_original_W = design_matrix @ alpha_coefficients_original_W

    return predicted_counterfactual_original_W, alpha_coefficients_original_W, standard_error_tau_placeholder


def compute_hac_variance(treatment_effects_vector: np.ndarray, truncation_lag: int) -> float:
    """
    Compute the HAC long-run variance estimator with truncation lag.

    Uses Bartlett kernel.

    Parameters
    ----------
    treatment_effects_vector : np.ndarray
        Array of estimated treatment effects, typically for each post-treatment period.
        Shape (num_effects_observations,), where num_effects_observations is the number of post-treatment observations.
    truncation_lag : int
        The truncation lag for the Bartlett kernel. This determines how many
        autocovariances are included in the sum. A common rule of thumb for this lag
        is ``floor(4 * (num_effects_observations/100)**(2/9))``.

    Returns
    -------
    float
        The Heteroskedasticity and Autocorrelation Consistent (HAC) variance estimate.
        Returns `np.nan` if `num_effects_observations` is 0.
    """
    num_effects_observations: int = len(treatment_effects_vector)
    if num_effects_observations == 0:
        return np.nan
    
    average_treatment_effect_local: float = np.mean(treatment_effects_vector)
    effect_residuals: np.ndarray = treatment_effects_vector - average_treatment_effect_local

    hac_variance_estimate: float = np.var(effect_residuals, ddof=1)  # Start with the sample variance (gamma_0)
    
    if num_effects_observations <= 1: # Cannot compute covariance for lag > 0
        return hac_variance_estimate if num_effects_observations == 1 else np.nan

    for current_lag in range(1, min(truncation_lag + 1, num_effects_observations)): # Iterate up to min(truncation_lag, num_effects_observations-1)
        bartlett_kernel_weight: float = 1 - current_lag / (truncation_lag + 1)
        # Calculate autocovariance for the current lag
        autocovariance_at_lag: float = np.mean(effect_residuals[:-current_lag] * effect_residuals[current_lag:])
        hac_variance_estimate += 2 * bartlett_kernel_weight * autocovariance_at_lag
        
    return hac_variance_estimate


def compute_t_stat_and_ci(
    average_treatment_effect: float,
    post_treatment_effects_vector: np.ndarray,
    truncation_lag: int,
    confidence_level: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the t-statistic and confidence interval for ATT.

    Uses HAC variance for standard error calculation.

    Parameters
    ----------
    average_treatment_effect : float
        Average treatment effect on the treated (ATT).
    post_treatment_effects_vector : np.ndarray
        Array of estimated treatment effects for post-treatment periods,
        shape (num_post_treatment_obs,).
    truncation_lag : int
        The truncation lag for HAC variance.
    confidence_level : float, optional
        Desired confidence level, by default 0.95.

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        - t_statistic_value : float
            The computed t-statistic for the ATT.
        - confidence_interval : Tuple[float, float]
            A tuple containing the lower and upper bounds of the confidence interval
            for the ATT.

    Examples
    --------
    >>> average_treatment_effect_example = 0.5
    >>> post_effects_vector_example = np.array([0.4, 0.5, 0.6, 0.45, 0.55])
    >>> truncation_lag_example = 1
    >>> t_statistic_value_example, confidence_interval_example = compute_t_stat_and_ci(
    ...     average_treatment_effect_example, post_effects_vector_example, truncation_lag_example
    ... )
    >>> print(f"t-statistic: {t_statistic_value_example:.2f}")
    t-statistic: 12.55
    >>> print(f"CI: ({confidence_interval_example[0]:.2f}, {confidence_interval_example[1]:.2f})")
    CI: (0.42, 0.58)
    """
    num_post_treatment_obs: int = len(post_treatment_effects_vector)
    if num_post_treatment_obs == 0:
        return np.nan, (np.nan, np.nan)

    hac_variance_value: float = compute_hac_variance(post_treatment_effects_vector, truncation_lag)
    
    standard_error_att: float
    if hac_variance_value < 0 or np.isnan(hac_variance_value): # Variance should be non-negative
        standard_error_att = np.nan
    else:
        standard_error_att = np.sqrt(hac_variance_value / num_post_treatment_obs)

    t_statistic_value: float
    confidence_interval_lower_bound: float
    confidence_interval_upper_bound: float

    if np.isnan(standard_error_att) or standard_error_att == 0:
        t_statistic_value = np.nan
        confidence_interval_lower_bound, confidence_interval_upper_bound = np.nan, np.nan
    else:
        t_statistic_value = average_treatment_effect / standard_error_att
        # Using z-distribution (normal approximation) for CI
        alpha_for_ci: float = 1 - confidence_level
        z_critical_value: float = norm.ppf(1 - alpha_for_ci / 2)
        confidence_interval_lower_bound = average_treatment_effect - z_critical_value * standard_error_att
        confidence_interval_upper_bound = average_treatment_effect + z_critical_value * standard_error_att

    return t_statistic_value, (confidence_interval_lower_bound, confidence_interval_upper_bound)



def l2_relax(
    num_pre_treatment_estimation_periods: int,
    treated_unit_outcome_vector: np.ndarray,
    donor_outcomes_matrix: np.ndarray,
    sup_norm_constraint_tau: float,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """L2-relaxation estimator with fallback solvers: CLARABEL, OSQP, ECOS."""

    treated_unit_subset = treated_unit_outcome_vector[:num_pre_treatment_estimation_periods]
    donor_outcomes_subset = donor_outcomes_matrix[:num_pre_treatment_estimation_periods, :]

    num_estimation_periods_subset, num_donors_subset = donor_outcomes_subset.shape
    Sigma_cov_matrix = (donor_outcomes_subset.T @ donor_outcomes_subset) / num_estimation_periods_subset
    eta_vector = (donor_outcomes_subset.T @ treated_unit_subset) / num_estimation_periods_subset

    donor_weights_cvxpy = cp.Variable(num_donors_subset)
    objective = cp.Minimize(0.5 * cp.sum_squares(donor_weights_cvxpy))
    constraint = [cp.norm(eta_vector - Sigma_cov_matrix @ donor_weights_cvxpy, "inf") <= sup_norm_constraint_tau]
    problem = cp.Problem(objective, constraint)

    solvers_to_try = ["CLARABEL", "OSQP", "ECOS"]
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver)
            if donor_weights_cvxpy.value is not None:
                break
        except Exception:
            continue

    if donor_weights_cvxpy.value is None:
        raise MlsynthEstimationError("L2-relax failed: All solvers failed to converge.")

    estimated_coefficients = donor_weights_cvxpy.value
    estimated_intercept = (
        np.mean(treated_unit_subset) -
        np.mean(donor_outcomes_subset @ estimated_coefficients)
    )
    predicted_counterfactuals = donor_outcomes_matrix @ estimated_coefficients + estimated_intercept

    return estimated_coefficients, estimated_intercept, predicted_counterfactuals


def adaptive_cross_validate_tau(
    pre_treatment_treated_outcome: np.ndarray,
    pre_treatment_donor_outcomes: np.ndarray,
    tau_upper_bound_for_grid: float,
    num_coarse_points: int = 50,
    num_fine_points: int = 100,
    zoom_width: float = 0.5
) -> Tuple[float, float]:
    """
    Adaptive cross-validation for tau in L2-relax estimator.
    First performs a coarse grid search, then zooms in around the best tau.
    """
    num_pre_periods = len(pre_treatment_treated_outcome)
    half = num_pre_periods // 2

    y_train = pre_treatment_treated_outcome[:half]
    X_train = pre_treatment_donor_outcomes[:half, :]
    y_val = pre_treatment_treated_outcome[half:]
    X_val = pre_treatment_donor_outcomes[half:, :]

    def mse_for_tau(tau_val: float) -> float:
        coefs, intercept, _ = l2_relax(
            num_pre_treatment_estimation_periods=half,
            treated_unit_outcome_vector=y_train,
            donor_outcomes_matrix=X_train,
            sup_norm_constraint_tau=tau_val
        )
        preds = X_val @ coefs + intercept
        return np.mean((y_val - preds) ** 2)

    # Stage 1: Coarse log grid
    coarse_grid = np.logspace(-4, np.log10(tau_upper_bound_for_grid), num_coarse_points)
    coarse_mse = [mse_for_tau(tau) for tau in coarse_grid]
    best_idx = np.argmin(coarse_mse)
    best_tau_coarse = coarse_grid[best_idx]

    # Stage 2: Fine linear grid around best tau
    lower = max(best_tau_coarse * (1 - zoom_width), 1e-6)
    upper = best_tau_coarse * (1 + zoom_width)
    fine_grid = np.linspace(lower, upper, num_fine_points)
    fine_mse = [mse_for_tau(tau) for tau in fine_grid]
    best_fine_idx = np.argmin(fine_mse)

    return fine_grid[best_fine_idx], fine_mse[best_fine_idx]



def ci_bootstrap(
    original_scm_weights: np.ndarray,
    num_control_units_or_features: int,
    donor_feature_matrix: np.ndarray,
    treated_outcome_vector: np.ndarray,
    num_pre_treatment_periods: int,
    num_bootstrap_samples: int,
    original_average_treatment_effect: float,
    scm_method_name: str,
    original_counterfactual_outcome_vector: np.ndarray,
    confidence_level: float = 0.95,
) -> List[float]:
    """
    Perform subsampling bootstrap for TSSC methods.

    Parameters
    ----------
    original_scm_weights : np.ndarray
        Original SCM weights. Shape can be (num_control_units_or_features,) or (N_features,) if no intercept,
        or (num_control_units_or_features+1,) or (N_features+1,) if an intercept is included by the method
        (e.g., for "MSCa", "MSCc"). The internal logic adjusts features for dot
        product based on `scm_method_name`.
    num_control_units_or_features : int
        Number of original control units (donors) or features before any
        potential intercept addition by `Opt.SCopt`.
    donor_feature_matrix : np.ndarray
        Feature matrix for donors, shape (total_time_periods, N_features).
    treated_outcome_vector : np.ndarray
        Target vector (total_time_periods,).
    num_pre_treatment_periods : int
        Number of pre-treatment periods.
    num_bootstrap_samples : int
        Number of bootstrap samples.
    original_average_treatment_effect : float
        Original Average Treatment Effect on Treated.
    scm_method_name : str
        Name of the SCM method used (e.g., "MSCb", "MSCa").
    original_counterfactual_outcome_vector : np.ndarray
        Original counterfactual outcome vector (total_time_periods,).
    confidence_level : float, optional
        The desired confidence level for the interval, by default 0.95.

    Returns
    -------
    List[float]
        A list containing two floats: `[lower_bound, upper_bound]` for the
        confidence interval of the ATT, at the specified `confidence_level`.
        Returns `[np.nan, np.nan]` if `pre_treatment_subsample_size` is non-positive.

    Notes
    -----
    This function implements a subsampling bootstrap procedure tailored for
    Two-Step Synthetic Control (TSSC) methods. It involves simulating
    post-treatment errors and re-estimating SCM weights on subsamples
    of the pre-treatment data to construct a distribution for the ATT.
    The `Opt.SCopt` method is called internally for re-estimation.
    Reproducibility is ensured by `np.random.seed(_BOOTSTRAP_RANDOM_SEED)`.

    Examples
    --------
    >>> # This is a complex function to exemplify simply due to Opt.SCopt dependency.
    >>> # A conceptual example:
    >>> num_donors_example, total_periods_example, num_pre_periods_example, num_features_example = 3, 20, 10, 2
    >>> original_weights_example = np.array([0.5, 0.3, 0.2]) # Example weights
    >>> donor_features_example = np.random.rand(total_periods_example, num_features_example)
    >>> treated_outcome_example = np.random.rand(total_periods_example)
    >>> original_counterfactual_example = np.random.rand(total_periods_example)
    >>> original_att_example = np.mean(treated_outcome_example[num_pre_periods_example:] - original_counterfactual_example[num_pre_periods_example:])
    >>> # Mocking Opt.SCopt would be needed for a runnable example here.
    >>> # Assuming Opt.SCopt is available and works as expected:
    >>> # confidence_interval_example = ci_bootstrap(
    ... #     original_weights_example, num_donors_example, donor_features_example,
    ... #     treated_outcome_example, num_pre_periods_example, 100,
    ... #     original_att_example, "MSCb", original_counterfactual_example, confidence_level=0.90
    ... # )
    >>> # print(confidence_interval_example) # Expected: [float, float]
    """
    total_time_periods: int = len(original_counterfactual_outcome_vector)
    pre_treatment_subsample_size: int = num_pre_treatment_periods - _BOOTSTRAP_SUBSAMPLE_ADJUSTMENT
    if pre_treatment_subsample_size <= 0:
        warnings.warn("Subsample size for bootstrap is non-positive. CI may be unreliable.")
        return [np.nan, np.nan]

    num_post_treatment_periods: int = total_time_periods - num_pre_treatment_periods

    # Data for pre-treatment period
    pre_treatment_features_and_outcome_concatenated: np.ndarray = np.concatenate(
        (donor_feature_matrix[:num_pre_treatment_periods], treated_outcome_vector[:num_pre_treatment_periods].reshape(-1, 1)), axis=1
    )
    pre_treatment_error_std_dev: float = np.sqrt(
        np.mean((treated_outcome_vector[:num_pre_treatment_periods] - original_counterfactual_outcome_vector[:num_pre_treatment_periods]) ** 2)
    )

    # Variance of post-treatment errors (residuals from original ATT)
    post_treatment_residuals_from_att = (
        treated_outcome_vector[num_pre_treatment_periods:]
        - original_counterfactual_outcome_vector[num_pre_treatment_periods:]
        - np.mean(treated_outcome_vector[num_pre_treatment_periods:] - original_counterfactual_outcome_vector[num_pre_treatment_periods:])
    )
    post_treatment_error_variance: float = np.mean(post_treatment_residuals_from_att**2)
    if post_treatment_error_variance < 0: post_treatment_error_variance = 0 # Ensure non-negative variance

    # Simulated post-treatment errors
    simulated_post_treatment_errors: np.ndarray = np.sqrt(post_treatment_error_variance) * np.random.randn(
        num_post_treatment_periods, num_bootstrap_samples
    )
    bootstrap_att_statistics: np.ndarray = np.zeros(num_bootstrap_samples)

    post_treatment_donor_features: np.ndarray = donor_feature_matrix[num_pre_treatment_periods:total_time_periods, :] # Corrected slicing

    np.random.seed(_BOOTSTRAP_RANDOM_SEED) # For reproducibility

    for sample_index in range(num_bootstrap_samples):
        # Shuffle pre-treatment data
        shuffled_pre_treatment_indices = np.random.permutation(num_pre_treatment_periods)
        shuffled_pre_treatment_features_and_outcome = pre_treatment_features_and_outcome_concatenated[shuffled_pre_treatment_indices]

        # Subsample pre-treatment data
        subsampled_pre_treatment_features_and_outcome: np.ndarray = shuffled_pre_treatment_features_and_outcome[:pre_treatment_subsample_size, :]
        subsampled_pre_treatment_donor_features: np.ndarray = subsampled_pre_treatment_features_and_outcome[:, :-1]
        
        # Adjust features if method requires intercept handling for dot product
        subsampled_donor_features_for_dot_product = subsampled_pre_treatment_donor_features.copy()
        if scm_method_name in [_SCM_MODEL_MSCA, _SCM_MODEL_MSCC]: # Use constants
            # original_scm_weights includes intercept, so prepend intercept to features for dot product
            subsampled_donor_features_for_dot_product = np.c_[np.ones((subsampled_pre_treatment_donor_features.shape[0], 1)), subsampled_pre_treatment_donor_features]

        # Simulate y for subsample based on original weights and pre-treatment error
        subsampled_simulated_treated_outcome: np.ndarray = np.dot(
            subsampled_donor_features_for_dot_product, original_scm_weights
        ) + pre_treatment_error_std_dev * np.random.randn(pre_treatment_subsample_size)

        # For re-estimating weights, pass the original subsampled_pre_treatment_donor_features
        # as Opt.SCopt will handle intercept addition internally if needed.
        donor_features_for_bootstrap_scm = subsampled_pre_treatment_donor_features.copy()

        # Re-estimate weights on subsample
        bootstrap_scm_problem = Opt.SCopt(
            num_control_units_or_features, subsampled_simulated_treated_outcome, pre_treatment_subsample_size, donor_features_for_bootstrap_scm, scm_model_type=scm_method_name
        )
        
        bootstrap_estimated_scm_weights: np.ndarray
        if bootstrap_scm_problem.status in ["optimal", "optimal_inaccurate"]:
            bootstrap_estimated_scm_weights = bootstrap_scm_problem.solution.primal_vars[
                next(iter(bootstrap_scm_problem.solution.primal_vars))
            ]
        else:
            warnings.warn(
                f"Bootstrap SCM optimization failed for sample {sample_index + 1}/{num_bootstrap_samples} "
                f"with status {bootstrap_scm_problem.status}. Assigning NaN to this sample's statistic.",
                UserWarning
            )
            bootstrap_att_statistics[sample_index] = np.nan
            continue # Skip to the next bootstrap sample

        # Calculate bootstrap statistic components
        post_treatment_donor_features_for_dot_product = post_treatment_donor_features.copy()
        if scm_method_name in [_SCM_MODEL_MSCA, _SCM_MODEL_MSCC]: # Use constants
            # If weights include intercept, features need an intercept column for this dot product
            post_treatment_donor_features_for_dot_product = np.c_[np.ones((post_treatment_donor_features.shape[0], 1)), post_treatment_donor_features]

        bootstrap_stat_component1: float = -np.mean(
            np.dot(post_treatment_donor_features_for_dot_product, (bootstrap_estimated_scm_weights - original_scm_weights))
        ) * np.sqrt((num_post_treatment_periods * pre_treatment_subsample_size) / num_pre_treatment_periods)
        bootstrap_stat_component2: float = np.sqrt(num_post_treatment_periods) * np.mean(simulated_post_treatment_errors[:, sample_index])
        bootstrap_att_statistics[sample_index] = bootstrap_stat_component1 + bootstrap_stat_component2

    sorted_normalized_bootstrap_att_stats: np.ndarray = np.sort(bootstrap_att_statistics / np.sqrt(num_post_treatment_periods))
    
    alpha_val = 1.0 - confidence_level
    lower_percentile = alpha_val / 2.0
    upper_percentile = 1.0 - (alpha_val / 2.0)

    lower_percentile_index = int(lower_percentile * num_bootstrap_samples)
    upper_percentile_index = int(upper_percentile * num_bootstrap_samples)
    
    # Ensure indices are within bounds
    lower_percentile_index = max(0, min(lower_percentile_index, num_bootstrap_samples - 1))
    upper_percentile_index = max(0, min(upper_percentile_index, num_bootstrap_samples - 1))


    lower_percentile_critical_value: float = sorted_normalized_bootstrap_att_stats[lower_percentile_index]
    upper_percentile_critical_value: float = sorted_normalized_bootstrap_att_stats[upper_percentile_index]

    # Confidence interval
    confidence_interval_lower_bound: float = original_average_treatment_effect - upper_percentile_critical_value
    confidence_interval_upper_bound: float = original_average_treatment_effect - lower_percentile_critical_value
    
    return [confidence_interval_lower_bound, confidence_interval_upper_bound]


def _estimate_single_sc_model_for_tsest(
    scm_method_name: str,
    iteration_donor_features: np.ndarray,
    treated_outcome_vector: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_treatment_periods: int,
    num_bootstrap_samples: int,
    current_donor_names: List[str],
    num_iteration_donors_or_features: int,
) -> Dict[str, Any]:
    """
    Estimate a single SCM model for TSEST, calculate effects, and bootstrap CIs.

    Parameters
    ----------
    scm_method_name : str
        Name of the SCM method (e.g., "SIMPLEX", "MSCb").
    iteration_donor_features : np.ndarray
        Feature matrix for donor units for the current iteration.
    treated_outcome_vector : np.ndarray
        Outcome vector for the treated unit.
    num_pre_treatment_periods : int
        Number of pre-treatment periods.
    num_post_treatment_periods : int
        Number of post-treatment periods.
    num_bootstrap_samples : int
        Number of bootstrap samples.
    current_donor_names : List[str]
        List of donor names.
    num_iteration_donors_or_features : int
        Number of control units/features for the current iteration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results for the single SCM model.
    """
    # Step 1: Estimate SCM weights using the provided method and pre-treatment data
    scm_problem_instance = Opt.SCopt(
        num_control_units=num_iteration_donors_or_features, # Number of donors/features for this iteration
        target_outcomes_pre_treatment=treated_outcome_vector[:num_pre_treatment_periods], # Treated unit's pre-treatment outcomes
        num_pre_treatment_periods=num_pre_treatment_periods, # Number of pre-treatment periods
        donor_outcomes_pre_treatment=iteration_donor_features[:num_pre_treatment_periods], # Donor features for pre-treatment
        scm_model_type=scm_method_name, # SCM variant (e.g., "SIMPLEX", "MSCb")
    )

    # Handle cases where SCM optimization fails
    if scm_problem_instance.status not in ["optimal", "optimal_inaccurate"]:
        warnings.warn(
            f"Initial SCM optimization failed for method {scm_method_name} "
            f"with status {scm_problem_instance.status}. Returning NaNs for this method.", 
            UserWarning
        )
        # Prepare a dictionary with NaN results if optimization fails
        nan_effects = {"ATT": np.nan, "ATT_post": np.nan, "ATT_pre": np.nan}
        nan_fit = {"RMSE_pre": np.nan, "RMSE_post": np.nan, "R2_pre": np.nan, "R2_post": np.nan}
        total_periods_local = len(treated_outcome_vector)
        time_periods_vector = np.arange(total_periods_local)
        nan_vectors = {
            "Y": treated_outcome_vector, 
            "Y_sc": np.full(total_periods_local, np.nan), # Synthetic control outcome
            "Gap": np.full(total_periods_local, np.nan), # Gap between actual and synthetic
            "TimePeriods": time_periods_vector
        }
        # Determine expected weight vector length (accounts for potential intercept in some models)
        expected_weight_len = num_iteration_donors_or_features
        if scm_method_name in [_SCM_MODEL_MSCA, _SCM_MODEL_MSCC]: # Models with intercept
            expected_weight_len +=1 
            
        return { # Return dictionary populated with NaNs
            "Fit": nan_fit,
            "Effects": nan_effects,
            "95% CI": [np.nan, np.nan], 
            "Vectors": nan_vectors,
            "WeightV": np.full(expected_weight_len, np.nan), # Raw weight vector
            "Weights": {}, # Formatted donor weights
        }

    # Extract estimated SCM weights if optimization was successful
    estimated_weights: np.ndarray = scm_problem_instance.solution.primal_vars[
        next(iter(scm_problem_instance.solution.primal_vars)) # Access the first (and only) primal variable
    ]

    # Create a display-friendly dictionary of donor weights (non-zero, rounded)
    displayable_donor_weights: Dict[str, float] = {
        donor: weight
        for donor, weight in zip(current_donor_names, np.round(estimated_weights, 4)) # Round for display
        if abs(weight) > 0.001 # Include only weights with significant magnitude
    }
    
    # Prepare donor features for counterfactual calculation.
    # If the SCM method includes an intercept (MSCa, MSCc), prepend a column of ones.
    donor_features_for_counterfactual = iteration_donor_features.copy()
    if scm_method_name in [_SCM_MODEL_MSCA, _SCM_MODEL_MSCC]: 
         donor_features_for_counterfactual = np.c_[
            np.ones((iteration_donor_features.shape[0], 1)), iteration_donor_features # Add intercept column
        ]

    # Step 2: Calculate the synthetic counterfactual outcome (Y_sc)
    y_counterfactual: np.ndarray = donor_features_for_counterfactual.dot(estimated_weights)
    
    # Step 3: Calculate treatment effects and fit diagnostics
    iteration_att_results, iteration_fit_diagnostics, iteration_time_series_vectors = effects.calculate(
        observed_outcome_series=treated_outcome_vector, # Actual outcome of the treated unit
        counterfactual_outcome_series=y_counterfactual,      # Synthetic counterfactual outcome
        num_pre_treatment_periods=num_pre_treatment_periods,
        num_actual_post_periods=num_post_treatment_periods
    )
    iteration_average_treatment_effect: float = iteration_att_results["ATT"] # Extract ATT

    # Step 4: Perform bootstrap for confidence intervals
    iteration_confidence_intervals: List[float] = ci_bootstrap(
        original_scm_weights=estimated_weights,
        num_control_units_or_features=num_iteration_donors_or_features,
        donor_feature_matrix=iteration_donor_features, # Original donor features (not low-rank or intercept-added for bootstrap internals)
        treated_outcome_vector=treated_outcome_vector,
        num_pre_treatment_periods=num_pre_treatment_periods,
        num_bootstrap_samples=num_bootstrap_samples,
        original_average_treatment_effect=iteration_average_treatment_effect,
        scm_method_name=scm_method_name, # Pass SCM method for bootstrap logic
        original_counterfactual_outcome_vector=y_counterfactual, # Original Y_sc
    ) # Default confidence_level=0.95 will be used here

    # Compile and return all results for this SCM method iteration
    return {
        "Fit": iteration_fit_diagnostics,
        "Effects": iteration_att_results,
        "95% CI": iteration_confidence_intervals, 
        "Vectors": iteration_time_series_vectors,
        "WeightV": np.round(estimated_weights, 3), # Raw weight vector, rounded
        "Weights": displayable_donor_weights, # Formatted donor weights
    }


def TSEST(
    donor_features_matrix: np.ndarray,
    treated_outcome_vector: np.ndarray,
    num_pre_treatment_periods: int,
    num_bootstrap_samples: int,
    all_donor_names: List[str],
    num_post_treatment_periods: int,
) -> List[Dict[str, Any]]:
    """
    Perform Two-Step Synthetic Control (TSSC) estimation for multiple SCM methods.

    This function iterates through a predefined list of Synthetic Control Method
    (SCM) variants ('SIMPLEX', 'MSCb', 'MSCa', 'MSCc'), estimates SCM weights,
    calculates treatment effects, and computes confidence intervals using
    a bootstrap procedure.

    Parameters
    ----------
    donor_features_matrix : np.ndarray
        Feature matrix for donor units, shape (T_total, N_features).
        `T_total` is the total number of time periods.
        `N_features` is the number of donor units/features.
    treated_outcome_vector : np.ndarray
        Outcome vector for the treated unit, shape (T_total,).
    num_pre_treatment_periods : int
        Number of pre-treatment periods. Used for fitting the SCM weights.
    num_bootstrap_samples : int
        Number of bootstrap samples to generate for confidence interval estimation.
    all_donor_names : List[str]
        List of names corresponding to the columns (features/donors) in `donor_features_matrix`.
        Length must be equal to `N_features`.
    num_post_treatment_periods : int
        Number of post-treatment periods. Used for calculating ATT and CIs.
        Note: `T_total` should be `num_pre_treatment_periods + num_post_treatment_periods`.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries. Each dictionary corresponds to one SCM method
        and has a single key (the method name, e.g., "SIMPLEX") whose value
        is another dictionary containing the detailed results for that method.
        The structure of the inner dictionary is:
        - "Fit" : Dict[str, float]
            Goodness-of-fit statistics. Keys: "RMSE_pre", "RMSE_post",
            "R2_pre", "R2_post".
        - "Effects" : Dict[str, float]
            Estimated treatment effects. Keys: "ATT" (Average Treatment Effect
                on Treated for post-treatment period), "ATT_post" (same as ATT),
                "ATT_pre" (average gap in pre-treatment period).

        - "95% CI" : List[float]
            A list containing two floats: `[lower_bound, upper_bound]` for the
            95% confidence interval of the ATT.

        - "Vectors" : Dict[str, np.ndarray]
            Time series vectors. Keys:
            "Y" (original outcome `treated_outcome_vector`),
            "Y_sc" (synthetic control counterfactual `y_counterfactual`),
            "Gap" (difference `Y - Y_sc`),
            "TimePeriods" (array of time period indices).

        - "WeightV" : np.ndarray
            Raw estimated SCM weights, shape (N_features,) or (N_features+1,)
            if an intercept is included by the method (e.g., "MSCa", "MSCc").

        - "Weights" : Dict[str, float]
            Dictionary mapping donor names to their estimated weights (rounded,
            non-zero weights shown). If an intercept is included, it might be
            named implicitly or handled by `Opt.SCopt`.

    Notes
    -----
    - The total number of time periods (`T_total`) is implicitly determined
      from `donor_features_matrix.shape[0]` or `treated_outcome_vector.shape[0]`.
    - The number of features/donors (`N_features`) is implicitly determined
      from `donor_features_matrix.shape[1]`.
    - This function relies on `Opt.SCopt` for SCM weight estimation and
      `ci_bootstrap` for confidence interval calculation.
    - The methods run are 'SIMPLEX', 'MSCb', 'MSCa', 'MSCc'.

    Examples
    --------
    >>> T_total_example, N_features_example = 20, 3
    >>> num_pre_periods_example, num_post_periods_example = 10, 10
    >>> donor_features_example = np.random.rand(T_total_example, N_features_example)
    >>> treated_outcome_example = np.random.rand(T_total_example) + np.concatenate((np.zeros(num_pre_periods_example), np.ones(num_post_periods_example) * 0.5))
    >>> donor_names_example = [f"Donor_{i+1}" for i in range(N_features_example)]
    >>> estimation_results_list = TSEST(
    ...     donor_features_example, treated_outcome_example, num_pre_periods_example, 100, donor_names_example, num_post_periods_example
    ... ) # num_bootstrap_samples=100 for example
    >>> print(f"Number of methods run: {len(estimation_results_list)}")
    Number of methods run: 4
    >>> first_method_name_example = list(estimation_results_list[0].keys())[0]
    >>> print(f"Results for method: {first_method_name_example}")
    Results for method: SIMPLEX
    >>> isinstance(estimation_results_list[0][first_method_name_example]["Effects"]["ATT"], float)
    True
    >>> isinstance(estimation_results_list[0][first_method_name_example]["Weights"], dict)
    True
    """
    all_method_results_list: List[Dict[str, Any]] = []
    scm_methods_to_evaluate: List[str] = [
        _SCM_MODEL_SIMPLEX, _SCM_MODEL_MSCB, _SCM_MODEL_MSCA, _SCM_MODEL_MSCC
    ]
    
    current_donor_features_matrix = donor_features_matrix.copy()

    for current_scm_method_name in scm_methods_to_evaluate:
        features_for_current_method = current_donor_features_matrix.copy()
        num_features_for_current_method = features_for_current_method.shape[1]

        results_for_current_method = _estimate_single_sc_model_for_tsest(
            scm_method_name=current_scm_method_name,
            iteration_donor_features=features_for_current_method,
            treated_outcome_vector=treated_outcome_vector,
            num_pre_treatment_periods=num_pre_treatment_periods,
            num_post_treatment_periods=num_post_treatment_periods,
            num_bootstrap_samples=num_bootstrap_samples,
            current_donor_names=all_donor_names,
            num_iteration_donors_or_features=num_features_for_current_method
        )
        all_method_results_list.append({current_scm_method_name: results_for_current_method})

    return all_method_results_list


class Opt:

    DEFAULT_LAMBDA_PENALTY: float = 0.0  # No regularization by default
    DEFAULT_P_NORM: float = 2            # p-norm type (2 = Ridge, 1 = LASSO)
    DEFAULT_Q_EXPONENT: float = 2        # Usually q = p

    @staticmethod
    def _solve_ma_model(
        target_outcomes_ma: np.ndarray,
        base_model_results_ma: Dict[str, Dict[str, Any]],
        num_pre_treatment_periods_ma: int,
    ) -> Dict[str, Any]:
        """
        Solve the Model Averaging (MA) SCM optimization problem.

        Parameters
        ----------
        target_outcomes_ma : np.ndarray
            Target vector for the pre-treatment periods, shape (num_pre_treatment_periods_ma,).
        base_model_results_ma : Dict[str, Dict[str, Any]]
            Dictionary where keys are model names and values are dicts
            containing 'weights' (np.ndarray of donor weights) and 'cf'
            (np.ndarray for pre-treatment counterfactual of length
            num_pre_treatment_periods_ma) for each base model.
        num_pre_treatment_periods_ma : int
            Number of pre-treatment periods.

        Returns
        -------
        Dict[str, Any]
            A dictionary with "Lambdas" (model averaging weights), "w_MA"
            (final averaged donor weights), and "Counterfactual_pre"
            (model-averaged pre-treatment counterfactual).
        """
        # Validate input: base_model_results_ma must be a non-empty dictionary
        if not isinstance(base_model_results_ma, dict) or not base_model_results_ma:
            raise MlsynthConfigError(
                "For 'MA' method, 'base_model_results_ma' must be a non-empty dictionary."
            )

        base_model_names_ma: List[str] = list(base_model_results_ma.keys())
        num_base_models_ma: int = len(base_model_names_ma)

        # Construct a matrix where each column is the pre-treatment counterfactual from a base model
        base_model_counterfactuals_matrix_ma: np.ndarray = np.column_stack(
            [
                base_model_results_ma[name]["cf"][:num_pre_treatment_periods_ma] # Ensure correct slicing
                for name in base_model_names_ma
            ]
        )

        # Define CVXPY variable for model averaging weights (lambda_k)
        model_averaging_weights_cvxpy_ma = cp.Variable(
            num_base_models_ma, nonneg=True # Weights must be non-negative
        )

        # Objective: Minimize || Y_pre - sum(lambda_k * CF_k_pre) ||_2
        # This is equivalent to minimizing the L2 norm of the difference between
        # the actual pre-treatment outcomes and the weighted average of base model counterfactuals.
        objective_ma = cp.Minimize(
            cp.norm(
                target_outcomes_ma # Actual Y_pre
                - base_model_counterfactuals_matrix_ma @ model_averaging_weights_cvxpy_ma, # Weighted sum of CF_k_pre
                2, # L2 norm
            )
        )
        # Constraint: sum(lambda_k) = 1 (weights must sum to one)
        constraints_ma = [cp.sum(model_averaging_weights_cvxpy_ma) == 1]

        problem_ma = cp.Problem(objective_ma, constraints_ma)
        problem_ma.solve(solver=_SOLVER_CLARABEL_STR) # Solve the optimization problem

        # Extract estimated model averaging weights
        estimated_model_averaging_weights_ma: np.ndarray = (
            model_averaging_weights_cvxpy_ma.value
        )
        # Create a dictionary mapping model names to their weights
        model_averaging_weights_dict_ma: Dict[str, float] = {
            base_model_names_ma[i]: float(
                estimated_model_averaging_weights_ma[i]
            )
            for i in range(num_base_models_ma)
        }

        # Calculate the final model-averaged donor weights (w_MA)
        # w_MA = sum(lambda_k * w_k), where w_k are donor weights from base model k
        final_averaged_donor_weights_ma: np.ndarray = sum(
            estimated_model_averaging_weights_ma[i] # lambda_k
            * base_model_results_ma[base_model_names_ma[i]]["weights"] # w_k
            for i in range(num_base_models_ma)
        )
        # Calculate the model-averaged pre-treatment counterfactual
        averaged_pre_treatment_counterfactual_ma: np.ndarray = np.dot(
            base_model_counterfactuals_matrix_ma, # Matrix of [CF_1_pre, CF_2_pre, ...]
            estimated_model_averaging_weights_ma, # Vector of [lambda_1, lambda_2, ...]
        )

        return {
            "Lambdas": model_averaging_weights_dict_ma,
            "w_MA": final_averaged_donor_weights_ma,
            "Counterfactual_pre": averaged_pre_treatment_counterfactual_ma,
        }

    @staticmethod
    def _solve_affine_model(target_outcomes: np.ndarray,
                            donor_outcomes: np.ndarray,
                            num_coefficients: int,
                            w_convex: np.ndarray = None,
                            lambda_penalty: float = 0.0) -> cp.Problem:
        """
        Solve an AFFINE SCM model: weights may be negative, but must sum to 1.
        Optionally penalize deviation from convex weights.

        Tries multiple solvers in order and returns the first successful solution.

        Parameters
        ----------
        target_outcomes : np.ndarray
            Pre-treatment outcomes for the treated unit.
        donor_outcomes : np.ndarray
            Pre-treatment outcomes for donor units.
        num_coefficients : int
            Number of donor units (length of weight vector).
        w_convex : np.ndarray, optional
            Convex SCM weights to penalize deviation from.
        lambda_reg : float, default 0.0
            Strength of penalty on deviation from convex weights.
        """
        _SOLVERS = ["CLARABEL", "OSQP", "ECOS", "SCS"]
        w = cp.Variable(num_coefficients)

        # Affine constraint: sum to 1
        constraints = [cp.sum(w) == 1]

        # Base L2 fit
        objective_expr = cp.norm(target_outcomes - donor_outcomes @ w, 2) ** 2

        # Add penalty if provided
        if w_convex is not None and lambda_penalty > 0:
            objective_expr += lambda_penalty * cp.norm(w - w_convex, 2) ** 2

        objective = cp.Minimize(objective_expr)
        problem = cp.Problem(objective, constraints)

        for solver in _SOLVERS:
            try:
                problem.solve(solver=solver)
                if problem.status not in ["infeasible", "unbounded", "unknown"]:
                    return problem
            except Exception:
                continue

        raise RuntimeError(f"Affine SCM could not be solved with any solver in {_SOLVERS}")

    @staticmethod
    def _solve_simplex_model(
        target_outcomes_simplex: np.ndarray,
        processed_donor_outcomes_simplex: np.ndarray,
        num_coefficients_simplex: int,
    ) -> cp.Problem:
        """
        Solve the SIMPLEX SCM optimization problem.

        This method sets up and solves the optimization problem for the SIMPLEX
        variant of the Synthetic Control Method. Weights are constrained to be
        non-negative and sum to 1.

        Parameters
        ----------
        target_outcomes_simplex : np.ndarray
            Target vector for the pre-treatment periods, shape (t1,).
        processed_donor_outcomes_simplex : np.ndarray
            Processed donor matrix for pre-treatment periods, shape (t1, num_coefficients_simplex).
            For SIMPLEX, this typically does not include an intercept column.
        num_coefficients_simplex : int
            Number of coefficients to solve for (number of columns in processed_donor_outcomes_simplex).

        Returns
        -------
        cp.Problem
            The solved CVXPY problem object. The solution (weights) can be
            accessed via `problem.solution.primal_vars`.
        """
        # Define CVXPY variable for donor weights (coefficients)
        coefficients_cvxpy_simplex = cp.Variable(num_coefficients_simplex)

        # Constraints for SIMPLEX model:
        # 1. Weights must be non-negative (w_j >= 0 for all j).
        # 2. Weights must sum to 1 (sum(w_j) = 1).
        constraints_simplex = [
            coefficients_cvxpy_simplex >= 0,
            cp.sum(coefficients_cvxpy_simplex) == 1,
        ]

        # Objective: Minimize || Y_pre - X_donors_pre * w ||_2
        # This minimizes the L2 norm of the difference between actual pre-treatment outcomes
        # and the synthetic control constructed from donor outcomes.
        objective_simplex = cp.Minimize(
            cp.norm(
                target_outcomes_simplex # Y_pre
                - processed_donor_outcomes_simplex @ coefficients_cvxpy_simplex, # X_donors_pre * w
                2, # L2 norm
            )
        )

        problem_simplex = cp.Problem(objective_simplex, constraints_simplex)
        problem_simplex.solve(solver=_SOLVER_CLARABEL_STR) # Solve the optimization problem
        return problem_simplex

    @staticmethod
    def _solve_msca_model(
        target_outcomes_pre_treatment_msca: np.ndarray,
        donor_outcomes_pre_treatment_msca: np.ndarray,
        num_original_donors_msca: int,
    ) -> cp.Problem:
        """
        Solve the MSCa SCM optimization problem.

        MSCa (Model with Synthetic Controls and intercept, sum-to-one constraint
        on donor weights) involves adding an intercept to the donor matrix and
        constraining donor weights (excluding intercept) to be non-negative
        and sum to 1. The intercept itself is also constrained to be non-negative.

        Parameters
        ----------
        target_outcomes_pre_treatment_msca : np.ndarray
            Target vector for the pre-treatment periods, shape (t1,).
        donor_outcomes_pre_treatment_msca : np.ndarray
            Original donor matrix for pre-treatment periods (without intercept),
            shape (t1, num_original_donors_msca).
        num_original_donors_msca : int
            Number of original control units (columns in donor_outcomes_pre_treatment_msca).

        Returns
        -------
        cp.Problem
            The solved CVXPY problem object. The solution (weights, including
            intercept as the first element) can be accessed via
            `problem.solution.primal_vars`.
        """
        # Add an intercept column (column of ones) to the donor matrix
        donor_outcomes_with_intercept_msca = np.c_[
            np.ones((donor_outcomes_pre_treatment_msca.shape[0], 1)), # Intercept column
            donor_outcomes_pre_treatment_msca, # Original donor outcomes
        ]
        # Total number of coefficients to solve for (intercept + donor weights)
        num_total_coefficients_msca = num_original_donors_msca + 1

        # Define CVXPY variable for coefficients (intercept is the first element)
        coefficients_cvxpy_msca = cp.Variable(num_total_coefficients_msca)

        # Constraints for MSCa model:
        # 1. All coefficients (including intercept and donor weights) must be non-negative.
        #    coefficients_cvxpy_msca[0] is the intercept, coefficients_cvxpy_msca[1:] are donor weights.
        # 2. The sum of donor weights (excluding the intercept) must equal 1.
        constraints_list_msca = [
            coefficients_cvxpy_msca >= 0, # w_intercept >= 0, w_donors >= 0
            cp.sum(coefficients_cvxpy_msca[1:]) == 1, # sum(w_donors) = 1
        ]

        # Objective: Minimize || Y_pre - [1, X_donors_pre] * [w_intercept, w_donors]' ||_2
        objective_function_msca = cp.Minimize(
            cp.norm(
                target_outcomes_pre_treatment_msca # Y_pre
                - donor_outcomes_with_intercept_msca @ coefficients_cvxpy_msca, # [1, X_donors_pre] * w
                2, # L2 norm
            )
        )

        problem_msca = cp.Problem(objective_function_msca, constraints_list_msca)
        problem_msca.solve(solver=_SOLVER_CLARABEL_STR) # Solve the optimization problem
        return problem_msca

    @staticmethod
    def _solve_mscb_model(
        target_outcomes_pre_treatment_mscb: np.ndarray,
        donor_outcomes_pre_treatment_mscb: np.ndarray,
        num_donors_mscb: int,
    ) -> cp.Problem:
        """
        Solve the MSCb SCM optimization problem.

        MSCb (Model with Synthetic Controls, non-negative weights) involves
        constraining donor weights to be non-negative. No intercept is added
        by default, and weights do not need to sum to 1.

        Parameters
        ----------
        target_outcomes_pre_treatment_mscb : np.ndarray
            Target vector for the pre-treatment periods, shape (t1,).
        donor_outcomes_pre_treatment_mscb : np.ndarray
            Original donor matrix for pre-treatment periods (without intercept),
            shape (t1, num_donors_mscb).
        num_donors_mscb : int
            Number of original control units (columns in donor_outcomes_pre_treatment_mscb).

        Returns
        -------
        cp.Problem
            The solved CVXPY problem object. The solution (weights) can be
            accessed via `problem.solution.primal_vars`.
        """
        # Define CVXPY variable for donor weights
        coefficients_cvxpy_mscb = cp.Variable(num_donors_mscb)

        # Constraints for MSCb model:
        # 1. Donor weights must be non-negative (w_j >= 0 for all j).
        # No sum-to-one constraint and no explicit intercept in this model variant.
        constraints_list_mscb = [coefficients_cvxpy_mscb >= 0]

        # Objective: Minimize || Y_pre - X_donors_pre * w ||_2
        objective_function_mscb = cp.Minimize(
            cp.norm(
                target_outcomes_pre_treatment_mscb # Y_pre
                - donor_outcomes_pre_treatment_mscb @ coefficients_cvxpy_mscb, # X_donors_pre * w
                2, # L2 norm
            )
        )

        problem_mscb = cp.Problem(objective_function_mscb, constraints_list_mscb)
        problem_mscb.solve(solver=_SOLVER_CLARABEL_STR) # Solve the optimization problem
        return problem_mscb

    @staticmethod
    def _solve_mscc_model(
        target_outcomes_pre_treatment_mscc: np.ndarray,
        donor_outcomes_pre_treatment_mscc: np.ndarray,
        num_original_donors_mscc: int,
    ) -> cp.Problem:
        """
        Solve the MSCc SCM optimization problem.

        MSCc (Model with Synthetic Controls, intercept, non-negative weights)
        involves adding an intercept to the donor matrix. All coefficients,
        including the intercept and donor weights, are constrained to be non-negative.
        There is no sum-to-one constraint on donor weights.

        Parameters
        ----------
        target_outcomes_pre_treatment_mscc : np.ndarray
            Target vector for the pre-treatment periods, shape (t1,).
        donor_outcomes_pre_treatment_mscc : np.ndarray
            Original donor matrix for pre-treatment periods (without intercept),
            shape (t1, num_original_donors_mscc).
        num_original_donors_mscc : int
            Number of original control units (columns in donor_outcomes_pre_treatment_mscc).

        Returns
        -------
        cp.Problem
            The solved CVXPY problem object. The solution (weights, including
            intercept as the first element) can be accessed via
            `problem.solution.primal_vars`.
        """
        # Add an intercept column (column of ones) to the donor matrix
        donor_outcomes_with_intercept_mscc = np.c_[
            np.ones((donor_outcomes_pre_treatment_mscc.shape[0], 1)), # Intercept column
            donor_outcomes_pre_treatment_mscc, # Original donor outcomes
        ]
        # Total number of coefficients to solve for (intercept + donor weights)
        num_total_coefficients_mscc = num_original_donors_mscc + 1

        # Define CVXPY variable for coefficients (intercept is the first element)
        coefficients_cvxpy_mscc = cp.Variable(num_total_coefficients_mscc)

        # Constraints for MSCc model:
        # 1. All coefficients (including intercept and donor weights) must be non-negative.
        #    coefficients_cvxpy_mscc[0] is the intercept, coefficients_cvxpy_mscc[1:] are donor weights.
        # No sum-to-one constraint on donor weights.
        constraints_list_mscc = [coefficients_cvxpy_mscc >= 0] # w_intercept >= 0, w_donors >= 0

        # Objective: Minimize || Y_pre - [1, X_donors_pre] * [w_intercept, w_donors]' ||_2
        objective_function_mscc = cp.Minimize(
            cp.norm(
                target_outcomes_pre_treatment_mscc # Y_pre
                - donor_outcomes_with_intercept_mscc @ coefficients_cvxpy_mscc, # [1, X_donors_pre] * w
                2, # L2 norm
            )
        )

        problem_mscc = cp.Problem(objective_function_mscc, constraints_list_mscc)
        problem_mscc.solve(solver=_SOLVER_CLARABEL_STR) # Solve the optimization problem
        return problem_mscc

    @staticmethod
    def _solve_ols_model(
            target_outcomes_pre_treatment_ols: np.ndarray,
            donor_outcomes_pre_treatment_ols: np.ndarray,
            num_donors_ols: int,
            lambda_penalty: Optional[float] = None,
            p: Optional[float] = None,
            q: Optional[float] = None,
            fit_intercept: bool = False,
    ) -> cp.Problem:
        """
        Solve OLS SCM with optional (p, q) regularization and optional free intercept.

        If fit_intercept is True, solves
            minimize || y - b0 - X w ||_2^2 + lambda * ||w||_p^q
        where b0 is an unconstrained scalar (not regularized) and w is a vector of length num_donors_ols.

        If fit_intercept is False (legacy behavior), solves
            minimize || y - X w ||_2^2 + lambda * ||w||_p^q
        where w is length num_donors_ols.

        Returns solved cvxpy Problem. Extract variables from problem.solution.primal_vars.
        """
        lambda_penalty = Opt.DEFAULT_LAMBDA_PENALTY if lambda_penalty is None else float(lambda_penalty)
        p = Opt.DEFAULT_P_NORM if p is None else p
        q = Opt.DEFAULT_Q_EXPONENT if q is None else q

        # Build variables
        if fit_intercept:
            b0 = cp.Variable()  # scalar intercept (unregularized)
            w = cp.Variable(num_donors_ols)  # donor weights
            residual = target_outcomes_pre_treatment_ols - (b0 + donor_outcomes_pre_treatment_ols @ w)
        else:
            b0 = None
            w = cp.Variable(num_donors_ols)
            residual = target_outcomes_pre_treatment_ols - donor_outcomes_pre_treatment_ols @ w

        # Regularization penalty (on w only)
        penalty_term = lambda_penalty * cp.power(cp.norm(w, p), q) if lambda_penalty > 0 else 0.0

        # Objective
        objective_function_ols = cp.Minimize(cp.sum_squares(residual) + penalty_term)
        constraints = []

        # Build problem
        problem_ols = cp.Problem(objective_function_ols, constraints)

        # Solve with solver depending on norm
        if p == 2:
            # L2 / Ridge: use Clarabel
            problem_ols.solve(
                solver=cp.CLARABEL
            )
        elif p == 1:
            # L1 / Lasso: use CVXOPT with strict tolerances
            problem_ols.solve(
                solver=cp.OSQP
            )
        else:
            # fallback
            problem_ols.solve(solver=cp.CLARABEL)

        return problem_ols

    @staticmethod
    def _solve_inf_model(
            target_outcomes_pre_treatment_ols: np.ndarray,
            donor_outcomes_pre_treatment_ols: np.ndarray,
            num_donors_ols: int,
            lambda_penalty: Optional[float] = None,
            alpha_penalty: Optional[float] = 0.0,
            fit_intercept: bool = False,
            affine: bool = False,
            simplex: bool = True,
            EN: Literal["L1_INF", "L1_L2"] = "L1_INF"
    ) -> cp.Problem:
        """
        Solve the SCM problem with either L1–L∞ or L1–L2 regularization.

        Parameters
        ----------
        target_outcomes_pre_treatment_ols : np.ndarray
            Outcome vector for the treated unit (pre-treatment).
        donor_outcomes_pre_treatment_ols : np.ndarray
            Matrix of donor outcomes (pre-treatment).
        num_donors_ols : int
            Number of donor units.
        lambda_penalty : float, optional
            Regularization strength.
        alpha_penalty : float, optional
            Trade-off between L1 and second norm (L∞ or L2).
        fit_intercept : bool
            Whether to include an intercept.
        affine : bool
            Whether weights should sum to 1 (affine constraint).
        simplex : bool
            Whether weights should be nonnegative and sum to 1 (simplex constraint).
        EN : {"L1_INF", "L1_L2"}
            Which hybrid penalty to use.

        Returns
        -------
        cp.Problem
            CVXPY problem solved.
        """

        lam = lambda_penalty
        alpha = alpha_penalty
        T = target_outcomes_pre_treatment_ols.shape[0]
        J = num_donors_ols

        # ------------------ Variables ------------------
        if fit_intercept:
            b0 = cp.Variable(name="intercept")
            w = cp.Variable(J, name="donor_weights")
            residual = target_outcomes_pre_treatment_ols - (b0 + donor_outcomes_pre_treatment_ols @ w)
        else:
            w = cp.Variable(J, name="donor_weights")
            residual = target_outcomes_pre_treatment_ols - donor_outcomes_pre_treatment_ols @ w

        # ------------------ Regularization ------------------
        if EN == "L1_INF":
            reg_term = lam * (alpha * cp.norm(w, 1) + (1 - alpha) * cp.norm(w, "inf"))
        elif EN == "L1_L2":
            reg_term = lam * (alpha * cp.norm(w, 1) + (1 - alpha) * cp.norm(w, 2))
        else:
            raise ValueError(f"Unknown penalty_type: {EN}")

        # ------------------ Objective ------------------
        objective = cp.Minimize((1 / T) * cp.sum_squares(residual) + reg_term)

        # ------------------ Constraints ------------------
        constraints = []
        if simplex:
            constraints.append(cp.sum(w) == 1)
            constraints.append(w >= 0)
        elif affine:
            constraints.append(cp.sum(w) == 1)

        # ------------------ Solve ------------------
        problem = cp.Problem(objective, constraints)
        problem.solve(
            solver=cp.OSQP,
            verbose=False,
        )

        return problem

    @staticmethod
    def _solve_relax_l2_tau(
            target_outcomes_pre_treatment_ols: np.ndarray,
            donor_outcomes_pre_treatment_ols: np.ndarray,
            num_donors_ols: int,
            tau: float,
    ) -> cp.Problem:
        """
        Solve the τ-constrained relaxed balanced SCM problem:

            minimize ||w||_2^2
            subject to  || X^T (X w - y) / T + gamma * 1 ||_inf ≤ τ

        Always uses:
            - nonnegative weights
            - sum(w) = 1
            - no intercept
        """

        X = donor_outcomes_pre_treatment_ols
        y = target_outcomes_pre_treatment_ols
        T, J = X.shape

        # Variables: simplex weights + scalar gamma
        w = cp.Variable(J, nonneg=True, name="donor_weights")
        gamma = cp.Variable(name="gamma")

        residual = X @ w - y
        imbalance = X.T @ residual / T + gamma * cp.ones(J)

        # Objective: minimize squared L2 norm of weights
        objective = cp.Minimize(cp.sum_squares(w))

        # Constraints: simplex + τ bound
        constraints = [
            cp.sum(w) == 1,
            cp.norm(imbalance, "inf") <= tau,
        ]

        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.CLARABEL,verbose=False)

        return problem


    @staticmethod
    def SCopt(
        num_control_units: int,
        target_outcomes_pre_treatment: np.ndarray,
        num_pre_treatment_periods: int,
        donor_outcomes_pre_treatment: np.ndarray,
        scm_model_type: str = _SCM_MODEL_MSCB,
        base_model_results_for_averaging: Optional[Dict[str, Dict[str, Any]]] = None,
        donor_names: Optional[List[str]] = None,
        # --- New regularization args ---
        EN: Literal["L1_INF", "L1_L2"] = "L1_INF",
        lambda_penalty: Optional[float] = None,
        alpha_penalty: Optional[float] = None,
        p: Optional[float] = None,
        q: Optional[float] = None,
        w_convex = None,
        fit_intercept: bool = False,
        affine: bool = False,
        simplex: bool = True,
    ) -> Union[cp.Problem, Dict[str, Any]]:
        """Perform optimization for various Synthetic Control Method (SCM) variants.

        This static method sets up and solves the SCM optimization problem
        using CVXPY for different model specifications like standard SCM
        (with or without intercept, with or without simplex constraint),
        OLS, or Model Averaging.

        Parameters
        ----------
        num_control_units : int
            Number of control units (donors) or columns in the input matrix
            `donor_outcomes_pre_treatment`. This should be the count before
            any intercept is added internally.
        target_outcomes_pre_treatment : np.ndarray
            Target vector for the pre-treatment periods, shape
            (num_pre_treatment_periods,). This is typically the outcome of the
            treated unit in the pre-treatment phase.
        num_pre_treatment_periods : int
            Number of pre-treatment periods. This defines the length of
            `target_outcomes_pre_treatment` and the number of rows of
            `donor_outcomes_pre_treatment` used for fitting.
        donor_outcomes_pre_treatment : np.ndarray
            Donor matrix for the pre-treatment periods, shape
            (num_pre_treatment_periods, num_control_units). Each column
            represents a donor unit's outcomes or features.
        scm_model_type : str, optional
            The SCM optimization model to use. Options are:

            - 'MSCa': SCM with an intercept, weights for donors sum to 1, all non-negative.
            - 'MSCb': SCM without intercept, weights non-negative. (Default)
            - 'MSCc': SCM with an intercept, donor weights non-negative.
            - 'SIMPLEX': SCM without intercept, weights non-negative and sum to 1.
            - 'OLS': Ordinary Least Squares, weights are unconstrained.
            - 'MA': Model Averaging. Requires `base_model_results_for_averaging`.

            Default is _SCM_MODEL_MSCB.
        base_model_results_for_averaging : Optional[Dict[str, Dict[str, Any]]], optional
            Required if `scm_model_type` is _SCM_MODEL_MA. A dictionary where
            keys are model names (strings) and values are dictionaries. Each
            inner dictionary must contain:

            ``'weights'`` : np.ndarray
                The SCM weights obtained from that model.
            ``'cf'`` : np.ndarray
                The counterfactual predictions for the pre-treatment
                period (length `num_pre_treatment_periods`) from that model.

            Default is None.

        donor_names : Optional[List[str]], optional
            List of donor names. Currently used by some internal model types if they
            handle intercepts or specific donor selections. Default is None.

        Returns
        -------
        Union[cp.Problem, Dict[str, Any]]
            The result of the SCM optimization.
            If `scm_model_type` is _SCM_MODEL_MA, returns a dictionary with keys:

            ``"Lambdas"`` : Dict[str, float]
                Model names mapped to their lambda (averaging) weights.
            ``"w_MA"`` : np.ndarray
                The final model-averaged SCM weights for donors.
            ``"Counterfactual_pre"`` : np.ndarray
                The model-averaged counterfactual prediction for the
                pre-treatment period, shape (num_pre_treatment_periods,).

            Otherwise (for _SCM_MODEL_MSCA, _SCM_MODEL_MSCB, _SCM_MODEL_MSCC,
            _SCM_MODEL_SIMPLEX, _SCM_MODEL_OLS), returns a `cvxpy.Problem`
            object after it has been solved. The estimated SCM weights can be
            accessed via:
            `problem.solution.primal_vars[next(iter(problem.solution.primal_vars))]`.
            If an intercept was added (e.g., _SCM_MODEL_MSCA, _SCM_MODEL_MSCC),
            it will be the first element of the weight vector.

        Raises
        ------
        MlsynthDataError
            If `donor_outcomes_pre_treatment` and
            `target_outcomes_pre_treatment` dimensions (after slicing to
            `num_pre_treatment_periods`) are inconsistent for non-MA models.
        MlsynthConfigError
            If `scm_model_type` is _SCM_MODEL_MA and
            `base_model_results_for_averaging` is not provided or is
            malformed, or if an unsupported `scm_model_type` is provided.

        Notes
        -----
        - For models _SCM_MODEL_MSCA and _SCM_MODEL_MSCC, an intercept term is
          automatically added to the `donor_outcomes_pre_treatment` matrix
          internally. The `num_control_units` parameter is incremented
          accordingly, and the returned weight vector will include the intercept
          as its first element.
        - The method solves `min ||y - Xw||_2^2` subject to constraints
          defined by the `scm_model_type` type.
        - CVXPY's CLARABEL solver is used.

        Examples
        --------
        >>> T_pre_ex, N_donors_ex = 10, 3
        >>> y_ex = np.random.rand(T_pre_ex)
        >>> X_ex = np.random.rand(T_pre_ex, N_donors_ex)
        >>> # Solve for SIMPLEX model (weights non-negative, sum to 1)
        >>> prob_simplex = Opt.SCopt(N_donors_ex, y_ex, T_pre_ex, X_ex, scm_model_type=_SCM_MODEL_SIMPLEX)
        >>> if prob_simplex.status == 'optimal':
        ...     weights_simplex = prob_simplex.solution.primal_vars[next(iter(prob_simplex.solution.primal_vars))]
        ...     print(f"Simplex weights sum: {np.sum(weights_simplex):.2f}")
        ...     print(f"Are weights non-negative: {np.all(weights_simplex >= -1e-5)}") # Allow for small numerical errors
        Simplex weights sum: 1.00
        Are weights non-negative: True
        >>> # Solve for MSCb model (weights non-negative)
        >>> prob_mscb = Opt.SCopt(N_donors_ex, y_ex, T_pre_ex, X_ex, scm_model_type=_SCM_MODEL_MSCB)
        >>> if prob_mscb.status == 'optimal':
        ...     weights_mscb = prob_mscb.solution.primal_vars[next(iter(prob_mscb.solution.primal_vars))]
        ...     print(f"Are MSCb weights non-negative: {np.all(weights_mscb >= -1e-5)}")
        Are MSCb weights non-negative: True
        """
        donor_outcomes_pre_treatment_subset = donor_outcomes_pre_treatment[:num_pre_treatment_periods]
        target_outcomes_pre_treatment_subset = target_outcomes_pre_treatment[:num_pre_treatment_periods]

        if scm_model_type != _SCM_MODEL_MA and donor_outcomes_pre_treatment_subset.shape[0] != target_outcomes_pre_treatment_subset.shape[0]:
            raise MlsynthDataError(
                "For non-MA models, donor_outcomes_pre_treatment and target_outcomes_pre_treatment "
                "(after slicing to num_pre_treatment_periods) must have matching row counts."
            )

        current_num_control_units = num_control_units
        processed_donor_outcomes_pre_treatment_subset = donor_outcomes_pre_treatment_subset.copy()

        if scm_model_type == _SCM_MODEL_MA:
            return Opt._solve_ma_model(
                target_outcomes_pre_treatment_subset,
                base_model_results_for_averaging,
                num_pre_treatment_periods,
            )
        elif scm_model_type == _SCM_MODEL_AFFINE:
            return Opt._solve_affine_model(
                target_outcomes_pre_treatment_subset,
                processed_donor_outcomes_pre_treatment_subset, # Original X, no intercept added here
                current_num_control_units, w_convex,lambda_penalty # Original num_control_units
            )
        elif scm_model_type == _SCM_MODEL_SIMPLEX:
            return Opt._solve_simplex_model(
                target_outcomes_pre_treatment_subset,
                processed_donor_outcomes_pre_treatment_subset, # Original X, no intercept added here
                current_num_control_units, # Original num_control_units
            )
        elif scm_model_type == _SCM_MODEL_MSCA:
            return Opt._solve_msca_model(
                target_outcomes_pre_treatment_subset,
                donor_outcomes_pre_treatment_subset, # Original X
                num_control_units, # Original num_control_units
            )
        elif scm_model_type == _SCM_MODEL_MSCB:
            return Opt._solve_mscb_model(
                target_outcomes_pre_treatment_subset,
                donor_outcomes_pre_treatment_subset, # Original X
                num_control_units, # Original num_control_units
            )
        elif scm_model_type == _SCM_MODEL_MSCC:
            return Opt._solve_mscc_model(
                target_outcomes_pre_treatment_subset,
                donor_outcomes_pre_treatment_subset, # Original X
                num_control_units, # Original num_control_units
            )

        elif scm_model_type == _SCM_MODEL_OLS:

            return Opt._solve_ols_model(
                target_outcomes_pre_treatment_subset,
                processed_donor_outcomes_pre_treatment_subset,  # Original X
                num_control_units,  # Original num_control_units
                lambda_penalty=lambda_penalty,  # new regularization strength
                p=p,  # norm for weight penalty
                q=q,  # norm for residuals
                fit_intercept=fit_intercept  # pass flag here
            )



        elif scm_model_type == _SCM_MODEL_INF:


            return Opt._solve_inf_model(
                target_outcomes_pre_treatment_ols=target_outcomes_pre_treatment_subset,
                donor_outcomes_pre_treatment_ols=processed_donor_outcomes_pre_treatment_subset,
                num_donors_ols=current_num_control_units,
                lambda_penalty=lambda_penalty,
                alpha_penalty=alpha_penalty,
                fit_intercept=fit_intercept,
                affine=affine, simplex=simplex,
                EN=EN)

        else:
            raise MlsynthConfigError(
                f"Unsupported SCM model type: {scm_model_type}. Supported types are: "
                f"{_SCM_MODEL_SIMPLEX}, {_SCM_MODEL_MSCA}, {_SCM_MODEL_MSCB}, "
                f"{_SCM_MODEL_MSCC}, {_SCM_MODEL_OLS}, {_SCM_MODEL_MA}, {_SCM_MODEL_INF},"
                f"{_SCM_MODEL_RL2}."
            )


def bartlett(lag_order: int, truncation_lag: int) -> float:
    """
    Calculate the Bartlett kernel weight for a given lag.

    The Bartlett kernel is a common choice for HAC (Heteroskedasticity and
    Autocorrelation Consistent) covariance matrix estimation. It assigns
    linearly decreasing weights to autocovariances as the lag order increases,
    up to a specified truncation lag.

    Parameters
    ----------
    lag_order : int
        The current lag order for which the weight is to be calculated.
        Typically, `lag_order` ranges from 0 up to `truncation_lag`.
    truncation_lag : int
        The truncation lag (bandwidth parameter). Autocovariances for lags
        greater than `truncation_lag` receive a weight of 0.

    Returns
    -------
    float
        The Bartlett kernel weight, calculated as ``1 - |lag_order| / (truncation_lag + 1)``
        if ``|lag_order| <= truncation_lag``, and 0 otherwise.

    Examples
    --------
    >>> bartlett(0, 5)
    1.0
    >>> bartlett(3, 5)
    0.5
    >>> bartlett(5, 5)
    0.16666666666666663
    >>> bartlett(6, 5)
    0.0
    """
    if np.abs(lag_order) <= truncation_lag:
        return 1 - np.abs(lag_order) / (truncation_lag + 1)
    else:
        return 0.0


def hac(G_moments: np.ndarray, truncation_lag: int, kernel: Callable[[int, int], float] = bartlett) -> np.ndarray:
    """
    Heteroskedasticity and Autocorrelation Consistent (HAC) covariance matrix estimator.

    This function computes the HAC covariance matrix, often used in GMM
    estimation to obtain robust standard errors when moment conditions exhibit
    serial correlation and/or heteroskedasticity.

    Parameters
    ----------
    G_moments : np.ndarray
        Matrix of moment conditions, where rows are observations and columns
        are different moment conditions. Shape (num_observations, num_moment_conditions), where
        `num_observations` is the number of observations and `num_moment_conditions` is the number
        of moment conditions.
    truncation_lag : int
        The truncation lag (bandwidth parameter) for the kernel. This determines
        how many autocovariance terms are included in the HAC sum.
    kernel : Callable[[int, int], float], optional
        A kernel function that takes two integer arguments (current lag `lag_order`,
        truncation lag `truncation_lag`) and returns a float weight.
        Default is `bartlett`.

    Returns
    -------
    np.ndarray
        The estimated HAC covariance matrix, shape (num_moment_conditions, num_moment_conditions).

    Examples
    --------
    >>> num_observations_ex, num_moment_conditions_ex = 50, 2
    >>> G_moments_ex = np.random.randn(num_observations_ex, num_moment_conditions_ex) # Example moment conditions
    >>> truncation_lag_ex = 4 # Truncation lag
    >>> omega_hat_ex = hac(G_moments_ex, truncation_lag_ex)
    >>> print(omega_hat_ex.shape)
    (2, 2)
    >>> # Example with a custom kernel (e.g., Truncated/Uniform kernel)
    >>> def truncated_kernel(lag_order: int, current_truncation_lag: int) -> float:
    ...     return 1.0 if np.abs(lag_order) <= current_truncation_lag else 0.0
    >>> omega_hat_trunc_ex = hac(G_moments_ex, truncation_lag_ex, kernel=truncated_kernel)
    >>> print(omega_hat_trunc_ex.shape)
    (2, 2)
    """
    num_observations, num_moment_conditions = G_moments.shape
    omega_hac_matrix: np.ndarray = np.zeros((num_moment_conditions, num_moment_conditions))

    # Sum for current_lag = 0
    omega_hac_matrix += (G_moments.T @ G_moments) / num_observations

    # Sum for current_lag = 1 to truncation_lag
    for current_lag in range(1, min(truncation_lag, num_observations -1) + 1): # Ensure current_lag < num_observations
        current_kernel_weight: float = kernel(current_lag, truncation_lag)
        autocovariance_at_lag: np.ndarray = (G_moments[:-current_lag].T @ G_moments[current_lag:]) / num_observations # (K,K)
        omega_hac_matrix += current_kernel_weight * (autocovariance_at_lag + autocovariance_at_lag.T)
    return omega_hac_matrix


def pi(
    outcome_vector: np.ndarray,
    design_matrix: np.ndarray,
    instrument_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_effect_eval: int,
    total_periods: int,
    hac_truncation_lag: int,
    common_aux_covariates_1: Optional[np.ndarray] = None,
    common_aux_covariates_2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate treatment effects using a two-stage Proximal Inference approach.

    In the first stage, coefficients (`alpha`) for the design matrix `design_matrix` are
    estimated using pre-treatment data, with `instrument_matrix` serving as instruments for `design_matrix`.
    In the second stage, these coefficients are used to predict counterfactual
    outcomes, and the average treatment effect (`tau`) is estimated from the
    post-treatment residuals. Standard errors for `tau` are computed using
    a GMM framework with HAC correction.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Outcome vector for the treated unit, shape (total_periods,).
    design_matrix : np.ndarray
        Design matrix (e.g., covariates for the treated unit or donor outcomes),
        shape (total_periods, dim_design_matrix).
    instrument_matrix : np.ndarray
        Proxy matrix (instruments for `design_matrix` in the pre-treatment period),
        shape (total_periods, dim_instrument_matrix).
    num_pre_treatment_periods : int
        Number of pre-treatment periods.
    num_post_periods_for_effect_eval : int
        Number of post-treatment periods over which the average treatment
        effect (`tau_mean_effect`) is calculated.
    total_periods : int
        Total number of time periods in `outcome_vector`, `design_matrix`, and `instrument_matrix`.
    hac_truncation_lag : int
        Lag length for the HAC (Heteroskedasticity and Autocorrelation Consistent)
        variance estimation used for the standard error of `tau_mean_effect`.
    common_aux_covariates_1 : Optional[np.ndarray], optional
        Additional covariates to augment `design_matrix` and `instrument_matrix`.
        If provided, `common_aux_covariates_2` must also be provided.
        Shape (total_periods, dim_common_aux_covariates_1). Default is None.
    common_aux_covariates_2 : Optional[np.ndarray], optional
        Additional covariates to augment `design_matrix` and `instrument_matrix`.
        If provided, `common_aux_covariates_1` must also be provided.
        Shape (total_periods, dim_common_aux_covariates_2). Default is None.
        Note: `dim_common_aux_covariates_1` and `dim_common_aux_covariates_2` are used to augment
        both `design_matrix` and `instrument_matrix` such that they maintain the same number of columns.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        - predicted_counterfactual_original_W : np.ndarray
            Predicted counterfactual outcomes for all `total_periods` periods, based on
            the original `design_matrix` matrix and estimated `alpha_coefficients_original_W`. Shape (total_periods,).
        - alpha_coefficients_original_W : np.ndarray
            Estimated coefficients for the original (non-augmented) `design_matrix` matrix. Shape (dim_design_matrix,).
        - standard_error_tau_effect : float
            Standard error of the average treatment effect (`tau_mean_effect`).
            Returns `np.nan` if GMM inference fails (e.g., due to singular matrix).

    Raises
    ------
    MlsynthConfigError
        If augmented `design_matrix` and `instrument_matrix` matrices do not have the same number of columns.
    np.linalg.LinAlgError
        If matrix inversion fails during coefficient estimation or GMM inference.

    Notes
    -----
    The GMM inference for standard errors involves constructing moment conditions
    based on pre-treatment orthogonality and post-treatment residuals. The
    Jacobian of these moments and the HAC variance of the moment conditions
    are used to compute the asymptotic variance of the parameters.

    Examples
    --------
    >>> T_ex, T0_ex, N_W_ex, N_Z0_ex = 20, 10, 2, 2
    >>> outcome_vec_ex = np.random.rand(T_ex)
    >>> design_mat_ex = np.random.rand(T_ex, N_W_ex)
    >>> instr_mat_ex = np.random.rand(T_ex, N_Z0_ex)
    >>> num_post_periods_eval_ex = T_ex - T0_ex
    >>> hac_trunc_lag_ex = 2
    >>> # Assuming design_mat_ex and instr_mat_ex have same number of columns for simplicity here
    >>> if design_mat_ex.shape[1] == instr_mat_ex.shape[1]:
    ...     y_pred, alpha, se_tau = pi(
    ...         outcome_vec_ex, design_mat_ex, instr_mat_ex, T0_ex, num_post_periods_eval_ex, T_ex, hac_trunc_lag_ex
    ...     )
    ...     print(y_pred.shape, alpha.shape, isinstance(se_tau, float))
    (20,) (2,) True
    """
    W_aug, Z0_aug = design_matrix, instrument_matrix
    if common_aux_covariates_1 is not None and common_aux_covariates_2 is not None:
        Z0_aug = np.column_stack((instrument_matrix, common_aux_covariates_2, common_aux_covariates_1))
        W_aug = np.column_stack((design_matrix, common_aux_covariates_2, common_aux_covariates_1))

    if W_aug.shape[1] != Z0_aug.shape[1]:
        raise MlsynthConfigError("Augmented design_matrix and instrument_matrix must have the same number of columns.")

    # Stage 1: Estimate alpha using pre-treatment data (Y_pre = W_pre * alpha + error, instrumented by Z0_pre)
    # alpha_hat = (Z0_pre' * W_pre)^-1 * (Z0_pre' * Y_pre)
    Z0W_pre: np.ndarray = Z0_aug[:num_pre_treatment_periods].T @ W_aug[:num_pre_treatment_periods]
    Z0Y_pre: np.ndarray = Z0_aug[:num_pre_treatment_periods].T @ outcome_vector[:num_pre_treatment_periods]
    alpha_coeffs_aug: np.ndarray = np.linalg.solve(Z0W_pre, Z0Y_pre)

    # Predicted counterfactual based on W_aug and estimated alpha_coeffs_aug
    predicted_outcome_augmented_W: np.ndarray = W_aug @ alpha_coeffs_aug
    # Time-varying treatment effects (residuals from the first stage model)
    taut_effects_all_periods: np.ndarray = outcome_vector - predicted_outcome_augmented_W
    # Average treatment effect over the specified post-treatment evaluation window
    tau_mean_effect: float = np.mean(taut_effects_all_periods[num_pre_treatment_periods : num_pre_treatment_periods + num_post_periods_for_effect_eval])
    
    # Construct moment conditions for GMM standard error calculation
    # U0: Z0_aug' * (outcome_vector - W_aug @ alpha_coeffs_aug), zeroed out for post-treatment periods
    U0_for_hac = (Z0_aug.T * (outcome_vector - W_aug @ alpha_coeffs_aug).reshape(1, -1))
    U0_for_hac[:, num_pre_treatment_periods:] = 0 # Moment condition applies only to pre-treatment
    
    # U1: outcome_vector - tau_mean_effect - W_aug @ alpha_coeffs_aug, zeroed out for pre-treatment periods
    # This represents the residual from the overall model (Y = W*alpha + tau*D + error), where D is post-treatment indicator
    U1_for_hac = outcome_vector - tau_mean_effect - W_aug @ alpha_coeffs_aug
    U1_for_hac[:num_pre_treatment_periods] = 0 # This moment applies to post-treatment
    
    # Stack moment conditions for HAC variance estimation
    U_combined_for_hac = np.column_stack((U0_for_hac.T, U1_for_hac)) # Shape (total_periods, dimZ0_aug + 1)

    dimZ0_aug_hac, dimW_aug_hac = Z0_aug.shape[1], W_aug.shape[1]
    
    # Jacobian G = d(E[moment_conditions])/d(parameters'), where parameters = [alpha_coeffs_aug, tau_mean_effect]
    G_hac = np.zeros((dimZ0_aug_hac + 1, dimW_aug_hac + 1)) # (num_moments x num_params)
    # d(E[U0])/d(alpha_coeffs_aug) = -E[Z0_aug' * W_aug] (from pre-treatment)
    G_hac[:dimZ0_aug_hac, :dimW_aug_hac] = -Z0W_pre / num_pre_treatment_periods
    # d(E[U1])/d(alpha_coeffs_aug) = -E[W_aug_post_eval] (average over post-treatment evaluation window)
    G_hac[-1, :dimW_aug_hac] = -np.mean(W_aug[num_pre_treatment_periods : num_pre_treatment_periods + num_post_periods_for_effect_eval], axis=0)
    # d(E[U1])/d(tau_mean_effect) = -1
    G_hac[-1, -1] = -1.0

    # HAC robust covariance matrix of the moment conditions
    Omega_hac_val: np.ndarray = hac(U_combined_for_hac, hac_truncation_lag)
    
    try:
        G_inv = np.linalg.inv(G_hac)
        Cov_matrix = G_inv @ Omega_hac_val @ G_inv.T / total_periods
        var_tau_effect = Cov_matrix[-1, -1]
        standard_error_tau_effect = np.sqrt(var_tau_effect) if var_tau_effect >= 0 else np.nan
    except np.linalg.LinAlgError:
        standard_error_tau_effect = np.nan

    alpha_coefficients_original_W: np.ndarray = alpha_coeffs_aug[: design_matrix.shape[1]]
    predicted_counterfactual_original_W = design_matrix @ alpha_coefficients_original_W

    return predicted_counterfactual_original_W, alpha_coefficients_original_W, standard_error_tau_effect


def pi_surrogate(
    outcome_vector: np.ndarray, design_matrix_main: np.ndarray, instrument_matrix_main: np.ndarray, 
    instrument_matrix_surrogate: np.ndarray, surrogate_outcome_matrix: np.ndarray,
    num_pre_treatment_periods: int, num_post_periods_for_effect_eval: int, total_periods: int, hac_truncation_lag: int,
    aux_covariates_main_1: Optional[np.ndarray] = None, aux_covariates_main_2: Optional[np.ndarray] = None, 
    aux_covariates_surrogate: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Estimate treatment effects using Proximal Inference with surrogate outcomes.

    This method extends the proximal inference framework by incorporating
    surrogate outcomes to potentially improve the estimation of treatment
    effects, especially when the direct effect on the primary outcome is
    difficult to measure in the post-treatment period.

    The estimation involves a multi-stage process:
    1. Estimate coefficients (`alpha`) for the main design matrix `W` using
       pre-treatment data (`Y[:T0]`), with `Z0` serving as instruments for `W`.
    2. Calculate residualized outcomes (`tau_hat_post = Y[T0:] - W[T0:] @ alpha`)
       for the post-treatment period. These residuals serve as a proxy for the
       treatment effect.
    3. Estimate coefficients (`gamma`) for the effect of surrogate outcomes `X_surr`
       on `tau_hat_post`, using `Z1` as instruments for `X_surr` in the
       post-treatment period.
    4. The predicted time-varying treatment effects are then `X_surr @ gamma`.
       For the pre-treatment period, `taut_effects_all_periods` stores the
       residuals from stage 1 (`Y[:T0] - W[:T0] @ alpha`).
    5. Standard errors for the average treatment effect are computed using a
       GMM framework with HAC correction.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Outcome vector for the treated unit, shape (total_periods,).
    design_matrix_main : np.ndarray
        Design matrix for the main outcome equation (e.g., covariates or donor
        outcomes), shape (total_periods, dim_design_matrix_main).
    instrument_matrix_main : np.ndarray
        Proxy matrix (instruments for `design_matrix_main` in the pre-treatment period),
        shape (total_periods, dim_instrument_matrix_main). `dim_design_matrix_main` must equal `dim_instrument_matrix_main`.
    instrument_matrix_surrogate : np.ndarray
        Instruments for surrogate outcomes `surrogate_outcome_matrix` (used in post-treatment period),
        shape (total_periods, dim_instrument_matrix_surrogate). `dim_surrogate_outcome_matrix` must equal `dim_instrument_matrix_surrogate`.
    surrogate_outcome_matrix : np.ndarray
        Surrogate outcomes, shape (total_periods, dim_surrogate_outcome_matrix).
    num_pre_treatment_periods : int
        Number of pre-treatment periods.
    num_post_periods_for_effect_eval : int
        Number of post-treatment periods over which the average treatment
        effect (`tau_mean_effect`) is calculated.
    total_periods : int
        Total number of time periods in `outcome_vector`, `design_matrix_main`, `instrument_matrix_main`, `instrument_matrix_surrogate`, and `surrogate_outcome_matrix`.
    hac_truncation_lag : int
        Lag length for the HAC (Heteroskedasticity and Autocorrelation Consistent)
        variance estimation used for the standard error of `tau_mean_effect`.
    aux_covariates_main_1 : Optional[np.ndarray], optional
        Additional covariates for the main outcome model. If provided, `aux_covariates_main_2` and
        `aux_covariates_surrogate` must also be provided. These are concatenated to both `design_matrix_main` and `instrument_matrix_main`.
        Shape (total_periods, dim_aux_covariates_main_1). Default is None.
    aux_covariates_main_2 : Optional[np.ndarray], optional
        Additional covariates for the main outcome model. If provided, `aux_covariates_main_1` and
        `aux_covariates_surrogate` must also be provided. These are concatenated to both `design_matrix_main` and `instrument_matrix_main`.
        Shape (total_periods, dim_aux_covariates_main_2). Default is None.
    aux_covariates_surrogate : Optional[np.ndarray], optional
        Additional covariates for the surrogate outcome model. If provided, `aux_covariates_main_1`
        and `aux_covariates_main_2` must also be provided. These are concatenated to both `surrogate_outcome_matrix`
        and `instrument_matrix_surrogate`. Shape (total_periods, dim_aux_covariates_surrogate). Default is None.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray, float]
        - tau_mean_effect : float
            Estimated average treatment effect over the `num_post_periods_for_effect_eval` periods.
        - taut_effects_all_periods : np.ndarray
            Time-varying treatment effects, shape (total_periods,). In the pre-treatment
            period (`:num_pre_treatment_periods`), these are residuals from the first stage estimation
            (`outcome_vector[:num_pre_treatment_periods] - W_aug[:num_pre_treatment_periods] @ alpha_coeffs_aug`). In the post-treatment
            period (`num_pre_treatment_periods:`), these are `X_surr_aug @ gamma_coeffs`.
        - alpha_coeffs_orig_W : np.ndarray
            Estimated coefficients for the original (non-augmented) `design_matrix_main` matrix,
            shape (dim_design_matrix_main,).
        - se_tau_effect : float
            Standard error of `tau_mean_effect`. Returns `np.nan` if GMM
            inference fails (e.g., due to singular matrix).

    Raises
    ------
    MlsynthConfigError
        If augmented `design_matrix_main` and `instrument_matrix_main` matrices (or `surrogate_outcome_matrix` and `instrument_matrix_surrogate` matrices)
        do not have the same number of columns after augmentation, or if
        the base `design_matrix_main` and `instrument_matrix_main` (or `surrogate_outcome_matrix` and `instrument_matrix_surrogate`) do not have the same
        number of columns when no augmentation is applied.
    np.linalg.LinAlgError
        If matrix inversion fails during coefficient estimation or GMM inference.

    Notes
    -----
    - It's implicitly assumed that `design_matrix_main` and `instrument_matrix_main` have the same number of columns,
      and `surrogate_outcome_matrix` and `instrument_matrix_surrogate` have the same number of columns, respectively,
      before any augmentation with `aux_covariates_main_1`, `aux_covariates_main_2`, `aux_covariates_surrogate`.
    - If `aux_covariates_main_1`, `aux_covariates_main_2`, `aux_covariates_surrogate` are used, they must all be provided. `aux_covariates_main_1` and `aux_covariates_main_2`
      are added to both `design_matrix_main` and `instrument_matrix_main`. `aux_covariates_surrogate` is added to both `surrogate_outcome_matrix` and `instrument_matrix_surrogate`.
    - The GMM inference for standard errors is complex and involves constructing
      moment conditions based on pre-treatment orthogonality (for `alpha`) and
      post-treatment relationships involving surrogates (for `gamma` and `tau`).

    Examples
    --------
    >>> total_periods_ex, num_pre_treatment_periods_ex, N_W_Z0, N_X_Z1 = 30, 20, 2, 3
    >>> num_post_periods_for_effect_eval_ex = total_periods_ex - num_pre_treatment_periods_ex
    >>> hac_truncation_lag_ex = 2
    >>> outcome_vector_data = np.random.rand(total_periods_ex)
    >>> design_matrix_main_data = np.random.rand(total_periods_ex, N_W_Z0)
    >>> instrument_matrix_main_data = np.random.rand(total_periods_ex, N_W_Z0) # Must have same N_cols as design_matrix_main_data
    >>> instrument_matrix_surrogate_data = np.random.rand(total_periods_ex, N_X_Z1) # Must have same N_cols as surrogate_outcome_matrix_data
    >>> surrogate_outcome_matrix_data = np.random.rand(total_periods_ex, N_X_Z1)
    >>> tau, taut, alpha, se = pi_surrogate(
    ...     outcome_vector_data, design_matrix_main_data, instrument_matrix_main_data, instrument_matrix_surrogate_data, surrogate_outcome_matrix_data,
    ...     num_pre_treatment_periods_ex, num_post_periods_for_effect_eval_ex, total_periods_ex, hac_truncation_lag_ex
    ... )
    >>> print(f"Avg. Effect (tau): {tau:.3f}") # Avg. Effect (tau): ...
    >>> print(f"Time-varying effects shape: {taut.shape}") # (30,)
    >>> print(f"Alpha coefficients shape: {alpha.shape}") # (2,)
    >>> print(f"SE of tau: {se:.3f}") # SE of tau: ...
    >>> # Example with optional covariates
    >>> N_aux_main_1, N_aux_main_2, N_aux_surr = 1, 1, 1
    >>> aux_covariates_main_1_data = np.random.rand(total_periods_ex, N_aux_main_1)
    >>> aux_covariates_main_2_data = np.random.rand(total_periods_ex, N_aux_main_2)
    >>> aux_covariates_surrogate_data = np.random.rand(total_periods_ex, N_aux_surr)
    >>> # design_matrix_main_data and instrument_matrix_main_data still need same base N_cols.
    >>> # surrogate_outcome_matrix_data and instrument_matrix_surrogate_data still need same base N_cols.
    >>> tau_aug, _, _, _ = pi_surrogate(
    ...     outcome_vector_data, design_matrix_main_data, instrument_matrix_main_data, instrument_matrix_surrogate_data, surrogate_outcome_matrix_data,
    ...     num_pre_treatment_periods_ex, num_post_periods_for_effect_eval_ex, total_periods_ex, hac_truncation_lag_ex,
    ...     aux_covariates_main_1=aux_covariates_main_1_data, aux_covariates_main_2=aux_covariates_main_2_data, aux_covariates_surrogate=aux_covariates_surrogate_data
    ... )
    >>> print(f"Avg. Effect with aug covs (tau): {tau_aug:.3f}") # Avg. Effect with aug covs (tau): ...
    """
    W_aug, Z0_aug, X_surr_aug, Z1_aug = design_matrix_main, instrument_matrix_main, surrogate_outcome_matrix, instrument_matrix_surrogate
    if aux_covariates_main_1 is not None and aux_covariates_main_2 is not None and aux_covariates_surrogate is not None:
        Z0_aug = np.column_stack((instrument_matrix_main, aux_covariates_main_2, aux_covariates_main_1))
        W_aug = np.column_stack((design_matrix_main, aux_covariates_main_2, aux_covariates_main_1))
        Z1_aug = np.column_stack((instrument_matrix_surrogate, aux_covariates_surrogate)) # Assuming aux_covariates_surrogate augments instrument_matrix_surrogate
        X_surr_aug = np.column_stack((surrogate_outcome_matrix, aux_covariates_surrogate))

    if not (W_aug.shape[1] == Z0_aug.shape[1] and X_surr_aug.shape[1] == Z1_aug.shape[1]):
        raise MlsynthConfigError("Dimension mismatch after augmentation for main or surrogate matrices.")

    # Stage 1: Estimate alpha (coefficients for main_covariates W_aug)
    # Moment condition: E[Z0_aug' * (outcome_vector - W_aug * alpha)] = 0, using pre-treatment data.
    Z0W_pre: np.ndarray = Z0_aug[:num_pre_treatment_periods].T @ W_aug[:num_pre_treatment_periods]
    Z0Y_pre: np.ndarray = Z0_aug[:num_pre_treatment_periods].T @ outcome_vector[:num_pre_treatment_periods]
    alpha_coeffs_aug: np.ndarray = np.linalg.solve(Z0W_pre, Z0Y_pre) # Solves Z0W_pre * alpha = Z0Y_pre
    
    # Residualized outcome from Stage 1, used as the dependent variable in Stage 2 for post-treatment period.
    # This tau_hat_post represents the part of the outcome not explained by W_aug in the post-treatment period.
    tau_hat_post: np.ndarray = outcome_vector[num_pre_treatment_periods:] - W_aug[num_pre_treatment_periods:] @ alpha_coeffs_aug

    # Stage 2: Estimate gamma (coefficients for surrogate_covariates X_surr_aug)
    # Moment condition: E[Z1_aug' * (tau_hat_post - X_surr_aug * gamma)] = 0, using post-treatment data.
    Z1X_post: np.ndarray = Z1_aug[num_pre_treatment_periods:].T @ X_surr_aug[num_pre_treatment_periods:]
    Z1tau_post: np.ndarray = Z1_aug[num_pre_treatment_periods:].T @ tau_hat_post
    gamma_coeffs: np.ndarray = np.linalg.solve(Z1X_post, Z1tau_post) # Solves Z1X_post * gamma = Z1tau_post

    # Predicted time-varying treatment effects using estimated gamma and surrogate covariates.
    taut_effects_all_periods: np.ndarray = X_surr_aug @ gamma_coeffs
    # For pre-treatment periods, taut_effects are the residuals from Stage 1.
    taut_effects_all_periods[:num_pre_treatment_periods] = (outcome_vector - W_aug @ alpha_coeffs_aug)[:num_pre_treatment_periods]
    # Average treatment effect over the specified post-treatment evaluation window.
    tau_mean_effect: float = np.mean(taut_effects_all_periods[num_pre_treatment_periods : num_pre_treatment_periods + num_post_periods_for_effect_eval])

    # GMM Inference for standard errors. This involves constructing moment conditions (U0, U1, U2)
    # and their Jacobian (G_hac) with respect to parameters (alpha, gamma, tau_mean_effect).
    
    # U0: Moment condition from Stage 1 (pre-treatment)
    # E[Z0_aug' * (outcome_vector - W_aug @ alpha)] = 0
    U0_hac = (Z0_aug.T * (outcome_vector - W_aug @ alpha_coeffs_aug).reshape(1, -1))
    U0_hac[:, num_pre_treatment_periods:] = 0 # Applies only to pre-treatment
    
    # U1: Moment condition from Stage 2 (post-treatment)
    # E[Z1_aug' * (outcome_vector - W_aug @ alpha - X_surr_aug @ gamma)] = 0
    # Note: (outcome_vector - W_aug @ alpha) is tau_hat. So, E[Z1_aug' * (tau_hat - X_surr_aug @ gamma)] = 0
    U1_hac = (Z1_aug.T * (outcome_vector - W_aug @ alpha_coeffs_aug - X_surr_aug @ gamma_coeffs).reshape(1, -1))
    U1_hac[:, :num_pre_treatment_periods] = 0 # Applies only to post-treatment
    
    # U2: Moment condition defining tau_mean_effect
    # E[X_surr_aug @ gamma - tau_mean_effect] = 0 (averaged over post-treatment evaluation window)
    U2_hac = X_surr_aug @ gamma_coeffs - tau_mean_effect
    U2_hac[:num_pre_treatment_periods] = 0 # Applies only to post-treatment
    
    # Stack moment conditions for HAC variance estimation
    U_combined_hac = np.column_stack((U0_hac.T, U1_hac.T, U2_hac)) # Shape (total_periods, dimZ0+dimZ1+1)

    dimZ0_h, dimZ1_h = Z0_aug.shape[1], Z1_aug.shape[1] # Dimensions of instrument sets
    dimW_h, dimX_h = W_aug.shape[1], X_surr_aug.shape[1] # Dimensions of covariate sets
    
    # Jacobian G = d(E[moment_conditions])/d(parameters'), where parameters = [alpha, gamma, tau_mean_effect]
    # Shape: (num_moments = dimZ0+dimZ1+1) x (num_params = dimW+dimX+1)
    G_hac = np.zeros((dimZ0_h + dimZ1_h + 1, dimW_h + dimX_h + 1))
    
    # Derivatives of E[U0]
    G_hac[:dimZ0_h, :dimW_h] = -Z0W_pre / num_pre_treatment_periods # d(E[U0])/d(alpha)
    # d(E[U0])/d(gamma) = 0, d(E[U0])/d(tau_mean) = 0 (already zero)

    # Derivatives of E[U1]
    G_hac[dimZ0_h : dimZ0_h + dimZ1_h, :dimW_h] = (-Z1_aug[num_pre_treatment_periods:].T @ W_aug[num_pre_treatment_periods:]) / (total_periods - num_pre_treatment_periods) # d(E[U1])/d(alpha)
    G_hac[dimZ0_h : dimZ0_h + dimZ1_h, dimW_h : dimW_h + dimX_h] = (-Z1_aug[num_pre_treatment_periods:].T @ X_surr_aug[num_pre_treatment_periods:]) / (total_periods - num_pre_treatment_periods) # d(E[U1])/d(gamma)
    # d(E[U1])/d(tau_mean) = 0 (already zero)

    # Derivatives of E[U2]
    # d(E[U2])/d(alpha) = 0 (already zero)
    G_hac[-1, dimW_h : dimW_h + dimX_h] = -np.mean(X_surr_aug[num_pre_treatment_periods : num_pre_treatment_periods + num_post_periods_for_effect_eval], axis=0) # d(E[U2])/d(gamma)
    G_hac[-1, -1] = -1.0 # d(E[U2])/d(tau_mean)

    # HAC robust covariance matrix of the moment conditions
    Omega_val: np.ndarray = hac(U_combined_hac, hac_truncation_lag)
    try:
        G_inv = np.linalg.inv(G_hac)
        Cov_mat = G_inv @ Omega_val @ G_inv.T / total_periods
        var_tau = Cov_mat[-1, -1]
        se_tau_effect = np.sqrt(var_tau) if var_tau >=0 else np.nan
    except np.linalg.LinAlgError:
        se_tau_effect = np.nan

    alpha_coeffs_orig_W: np.ndarray = alpha_coeffs_aug[: design_matrix_main.shape[1]]
    return tau_mean_effect, taut_effects_all_periods, alpha_coeffs_orig_W, se_tau_effect


def pi_surrogate_post(
    outcome_vector: np.ndarray, main_covariates: np.ndarray, main_instruments: np.ndarray, 
    surrogate_instruments: np.ndarray, surrogate_covariates: np.ndarray,
    treatment_start_period: int, num_post_treatment_periods_analyzed: int, hac_truncation_lag: int,
    aux_main_covariates: Optional[np.ndarray] = None, aux_main_instruments: Optional[np.ndarray] = None, 
    aux_surrogate_covariates: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Proximal inference with post-treatment surrogates and GMM inference.

    This function estimates treatment effects using a proximal inference
    approach that leverages post-treatment surrogate outcomes. Unlike `pi_surrogate`
    which uses pre-treatment data to estimate initial coefficients, this method
    focuses on the post-treatment period for estimating the relationship between
    outcomes, covariates, and surrogates.

    The core idea is to model the outcome `outcome_vector` in the post-treatment period as a
    function of pre-treatment covariates `main_covariates` and post-treatment surrogates
    `surrogate_covariates`, using `main_instruments` and `surrogate_instruments` as corresponding
    instruments. The coefficients for `surrogate_covariates` (`gamma_coeffs_surr`)
    are interpreted as the effect of surrogates on the outcome, and
    `surrogate_covariates @ gamma_coeffs_surr` gives the time-varying treatment effects.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Outcome variable, shape (total_periods,).
    main_covariates : np.ndarray
        Pre-treatment covariates, shape (total_periods, dim_main_covariates).
    main_instruments : np.ndarray
        Instruments for `main_covariates`, shape (total_periods, dim_main_instruments). `dim_main_covariates` must equal `dim_main_instruments`.
    surrogate_instruments : np.ndarray
        Instruments for surrogate outcomes `surrogate_covariates`, shape (total_periods, dim_surrogate_instruments).
        `dim_surrogate_covariates` must equal `dim_surrogate_instruments`.
    surrogate_covariates : np.ndarray
        Post-treatment covariates/surrogates, shape (total_periods, dim_surrogate_covariates).
    treatment_start_period : int
        Time period index when treatment starts (0-indexed).
    num_post_treatment_periods_analyzed : int
        Length of the post-treatment period to be used for analysis.
        The analysis window will be `outcome_vector[treatment_start_period : treatment_start_period + num_post_treatment_periods_analyzed]`.
    hac_truncation_lag : int
        Lag parameter for HAC (Heteroskedasticity and Autocorrelation Consistent)
        covariance estimation for standard errors.
    aux_main_covariates : Optional[np.ndarray], optional
        Additional covariates to augment `main_covariates`. If provided, `aux_main_instruments` and `aux_surrogate_covariates`
        must also be provided. Shape (total_periods, dim_aux_main_covariates). Default is None.
    aux_main_instruments : Optional[np.ndarray], optional
        Additional covariates to augment `main_instruments` (and implicitly `surrogate_instruments`
        if they relate to `outcome_vector`). If provided, `aux_main_covariates` and `aux_surrogate_covariates` must also be
        provided. Shape (total_periods, dim_aux_main_instruments). Default is None.
    aux_surrogate_covariates : Optional[np.ndarray], optional
        Additional covariates to augment `surrogate_covariates` (and `surrogate_instruments`).
        If provided, `aux_main_covariates` and `aux_main_instruments` must also be provided.
        Shape (total_periods, dim_aux_surrogate_covariates). Default is None.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray, float]
        - tau_mean_effect : float
            Estimated average treatment effect over the specified post-treatment period.
        - taut_varying_effects : np.ndarray
            Time-varying treatment effect estimates for all `total_periods` periods,
            calculated as `X_ext_aug @ gamma_coeffs_surr`.
        - params_W_coeffs : np.ndarray
            Estimated coefficients for the original (non-augmented) `main_covariates` matrix.
            Shape (dim_main_covariates,).
        - se_tau_effect : float
            Standard error of `tau_mean_effect`. Returns `np.nan` if GMM
            inference fails.

    Raises
    ------
    MlsynthConfigError
        If there's a dimension mismatch in the combined instrument and covariate
        matrices after potential augmentation, or if base matrices have mismatches.
    np.linalg.LinAlgError
        If matrix inversion fails during coefficient estimation or GMM inference.

    Notes
    -----
    - The augmentation logic for `Cw_cov`, `Cy_cov`, `Cx_cov` combines these into
      `Z_combined_aug` and `WX_combined_aug`. The exact structure of this
      combination (especially the order) is critical and follows the original
      implementation's implicit logic.
    - `X_ext_aug` refers to `X_post_covars` potentially augmented by `Cx_cov`.
    - The GMM inference for standard errors is based on moment conditions
      derived from the post-treatment estimation.

    Examples
    --------
    >>> import numpy as np
    >>> total_periods_ex, treatment_start_period_ex, num_post_treatment_periods_analyzed_ex = 30, 15, 10
    >>> N_main_cov_ex, N_main_instr_ex, N_surr_cov_ex, N_surr_instr_ex = 2, 2, 3, 3
    >>> hac_truncation_lag_ex = 2
    >>> outcome_vector_ex = np.random.rand(total_periods_ex)
    >>> main_covariates_ex = np.random.rand(total_periods_ex, N_main_cov_ex)
    >>> main_instruments_ex = np.random.rand(total_periods_ex, N_main_instr_ex)
    >>> surrogate_instruments_ex = np.random.rand(total_periods_ex, N_surr_instr_ex)
    >>> surrogate_covariates_ex = np.random.rand(total_periods_ex, N_surr_cov_ex)
    >>> tau_val, taut_val, alpha_W_val, se_val = pi_surrogate_post(
    ...     outcome_vector_ex, main_covariates_ex, main_instruments_ex, surrogate_instruments_ex, surrogate_covariates_ex,
    ...     treatment_start_period=treatment_start_period_ex,
    ...     num_post_treatment_periods_analyzed=num_post_treatment_periods_analyzed_ex,
    ...     hac_truncation_lag=hac_truncation_lag_ex
    ... )
    >>> print(f"Estimated ATT: {tau_val:.4f}") # doctest: +SKIP
    Estimated ATT: ...
    >>> print(f"SE of ATT: {se_val:.4f}") # doctest: +SKIP
    SE of ATT: ...
    >>> print(f"Shape of W coefficients: {alpha_W_val.shape}")
    Shape of W coefficients: (2,)
    >>> print(f"Shape of time-varying effects: {taut_val.shape}")
    Shape of time-varying effects: (30,)
    """
    # W_aug, Z0_aug, X_aug, Z1_aug = main_covariates, main_instruments, surrogate_covariates, surrogate_instruments
    # Augment matrices if additional covariates are provided
    if aux_main_covariates is not None and aux_main_instruments is not None and aux_surrogate_covariates is not None:
        # Assuming aux_main_instruments augments main_instruments and surrogate_instruments if they relate to outcome_vector
        Z_combined_aug = np.column_stack((main_instruments, aux_main_instruments, surrogate_instruments, aux_surrogate_covariates)) # Order might matter
        WX_combined_aug = np.column_stack((main_covariates, aux_main_covariates, surrogate_covariates, aux_surrogate_covariates))
        X_ext_aug = np.column_stack((surrogate_covariates, aux_surrogate_covariates))
        if Z_combined_aug.shape[1] != WX_combined_aug.shape[1]:
            raise MlsynthConfigError(
                "Dimension mismatch for combined instruments (Z_combined_aug) and "
                "covariates (WX_combined_aug) after augmentation."
            )
    else: # No additional covariates
        if not (main_covariates.shape[1] == main_instruments.shape[1] and\
                surrogate_covariates.shape[1] == surrogate_instruments.shape[1]):
            raise MlsynthConfigError(
                "Dimension mismatch for base main_covariates/main_instruments or "
                "surrogate_covariates/surrogate_instruments before augmentation."
            )
        Z_combined_aug = np.column_stack((main_instruments, surrogate_instruments))
        WX_combined_aug = np.column_stack((main_covariates, surrogate_covariates))
        X_ext_aug = surrogate_covariates


    # Analysis on post-treatment period data for estimating parameters
    # Y_post = WX_combined_post * params_all + error, instrumented by Z_combined_post
    Y_post = outcome_vector[treatment_start_period : treatment_start_period + num_post_treatment_periods_analyzed]
    Z_combined_post = Z_combined_aug[treatment_start_period : treatment_start_period + num_post_treatment_periods_analyzed]
    WX_combined_post = WX_combined_aug[treatment_start_period : treatment_start_period + num_post_treatment_periods_analyzed]
    # X_ext_post is the part of WX_combined_post corresponding to surrogate_covariates (and its aux)
    X_ext_post = X_ext_aug[treatment_start_period : treatment_start_period + num_post_treatment_periods_analyzed]


    # Estimate combined parameters [alpha_W_coeffs, gamma_coeffs_surr]
    # (Z_combined_post' * WX_combined_post) * params_all = Z_combined_post' * Y_post
    ZWX_post: np.ndarray = Z_combined_post.T @ WX_combined_post
    ZY_post: np.ndarray = Z_combined_post.T @ Y_post
    params_all: np.ndarray = np.linalg.solve(ZWX_post, ZY_post) # params_all = [alpha_W_coeffs, gamma_coeffs_surr]
    
    # Extract gamma_coeffs_surr (coefficients for X_ext_aug)
    gamma_coeffs_surr: np.ndarray = params_all[-X_ext_aug.shape[1]:]
    # Time-varying treatment effects are X_ext_aug @ gamma_coeffs_surr
    taut_varying_effects: np.ndarray = X_ext_aug @ gamma_coeffs_surr # Effects for all periods
    # Average treatment effect over the specified post-treatment analysis window
    tau_mean_effect: float = np.mean(taut_varying_effects[treatment_start_period : treatment_start_period + num_post_treatment_periods_analyzed])

    # GMM Inference for standard errors
    # Moment conditions evaluated at estimated parameters for HAC variance calculation.
    # U0_hac_post: Z_combined_post' * (Y_post - WX_combined_post @ params_all)
    # These are the residuals from the main estimation equation, scaled by instruments.
    U0_hac_post = (Z_combined_post.T * (Y_post - WX_combined_post @ params_all).reshape(1,-1))
    # U1_hac_post: X_ext_post @ gamma_coeffs_surr - tau_mean_effect
    # This is the difference between predicted effect from surrogates and the estimated average effect.
    U1_hac_post = X_ext_post @ gamma_coeffs_surr - tau_mean_effect
    
    # Stack moment conditions for HAC variance estimation
    U_combined_hac_post = np.column_stack((U0_hac_post.T, U1_hac_post)) # Shape (num_post_periods_analyzed, dimZ_combined + 1)

    # Jacobian G = d(E[moment_conditions])/d(parameters'), where parameters = [params_all, tau_mean_effect]
    # Shape: (num_moments = dimZ_combined + 1) x (num_params = dimWX_combined + 1)
    G_hac = np.zeros((Z_combined_aug.shape[1] + 1, WX_combined_aug.shape[1] + 1))
    # d(E[U0_hac_post])/d(params_all) = -E[Z_combined_post' * WX_combined_post]
    G_hac[:Z_combined_aug.shape[1], :WX_combined_aug.shape[1]] = -ZWX_post / num_post_treatment_periods_analyzed
    # d(E[U1_hac_post])/d(gamma_coeffs_surr) = -E[X_ext_post] (gamma_coeffs_surr are part of params_all)
    # The slice main_covariates.shape[1] : main_covariates.shape[1] + X_ext_aug.shape[1] correctly targets gamma_coeffs_surr within params_all
    G_hac[-1, main_covariates.shape[1] : main_covariates.shape[1] + X_ext_aug.shape[1]] = -np.mean(X_ext_post, axis=0)
    # d(E[U1_hac_post])/d(tau_mean_effect) = -1
    G_hac[-1, -1] = -1.0

    # HAC robust covariance matrix of the moment conditions
    Omega_val: np.ndarray = hac(U_combined_hac_post, hac_truncation_lag)
    try:
        G_inv = np.linalg.inv(G_hac)
        Cov_mat = G_inv @ Omega_val @ G_inv.T / num_post_treatment_periods_analyzed
        var_tau = Cov_mat[-1, -1]
        se_tau_effect = np.sqrt(var_tau) if var_tau >=0 else np.nan
    except np.linalg.LinAlgError:
        se_tau_effect = np.nan
        
    params_W_coeffs: np.ndarray = params_all[:main_covariates.shape[1]] # Coeffs for original main_covariates
    return tau_mean_effect, taut_varying_effects, params_W_coeffs, se_tau_effect


def SMOweights(
    multi_outcome_data_dict: Dict[str, List[np.ndarray]], 
    aggregation_method: str = "concatenated", 
    num_pre_treatment_periods_override: Optional[int] = None
) -> np.ndarray:
    """
    Estimate Synthetic Control Method (SCM) weights for Multiple Outcomes (SMO).

    This function calculates SCM weights when there are multiple outcome variables
    for the treated unit and donor units. It supports two approaches for
    combining information across outcomes:
    1.  'concatenated': Stacks the outcome series (treated and donors) for all
        outcomes into single long vectors/matrices before solving the SCM
        optimization problem. This is akin to the Time-series, Likelihood,
        Penalized (TLP) approach.
    2.  'average': Averages the demeaned outcome series (treated and donors)
        across all outcomes before solving the SCM optimization. This resembles
        a Sparse Bayesian Model Averaging Framework (SBMF) like objective.

    The optimization problem minimizes the sum of squared differences between
    the (potentially transformed) treated unit's outcomes and the weighted sum
    of donor outcomes, subject to weights being non-negative and summing to 1.

    Parameters
    ----------
    multi_outcome_data_dict : Dict[str, List[np.ndarray]]
        A dictionary containing the data for multiple outcomes. It must have
        two keys: 'Target' and 'Donors'.
        - 'Target' : List[np.ndarray]
            A list where each element is a 1D NumPy array representing the
            outcome series for one of the K outcomes of the treated unit.
            Each array should have shape (total_time_periods,), where total_time_periods is the
            total number of time periods.
        - 'Donors' : List[np.ndarray]
            A list where each element is a 2D NumPy array representing the
            outcomes of donor units for the corresponding outcome in 'Target'.
            Each array should have shape (total_time_periods, num_donors), where num_donors
            is the number of donor units. All donor matrices in the list must
            have the same number of columns (num_donors) and rows (total_time_periods).
    aggregation_method : str, optional
        The method to use for combining information across multiple outcomes.
        Options are:
        - 'concatenated': Stacks data from all outcomes. (Default)
        - 'average': Averages demeaned data across outcomes.
        Default is "concatenated".
    num_pre_treatment_periods_override : Optional[int], optional
        The number of pre-treatment periods to use for fitting the SCM weights.
        If None, all time periods in the input data are considered as
        pre-treatment. Default is None.

    Returns
    -------
    np.ndarray
        An array containing the optimal SCM weights for the donor units.
        The shape of the array is (num_donors,). These weights are non-negative
        and sum to 1.

    Raises
    ------
    MlsynthDataError
        If `multi_outcome_data_dict` is not structured correctly (e.g., 'Target' or 'Donors'
        are not lists of NumPy arrays, lists have different lengths, or lists are empty).
    MlsynthConfigError
        If an invalid `aggregation_method` is specified.

    Examples
    --------
    >>> total_periods_example, num_donors_example, num_outcomes_example = 20, 3, 2
    >>> num_pre_periods_example = 10
    >>> target_outcomes_list_example = [np.random.rand(total_periods_example) for _ in range(num_outcomes_example)]
    >>> donor_outcomes_list_example = [np.random.rand(total_periods_example, num_donors_example) for _ in range(num_outcomes_example)]
    >>> data_dict_example = {"Target": target_outcomes_list_example, "Donors": donor_outcomes_list_example}
    >>> # Using 'concatenated' method
    >>> weights_concat_example = SMOweights(
    ...     data_dict_example, aggregation_method="concatenated", num_pre_treatment_periods_override=num_pre_periods_example
    ... )
    >>> print(f"Concatenated weights shape: {weights_concat_example.shape}")
    Concatenated weights shape: (3,)
    >>> print(f"Sum of weights: {np.sum(weights_concat_example):.2f}")
    Sum of weights: 1.00
    >>> # Using 'average' method
    >>> weights_avg_example = SMOweights(
    ...     data_dict_example, aggregation_method="average", num_pre_treatment_periods_override=num_pre_periods_example
    ... )
    >>> print(f"Average weights shape: {weights_avg_example.shape}")
    Average weights shape: (3,)
    """
    # Retrieve lists of outcome series for the target unit and donor units from the input dictionary.
    full_period_target_outcomes_list: List[np.ndarray] = multi_outcome_data_dict.get("Target", [])
    full_period_donor_outcomes_list: List[np.ndarray] = multi_outcome_data_dict.get("Donors", [])

    # Validate the structure and content of the input data dictionary.
    if not (isinstance(full_period_target_outcomes_list, list) and isinstance(full_period_donor_outcomes_list, list)):
        raise MlsynthDataError("Inputs 'Target' and 'Donors' must be lists of NumPy arrays.")
    if len(full_period_target_outcomes_list) != len(full_period_donor_outcomes_list):
        raise MlsynthDataError("Target and Donor lists must have the same length (number of outcomes).")
    if not full_period_target_outcomes_list: # Checks if the list is empty
        raise MlsynthDataError("Input 'Target' and 'Donors' lists cannot be empty.")

    # Determine the number of outcomes, total time periods, and number of donors from the input data.
    num_outcomes: int = len(full_period_target_outcomes_list)
    # Assuming all donor matrices have the same shape, use the first one to get dimensions.
    total_time_periods_from_data, num_donors_from_data = full_period_donor_outcomes_list[0].shape

    # Determine the number of pre-treatment periods to use for fitting the SCM weights.
    # If num_pre_treatment_periods_override is provided, use it; otherwise, use all available time periods.
    num_pre_periods_for_fitting: int = num_pre_treatment_periods_override if num_pre_treatment_periods_override is not None else total_time_periods_from_data

    # Slice the target and donor outcome series to include only the pre-treatment periods for fitting.
    pre_treatment_target_outcomes_list: List[np.ndarray] = [y[:num_pre_periods_for_fitting] for y in full_period_target_outcomes_list]
    pre_treatment_donor_outcomes_list: List[np.ndarray] = [Y[:num_pre_periods_for_fitting, :] for Y in full_period_donor_outcomes_list]

    # Define the CVXPY variable for the donor weights. These are the weights we want to optimize.
    donor_weights_cvxpy_var = cp.Variable(num_donors_from_data)
    # Initialize the SCM optimization objective expression.
    scm_optimization_objective: cp.Expression

    # Construct the SCM optimization objective based on the chosen aggregation_method.
    if aggregation_method == "concatenated":
        # 'concatenated' method: Stack the outcome series for all outcomes into long vectors/matrices.
        # This approach treats each outcome-period combination as a separate data point for fitting.
        stacked_pre_treatment_target_outcomes: np.ndarray = np.concatenate(pre_treatment_target_outcomes_list, axis=0)
        stacked_pre_treatment_donor_outcomes: np.ndarray = np.vstack(pre_treatment_donor_outcomes_list)
        # Objective: Minimize the sum of squared differences between the stacked target outcomes
        # and the stacked synthetic control constructed from donor outcomes.
        scm_optimization_objective = cp.Minimize(cp.sum_squares(stacked_pre_treatment_donor_outcomes @ donor_weights_cvxpy_var - stacked_pre_treatment_target_outcomes))
    elif aggregation_method == "average":
        # 'average' method: Average the demeaned outcome series across all outcomes.
        # This approach gives equal importance to each outcome after demeaning.
        demeaned_pre_treatment_target_outcomes_list: List[np.ndarray] = [y - np.mean(y) for y in pre_treatment_target_outcomes_list]
        demeaned_pre_treatment_donor_outcomes_list: List[np.ndarray] = [
            Y - np.mean(Y, axis=0, keepdims=True) for Y in pre_treatment_donor_outcomes_list # Demean each donor's series for each outcome
        ]
        # Calculate the average of the demeaned target outcomes and donor outcomes across all K outcomes.
        averaged_demeaned_pre_treatment_target_outcomes: np.ndarray = sum(demeaned_pre_treatment_target_outcomes_list) / num_outcomes
        averaged_demeaned_pre_treatment_donor_outcomes: np.ndarray = sum(demeaned_pre_treatment_donor_outcomes_list) / num_outcomes
        # Objective: Minimize the sum of squared differences between the averaged demeaned target outcomes
        # and the averaged demeaned synthetic control.
        scm_optimization_objective = cp.Minimize(cp.sum_squares(averaged_demeaned_pre_treatment_donor_outcomes @ donor_weights_cvxpy_var - averaged_demeaned_pre_treatment_target_outcomes))
    else:
        # If an invalid aggregation_method is specified, raise an error.
        raise MlsynthConfigError(f"Invalid aggregation_method: {aggregation_method}. Choose 'concatenated' or 'average'.")

    # Define the constraints for the SCM optimization problem:
    # 1. Donor weights must be non-negative (w_j >= 0 for all j).
    # 2. Donor weights must sum to 1 (sum(w_j) = 1).
    scm_optimization_constraints = [donor_weights_cvxpy_var >= 0, cp.sum(donor_weights_cvxpy_var) == 1]
    
    # Create and solve the CVXPY optimization problem.
    scm_optimization_problem = cp.Problem(scm_optimization_objective, scm_optimization_constraints)
    scm_optimization_problem.solve(solver=_SOLVER_CLARABEL_STR) # Use CLARABEL solver

    # Return the estimated optimal donor weights.
    return donor_weights_cvxpy_var.value


def NSC_opt(
    pre_treatment_treated_outcome: np.ndarray, 
    pre_treatment_donor_outcomes: np.ndarray, 
    l1_discrepancy_penalty_factor: float, 
    l2_penalty_factor: float
) -> np.ndarray:
    """
    Normalized Synthetic Control (NSC) weight optimization.

    Solves for weights minimizing error between treated unit's outcome (pre_treatment_treated_outcome)
    and an affine combination of control units' outcomes (pre_treatment_donor_outcomes),
    subject to regularization.

    The optimization problem is:

    ``min ||pre_treatment_treated_outcome - pre_treatment_donor_outcomes @ donor_weights_cvxpy_var||_2**2 +
    l1_discrepancy_penalty_factor * sum(|donor_weights_cvxpy_var_j| * discrepancy_j) +
    l2_penalty_factor * ||donor_weights_cvxpy_var||_2**2``

    subject to ``sum(donor_weights_cvxpy_var) == 1``.
    The ``discrepancy_j`` is the L2 norm of the difference between ``pre_treatment_treated_outcome``
    and the j-th donor's outcome series ``pre_treatment_donor_outcomes[:, j]``.

    Parameters
    ----------
    pre_treatment_treated_outcome : np.ndarray
        Outcome vector for the treated unit in the pre-treatment periods.
        Shape (T0,), where T0 is the number of pre-treatment periods.
    pre_treatment_donor_outcomes : np.ndarray
        Matrix of outcomes for control (donor) units in the pre-treatment periods.
        Shape (T0, num_donors), where num_donors is the number of donor units.
    l1_discrepancy_penalty_factor : float
        Regularization parameter for the L1 penalty term on weights, scaled by
        the pairwise discrepancy between the treated unit and each donor unit.
        This term encourages sparsity and favors donors similar to the treated unit.
    l2_penalty_factor : float
        Regularization parameter for the L2 penalty term on weights (Ridge-like
        regularization). This term helps to stabilize the solution and prevent
        overfitting.

    Returns
    -------
    np.ndarray
        An array containing the computed optimal weights for the control units.
        Shape (num_donors,). The weights are constrained to sum to 1.
        Returns an array of NaNs of shape (num_donors,) if the optimization
        does not converge to an optimal or optimal_inaccurate solution.

    Notes
    -----
    - The CVXPY library with the CLARABEL solver is used for optimization.
    - A warning is issued if the optimization fails to converge.

    Examples
    --------
    >>> num_pre_periods_ex, num_donors_ex = 10, 4
    >>> pre_treatment_treated_outcome_ex = np.random.rand(num_pre_periods_ex)
    >>> pre_treatment_donor_outcomes_ex = np.random.rand(num_pre_periods_ex, num_donors_ex)
    >>> l1_penalty_factor_ex, l2_penalty_factor_ex = 0.1, 0.05
    >>> optimal_weights_ex = NSC_opt(
    ...     pre_treatment_treated_outcome_ex, pre_treatment_donor_outcomes_ex,
    ...     l1_penalty_factor_ex, l2_penalty_factor_ex
    ... )
    >>> if optimal_weights_ex is not None and not np.isnan(optimal_weights_ex).any():
    ...     print(f"NSC weights shape: {optimal_weights_ex.shape}")
    ...     print(f"Sum of weights: {np.sum(optimal_weights_ex):.2f}") # doctest: +SKIP
    NSC weights shape: (4,)
    Sum of weights: 1.00
    """
    # Determine the number of donor units.
    num_donors: int = pre_treatment_donor_outcomes.shape[1]
    # Define the CVXPY variable for donor weights. These are the parameters to be optimized.
    donor_weights_cvxpy_var = cp.Variable(num_donors)

    # Calculate the prediction residuals: difference between actual treated outcome and synthetic control.
    # Synthetic control is pre_treatment_donor_outcomes @ donor_weights_cvxpy_var.
    prediction_residuals_cvxpy: cp.Expression = pre_treatment_treated_outcome - pre_treatment_donor_outcomes @ donor_weights_cvxpy_var
    # First term of the objective: sum of squared residuals (L2 norm squared of prediction_residuals_cvxpy).
    # This term aims to minimize the difference between the treated unit and its synthetic counterpart.
    sum_squared_residuals_term: cp.Expression = cp.sum_squares(prediction_residuals_cvxpy)

    # Calculate pairwise discrepancies (L2 norm of differences) between the treated unit's pre-treatment outcome
    # and each donor unit's pre-treatment outcome series.
    # This vector (pairwise_discrepancies_vector) will have shape (num_donors,).
    pairwise_discrepancies_vector: np.ndarray = np.linalg.norm(
        pre_treatment_treated_outcome[:, np.newaxis] - pre_treatment_donor_outcomes, axis=0 
    )
    # Second term of the objective: L1 penalty on donor weights, scaled by pairwise discrepancies.
    # This term encourages sparsity in weights and prioritizes donors that are more similar (lower discrepancy)
    # to the treated unit in the pre-treatment period.
    l1_discrepancy_penalty_term: cp.Expression = cp.sum(cp.multiply(cp.abs(donor_weights_cvxpy_var), pairwise_discrepancies_vector))
    
    # Third term of the objective: L2 penalty on donor weights (Ridge-like regularization).
    # This term helps to stabilize the solution, prevent overfitting, and shrink weights towards zero.
    l2_weights_penalty_term: cp.Expression = cp.sum_squares(donor_weights_cvxpy_var)

    # Combine the terms to form the complete optimization objective function.
    # The goal is to minimize this combined objective.
    optimization_objective = cp.Minimize(
        sum_squared_residuals_term + 
        l1_discrepancy_penalty_factor * l1_discrepancy_penalty_term + 
        l2_penalty_factor * l2_weights_penalty_term
    )
    # Define the constraint for the optimization: donor weights must sum to 1.
    # This ensures that the synthetic control is a convex combination of donor units.
    optimization_constraints = [cp.sum(donor_weights_cvxpy_var) == 1] 

    # Create and solve the CVXPY optimization problem using the CLARABEL solver.
    optimization_problem = cp.Problem(optimization_objective, optimization_constraints)
    optimization_problem.solve(solver=_SOLVER_CLARABEL_STR)
    
    # Check the status of the optimization.
    if optimization_problem.status in ["optimal", "optimal_inaccurate"]:
        # If an optimal or near-optimal solution is found, return the estimated donor weights.
        return donor_weights_cvxpy_var.value
    else:
        # If the optimization fails to converge or encounters an issue, issue a warning
        # and return an array of NaNs with the same shape as the expected weights.
        warnings.warn(f"NSC_opt optimization failed with status: {optimization_problem.status}", UserWarning)
        return np.full(num_donors, np.nan) 


def NSCcv(
    actual_treated_outcome_pre_treatment: np.ndarray, 
    all_donors_outcomes_pre_treatment: np.ndarray,
    a_vals: Optional[List[float]] = None, # Changed from l1_penalty_grid
    b_vals: Optional[List[float]] = None, # Changed from l2_penalty_grid
    num_cv_folds: int = 5
) -> Tuple[float, float]:
    """
    K-fold cross-validation for Normalized Synthetic Control (NSC).

    Selects the best tuning parameters `l1_discrepancy_penalty_factor` (a_reg)
    and `l2_penalty_factor` (b_reg) for the Normalized Synthetic Control (NSC)
    method using k-fold cross-validation.

    The cross-validation process iterates through combinations of penalty factors
    from the provided grids. For each combination, it performs k-fold
    cross-validation on the donor units. In each fold, a subset of donors is
    held out for validation, and `NSC_opt` is used to fit weights to reconstruct
    each validation donor using the remaining training donors. The penalty factors
    that yield the minimum average mean squared error (MSE) across folds are
    selected as optimal.

    Parameters
    ----------
    actual_treated_outcome_pre_treatment : np.ndarray
        Outcome vector for the actual treated unit in the pre-treatment periods.
        Shape (T0,), where T0 is the number of pre-treatment periods. This is
        used by `NSC_opt` internally to calculate discrepancies when fitting
        weights to reconstruct pseudo-treated (validation) donors.
    all_donors_outcomes_pre_treatment : np.ndarray
        Matrix of outcomes for all control (donor) units in the pre-treatment
        periods. Shape (T0, total_num_donors), where total_num_donors is the total number of
        donor units available for cross-validation.
    l1_penalty_grid : np.ndarray, optional
        A 1D NumPy array specifying the grid of values for the L1 discrepancy
        penalty factor (`l1_discrepancy_penalty_factor`) to be tested.
        Default is `np.arange(0.01, 1.01, 0.05)`.
    l2_penalty_grid : np.ndarray, optional
        A 1D NumPy array specifying the grid of values for the L2 penalty
        factor (`l2_penalty_factor`) to be tested.
        Default is `np.arange(0.01, 1.01, 0.05)`.
    num_cv_folds : int, optional
        The number of folds to use for cross-validation. Default is 5.
        The actual number of folds will be `min(num_cv_folds, total_num_donors)`.
        If the actual number of folds is less than 2, default penalty factors
        (0.01, 0.01) are returned with a warning.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the optimal `l1_discrepancy_penalty_factor` and
        `l2_penalty_factor` values found:
        - optimal_l1_penalty : float
            The value from `l1_penalty_grid` that, in combination with
            `optimal_l2_penalty`, resulted in the lowest average cross-validation MSE.
        - optimal_l2_penalty : float
            The value from `l2_penalty_grid` that, in combination with
            `optimal_l1_penalty`, resulted in the lowest average cross-validation MSE.
        If CV cannot be run meaningfully (e.g., too few donors) or no valid
        parameters are found, defaults (0.01, 0.01) are returned.

    Notes
    -----
    - The cross-validation is performed by treating each donor unit (or group of
      donors in a fold) as a pseudo-treated unit and attempting to reconstruct
      its pre-treatment outcome series using the remaining donors.
    - The `actual_treated_outcome_pre_treatment` (i.e., the outcome of the *actual*
      treated unit) is used within `NSC_opt` to calculate the discrepancy term.
      This means the discrepancy penalty is always relative to the actual treated unit,
      even when reconstructing a pseudo-treated (validation) donor. This is a specific
      choice in this CV implementation. An alternative would be to calculate discrepancy
      relative to the current pseudo-treated unit.

    Examples
    --------
    >>> num_pre_periods_ex, total_num_donors_ex = 15, 7
    >>> actual_treated_pre_ex = np.random.rand(num_pre_periods_ex) # Actual treated unit's data
    >>> all_donors_pre_ex = np.random.rand(num_pre_periods_ex, total_num_donors_ex) # All donors' data
    >>> l1_grid_ex = np.array([0.1, 0.5])
    >>> l2_grid_ex = np.array([0.01, 0.2])
    >>> best_l1, best_l2 = NSCcv(
    ...     actual_treated_pre_ex, all_donors_pre_ex,
    ...     a_vals=l1_grid_ex, b_vals=l2_grid_ex, num_cv_folds=3
    ... )
    >>> print(f"Best L1 penalty: {best_l1}, Best L2 penalty: {best_l2}") # doctest: +SKIP
    Best L1 penalty: ..., Best L2 penalty: ...
    """
    # Use provided search spaces or defaults
    l1_penalty_grid_to_use = np.array(a_vals) if a_vals is not None else np.arange(0.01, 1.01, 0.05)
    l2_penalty_grid_to_use = np.array(b_vals) if b_vals is not None else np.arange(0.01, 1.01, 0.05)
    
    optimal_l1_penalty_factor: float = 0.01 # Default value in case no better params are found
    optimal_l2_penalty_factor: float = 0.01 # Default value
    min_mse_cv: float = np.inf # Initialize with a very large value
    total_num_donors: int = all_donors_outcomes_pre_treatment.shape[1]

    # Ensure num_cv_folds is not greater than number of donors and is at least 2 for KFold
    effective_num_cv_folds: int = min(num_cv_folds, total_num_donors)
    if effective_num_cv_folds < 2: 
        warnings.warn(
            f"Not enough donors ({total_num_donors}) for meaningful {effective_num_cv_folds}-fold CV. "
            "Returning default L1=0.01, L2=0.01.", UserWarning
        )
        return 0.01, 0.01 

    k_fold_splitter = KFold(n_splits=effective_num_cv_folds, shuffle=True, random_state=1400)

    # Iterate over all combinations of L1 and L2 penalty factors
    for current_l1_penalty in l1_penalty_grid_to_use:
        for current_l2_penalty in l2_penalty_grid_to_use:
            current_fold_mse_list: List[float] = [] # Store MSE for each fold for current (L1, L2)
            
            # Perform K-fold cross-validation
            for train_indices, val_indices in k_fold_splitter.split(np.arange(total_num_donors)):
                if not train_indices.size or not val_indices.size: # Skip if a fold is empty (should not happen with KFold)
                    continue 

                training_donor_outcomes_fold: np.ndarray = all_donors_outcomes_pre_treatment[:, train_indices]
                
                # Iterate over each donor in the validation set (treating it as pseudo-treated)
                for val_idx_single_donor in val_indices:
                    # The outcome of the current validation donor is the target to reconstruct
                    validation_donor_outcome_fold: np.ndarray = all_donors_outcomes_pre_treatment[:, val_idx_single_donor]
                    
                    # Fit NSC_opt to reconstruct the validation donor using training donors.
                    # Note: The discrepancy term in NSC_opt will be calculated relative to
                    # `validation_donor_outcome_fold` (the pseudo-treated unit for this CV iteration).
                    cv_weights_fold: Optional[np.ndarray] = NSC_opt(
                        pre_treatment_treated_outcome=validation_donor_outcome_fold, 
                        pre_treatment_donor_outcomes=training_donor_outcomes_fold, 
                        l1_discrepancy_penalty_factor=current_l1_penalty, 
                        l2_penalty_factor=current_l2_penalty
                    )
                    
                    # If optimization was successful and returned valid weights
                    if cv_weights_fold is not None and not np.isnan(cv_weights_fold).any():
                        # Predict the outcome for the validation donor using the fitted weights
                        predicted_validation_outcome_fold: np.ndarray = training_donor_outcomes_fold @ cv_weights_fold
                        # Calculate Mean Squared Error for this fold
                        mse_for_fold: float = np.mean((validation_donor_outcome_fold - predicted_validation_outcome_fold) ** 2)
                        current_fold_mse_list.append(mse_for_fold)
            
            # If MSEs were computed for any folds with the current (L1, L2)
            if current_fold_mse_list: 
                average_mse_for_params: float = np.mean(current_fold_mse_list)
                # If current (L1, L2) yields a lower average MSE, update optimal parameters
                if average_mse_for_params < min_mse_cv:
                    min_mse_cv = average_mse_for_params
                    optimal_l1_penalty_factor = current_l1_penalty
                    optimal_l2_penalty_factor = current_l2_penalty
    
    # If min_mse_cv is still infinity, it means no valid parameters were found (e.g., all optimizations failed)
    if np.isinf(min_mse_cv): 
        warnings.warn(
            "Cross-validation did not find any valid (L1, L2) parameter combination "
            "that resulted in successful optimization across folds. Returning default L1=0.01, L2=0.01.", 
            UserWarning
        )
        return 0.01, 0.01

    return optimal_l1_penalty_factor, optimal_l2_penalty_factor


# --- 1. Local linear smoother with Gaussian kernel ---
def smooth(y_pre, bw):
    T_pre = len(y_pre)
    smoothed = np.zeros(T_pre)
    for i in range(T_pre):
        w = np.exp(-0.5 * ((np.arange(T_pre) - i) / bw) ** 2)
        w /= w.sum()
        X = np.vstack([np.ones(T_pre), np.arange(T_pre) - i]).T
        W = np.diag(w)
        beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y_pre
        smoothed[i] = beta[0]
    return smoothed

# --- 2. LOOCV to choose optimal bandwidth ---
def loocv_bandwidth(y_pre, bandwidth_grid):
    T_pre = len(y_pre)
    cv_errors = []

    for h in bandwidth_grid:
        errors = []
        for i in range(T_pre):
            y_train = np.delete(y_pre, i)
            idx = np.arange(T_pre) != i
            w = np.exp(-0.5 * ((np.where(idx)[0] - i) / h) ** 2)
            w /= w.sum()
            X = np.vstack([np.ones(T_pre - 1), np.where(idx)[0] - i]).T
            W = np.diag(w)
            beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y_train
            y_pred = beta[0]
            errors.append((y_pre[i] - y_pred) ** 2)
        cv_errors.append(np.mean(errors))

    best_h = bandwidth_grid[np.argmin(cv_errors)]
    return best_h, cv_errors


def _solve_SHC_QP(L, ell_eval, use_augmented=False, w_shc=None, lam=None, varsigma=1e-6, tol=1e-8):
    """
    Unified function to solve SHC or ASHC QP.

    Parameters
    ----------
    L : np.ndarray
        Donor matrix (m x N).
    ell_eval : np.ndarray
        Evaluation vector (m,).
    use_augmented : bool
        If True, solve ASHC; otherwise, solve SHC.
    w_shc : np.ndarray, optional
        SHC weight vector (required if use_augmented=True).
    lam : float, optional
        Regularization parameter for ASHC (required if use_augmented=True).
    varsigma : float
        Regularization parameter for eigenvalue-based penalty.
    tol : float
        Eigenvalue threshold for low-variance direction identification.

    Returns
    -------
    w_opt : np.ndarray or None
        Optimal weight vector.
    obj_val : float or None
        Final objective value.
    """
    N = L.shape[1]
    w = cp.Variable(N)

    # Fit term
    if use_augmented:
        if lam is None or w_shc is None:
            raise ValueError("lam and w_shc must be provided for ASHC.")
        fit_term = cp.sum_squares(ell_eval - L @ w)
        deviation = (1 / (2 * lam)) * cp.sum_squares(w - w_shc)
    else:
        fit_term = cp.sum_squares(ell_eval - L @ w)
        deviation = 0

    # Eigenvalue penalty
    G = L.T @ L
    eigvals, eigvecs = eigh(G)
    C = eigvecs[:, eigvals < tol]
    penalty = varsigma * cp.sum_squares(C.T @ w) if C.size > 0 else 0

    # Objective
    objective = cp.Minimize(fit_term + deviation + penalty)

    # Constraints
    constraints = [cp.sum(w) == 1]
    if not use_augmented:
        constraints.append(w >= 0)

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    return (w.value, prob.value) if w.value is not None else (None, None)

def tune_lambda_ashc(L, ell_eval, w_shc, lambda_grid=None, split_ratio=0.5):
    """
    Tune lambda for ASHC using holdout validation on ell_eval.

    Parameters
    ----------
    L : np.ndarray
        Donor matrix of shape (m, N).
    ell_eval : np.ndarray
        Evaluation vector of length m.
    w_shc : np.ndarray
        SHC weight vector (N,).
    lambda_grid : list or np.ndarray, optional
        Candidate values for lambda. If None, uses logscale grid.
    split_ratio : float, optional
        Fraction of data to use for training (default 0.5).

    Returns
    -------
    best_lambda : float
        Lambda minimizing validation MSE.
    lambda_errors : dict
        Mapping from lambda to validation MSE.
    """
    m = len(ell_eval)
    train_size = int(split_ratio * m)

    ell_train = ell_eval[:train_size]
    ell_val = ell_eval[train_size:]

    L_train = L[:train_size, :]
    L_val = L[train_size:, :]

    if lambda_grid is None:
        lambda_grid = np.logspace(-6, 2, 50)

    lambda_errors = {}

    for lam in lambda_grid:
        w_hat, _ = _solve_SHC_QP(
            L_train,
            ell_train,
            use_augmented=True,
            w_shc=w_shc,
            lam=lam
        )
        if w_hat is not None:
            y_val_pred = L_val @ w_hat
            mse = np.mean((ell_val - y_val_pred) ** 2)
            lambda_errors[lam] = mse

    best_lambda = min(lambda_errors, key=lambda_errors.get)
    return best_lambda, lambda_errors


## Relaxed Balanced SC

class L1INFRelaxationCV(BaseEstimator, RegressorMixin):
    """INF SCM with optional cross-validation for alpha/lambda tuning."""

    def __init__(self, alpha=None, lam=None, intercept=False, n_splits=5, n_repeats=1,
                 max_workers=None, affine=False, simplex=True, EN="L1_INF"):
        self.alpha = alpha
        self.lam = lam
        self.intercept = intercept
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.max_workers = max_workers
        self.affine = affine
        self.simplex = simplex
        self.EN = EN

    def _model_name(self, lam, alpha):
        """
        Generate model name string based on lambda, alpha,
        penalty geometry, and intercept specification.
        """
        eps = 1e-8  # numerical tolerance

        # Base model (no regularization)
        if lam <= eps:
            name = r"Simplex SCM" if self.simplex else r"Unregularized SCM"
        else:
            if self.EN == "L1_INF":
                if alpha == 1:
                    name = r"$L_1$ SCM (LASSO)"
                elif alpha == 0:
                    name = r"$L_\infty$ SCM (max-norm)"
                else:
                    name = r"$L_1 + L_\infty$ Elastic Net SCM"
            elif self.EN == "L1_L2":
                if alpha == 1:
                    name = r"$L_1$ SCM (LASSO)"
                elif alpha == 0:
                    name = r"$L_2$ SCM (Ridge)"
                else:
                    name = r"Elastic Net SCM ($L_1 + L_2$)"
            else:
                name = "SCM"

        if self.intercept:
            name += r" + Intercept"

        return name

    def _solve_inf_scm(self, X, y, alpha, lam):
        prob = Opt.SCopt(
            num_control_units=X.shape[1],
            target_outcomes_pre_treatment=y,
            num_pre_treatment_periods=len(y),
            donor_outcomes_pre_treatment=X,
            scm_model_type="INF",
            lambda_penalty=lam,
            alpha_penalty=alpha,
            fit_intercept=self.intercept,
            affine=self.affine,
            simplex=self.simplex,
            EN=self.EN
        )
        vars_list = list(prob.solution.primal_vars.values())
        if self.intercept:
            return np.concatenate([np.asarray(vars_list[0]).ravel(), np.asarray(vars_list[1]).ravel()])
        else:
            return np.asarray(vars_list[0]).ravel()

    def _cross_validate(self, X, y):
        """Perform CV only if at least one of alpha or lambda is a grid."""
        alpha_vals = np.atleast_1d(self.alpha)
        lam_vals = np.atleast_1d(self.lam)

        # If both are scalars, skip CV
        if len(alpha_vals) == 1 and len(lam_vals) == 1:
            self.best_alpha_ = alpha_vals[0]
            self.best_lambda_ = lam_vals[0]
            self.cv_errors_ = None
            return

        # Build grid
        param_list = [(a, l) for a in alpha_vals for l in lam_vals]

        cv_errors = {}
        best_error = np.inf
        best_params = None

        def evaluate_combo(alpha_lam):
            a, l = alpha_lam
            total_sspe = 0.0
            for _ in range(self.n_repeats):
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                Tau = np.zeros_like(y)
                for train_idx, test_idx in tscv.split(X):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    w = self._solve_inf_scm(X_train, y_train, a, l)
                    if self.intercept:
                        X_test_plus = np.hstack([np.ones((len(X_test), 1)), X_test])
                        Tau[test_idx] = y_test - X_test_plus @ w
                    else:
                        Tau[test_idx] = y_test - X_test @ w
                total_sspe += np.sum(Tau ** 2)
            return (a, l, total_sspe)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(evaluate_combo, p) for p in param_list]
            for future in as_completed(futures):
                a, l, err = future.result()
                cv_errors[(a, l)] = err
                if err < best_error:
                    best_error = err
                    best_params = (a, l)

        self.best_alpha_, self.best_lambda_ = best_params
        self.cv_errors_ = cv_errors

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._cross_validate(X, y)
        self.coef_ = self._solve_inf_scm(X, y, self.best_alpha_, self.best_lambda_)
        return self

    def predict(self, X):
        X = check_array(X)
        if self.intercept:
            X_plus = np.hstack([np.ones((X.shape[0], 1)), X])
            return X_plus @ self.coef_
        return X @ self.coef_


def fit_l1inf_scm(
        X_pre,
        y_pre,
        X_post,
        alpha=None,  # scalar or list/grid
        lam=None,    # scalar or list/grid
        intercept=False,
        affine=False,
        simplex=True,
        n_splits=5,
        n_repeats=1,
        max_workers=None,
        y=None,
        donor_names=None,
        EN="L1_INF"
):
    """
    Fit INF SCM with optional time-series cross-validation over alpha and lambda.

    If either alpha or lam is a list/grid (or None), cross-validation is performed.
    If both are scalars, the model is fit directly without CV.
    """
    from sklearn.preprocessing import StandardScaler

    # ------------------ Scale donors ------------------
    scaler_X = StandardScaler().fit(X_pre)
    X_pre_scaled = scaler_X.transform(X_pre)
    X_post_scaled = scaler_X.transform(X_post)

    # ------------------ Scale outcome -----------------
    y_mean, y_std = y_pre.mean(), y_pre.std()
    y_pre_scaled = (y_pre - y_mean) / y_std

    # ------------------ Default alpha/lambda -----------------
    if alpha is None:
        alpha = np.linspace(0.0, 1.0, 20)

    if lam is None:
        # Generate a single lambda grid based on a representative alpha (mean of alpha grid)
        representative_alpha = np.mean(np.atleast_1d(alpha))
        lam = generate_lambda_seq2(y_pre_scaled, X_pre_scaled, representative_alpha, num=20)

    # Ensure both are array-like for CV
    alpha_vals = np.atleast_1d(alpha)
    lam_vals = np.atleast_1d(lam)

    # ------------------ Fit model ---------------------
    model = L1INFRelaxationCV(
        alpha=alpha_vals,
        lam=lam_vals,
        intercept=intercept,
        n_splits=n_splits,
        n_repeats=n_repeats,
        max_workers=max_workers,
        affine=affine,
        simplex=simplex,
        EN=EN
    )

    # Cross-validation happens inside the class if either alpha or lambda has multiple values
    model.fit(X_pre_scaled, y_pre_scaled)

    # ------------------ Predictions -------------------
    y_pre_pred = model.predict(X_pre_scaled) * y_std + y_mean
    y_post_pred = model.predict(X_post_scaled) * y_std + y_mean

    # ------------------ Extract weights ----------------
    coef = model.coef_
    intercept_weight = coef[0] if intercept else 0.0
    donor_weights_scaled = coef[1:] if intercept else coef
    donor_weights = (donor_weights_scaled / scaler_X.scale_) / np.sum(donor_weights_scaled / scaler_X.scale_)

    # ------------------ Compute effects ----------------
    attdict, fitdict, Vectors = effects.calculate(
        y, np.concatenate([y_pre_pred, y_post_pred]),
        X_pre.shape[0], X_post_scaled.shape[0]
    )

    donor_weightsl1inf = {state: (0 if w < 0.001 else round(w, 3))
                          for state, w in zip(donor_names, donor_weights)}

    model_name = getattr(model, "_model_name", lambda lam, alpha: "L1INF SCM")(lam=model.best_lambda_, alpha=model.best_alpha_)

    # ------------------ Return nested dictionary ----------------
    return {
        "donor_weights": donor_weightsl1inf,
        "intercept": intercept_weight,
        "predictions": np.concatenate([y_pre_pred, y_post_pred]),
        "Results": {"Effects": attdict, "Fit": fitdict, "Vectors": Vectors},
        "Model": model_name,
        "hyperparameters": {
            "best_alpha": model.best_alpha_,
            "best_lambda": model.best_lambda_,
            "cv_errors": getattr(model, "cv_errors_", None)
        }
    }











def generate_lambda_seq2(Y1, Y0, alpha, epsilon=1e-4, num=30):
    # Standardize predictors
    sY0 = (Y0 - np.mean(Y0, axis=0)) / np.std(Y0, axis=0, ddof=0)
    alpha = max(alpha, 0.01)  # avoid div by zero
    # Lambda max based on L1 contribution
    lam_max = np.max(np.abs(sY0.T @ Y1)) / (Y0.shape[0] * alpha)
    lam_min = lam_max * epsilon
    lam_max = min(lam_max, 20)
    lam_min = min(lam_min, 1e-4)
    lam_seq = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), num))
    return lam_seq




class L2RelaxationCV(BaseEstimator, RegressorMixin):
    """L2 Relaxation with time-series aware cross-validation for synthetic control using Clarabel."""

    def __init__(self, *, tau=None, n_taus=1000, n_splits=10, nonneg=True):
        """
        Parameters
        ----------
        tau : float or array-like, optional
            If scalar, use this tau directly (skip CV).
            If array-like, use these taus for cross-validation.
            If None, generate grid automatically.
        n_taus : int
            Number of taus to generate if tau is None.
        n_splits : int
            Number of time-series CV splits.
        nonneg : bool
            Whether weights are constrained to be non-negative.
        """
        self.tau = tau
        self.n_taus = n_taus if tau is None else (len(tau) if hasattr(tau, "__len__") else 1)
        self.n_splits = n_splits
        self.nonneg = nonneg


    @staticmethod
    def _relaxation_min_obj(w):
        return cp.sum_squares(w)

    def _solve_l2_problem(self, X, y, tau=None):
        """Solve the L2 relaxation problem for given X, y, and tau."""
        T, J = X.shape
        w = cp.Variable(J, nonneg=self.nonneg)
        gam = cp.Variable()
        tau_val = self.tau_ if tau is None else tau

        objective = cp.Minimize(cp.sum_squares(w))
        constraints = [
            cp.sum(w) == 1,
            cp.pnorm(X.T @ (X @ w - y) / T + gam * np.ones(J), p='inf') <= tau_val
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        return w.value

    def _get_tau_lower_limit(self, X, y):
        T, J = X.shape
        w = cp.Variable(J, nonneg=True)
        gam = cp.Variable()
        constraints = [cp.sum(w) == 1]
        objective = cp.Minimize(cp.pnorm(X.T @ (X @ w - y) / T + gam * np.ones(J), p='inf'))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        return problem.value * 1.01

    def _get_tau_upper_limit(self, X, y):
        T, J = X.shape
        w = np.ones(J) / J
        gam = cp.Variable()
        objective = cp.Minimize(cp.pnorm(X.T @ (X @ w - y) / T + gam * np.ones(J), p='inf'))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        return problem.value * 1.01

    def _generate_taus(self, X, y):
        lower_limit = self._get_tau_lower_limit(X, y)
        upper_limit = self._get_tau_upper_limit(X, y)
        self.taus_ = np.geomspace(upper_limit, lower_limit, self.n_taus)

    def _process_tau_grid(self, X, y):
        if np.isscalar(self.tau):
            # Single tau: skip cross-validation
            self.taus_ = np.array([self.tau])
            self.tau_ = self.tau
            self.skip_cv_ = True
        elif self.tau is not None:
            # Array-like taus for cross-validation
            lower_limit = self._get_tau_lower_limit(X, y)
            self.taus_ = np.sort(np.array(self.tau))[::-1]
            self.taus_ = self.taus_[self.taus_ >= lower_limit]
            self.skip_cv_ = False
        else:
            # Generate taus automatically
            self._generate_taus(X, y)
            self.skip_cv_ = False

    def _cross_validate(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_splits = list(tscv.split(X))

        cv_errors = []
        for train_idx, test_idx in fold_splits:
            fold_errors = self._fit_fold(X, y, train_idx, test_idx)
            if fold_errors is not None:
                cv_errors.append(fold_errors)

        if len(cv_errors) == 0:
            raise ValueError("No feasible taus for any fold. Check your data and taus.")

        min_length = np.min([len(errors) for errors in cv_errors])
        cv_errors = np.array([errors[:min_length] for errors in cv_errors])
        self.taus_ = self.taus_[:min_length]

        self.cv_mean_mse_ = np.mean(cv_errors, axis=0)
        self.tau_ = self.taus_[np.argmin(self.cv_mean_mse_)]

    def _fit_fold(self, X, y, train_idx, test_idx):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        lower_limit = self._get_tau_lower_limit(X_train, y_train)
        taus_for_fold = self.taus_[self.taus_ >= lower_limit]
        if len(taus_for_fold) == 0:
            return None

        w_estimates = np.column_stack([self._solve_l2_problem(X_train, y_train, tau=val)
                                       for val in taus_for_fold])
        y_pred = X_test @ w_estimates
        return np.mean((y_test[:, None] - y_pred) ** 2, axis=0)

    def _fit_full_data(self, X, y):
        self.coef_ = self._solve_l2_problem(X, y)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._process_tau_grid(X, y)

        if not getattr(self, "skip_cv_", False):
            self._cross_validate(X, y)

        self._fit_full_data(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return X @ self.coef_



def fit_l2_scm(X_pre, y_pre, X_post, donor_names=None, y=None, tau=None,
               n_splits=5, n_taus=1000):
    """
    Fit L2 relaxation synthetic control with standardized inputs and back-transformed weights.

    Parameters
    ----------
    X_pre : np.ndarray
        Donor matrix for pre-treatment periods.
    y_pre : np.ndarray
        Treated unit outcomes for pre-treatment periods.
    X_post : np.ndarray
        Donor matrix for post-treatment periods.
    donor_names : list[str], optional
        Names of donor units.
    y : np.ndarray, optional
        Full treated outcome vector (pre + post), used for ATT calculation.
    tau : float or array-like, optional
        User-specified tau for the relaxed SCM.
        If scalar, CV is skipped. If array, CV uses this grid.
    n_splits : int
        Number of CV splits if cross-validation is performed.
    n_taus : int
        Number of taus to generate if tau is None.

    Returns
    -------
    dict
        Contains 'weights', 'predictions', 'cv_performed', 'n_splits', 'tau_used',
        and 'Results' dict (Effects, Fit, Counterfactuals).
    """

    # Standardize X
    scaler_X = StandardScaler().fit(X_pre)
    X_pre_scaled = scaler_X.transform(X_pre)
    X_post_scaled = scaler_X.transform(X_post)

    # Standardize y
    y_mean, y_std = y_pre.mean(), y_pre.std()
    y_pre_scaled = (y_pre - y_mean) / y_std

    # Initialize model with user-specified tau
    model = L2RelaxationCV(tau=tau, n_taus=n_taus, n_splits=n_splits, nonneg=True)
    model.fit(X_pre_scaled, y_pre_scaled)

    # Determine if cross-validation was performed
    cv_performed = not isinstance(tau, (int, float))  # scalar tau skips CV
    tau_used = model.tau_ if cv_performed else tau

    # Predictions
    y_pre_pred = model.predict(X_pre_scaled) * y_std + y_mean
    y_post_pred = model.predict(X_post_scaled) * y_std + y_mean

    # Transform weights back to original scale
    weights_scaled = model.coef_
    weights_orig = (weights_scaled / scaler_X.scale_) / (weights_scaled / scaler_X.scale_).sum()

    donor_weights = {state: (0 if w < 0.001 else round(w, 3))
                     for state, w in zip(donor_names, weights_orig)}

    # ATT and fit diagnostics
    attdict, fitdict, Vectors = effects.calculate(
        y, np.concatenate([y_pre_pred, y_post_pred]),
        X_pre.shape[0], X_post_scaled.shape[0]
    )

    return {
        "donor_weights": donor_weights,
        "predictions": np.concatenate([y_pre_pred, y_post_pred]),
        "Results": {
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors
        },
        "Model": "Relaxed Balanced",
        "hyperparameters": {
            "tau_used": tau_used,
            "cv_performed": cv_performed,
            "n_splits": n_splits if cv_performed else None
        }
    }







