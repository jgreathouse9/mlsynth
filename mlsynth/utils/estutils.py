import numpy as np
import cvxpy as cp
from mlsynth.utils.denoiseutils import (
    universal_rank,
    svt,
    RPCA,
    spectral_rank,
    RPCA_HQF,
)
from mlsynth.utils.resultutils import effects
from mlsynth.utils.selector_helpers import (
    determine_optimal_clusters,
    SVDCluster,
    PDAfs,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Not used in this file
from sklearn.model_selection import KFold
from functools import partial # Not used in this file
from statsmodels.tsa.stattools import acf
from scipy.stats import norm # zconfint was not explicitly imported and not used
from screenot.ScreeNOT import adaptiveHardThresholding # Not used in this file
from sklearn.linear_model import LassoCV
from scipy.stats import t as t_dist
from mlsynth.utils.bayesutils import BayesSCM
from mlsynth.utils.selector_helpers import fpca
import warnings # For RPCASYNTH warning
import pandas as pd

from typing import Any, Tuple, List, Optional, Dict, Union, Callable
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError

# --- Constants for SCM and PDA Method Names ---
_SCM_MODEL_SIMPLEX = "SIMPLEX"
_SCM_MODEL_MSCA = "MSCa"
_SCM_MODEL_MSCB = "MSCb"
_SCM_MODEL_MSCC = "MSCc"
_SCM_MODEL_OLS = "OLS"
_SCM_MODEL_MA = "MA" # Model Averaging

_PDA_METHOD_FS = "fs" # Forward Selection
_PDA_METHOD_LASSO = "LASSO"
_PDA_METHOD_L2 = "l2" # L2-relaxation

# --- Constants for CVXPY Solvers --- #
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

    estimated_coefficients = donor_weights_cvxpy.value
    estimated_intercept = (
        np.mean(treated_unit_subset) -
        np.mean(donor_outcomes_subset @ estimated_coefficients)
    )
    predicted_counterfactuals = donor_outcomes_matrix @ estimated_coefficients + estimated_intercept

    return estimated_coefficients, estimated_intercept, predicted_counterfactuals



def cross_validate_tau(
    pre_treatment_treated_outcome: np.ndarray,
    pre_treatment_donor_outcomes: np.ndarray,
    tau_upper_bound_for_grid: float,
    num_coarse_points: int = 10,
    num_fine_points: int = 20,
    zoom_width: float = 0.5
) -> Tuple[float, float]:
    """
    Adaptive cross-validation for tau in L2-relax estimator.
    Skips over tau values that cause solver failure.
    """
    num_pre_periods = len(pre_treatment_treated_outcome)
    half = num_pre_periods // 2

    y_train = pre_treatment_treated_outcome[:half]
    X_train = pre_treatment_donor_outcomes[:half, :]
    y_val = pre_treatment_treated_outcome[half:]
    X_val = pre_treatment_donor_outcomes[half:, :]

    def mse_for_tau(tau_val: float) -> float:
        try:
            coefs, _, _ = l2_relax(
                num_pre_treatment_estimation_periods=half,
                treated_unit_outcome_vector=y_train,
                donor_outcomes_matrix=X_train,
                sup_norm_constraint_tau=tau_val
            )
            preds = X_val @ coefs
            return np.mean((y_val - preds) ** 2)
        except Exception:
            return 1e6  # large MSE on failure

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


def _solve_pda_fs(
    prepped_data: Dict[str, Any], max_donors_to_select: int
) -> Dict[str, Any]:
    """
    Solve Panel Data Approach (PDA) using Forward Selection ('fs').

    Parameters
    ----------
    prepped_data : Dict[str, Any]
        Preprocessed data dictionary. Expected keys:
        - 'y' (outcome_vector_treated): np.ndarray, outcome vector for the treated unit.
        - 'donor_matrix' (donor_outcomes_matrix): np.ndarray, matrix of outcomes/features for donor units.
        - 'donor_names' (all_donor_names): Union[pd.Index, List[str]], names of the original donor units.
        - 'pre_periods' (num_pre_treatment_periods): int, number of pre-treatment time periods.
        - 'post_periods' (num_post_treatment_periods): int, number of post-treatment time periods.
    max_donors_to_select : int
        Maximum number of donor units to select by the forward selection algorithm.

    Returns
    -------
    Dict[str, Any]
        Results dictionary for the 'fs' method.
    """
    # Extract data from the preprocessed dictionary
    outcome_vector_treated: np.ndarray = prepped_data["y"]
    donor_outcomes_matrix: np.ndarray = prepped_data["donor_matrix"]
    all_donor_names = prepped_data["donor_names"]
    num_pre_treatment_periods: int = prepped_data["pre_periods"]
    num_post_treatment_periods: int = prepped_data["post_periods"]

    actual_total_donors = donor_outcomes_matrix.shape[1]
    forward_selection_results: Dict[str, Any] = PDAfs(
        outcome_vector_treated,
        donor_outcomes_matrix,
        num_pre_treatment_periods,
        num_pre_treatment_periods + num_post_treatment_periods, # total_time_periods
        actual_total_donors, # total_num_donors
        # max_donors_to_select is not directly used by PDAfs signature if it expects total_num_donors
        # If PDAfs was meant to use max_donors_to_select as a cap, its signature/logic would need change.
        # For now, ensuring total_num_donors is correct based on the matrix.
    )
    selected_donor_indices: np.ndarray = forward_selection_results["selected_donor_indices"]
    forward_selection_model_coefficients: np.ndarray = forward_selection_results[
        "final_model_coefficients"
    ]

    selected_donor_names_fs: List[str] = all_donor_names[
        selected_donor_indices
    ].tolist()

    forward_selection_donor_coefficients_dict: Dict[str, float] = {
        name: round(coef, 2)
        for name, coef in zip(
            selected_donor_names_fs, forward_selection_model_coefficients[1:]
        )
    }

    selected_donor_outcomes_fs: np.ndarray = donor_outcomes_matrix[
        :, selected_donor_indices
    ]
    forward_selection_intercept: float = forward_selection_model_coefficients[0]
    assert selected_donor_outcomes_fs.shape[1] == forward_selection_model_coefficients[1:].shape[0],\
    f"Mismatch: donors={selected_donor_outcomes_fs.shape[1]}, coefs={forward_selection_model_coefficients[1:].shape[0]}"

    forward_selection_counterfactual_outcome: np.ndarray = (
        forward_selection_intercept
        + selected_donor_outcomes_fs @ forward_selection_model_coefficients[1:]
    )

    forward_selection_gap_vector: np.ndarray = (
        outcome_vector_treated[num_pre_treatment_periods:]
        - forward_selection_counterfactual_outcome[num_pre_treatment_periods:]
    )
    long_run_variance_lag_fs: int = int(
        np.floor(4 * (num_post_treatment_periods / 100) ** (2 / 9))
    )

    long_run_variance_fs: float
    if (
        long_run_variance_lag_fs > 0
        and len(forward_selection_gap_vector) > long_run_variance_lag_fs
    ):
        autocovariances_fs: np.ndarray = acf(
            forward_selection_gap_vector, nlags=long_run_variance_lag_fs, fft=False
        )
        autocovariance_weights_fs: np.ndarray = 1 - (
            np.arange(1, long_run_variance_lag_fs + 1)
        ) / (long_run_variance_lag_fs + 1)
        variance_of_gap_fs = np.var(forward_selection_gap_vector, ddof=0)
        long_run_variance_fs = variance_of_gap_fs * (
            1 + 2 * np.sum(autocovariance_weights_fs * autocovariances_fs[1:])
        )
    else:
        long_run_variance_fs = np.var(forward_selection_gap_vector, ddof=1)

    long_run_standard_error_ate_fs: float = (
        np.sqrt(long_run_variance_fs / num_post_treatment_periods)
        if num_post_treatment_periods > 0
        else np.nan
    )
    mean_of_gap_fs: float = np.mean(forward_selection_gap_vector)

    z_statistic_fs: float = (
        mean_of_gap_fs / long_run_standard_error_ate_fs
        if long_run_standard_error_ate_fs > 0
        else np.nan
    )
    p_value_fs: float = (
        2 * (1 - norm.cdf(np.abs(z_statistic_fs)))
        if not np.isnan(z_statistic_fs)
        else np.nan
    )

    critical_value_fs: float = norm.ppf(1 - 0.05 / 2)
    confidence_interval_lower_bound_fs: float = (
        mean_of_gap_fs - critical_value_fs * long_run_standard_error_ate_fs
    )
    confidence_interval_upper_bound_fs: float = (
        mean_of_gap_fs + critical_value_fs * long_run_standard_error_ate_fs
    )
    confidence_interval_fs: Tuple[float, float] = (
        confidence_interval_lower_bound_fs,
        confidence_interval_upper_bound_fs,
    )

    inference_results_fs: Dict[str, Any] = {
        "t_stat": z_statistic_fs,
        "SE": long_run_standard_error_ate_fs,
        "95% CI": confidence_interval_fs,
        "p_value": p_value_fs,
    }
    att_results_fs, fit_diagnostics_fs, time_series_vectors_fs = effects.calculate(
        observed_outcome_series=outcome_vector_treated,
        counterfactual_outcome_series=forward_selection_counterfactual_outcome,
        num_pre_treatment_periods=num_pre_treatment_periods,
        num_actual_post_periods=num_post_treatment_periods,
    )
    return {
        "method": "fs",
        "Betas": forward_selection_donor_coefficients_dict,
        "Effects": att_results_fs,
        "Fit": fit_diagnostics_fs,
        "Vectors": time_series_vectors_fs,
        "Inference": inference_results_fs,
    }


def _solve_pda_lasso(prepped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve Panel Data Approach (PDA) using LASSO.

    Parameters
    ----------
    prepped_data : Dict[str, Any]
        Preprocessed data dictionary. Expected keys:
        - 'y' (outcome_vector_treated): np.ndarray, outcome vector for the treated unit.
        - 'donor_matrix' (donor_outcomes_matrix): np.ndarray, matrix of outcomes/features for donor units.
        - 'donor_names' (all_donor_names): Union[pd.Index, List[str]], names of the original donor units.
        - 'pre_periods' (num_pre_treatment_periods): int, number of pre-treatment time periods.
        - 'post_periods' (num_post_treatment_periods): int, number of post-treatment time periods.

    Returns
    -------
    Dict[str, Any]
        Results dictionary for the 'LASSO' method.
    """
    # Extract data from the preprocessed dictionary
    outcome_vector_treated: np.ndarray = prepped_data["y"]
    donor_outcomes_matrix: np.ndarray = prepped_data["donor_matrix"]
    all_donor_names = prepped_data["donor_names"]
    num_pre_treatment_periods: int = prepped_data["pre_periods"]
    num_post_treatment_periods: int = prepped_data["post_periods"]

    pre_treatment_outcome_lasso: np.ndarray = outcome_vector_treated[
        :num_pre_treatment_periods
    ]
    pre_treatment_donor_outcomes_lasso: np.ndarray = donor_outcomes_matrix[
        :num_pre_treatment_periods, :
    ]

    lasso_cv_model = LassoCV(
        cv=min(num_pre_treatment_periods, 5), random_state=1400
    )
    lasso_cv_model.fit(
        pre_treatment_donor_outcomes_lasso, pre_treatment_outcome_lasso
    )
    lasso_counterfactual_outcome: np.ndarray = lasso_cv_model.predict(
        donor_outcomes_matrix
    )

    non_zero_indices_lasso: np.ndarray = np.where(lasso_cv_model.coef_ != 0)[0]
    non_zero_names_lasso: List[str] = [
        all_donor_names[i] for i in non_zero_indices_lasso
    ]
    non_zero_coefficients_dict_lasso: Dict[str, float] = dict(
        zip(
            non_zero_names_lasso,
            np.round(lasso_cv_model.coef_[non_zero_indices_lasso], 3),
        )
    )
    att_results_lasso, fit_diagnostics_lasso, time_series_vectors_lasso = (
        effects.calculate(
            observed_outcome_series=outcome_vector_treated,
            counterfactual_outcome_series=lasso_counterfactual_outcome,
            num_pre_treatment_periods=num_pre_treatment_periods,
            num_actual_post_periods=num_post_treatment_periods,
        )
    )

    def compute_lasso_inference(
        observed_pre_treatment_outcome: np.ndarray,
        predicted_full_period_outcome: np.ndarray,
        pre_treatment_donor_outcomes_inference: np.ndarray,
        num_pre_treatment_periods_inference: int,
        num_post_treatment_periods_inference: int,
        average_treatment_effect_inference: float,
        significance_level_inference: float = 0.05,
    ) -> Dict[str, Any]:
        pre_treatment_residuals_inference: np.ndarray = (
            observed_pre_treatment_outcome
            - predicted_full_period_outcome[:num_pre_treatment_periods_inference]
        )
        error_variance_inference: float = np.mean(
            pre_treatment_residuals_inference**2
        )
        mean_of_pre_treatment_donors_inference: np.ndarray = np.mean(
            pre_treatment_donor_outcomes_inference, axis=0
        ).reshape(-1, 1)
        covariance_structure_of_pre_treatment_donors_inference: np.ndarray = (
            pre_treatment_donor_outcomes_inference.T
            @ pre_treatment_donor_outcomes_inference
            / num_pre_treatment_periods_inference
        )
        try:
            inverse_covariance_structure_inference = np.linalg.inv(
                covariance_structure_of_pre_treatment_donors_inference
            )
        except np.linalg.LinAlgError:
            inverse_covariance_structure_inference = np.linalg.pinv(
                covariance_structure_of_pre_treatment_donors_inference
            )
        omega_component1_inference: float = (
            error_variance_inference
            * mean_of_pre_treatment_donors_inference.T
        ) @ inverse_covariance_structure_inference @ mean_of_pre_treatment_donors_inference
        total_omega_inference: float = (
            num_post_treatment_periods_inference
            / num_pre_treatment_periods_inference
        ) * omega_component1_inference + error_variance_inference
        standard_error_inference: float = (
            np.sqrt(total_omega_inference / num_post_treatment_periods_inference).item()
            if num_post_treatment_periods_inference > 0
            else np.nan
        )
        t_statistic_inference: float = (
            average_treatment_effect_inference / standard_error_inference
            if standard_error_inference > 0
            else np.nan
        )
        critical_value_inference: float = norm.ppf(
            1 - significance_level_inference / 2
        )
        confidence_interval_lower_bound_inference: float = (
            average_treatment_effect_inference
            - critical_value_inference * standard_error_inference
        )
        confidence_interval_upper_bound_inference: float = (
            average_treatment_effect_inference
            + critical_value_inference * standard_error_inference
        )
        confidence_interval_inference: Tuple[float, float] = (
            confidence_interval_lower_bound_inference,
            confidence_interval_upper_bound_inference,
        )
        return {
            "CI": confidence_interval_inference,
            "t_stat": t_statistic_inference,
            "SE": standard_error_inference,
        }

    inference_results_lasso: Dict[str, Any] = compute_lasso_inference(
        pre_treatment_outcome_lasso,
        lasso_counterfactual_outcome,
        pre_treatment_donor_outcomes_lasso,
        num_pre_treatment_periods,
        num_post_treatment_periods,
        att_results_lasso["ATT"],
    )
    return {
        "method": "LASSO",
        "non_zero_coef_dict": non_zero_coefficients_dict_lasso,
        "Inference": inference_results_lasso,
        "Effects": att_results_lasso,
        "Fit": fit_diagnostics_lasso,
        "Vectors": time_series_vectors_lasso,
    }


def _solve_pda_l2(
    prepped_data: Dict[str, Any], l2_regularization_parameter: Optional[float]
) -> Dict[str, Any]:
    """
    Solve Panel Data Approach (PDA) using L2-relaxation.

    Parameters
    ----------
    prepped_data : Dict[str, Any]
        Preprocessed data dictionary. Expected keys:
        - 'y' (outcome_vector_treated): np.ndarray, outcome vector for the treated unit.
        - 'donor_matrix' (donor_outcomes_matrix): np.ndarray, matrix of outcomes/features for donor units.
        - 'donor_names' (all_donor_names): Union[pd.Index, List[str]], names of the original donor units.
        - 'pre_periods' (num_pre_treatment_periods): int, number of pre-treatment time periods.
        - 'post_periods' (num_post_treatment_periods): int, number of post-treatment time periods.
    l2_regularization_parameter : Optional[float]
        Tuning parameter (tau) for L2-relaxation. If None, it's cross-validated.

    Returns
    -------
    Dict[str, Any]
        Results dictionary for the 'l2' method.
    """
    # Extract data from the preprocessed dictionary
    outcome_vector_treated: np.ndarray = prepped_data["y"]
    donor_outcomes_matrix: np.ndarray = prepped_data["donor_matrix"]
    all_donor_names = prepped_data["donor_names"]
    num_pre_treatment_periods: int = prepped_data["pre_periods"]
    num_post_treatment_periods: int = prepped_data["post_periods"]

    pre_treatment_outcome_l2: np.ndarray = outcome_vector_treated[
        :num_pre_treatment_periods
    ]
    pre_treatment_donor_outcomes_l2: np.ndarray = donor_outcomes_matrix[
        :num_pre_treatment_periods, :
    ]

    initial_tau_for_cv_l2: float = 1.5

    selected_l2_regularization_parameter: float
    if l2_regularization_parameter is not None:
        selected_l2_regularization_parameter = l2_regularization_parameter
    else:
        selected_l2_regularization_parameter, min_mse = adaptive_cross_validate_tau(
            pre_treatment_treated_outcome=pre_treatment_outcome_l2,
            pre_treatment_donor_outcomes=pre_treatment_donor_outcomes_l2,
            tau_upper_bound_for_grid=5)

    estimated_coefficients_l2, estimated_intercept_l2, _ = l2_relax(
        num_pre_treatment_periods,
        outcome_vector_treated,
        donor_outcomes_matrix,
        selected_l2_regularization_parameter,
    )
    l2_counterfactual_outcome: np.ndarray = (
        donor_outcomes_matrix @ estimated_coefficients_l2 + estimated_intercept_l2
    )

    att_results_l2, fit_diagnostics_l2, time_series_vectors_l2 = effects.calculate(
        observed_outcome_series=outcome_vector_treated,
        counterfactual_outcome_series=l2_counterfactual_outcome,
        num_pre_treatment_periods=num_pre_treatment_periods,
        num_actual_post_periods=num_post_treatment_periods,
    )

    def compute_l2_inference(
        average_treatment_effect_l2_inference: float,
        post_treatment_gap_vector_l2_inference: np.ndarray,
        num_post_treatment_periods_l2_inference: int,
        confidence_level_l2_inference: float = 0.95,
    ) -> Dict[str, Any]:
        hac_lag_l2_inference: int = int(
            np.floor(
                4 * (num_post_treatment_periods_l2_inference / 100) ** (2 / 9)
            )
        )
        hac_variance_l2_inference: float = compute_hac_variance(
            post_treatment_gap_vector_l2_inference, hac_lag_l2_inference
        )
        standard_error_l2_inference: float = (
            np.sqrt(
                hac_variance_l2_inference / num_post_treatment_periods_l2_inference
            )
            if num_post_treatment_periods_l2_inference > 0
            and hac_variance_l2_inference >= 0
            else np.nan
        )
        t_statistic_l2_inference: float = (
            average_treatment_effect_l2_inference / standard_error_l2_inference
            if standard_error_l2_inference > 0
            else np.nan
        )
        degrees_of_freedom_l2_inference: int = (
            num_post_treatment_periods_l2_inference - 1
            if num_post_treatment_periods_l2_inference > 1
            else 1
        )
        p_value_l2_inference: float = (
            2
            * (
                1
                - t_dist.cdf(
                    np.abs(t_statistic_l2_inference),
                    df=degrees_of_freedom_l2_inference,
                )
            )
            if not np.isnan(t_statistic_l2_inference)
            else np.nan
        )
        t_critical_value_l2_inference: float = t_dist.ppf(
            1 - (1 - confidence_level_l2_inference) / 2,
            df=degrees_of_freedom_l2_inference,
        )
        margin_of_error_l2_inference: float = (
            t_critical_value_l2_inference * standard_error_l2_inference
        )
        confidence_interval_lower_bound_l2_inference: float = (
            average_treatment_effect_l2_inference - margin_of_error_l2_inference
        )
        confidence_interval_upper_bound_l2_inference: float = (
            average_treatment_effect_l2_inference + margin_of_error_l2_inference
        )
        return {
            "t_stat": t_statistic_l2_inference,
            "standard_error": standard_error_l2_inference,
            "p_value": p_value_l2_inference,
            "confidence_interval": (
                confidence_interval_lower_bound_l2_inference,
                confidence_interval_upper_bound_l2_inference,
            ),
        }

    inference_results_l2: Dict[str, Any] = compute_l2_inference(
        att_results_l2["ATT"],
        time_series_vectors_l2["Gap"][-num_post_treatment_periods:],
        num_post_treatment_periods,
    )
    l2_donor_coefficients_dict: Dict[str, float] = {
        name: round(coef, 4)
        for name, coef in zip(all_donor_names, estimated_coefficients_l2)
    }
    return {
        "method": r"l2 relaxation",
        "optimal_tau": selected_l2_regularization_parameter,
        "Betas": l2_donor_coefficients_dict,
        "Inference": inference_results_l2,
        "Effects": att_results_l2,
        "Fit": fit_diagnostics_l2,
        "Vectors": time_series_vectors_l2,
        "Intercept": estimated_intercept_l2,
    }


def pcr(
    donor_outcomes_matrix: np.ndarray,
    treated_unit_outcome_vector: np.ndarray,
    scm_objective_model_type: str,
    all_donor_names: List[str],
    num_pre_treatment_periods: int = 10,
    enable_clustering: bool = False,
    use_frequentist_scm: bool = False,
) -> Dict[str, Any]:
    # Minimum number of pre-treatment periods required for PCR.
    # This is necessary for meaningful SVT and SCM fitting.
    MIN_PRE_PERIODS_FOR_PCR = 2 
    if num_pre_treatment_periods < MIN_PRE_PERIODS_FOR_PCR:
        raise MlsynthDataError(
            f"PCR requires at least {MIN_PRE_PERIODS_FOR_PCR} pre-treatment periods. "
            f"Provided: {num_pre_treatment_periods}."
        )
    """
    Perform Principal Component Regression (PCR) based Synthetic Control.

    This method first applies Singular Value Thresholding (SVT) to the
    pre-treatment donor matrix to obtain a low-rank representation.
    Optionally, SVD-based clustering can be performed on donors before SVT.
    Then, either a Frequentist SCM optimization (via `Opt.SCopt`) or a
    Bayesian SCM (via `BayesSCM`) is used to find weights for the
    (potentially transformed and clustered) donors. These weights are then
    applied to the original (selected) donor data to construct the counterfactual.

    Parameters
    ----------
    donor_outcomes_matrix : np.ndarray
        Matrix of donor outcomes or features, shape (T_total, N_donors).
        `T_total` is the total number of time periods.
        `N_donors` is the number of donor units.
    treated_unit_outcome_vector : np.ndarray
        Vector of treated unit outcomes, shape (T_total,).
    scm_objective_model_type : str
        SCM optimization model type (e.g., "MSCb", "SIMPLEX") to be used
        if `use_frequentist_scm` is True. Passed to `Opt.SCopt`.
    all_donor_names : List[str]
        List of names corresponding to the columns (donors) in `donor_outcomes_matrix`.
        Length must be equal to `N_donors`.
    num_pre_treatment_periods : int, optional
        Number of pre-treatment periods to use for SVT, clustering (if enabled),
        and SCM weight fitting. Default is 10.
    enable_clustering : bool, optional
        If True, perform SVD-based clustering on donors using pre-treatment data
        before applying SVT and fitting SCM weights. Default is False.
    use_frequentist_scm : bool, optional
        If True, use a Frequentist SCM optimization approach (`Opt.SCopt`) to find
        weights. If False, use a Bayesian SCM approach (`BayesSCM`). Default is False.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results:
        - "weights" : Dict[str, float]
            Dictionary mapping donor names (original or selected by clustering)
            to their estimated SCM weights.
        - "cf_mean" : np.ndarray
            The estimated counterfactual outcome vector for the treated unit over
            all `T_total` periods, shape (T_total,). For the Bayesian approach,
            this is the median of the posterior predictive samples.
        - "credible_interval" : Tuple[np.ndarray, np.ndarray], optional
            Only present if `use_frequentist_scm` is False. A tuple containing two
            arrays: `(lower_bound, upper_bound)`. Each array has shape (T_total,)
            and represents the 2.5th and 97.5th percentiles of the posterior
            predictive distribution for the counterfactual outcome.

    Notes
    -----
    - `T_total` (total time periods) and `N_donors` (number of donors) are
      inferred from the shapes of `donor_outcomes_matrix` and `treated_unit_outcome_vector`.
    - SVT is always applied to the pre-treatment portion of the donor matrix
      (or the clustered donor matrix if `enable_clustering` is True).
    - The counterfactual prediction `cf_mean` (and `credible_interval`) is
      generated by applying the learned weights to the original `donor_outcomes_matrix`
      (or the subset selected by clustering if `enable_clustering` is True),
      not the SVT-transformed matrix.

    Examples
    --------
    >>> T_total, N_donors_ex = 20, 5
    >>> pre_p = 10
    >>> X_d = np.random.rand(T_total, N_donors_ex)
    >>> y_t = np.random.rand(T_total) + np.arange(T_total) * 0.1
    >>> donor_names_ex = [f"D{i}" for i in range(N_donors_ex)]
    >>> # Frequentist example
    >>> results_freq = pcr(X_d, y_t, "MSCb", donor_names_ex, num_pre_treatment_periods=pre_p, use_frequentist_scm=True)
    >>> print(list(results_freq.keys()))
    ['weights', 'cf_mean']
    >>> isinstance(results_freq["weights"], dict)
    True
    >>> results_freq["cf_mean"].shape
    (20,)
    >>> # Bayesian example
    >>> results_bayes = pcr(X_d, y_t, "MSCb", donor_names_ex, num_pre_treatment_periods=pre_p, use_frequentist_scm=False)
    >>> print(sorted(list(results_bayes.keys())))
    ['cf_mean', 'credible_interval', 'weights']
    >>> results_bayes["credible_interval"][0].shape
    (20,)

    """
    # Slice data to pre-treatment period for initial processing (SVT, clustering, weight fitting)
    pre_treatment_donor_outcomes: np.ndarray = donor_outcomes_matrix[
        :num_pre_treatment_periods
    ]
    pre_treatment_treated_outcome: np.ndarray = treated_unit_outcome_vector[
        :num_pre_treatment_periods
    ]

    # Perform Singular Value Thresholding (SVT) on the pre-treatment donor matrix.
    # SVT helps to de-noise the data and extract a low-rank representation,
    # which can be beneficial for SCM weight estimation.
    # The svt function from denoiseutils returns: Y0_rank (low-rank matrix), n2 (Frobenius norm), u_rank, s_rank, v_rank.
    # We primarily need Y0_rank (low_rank_pre_treatment_donors).
    low_rank_pre_treatment_donors, _, _, _, _ = svt(
        pre_treatment_donor_outcomes # Using only pre-period data for SVT
    )  

    # Initialize variables that might be modified by clustering.
    # By default, use all donors and their SVT-transformed pre-treatment data.
    selected_donor_outcomes_all_periods: np.ndarray = donor_outcomes_matrix # Full period data for selected donors (initially all donors)
    selected_low_rank_pre_treatment_donors: np.ndarray = low_rank_pre_treatment_donors # Low-rank pre-treatment data for selected donors (initially all SVT'd donors)
    current_selected_donor_names: List[str] = all_donor_names # Names of selected donors (initially all donor names)

    if enable_clustering:
        # If clustering is enabled, perform SVD-based clustering on pre-treatment donor outcomes
        # to identify a more relevant subset of donors.
        # SVDCluster uses the treated unit's pre-treatment outcome to find the cluster most similar to it.
        # It returns:
        #   - clustered_pre_treatment_donor_outcomes_subset: Pre-treatment outcomes of donors in the selected cluster.
        #   - clustered_selected_donor_names: Names of donors in the selected cluster.
        #   - cluster_selected_donor_indices: Indices of these donors in the original donor_outcomes_matrix.
        (
            clustered_pre_treatment_donor_outcomes_subset, 
            clustered_selected_donor_names, 
            cluster_selected_donor_indices, 
        ) = SVDCluster(
            pre_treatment_donor_outcomes, # Original pre-treatment donor data for clustering
            pre_treatment_treated_outcome, # Pre-treatment outcome of the treated unit (for determining cluster relevance)
            all_donor_names,  # Full list of original donor names
        )
        
        # Apply SVT to the pre-treatment outcomes of the *selected cluster* of donors.
        # This re-calculates the low-rank representation based only on the clustered subset.
        selected_low_rank_pre_treatment_donors, _, _, _, _ = svt(
            clustered_pre_treatment_donor_outcomes_subset
        )  
        current_selected_donor_names = clustered_selected_donor_names # Update donor names to those in the selected cluster
        
        # Update the matrix used for counterfactual construction to include only clustered donors for *all* time periods.
        selected_donor_outcomes_all_periods = donor_outcomes_matrix[
            :, cluster_selected_donor_indices 
        ]  
        
        # Ensure there are enough donors after clustering for a meaningful SCM.
        # This prevents issues if clustering results in too few donors (e.g., less than 2).
        MIN_DONORS_FOR_CLUSTERED_SCM = 2 
        num_selected_donors_after_clustering = selected_donor_outcomes_all_periods.shape[1]
        if num_selected_donors_after_clustering < MIN_DONORS_FOR_CLUSTERED_SCM:
            raise MlsynthDataError(
                f"Clustering resulted in {num_selected_donors_after_clustering} selected donor(s). "
                f"At least {MIN_DONORS_FOR_CLUSTERED_SCM} donors are required for a meaningful clustered SCM when 'cluster=True'. "
                "This may be due to too few input donors or restrictive clustering."
            )

    # Choose between Frequentist or Bayesian SCM based on `use_frequentist_scm` flag.
    if use_frequentist_scm:
        # Frequentist SCM approach:
        # Use Opt.SCopt with the (potentially clustered and) SVT-transformed pre-treatment donor data.
        # Opt.SCopt expects: num_donors, treated_pre_outcome, num_pre_periods, transformed_donors_pre_data.
        # The number of control units is the number of features in the (SVT-transformed, possibly clustered) matrix.
        scm_problem_frequentist = Opt.SCopt(
            num_control_units=selected_low_rank_pre_treatment_donors.shape[1], 
            target_outcomes_pre_treatment=pre_treatment_treated_outcome,
            num_pre_treatment_periods=num_pre_treatment_periods,  
            donor_outcomes_pre_treatment=selected_low_rank_pre_treatment_donors, # Use low-rank pre-treatment data for fitting weights
            scm_model_type=scm_objective_model_type, 
        )
        # Extract the estimated SCM weights from the CVXPY problem solution.
        frequentist_scm_weights: np.ndarray = scm_problem_frequentist.solution.primal_vars[
            next(iter(scm_problem_frequentist.solution.primal_vars))
        ]
        # Create a dictionary mapping selected donor names to their weights.
        frequentist_scm_weights_dict: Dict[str, float] = dict(
            zip(current_selected_donor_names, frequentist_scm_weights) 
        )

        # Construct the counterfactual outcome.
        # IMPORTANT: Use the learned weights with the *original* (but potentially cluster-selected) donor data for all periods,
        # NOT the SVT-transformed data. SVT is used for robust weight estimation, but the counterfactual
        # should be in the scale of the original outcomes.
        frequentist_counterfactual_mean: np.ndarray = np.dot(
            selected_donor_outcomes_all_periods, frequentist_scm_weights
        )
        return {
            "weights": frequentist_scm_weights_dict,
            "cf_mean": frequentist_counterfactual_mean,
        }
    else:  # Bayesian SCM approach
        # Define a prior for the Bayesian SCM. This could be made configurable.
        alpha_prior: float = 1.0  
        
        # Perform Bayesian SCM using the (potentially clustered and) SVT-transformed pre-treatment donor data.
        # BayesSCM returns posterior mean and covariance of the weights.
        (
            bayesian_donor_coefficients, # Posterior mean of weights
            bayesian_coefficients_covariance, # Posterior covariance of weights
            _, # Placeholder for other BayesSCM outputs if any
            _, # Placeholder
        ) = BayesSCM(
            denoised_donor_matrix=selected_low_rank_pre_treatment_donors, # Low-rank pre-treatment donor data for fitting
            target_outcome_pre_intervention=pre_treatment_treated_outcome, # Pre-treatment treated outcome
            observation_noise_variance=np.var(pre_treatment_treated_outcome, ddof=1), # Prior for error variance (sample variance of treated outcome)
            weights_prior_precision=alpha_prior, # Prior for coefficients' precision
        )
        # Create a dictionary mapping selected donor names to their posterior mean weights.
        bayesian_scm_weights_dict: Dict[str, float] = dict(
            zip(current_selected_donor_names, bayesian_donor_coefficients) 
        )
        # Number of samples to draw from the posterior predictive distribution for credible intervals.
        num_bayesian_samples: int = 1000  

        # Sample SCM weights from their posterior distribution (multivariate normal).
        bayesian_coefficient_samples: np.ndarray = np.random.multivariate_normal(
            mean=bayesian_donor_coefficients,
            cov=bayesian_coefficients_covariance,
            size=num_bayesian_samples,
        )
        # Generate posterior predictive samples for the counterfactual.
        # Again, use the *original* (but potentially cluster-selected) donor data for all periods.
        # Resulting shape: (total_time_periods, num_bayesian_samples)
        bayesian_counterfactual_samples: np.ndarray = (
            selected_donor_outcomes_all_periods @ bayesian_coefficient_samples.T 
        )

        # Calculate credible interval (e.g., 95%) and median from posterior samples.
        # The median is often used as the point estimate for the Bayesian counterfactual.
        bayesian_credible_interval_lower_bound: np.ndarray = np.percentile(
            bayesian_counterfactual_samples, 2.5, axis=1 # 2.5th percentile for each time period
        )
        bayesian_credible_interval_upper_bound: np.ndarray = np.percentile(
            bayesian_counterfactual_samples, 97.5, axis=1 # 97.5th percentile for each time period
        )
        bayesian_counterfactual_median: np.ndarray = np.median(
            bayesian_counterfactual_samples, axis=1 # Median for each time period
        )

        return {
            "weights": bayesian_scm_weights_dict,
            "cf_mean": bayesian_counterfactual_median, # Use median as point estimate for counterfactual
            "credible_interval": (
                bayesian_credible_interval_lower_bound,
                bayesian_credible_interval_upper_bound,
            ),
        }


class Opt:
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
    ) -> cp.Problem:
        """
        Solve the OLS SCM optimization problem.

        OLS (Ordinary Least Squares) in this SCM context involves fitting
        the model without any constraints on the weights (neither non-negativity
        nor sum-to-one). No intercept is added by default within this helper,
        as `Opt.SCopt` passes `X_fit` which is `X[:t1]`.

        Parameters
        ----------
        target_outcomes_pre_treatment_ols : np.ndarray
            Target vector for the pre-treatment periods, shape (t1,).
        donor_outcomes_pre_treatment_ols : np.ndarray
            Original donor matrix for pre-treatment periods, shape (t1, num_donors_ols).
        num_donors_ols : int
            Number of original control units (columns in donor_outcomes_pre_treatment_ols).

        Returns
        -------
        cp.Problem
            The solved CVXPY problem object. The solution (weights) can be
            accessed via `problem.solution.primal_vars`.
        """
        # Define CVXPY variable for donor weights
        coefficients_cvxpy_ols = cp.Variable(num_donors_ols)
        
        # OLS (Ordinary Least Squares) has no constraints on the weights in this SCM context.
        # It simply minimizes the sum of squared residuals.
        # No intercept is explicitly added here; if an intercept is desired for OLS,
        # it should be part of `donor_outcomes_pre_treatment_ols`.
        constraints_ols = []

        # Objective: Minimize || Y_pre - X_donors_pre * w ||_2
        objective_function_ols = cp.Minimize(
            cp.norm(
                target_outcomes_pre_treatment_ols # Y_pre
                - donor_outcomes_pre_treatment_ols @ coefficients_cvxpy_ols, # X_donors_pre * w
                2, # L2 norm
            )
        )

        problem_ols = cp.Problem(objective_function_ols, constraints_ols)
        problem_ols.solve(solver=_SOLVER_CLARABEL_STR) # Solve the optimization problem
        return problem_ols

    @staticmethod
    def SCopt(
        num_control_units: int,
        target_outcomes_pre_treatment: np.ndarray,
        num_pre_treatment_periods: int,
        donor_outcomes_pre_treatment: np.ndarray,
        scm_model_type: str = _SCM_MODEL_MSCB, # Use constant
        base_model_results_for_averaging: Optional[Dict[str, Dict[str, Any]]] = None,
        donor_names: Optional[List[str]] = None,
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
                processed_donor_outcomes_pre_treatment_subset, # Original X
                num_control_units, # Original num_control_units
            )
        else: 
            raise MlsynthConfigError(
                f"Unsupported SCM model type: {scm_model_type}. Supported types are: "
                f"{_SCM_MODEL_SIMPLEX}, {_SCM_MODEL_MSCA}, {_SCM_MODEL_MSCB}, "
                f"{_SCM_MODEL_MSCC}, {_SCM_MODEL_OLS}, {_SCM_MODEL_MA}."
            )


def pda(
    preprocessed_data: Dict[str, Any],
    num_donors_for_fs: int,
    pda_method_type: str = _PDA_METHOD_FS, # Use constant
    l2_regularization_param_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Estimate counterfactual outcomes using Panel Data Approach (PDA) variants.

    This function implements three PDA methods:
    1. 'fs': Forward Selection to choose a subset of donor units.
    2. 'LASSO': LASSO regression to select donors and estimate weights.
    3. 'l2': L2-relaxation (Ridge-like) regression.

    Parameters
    ----------
    preprocessed_data : Dict[str, Any]
        A dictionary containing preprocessed data. Expected keys are:
        - 'y' : np.ndarray
            Outcome vector for the treated unit, shape (T_total,).
        - 'donor_matrix' : np.ndarray
            Matrix of outcomes/features for donor units, shape (T_total, N_donors_original).
        - 'donor_names' : Union[pd.Index, List[str]]
            Names of the original donor units, corresponding to columns of `donor_matrix`.
        - 'pre_periods' : int
            Number of pre-treatment time periods.
        - 'post_periods' : int
            Number of post-treatment time periods.
        - 'total_periods' : int
            Total number of time periods (`pre_periods + post_periods`).
    num_donors_for_fs : int
        Number of donor units to consider.
        - For 'fs' method: This is the maximum number of donor units to be selected
          by the forward selection algorithm.
        - For 'LASSO' and 'l2' methods: This parameter is not directly used by these
          methods as they typically consider all available donors from `preprocessed_data['donor_matrix']`.
    pda_method_type : str, optional
        The Panel Data Approach variant to use. Options are:
        - "fs": Forward Selection (default).
        - "LASSO": LASSO regression.
        - "l2": L2-relaxation.
        Default is _PDA_METHOD_FS.
    l2_regularization_param_override : Optional[float], optional
        Tuning parameter for the L2-relaxation method (if `pda_method_type` is "l2").
        If None, the parameter is determined using cross-validation via
        `cross_validate_tau`. Default is None.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results of the estimation. The structure
        of this dictionary varies depending on the `pda_method_type` used:

        **Common keys for all methods (derived from `effects.calculate`):**
        - "Effects" : Dict[str, float]
            Contains "ATT", "ATT_post", "ATT_pre".
        - "Fit" : Dict[str, float]
            Contains "RMSE_pre", "RMSE_post", "R2_pre", "R2_post".
        - "Vectors" : Dict[str, np.ndarray]
            Contains "Y" (original treated outcome), "Y_sc" (counterfactual),
            "Gap" (Y - Y_sc), "TimePeriods".

        **Method-specific keys:**
        - If `pda_method_type` is "fs":
            - "method": "fs"
            - "Betas": Dict[str, float], selected donor names to coefficients.
            - "Inference": Dict with "t_stat", "SE", "95% CI", "p_value".
        - If `pda_method_type` is "LASSO":
            - "method": "LASSO"
            - "non_zero_coef_dict": Dict[str, float], names of donors with
              non-zero LASSO coefficients to their coefficient values.
            - "Inference": Dict with "CI", "t_stat", "SE".
        - If `pda_method_type` is "l2":
            - "method": "l2 relaxation"
            - "optimal_tau": float, the tau value used.
            - "Betas": Dict[str, float], all donor names to L2 coefficients.
            - "Inference": Dict with "t_stat", "standard_error", "p_value",
              "confidence_interval".
            - "Intercept": float, the estimated intercept for L2 method.

    Raises
    ------
    MlsynthConfigError
        If an invalid `pda_method_type` is specified.

    Examples
    --------
    >>> import pandas as pd
    >>> T, N_orig, T0, T1 = 20, 5, 10, 10
    >>> y_data = np.random.rand(T)
    >>> donor_data = np.random.rand(T, N_orig)
    >>> donor_names_data = [f"D{i}" for i in range(N_orig)]
    >>> preprocessed_data_ex = {
    ...     "y": y_data,
    ...     "donor_matrix": donor_data,
    ...     "donor_names": pd.Index(donor_names_data),
    ...     "pre_periods": T0,
    ...     "post_periods": T1,
    ...     "total_periods": T
    ... }
    >>> # Forward Selection example
    >>> results_fs = pda(preprocessed_data_ex, num_donors_for_fs=3, pda_method_type=_PDA_METHOD_FS)
    >>> print(results_fs["method"])
    fs
    >>> isinstance(results_fs["Betas"], dict)
    True
    >>> # LASSO example
    >>> results_lasso = pda(preprocessed_data_ex, num_donors_for_fs=N_orig, pda_method_type=_PDA_METHOD_LASSO)
    >>> print(results_lasso["method"])
    LASSO
    >>> isinstance(results_lasso["non_zero_coef_dict"], dict)
    True
    """
    # The following lines were commented out as preprocessed_data is passed directly to helpers.
    # outcome_vector_treated_full_period: np.ndarray = preprocessed_data["y"]
    # donor_outcomes_matrix_full_period: np.ndarray = preprocessed_data["donor_matrix"]
    # all_donor_names_full_period = preprocessed_data["donor_names"]
    # num_pre_treatment_periods_pda: int = preprocessed_data["pre_periods"]
    # num_post_treatment_periods_pda: int = preprocessed_data["post_periods"]

    # Delegate to the appropriate solver based on pda_method_type
    if pda_method_type == _PDA_METHOD_FS:
        return _solve_pda_fs(preprocessed_data, num_donors_for_fs)
    elif pda_method_type == _PDA_METHOD_LASSO:
        return _solve_pda_lasso(preprocessed_data)
    elif pda_method_type == _PDA_METHOD_L2:
        return _solve_pda_l2(preprocessed_data, l2_regularization_param_override)
    else:
        raise MlsynthConfigError(
            f"Invalid PDA method specified: {pda_method_type}. Choose from '{_PDA_METHOD_FS}', '{_PDA_METHOD_LASSO}', or '{_PDA_METHOD_L2}'."
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


def get_theta(treated_outcome_pre_treatment: np.ndarray, donor_outcomes_pre_treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate alignment coefficients (theta) and the aligned donor matrix.

    This function is a helper for the Synthetic Regressing Control (SRC) method.
    It computes alignment coefficients `alignment_coefficients` for each donor unit based on
    pre-treatment data. These coefficients aim to align each donor's outcome
    series with the treated unit's outcome series by scaling the demeaned
    donor outcomes.

    The alignment coefficient for each donor `j` is calculated as:

    ``theta_j = (donor_outcomes_pre_treatment_demeaned_j' @ treated_outcome_pre_treatment_demeaned) / (donor_outcomes_pre_treatment_demeaned_j' @ donor_outcomes_pre_treatment_demeaned_j)``

    where ``donor_outcomes_pre_treatment_demeaned_j`` is the demeaned outcome series for donor `j` in the
    pre-treatment period, and ``treated_outcome_pre_treatment_demeaned`` is the demeaned outcome series for
    the treated unit in the pre-treatment period.

    The aligned donor matrix `aligned_donor_outcomes_pre_treatment` is then `donor_outcomes_pre_treatment * alignment_coefficients`,
    where the multiplication is element-wise broadcasted.

    Parameters
    ----------
    treated_outcome_pre_treatment : np.ndarray
        Outcome vector for the treated unit in the pre-treatment period,
        shape (T0,), where T0 is the number of pre-treatment periods.
    donor_outcomes_pre_treatment : np.ndarray
        Matrix of outcomes for donor units in the pre-treatment period,
        shape (T0, J), where J is the number of donor units.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - alignment_coefficients : np.ndarray
            Vector of estimated alignment coefficients for each donor unit,
            shape (J,).
        - aligned_donor_outcomes_pre_treatment : np.ndarray
            The donor matrix `donor_outcomes_pre_treatment` with each column scaled by its
            corresponding `alignment_coefficients` coefficient, shape (T0, J).

    Examples
    --------
    >>> T0_ex, J_ex = 10, 3
    >>> treated_outcome_pre_ex = np.random.rand(T0_ex)
    >>> donor_outcomes_pre_ex = np.random.rand(T0_ex, J_ex)
    >>> theta_coeffs, Y_aligned = get_theta(treated_outcome_pre_ex, donor_outcomes_pre_ex)
    >>> print(theta_coeffs.shape)
    (3,)
    >>> print(Y_aligned.shape)
    (10, 3)
    """
    treated_outcome_pre_treatment_demeaned: np.ndarray = treated_outcome_pre_treatment - np.mean(treated_outcome_pre_treatment)
    donor_outcomes_pre_treatment_demeaned: np.ndarray = donor_outcomes_pre_treatment - np.mean(donor_outcomes_pre_treatment, axis=0)

    alignment_numerators: np.ndarray = donor_outcomes_pre_treatment_demeaned.T @ treated_outcome_pre_treatment_demeaned
    alignment_denominators: np.ndarray = np.sum(donor_outcomes_pre_treatment_demeaned**2, axis=0)
    # Avoid division by zero if a donor has no variance in pre-period
    # Calculate alignment coefficients.
    # If a denominator is zero (donor has no variance in pre-period), the coefficient is set to 0.
    alignment_coefficients: np.ndarray = np.divide(alignment_numerators, alignment_denominators, out=np.zeros_like(alignment_numerators), where=alignment_denominators!=0)

    # Apply the alignment coefficients to the original pre-treatment donor outcomes.
    # This scales each donor's outcome series to better align with the treated unit's series.
    aligned_donor_outcomes_pre_treatment: np.ndarray = donor_outcomes_pre_treatment * alignment_coefficients # Element-wise broadcast
    return alignment_coefficients, aligned_donor_outcomes_pre_treatment


def get_sigmasq(treated_outcome_pre_treatment: np.ndarray, donor_outcomes_pre_treatment: np.ndarray) -> float:
    """
    Estimate noise variance (sigma^2) for the Synthetic Regressing Control (SRC) method.

    This function estimates the variance of the noise term in the conceptual model
    ``treated_outcome = donor_outcomes * alignment_coefficients + error_term``,
    where ``error_term`` is assumed to be white noise with variance ``sigma**2``.
    The estimation is based on the residuals obtained after projecting the
    demeaned pre-treatment outcome of the treated unit (`treated_outcome_pre_treatment`)
    onto the space spanned by the demeaned pre-treatment outcomes of the donor
    units (`donor_outcomes_pre_treatment`).

    The projection involves a `demeaning_matrix` and a `donor_based_projection_matrix`
    derived from `donor_outcomes_pre_treatment`. The residual error is effectively
    `demeaning_matrix @ treated_outcome_pre_treatment - demeaning_matrix @ donor_based_projection_matrix @ demeaning_matrix @ treated_outcome_pre_treatment`,
    and the noise variance (`sigma^2`) is estimated as the squared L2-norm of this residual error.

    Parameters
    ----------
    treated_outcome_pre_treatment : np.ndarray
        Outcome vector for the treated unit in the pre-treatment period,
        shape (num_pre_treatment_periods,), where num_pre_treatment_periods is the number of pre-treatment periods.
    donor_outcomes_pre_treatment : np.ndarray
        Matrix of outcomes for donor units in the pre-treatment period,
        shape (num_pre_treatment_periods, num_donors), where num_donors is the number of donor units.

    Returns
    -------
    float
        Estimated noise variance (`sigma^2`).

    Examples
    --------
    >>> num_pre_periods_example, num_donors_example = 10, 3
    >>> treated_outcome_pre_example = np.random.rand(num_pre_periods_example)
    >>> donor_outcomes_pre_example = np.random.rand(num_pre_periods_example, num_donors_example)
    >>> noise_variance_estimate_example = get_sigmasq(treated_outcome_pre_example, donor_outcomes_pre_example)
    >>> print(f"Estimated sigma^2: {noise_variance_estimate_example:.4f}") # doctest: +SKIP
    Estimated sigma^2: ...
    """
    num_pre_treatment_periods: int = len(treated_outcome_pre_treatment)
    demeaning_matrix: np.ndarray = np.eye(num_pre_treatment_periods) -\
                                  np.ones((num_pre_treatment_periods, num_pre_treatment_periods)) / num_pre_treatment_periods

    # G_matrix = Y0_demeaned.T @ Y0_demeaned
    # This calculates Y0_demeaned.T @ Y0_demeaned, where Y0_demeaned = demeaning_matrix @ donor_outcomes_pre_treatment.
    # However, the demeaning_matrix is applied after the transpose and before the second donor_outcomes_pre_treatment.
    # This effectively computes (donor_outcomes_pre_treatment.T @ demeaning_matrix) @ donor_outcomes_pre_treatment.
    # If donor_outcomes_pre_treatment is (T0, J), then donor_outcomes_pre_treatment.T is (J, T0).
    # demeaning_matrix is (T0, T0). So (J, T0) @ (T0, T0) -> (J, T0).
    # Then (J, T0) @ (T0, J) -> (J, J). This is the covariance matrix of demeaned donor outcomes if Y0 was centered first.
    # Here, it's more like sum of (Y0_j - mean(Y0_j)) * (Y0_k - mean(Y0_k)) for each pair of donors j,k.
    # The diagonal elements are sum((Y0_j - mean(Y0_j))^2).
    donor_demeaned_inner_product_matrix: np.ndarray = donor_outcomes_pre_treatment.T @ demeaning_matrix @ donor_outcomes_pre_treatment
    diagonal_of_donor_demeaned_inner_product: np.ndarray = np.diag(donor_demeaned_inner_product_matrix) # These are sum((Y0_j - mean(Y0_j))^2) for each donor j.
    
    # Avoid division by zero for diagonal_of_donor_demeaned_inner_product.
    # This inverse_diagonal is 1 / sum((Y0_j - mean(Y0_j))^2).
    inverse_diagonal_of_donor_demeaned_inner_product: np.ndarray = np.divide(
        1.0, diagonal_of_donor_demeaned_inner_product,
        out=np.zeros_like(diagonal_of_donor_demeaned_inner_product), # if denominator is 0, result is 0
        where=diagonal_of_donor_demeaned_inner_product != 0
    )

    # Construct the projection matrix P_Z = Z (Z' Q Z)^-1 Z' Q, where Z is donor_outcomes_pre_treatment and Q is demeaning_matrix.
    # Here, it's simplified to Z diag(1/(Z_j'QZ_j)) Z' Q.
    # This projects onto the space spanned by the *individually scaled* demeaned donor outcomes.
    # donor_based_projection_matrix shape: (T0, T0)
    donor_based_projection_matrix: np.ndarray = donor_outcomes_pre_treatment @\
                                              np.diag(inverse_diagonal_of_donor_demeaned_inner_product) @\
                                              donor_outcomes_pre_treatment.T
    
    # Project the demeaned treated outcome onto the space defined by the donor_based_projection_matrix.
    # projected_treated_outcome_demeaned = P_Z @ (Q @ treated_outcome_pre_treatment)
    projected_treated_outcome_demeaned: np.ndarray = donor_based_projection_matrix @ demeaning_matrix @ treated_outcome_pre_treatment
    
    # Calculate the residual error.
    # This is Q @ treated_outcome_pre_treatment - Q @ P_Z @ Q @ treated_outcome_pre_treatment.
    # Since P_Z is constructed with Z and Q, and Q is idempotent (Q*Q=Q),
    # Q @ P_Z @ Q simplifies depending on P_Z structure.
    # Here, it's effectively (demeaned treated outcome) - (demeaned projection of treated outcome).
    estimation_residual_error: np.ndarray = demeaning_matrix @ treated_outcome_pre_treatment -\
                                           demeaning_matrix @ projected_treated_outcome_demeaned # This is Qy1 - Q * (P_Z Q y1)
    
    # The noise variance is estimated as the squared L2-norm of this residual error.
    noise_variance_estimate: float = np.linalg.norm(estimation_residual_error) ** 2
    return noise_variance_estimate


def __SRC_opt(
    treated_outcome_pre_treatment: np.ndarray, donor_outcomes_pre_treatment: np.ndarray, 
    donor_outcomes_post_treatment: np.ndarray,
    aligned_donor_outcomes_pre_treatment: np.ndarray, alignment_coefficients: np.ndarray, 
    noise_variance_estimate: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Synthetic Regressing Control (SRC) weight optimization and prediction.

    This is a helper function for `SRCest`. It solves the SRC optimization
    problem to find donor weights `optimal_donor_weights` and then constructs
    pre-treatment predictions and post-treatment counterfactuals.

    The optimization problem is:

    ``min ||treated_outcome_pre_treatment - aligned_donor_outcomes_pre_treatment @ donor_weights||_2**2 +
    2 * noise_variance_estimate * sum(donor_weights)``

    subject to ``donor_weights >= 0`` and ``sum(donor_weights) == 1``.

    Parameters
    ----------
    treated_outcome_pre_treatment : np.ndarray
        Pre-treatment outcomes for the treated unit, shape (num_pre_treatment_periods,).
    donor_outcomes_pre_treatment : np.ndarray
        Pre-treatment outcomes for donor units, shape (num_pre_treatment_periods, num_donors).
    donor_outcomes_post_treatment : np.ndarray
        Post-treatment outcomes for donor units, shape (num_post_treatment_periods, num_donors).
    aligned_donor_outcomes_pre_treatment : np.ndarray
        Pre-treatment donor outcomes aligned by `alignment_coefficients`,
        shape (num_pre_treatment_periods, num_donors).
    alignment_coefficients : np.ndarray
        Estimated alignment coefficients, shape (num_donors,).
    noise_variance_estimate : float
        Estimated noise variance from `get_sigmasq`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - predicted_treated_outcome_pre_treatment : np.ndarray
            In-sample predicted treated unit outcomes for the pre-treatment
            period, shape (num_pre_treatment_periods,).
        - predicted_counterfactual_post_treatment : np.ndarray
            Predicted counterfactual outcomes for the treated unit in the
            post-treatment period, shape (num_post_treatment_periods,).
        - optimal_donor_weights : np.ndarray
            Optimal donor weights, shape (num_donors,).
        - alignment_coefficients : np.ndarray
            The input alignment coefficients, passed through, shape (num_donors,).

    Notes
    -----
    Uses OSQP solver for the quadratic program.
    """
    num_donors: int = donor_outcomes_pre_treatment.shape[1]
    # Define the CVXPY variable for donor weights. Weights are constrained to be non-negative.
    donor_weights_cvxpy = cp.Variable(num_donors, nonneg=True)

    # L2 loss term: minimize the squared difference between the actual pre-treatment outcome
    # and the synthetic control constructed from aligned donor outcomes.
    l2_loss_term = cp.sum_squares(treated_outcome_pre_treatment - aligned_donor_outcomes_pre_treatment @ donor_weights_cvxpy)
    # Penalty term: 2 * sigma^2 * sum(weights). This penalizes larger weights,
    # with the penalty scaled by the estimated noise variance.
    penalty_term = 2 * noise_variance_estimate * cp.sum(donor_weights_cvxpy)
    
    # Objective function: Minimize the sum of the L2 loss and the penalty term.
    optimization_objective = cp.Minimize(l2_loss_term + penalty_term)
    # Constraint: Donor weights must sum to 1, ensuring a convex combination.
    optimization_constraints = [cp.sum(donor_weights_cvxpy) == 1]
    
    # Create and solve the optimization problem using the OSQP solver, suitable for quadratic programs.
    optimization_problem = cp.Problem(optimization_objective, optimization_constraints)
    optimization_problem.solve(solver=_SOLVER_OSQP_STR) 

    # Extract the optimal donor weights from the solution.
    optimal_donor_weights: np.ndarray = donor_weights_cvxpy.value

    # Calculate means for pre-treatment outcomes (treated and donors) for constructing post-treatment counterfactual.
    mean_treated_outcome_pre_treatment: float = np.mean(treated_outcome_pre_treatment)
    mean_donor_outcomes_pre_treatment: np.ndarray = np.mean(donor_outcomes_pre_treatment, axis=0)
    
    # In-sample prediction for the pre-treatment period.
    # This uses the original (non-aligned) donor outcomes, scaled by the product of optimal weights and alignment coefficients.
    # This reconstructs the treated unit's pre-treatment outcome using the SRC model.
    predicted_treated_outcome_pre_treatment: np.ndarray = donor_outcomes_pre_treatment @ (optimal_donor_weights * alignment_coefficients)
    
    # Post-treatment counterfactual prediction.
    # This is constructed by taking the mean of the treated unit's pre-treatment outcome,
    # and adding the weighted and aligned deviations of post-treatment donor outcomes from their pre-treatment means.
    # Formula: Y_mean_pre_treated + (Y_donors_post - Y_mean_donors_pre) @ (weights * theta)
    predicted_counterfactual_post_treatment: np.ndarray = mean_treated_outcome_pre_treatment +\
        (donor_outcomes_post_treatment - mean_donor_outcomes_pre_treatment) @ (optimal_donor_weights * alignment_coefficients)

    return predicted_treated_outcome_pre_treatment, predicted_counterfactual_post_treatment, optimal_donor_weights, alignment_coefficients


def SRCest(
    treated_outcome_full_period: np.ndarray, 
    donor_outcomes_full_period: np.ndarray, 
    num_post_treatment_periods: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate counterfactual outcomes using the Synthetic Regressing Control (SRC) method.

    The SRC method involves three main steps:
    1.  **Alignment Coefficient Estimation (`get_theta`):** Estimate alignment
        coefficients (`estimated_alignment_coefficients`) for each donor unit by regressing the demeaned
        pre-treatment outcome of the treated unit on the demeaned pre-treatment
        outcomes of each donor unit individually.
    2.  **Noise Variance Estimation (`get_sigmasq`):** Estimate the variance
        (`estimated_noise_variance`) of the noise in the relationship between the treated unit's
        outcome and the aligned donor outcomes in the pre-treatment period.
    3.  **Weight Optimization and Prediction (`__SRC_opt`):** Solve an optimization
        problem to find donor weights (`optimal_donor_weights_result`) that minimize a penalized sum of
        squared errors between the treated unit's pre-treatment outcomes and the
        weighted sum of aligned donor outcomes. The penalty term involves `estimated_noise_variance`.
        These weights are then used to construct the counterfactual outcome series
        for both pre-treatment (in-sample prediction) and post-treatment periods.

    Parameters
    ----------
    treated_outcome_full_period : np.ndarray
        Outcomes for the treated unit across all time periods, shape (total_time_periods,).
        `total_time_periods` is the total number of time periods.
    donor_outcomes_full_period : np.ndarray
        Outcomes for donor units across all time periods, shape (total_time_periods, num_donors).
        `num_donors` is the number of donor units.
    num_post_treatment_periods : int
        Number of post-treatment periods. The number of pre-treatment periods
        (num_pre_treatment_periods) is inferred as `total_time_periods - num_post_treatment_periods`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - counterfactual_full_period : np.ndarray
            The estimated counterfactual outcome series for the treated unit
            across all `total_time_periods` periods, shape (total_time_periods,).
        - optimal_donor_weights_result : np.ndarray
            The optimal weights assigned to each donor unit, shape (num_donors,).
            These weights sum to 1 and are non-negative.
        - estimated_alignment_coefficients : np.ndarray
            The estimated alignment coefficients for each donor unit, shape (num_donors,).

    Examples
    --------
    >>> total_periods_example, num_donors_example = 20, 4
    >>> num_post_periods_example = 5
    >>> treated_outcome_example = np.random.rand(total_periods_example) + np.arange(total_periods_example) * 0.2
    >>> donor_outcomes_example = np.random.rand(total_periods_example, num_donors_example) + \
    ...                          np.arange(total_periods_example)[:, None] * \
    ...                          np.random.rand(num_donors_example) * 0.1
    >>> counterfactual_est, weights_est, theta_est = SRCest(
    ...     treated_outcome_example, donor_outcomes_example, num_post_periods_example
    ... )
    >>> print(f"Counterfactual shape: {counterfactual_est.shape}")
    Counterfactual shape: (20,)
    >>> print(f"Weights shape: {weights_est.shape}")
    Weights shape: (4,)
    >>> print(f"Theta coefficients shape: {theta_est.shape}")
    Theta coefficients shape: (4,)
    >>> print(f"Sum of weights: {np.sum(weights_est):.2f}") # Should be close to 1 # doctest: +SKIP
    Sum of weights: 1.00
    """
    total_time_periods, num_donors = donor_outcomes_full_period.shape
    # Infer the number of pre-treatment periods based on total periods and post-treatment periods.
    num_pre_treatment_periods: int = total_time_periods - num_post_treatment_periods

    # Slice the data into pre-treatment and post-treatment subsets.
    treated_outcome_pre_treatment_subset: np.ndarray = treated_outcome_full_period[:num_pre_treatment_periods]
    donor_outcomes_pre_treatment_subset: np.ndarray = donor_outcomes_full_period[:num_pre_treatment_periods]
    donor_outcomes_post_treatment_subset: np.ndarray = donor_outcomes_full_period[num_pre_treatment_periods:]

    # Step 1: Estimate alignment coefficients (theta) and get the aligned pre-treatment donor matrix.
    # `get_theta` aligns each donor's pre-treatment outcome series with the treated unit's pre-treatment outcome series.
    estimated_alignment_coefficients, aligned_donor_outcomes_pre_treatment_subset = get_theta(
        treated_outcome_pre_treatment_subset, donor_outcomes_pre_treatment_subset
    )
    
    # Step 2: Estimate the noise variance (sigma^2) from the pre-treatment period.
    # `get_sigmasq` estimates the variance of the noise in the relationship between the treated unit's outcome
    # and the (aligned) donor outcomes in the pre-treatment period.
    estimated_noise_variance = get_sigmasq(
        treated_outcome_pre_treatment_subset, donor_outcomes_pre_treatment_subset
    )

    # Step 3: Perform SRC weight optimization and predict outcomes.
    # `__SRC_opt` solves for donor weights and constructs pre-treatment predictions and post-treatment counterfactuals.
    # It uses the aligned donor outcomes and the estimated noise variance in its optimization.
    predicted_treated_outcome_pre_treatment_subset, predicted_counterfactual_post_treatment_subset,\
    optimal_donor_weights_result, _ = __SRC_opt( # The last returned item (alignment_coefficients) is ignored here as we already have it.
        treated_outcome_pre_treatment_subset, 
        donor_outcomes_pre_treatment_subset, # Original pre-treatment donor outcomes for constructing predictions
        donor_outcomes_post_treatment_subset, # Original post-treatment donor outcomes for constructing counterfactuals
        aligned_donor_outcomes_pre_treatment_subset, # Aligned pre-treatment donor outcomes for optimization
        estimated_alignment_coefficients, 
        estimated_noise_variance
    )
    
    # Concatenate the in-sample pre-treatment predictions and the post-treatment counterfactual predictions
    # to form the full counterfactual series for the treated unit.
    counterfactual_full_period: np.ndarray = np.concatenate([
        predicted_treated_outcome_pre_treatment_subset, 
        predicted_counterfactual_post_treatment_subset
    ])
    
    # Return the full counterfactual series, the optimal donor weights, and the estimated alignment coefficients.
    return counterfactual_full_period, optimal_donor_weights_result, estimated_alignment_coefficients


def RPCASYNTH(
    panel_dataframe: pd.DataFrame,
    estimation_config: Dict[str, Any],
    preprocessed_data_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Robust Principal Component Analysis Synthetic Control (RPCASYNTH).

    This method combines Robust PCA with Synthetic Control. It first performs
    functional PCA-based clustering on pre-treatment outcomes to identify a
    relevant donor pool. Then, Robust PCA (either PCP or HQF) is applied to
    the outcomes of these clustered donors. Finally, Synthetic Control Method
    weights are estimated using the low-rank representation of the donors from
    RPCA, and these weights are used to construct the counterfactual.

    Parameters
    ----------
    panel_dataframe : pd.DataFrame
        Panel data, typically in long format, containing outcomes for all units
        over time. It will be pivoted to a wide format internally. Must contain
        columns specified by `estimation_config['unitid']`, `estimation_config['time']`, and
        `estimation_config['outcome']`.
    estimation_config : Dict[str, Any]
        Configuration dictionary for the estimation. Expected keys are:
        - 'unitid' : str
            Column name in `panel_dataframe` identifying unique units.
        - 'time' : str
            Column name in `panel_dataframe` identifying time periods.
        - 'outcome' : str
            Column name in `panel_dataframe` for the outcome variable.
        - 'ROB' : str, optional
            Robust PCA method to use. Options are 'PCP' (Principal Component
            Pursuit) or 'HQF' (High Quality Factorization). Defaults to 'PCP'
            if not provided in `estimation_config`.
    preprocessed_data_dict : Dict[str, Any]
        A dictionary containing preprocessed data, typically output from a data
        preparation utility. Expected keys are:
        - 'treated_unit_name' : Union[str, int]
            Identifier of the treated unit. Must match a value in
            `panel_dataframe[estimation_config['unitid']]`.
        - 'pre_periods' : int
            Number of pre-treatment time periods. Used to slice the data for
            fitting various components (clustering, RPCA, SCM).
        - 'post_periods' : int
            Number of post-treatment time periods. Used for calculating effects.
        - 'y' : np.ndarray
            Outcome vector for the treated unit across all time periods (pre and
            post). Shape (total_time_periods,). This is used by `effects.calculate` as the
            observed series for the treated unit.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the estimation results. The keys are:
        - 'name' : str
            The name of the method, set to 'RPCA'.
        - 'weights' : Dict[str, float]
            A dictionary mapping donor names (from the selected cluster,
            excluding the treated unit) to their estimated SCM weights. Only
            non-zero weights (rounded to 3 decimal places, absolute value > 0.001)
            are included.
        - 'Effects' : Dict[str, float]
            Dictionary of treatment effect estimates (e.g., 'ATT', 'ATT_post',
            'ATT_pre') as returned by `effects.calculate`.
        - 'Fit' : Dict[str, float]
            Dictionary of goodness-of-fit statistics (e.g., 'RMSE_pre', 'R2_pre')
            as returned by `effects.calculate`.
        - 'Vectors' : Dict[str, np.ndarray]
            Dictionary of time series vectors (e.g., 'Y' (original treated
            outcome from `preprocessed_data_dict['y']`), 'Y_sc' (synthetic counterfactual),
            'Gap', 'TimePeriods') as returned by `effects.calculate`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data
    >>> data_example = {
    ...     'id_col': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4],
    ...     'time_col': [2000,2001,2002,2003,2004] * 4,
    ...     'outcome_val': (
    ...         np.array([10,11,12,13,14]) +  # Unit 1 (Treated)
    ...         np.array([20,21,22,18,19]) +  # Unit 2
    ...         np.array([15,16,17,18,19]) +  # Unit 3
    ...         np.array([12,13,14,10,11])    # Unit 4
    ...     )
    ... }
    >>> panel_df_example = pd.DataFrame(data_example)
    >>> config_example = {
    ...     'unitid': 'id_col', 'time': 'time_col', 'outcome': 'outcome_val', 'ROB': 'PCP'
    ... }
    >>> y_treated_example = panel_df_example[panel_df_example['id_col'] == 1]['outcome_val'].values
    >>> preproc_data_example = {
    ...     'treated_unit_name': 1, 'pre_periods': 3, 'post_periods': 2, 'y': y_treated_example
    ... }
    >>> results_rpca_example = RPCASYNTH(panel_df_example, config_example, preproc_data_example)
    >>> print(results_rpca_example['name'])
    RPCA
    >>> isinstance(results_rpca_example['weights'], dict)
    True
    >>> print(f"ATT: {results_rpca_example['Effects']['ATT']:.2f}") # Example output # doctest: +SKIP
    ATT: ...
    """
    unit_id_column_name: str = estimation_config["unitid"]
    time_period_column_name: str = estimation_config["time"]
    # Extract configuration parameters
    outcome_variable_column_name: str = estimation_config["outcome"]
    treated_unit_identifier = preprocessed_data_dict["treated_unit_name"]
    num_pre_treatment_periods_config: int = preprocessed_data_dict["pre_periods"]
    num_post_treatment_periods_config: int = preprocessed_data_dict["post_periods"]
    # Determine the Robust PCA method to use, defaulting to 'PCP' if not specified.
    robust_pca_method_type: str = estimation_config.get("ROB", "PCP")

    # Pivot the input panel data from long to wide format.
    # Rows become units, columns become time periods, values are outcomes.
    panel_data_wide_format = panel_dataframe.pivot_table(
        index=estimation_config["unitid"], # Unit identifier column
        columns=estimation_config["time"],   # Time period column
        values=outcome_variable_column_name, # Outcome variable column
        sort=False # Maintain original unit order if possible
    )
    # Extract pre-treatment outcomes for all units from the wide-format data.
    pre_treatment_outcomes_wide_format: np.ndarray = panel_data_wide_format.iloc[:, :num_pre_treatment_periods_config].to_numpy()

    # Perform functional PCA (fPCA) on pre-treatment outcomes to get features for clustering.
    # fpca returns: optimal number of clusters, transformed features, and principal components.
    # This step aims to group donors based on the similarity of their pre-treatment outcome trajectories.
    optimal_number_of_clusters, transformed_features_for_clustering, _ = fpca(pre_treatment_outcomes_wide_format)
    
    # Apply K-Means clustering using the features derived from fPCA.
    kmeans_clustering_model = KMeans(
        n_clusters=optimal_number_of_clusters, # Number of clusters determined by fPCA
        random_state=0, # For reproducibility
        init="k-means++", # Smart initialization for K-Means
        n_init='auto' # Automatically determine number of initializations
    )
    # Assign each unit (row in panel_data_wide_format) to a cluster.
    panel_data_wide_format["cluster_assignment"] = kmeans_clustering_model.fit_predict(transformed_features_for_clustering)

    # Identify the cluster to which the treated unit belongs.
    treated_unit_cluster_assignment: int = panel_data_wide_format.loc[treated_unit_identifier, "cluster_assignment"]
    # Select only the donor units that belong to the same cluster as the treated unit.
    donors_in_treated_unit_cluster_df = panel_data_wide_format[
        panel_data_wide_format["cluster_assignment"] == treated_unit_cluster_assignment
    ].drop("cluster_assignment", axis=1) # Remove the cluster assignment column

    # Extract outcome data for all units (treated and donors) within the selected cluster.
    all_outcomes_for_donors_in_cluster: np.ndarray = donors_in_treated_unit_cluster_df.to_numpy()
    # Find the row index of the treated unit within this clustered DataFrame.
    treated_unit_row_index_in_cluster_df: int = donors_in_treated_unit_cluster_df.index.get_loc(treated_unit_identifier)
    
    # Separate the treated unit's outcomes from the donor outcomes within the cluster.
    treated_unit_outcomes_from_cluster_df: np.ndarray = all_outcomes_for_donors_in_cluster[treated_unit_row_index_in_cluster_df]
    # Donor outcomes for all periods, excluding the treated unit.
    donor_outcomes_from_cluster_df_all_periods: np.ndarray = np.delete(all_outcomes_for_donors_in_cluster, treated_unit_row_index_in_cluster_df, axis=0)
    
    # Get the names of the donor units in the selected cluster.
    donor_names_in_cluster: List[str] = [
        name for name in donors_in_treated_unit_cluster_df.index if name != treated_unit_identifier
    ]

    # Apply Robust PCA to the outcomes of the selected donor pool (transposed, as RPCA expects features as columns).
    # This step aims to extract a low-rank (denoised) representation of the donor outcomes.
    low_rank_donor_outcomes_matrix: np.ndarray
    if robust_pca_method_type == "PCP": # Principal Component Pursuit
        low_rank_donor_outcomes_matrix = RPCA(donor_outcomes_from_cluster_df_all_periods.T).T
    elif robust_pca_method_type == "HQF": # High Quality Factorization
        hqf_m_dimension, hqf_n_dimension = donor_outcomes_from_cluster_df_all_periods.T.shape
        _, hqf_singular_values, _ = np.linalg.svd(donor_outcomes_from_cluster_df_all_periods.T, full_matrices=False)
        # Determine rank based on energy threshold of singular values.
        hqf_rank: int = spectral_rank(hqf_singular_values, energy_threshold=0.999)
        # Set regularization parameter for HQF.
        hqf_lambda_1: float = 1 / np.sqrt(max(hqf_m_dimension, hqf_n_dimension))
        low_rank_donor_outcomes_matrix_transposed_hqf = RPCA_HQF(
            observed_matrix_with_noise=donor_outcomes_from_cluster_df_all_periods.T, 
            target_rank_for_low_rank_component=hqf_rank, 
            max_iterations=1000,
            noise_scale_adaptation_factor=1.0, 
            factor_regularization_penalty=hqf_lambda_1 
        )
        low_rank_donor_outcomes_matrix = low_rank_donor_outcomes_matrix_transposed_hqf.T
    else: # Default to PCP if an invalid method is specified.
        warnings.warn(f"Invalid robust method '{robust_pca_method_type}'. Defaulting to 'PCP'.", UserWarning)
        low_rank_donor_outcomes_matrix = RPCA(donor_outcomes_from_cluster_df_all_periods.T).T

    # Extract pre-treatment outcomes for the treated unit (from the clustered data).
    treated_unit_outcomes_from_cluster_pre_treatment = treated_unit_outcomes_from_cluster_df[:num_pre_treatment_periods_config]
    # Extract pre-treatment low-rank donor outcomes and transpose for SCM fitting (donors as columns).
    low_rank_donor_outcomes_pre_treatment_transposed_for_scm = low_rank_donor_outcomes_matrix[:, :num_pre_treatment_periods_config].T

    # Estimate SCM weights using the low-rank pre-treatment donor outcomes and the treated unit's pre-treatment outcomes.
    # Opt.SCopt solves the SCM optimization problem. Here, MSCb (non-negative weights, no sum-to-one constraint by default) is used.
    scm_optimization_problem_rpca = Opt.SCopt(
        num_control_units=low_rank_donor_outcomes_pre_treatment_transposed_for_scm.shape[1], # Number of features in low-rank matrix
        target_outcomes_pre_treatment=treated_unit_outcomes_from_cluster_pre_treatment,
        num_pre_treatment_periods=num_pre_treatment_periods_config,
        donor_outcomes_pre_treatment=low_rank_donor_outcomes_pre_treatment_transposed_for_scm,
        scm_model_type=_SCM_MODEL_MSCB # Using MSCb model type for SCM
    )
    # Extract the estimated SCM weights.
    estimated_scm_weights_rpca: np.ndarray = scm_optimization_problem_rpca.solution.primal_vars[
        next(iter(scm_optimization_problem_rpca.solution.primal_vars))
    ]
    
    # Construct the counterfactual outcome for the treated unit.
    # This is done by applying the learned SCM weights to the low-rank donor outcomes matrix (transposed back).
    # The counterfactual is generated for all time periods.
    counterfactual_outcome_rpca: np.ndarray = np.dot(low_rank_donor_outcomes_matrix.T, estimated_scm_weights_rpca)
    
    # Create a dictionary of donor weights, showing only non-zero weights (rounded).
    final_donor_weights_dict_rpca: Dict[str, float] = {
        name: round(weight,3) for name, weight in zip(donor_names_in_cluster, estimated_scm_weights_rpca) if abs(weight) > 0.001
    }

    # Calculate treatment effects, fit diagnostics, and time series vectors using the observed treated outcome
    # (from preprocessed_data_dict['y']) and the constructed counterfactual.
    treatment_effects_results_rpca, fit_diagnostics_results_rpca, time_series_vectors_rpca = effects.calculate(
        observed_outcome_series=preprocessed_data_dict["y"], # Original observed outcome for the treated unit
        counterfactual_outcome_series=counterfactual_outcome_rpca,
        num_pre_treatment_periods=num_pre_treatment_periods_config,
        num_actual_post_periods=num_post_treatment_periods_config
    )
    # Return the results in a structured dictionary.
    return {
        "name": "RPCA", 
        "weights": final_donor_weights_dict_rpca, 
        "Effects": treatment_effects_results_rpca,
        "Fit": fit_diagnostics_results_rpca, 
        "Vectors": time_series_vectors_rpca,
    }


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
