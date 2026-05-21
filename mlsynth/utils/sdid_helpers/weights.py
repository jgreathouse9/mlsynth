"""Unit-weight, time-weight, and regularization solvers for SDID.

Verbatim from the previous monolithic ``sdidutils.py``; kept untouched so the
Prop 99 ATT remains numerically identical to the pre-refactor result.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError


def fit_time_weights(
    donor_outcomes_pre_treatment: np.ndarray, mean_donor_outcomes_post_treatment: np.ndarray
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Fit time weights for SDID.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    mean_donor_outcomes_post_treatment : np.ndarray
        Mean outcome of each donor unit in post-treatment period, shape (N_donors,).

    Returns
    -------
    Tuple[Optional[float], Optional[np.ndarray]]
        - intercept : Optional[float]
            The estimated intercept term (beta_0 in some notations).
            Returns `None` if the optimization fails or does not converge.
        - time_weights : Optional[np.ndarray]
            The estimated time weights (lambda_t in some notations).
            Shape (num_pre_treatment_periods,). These weights sum to 1 and are non-negative.
            Returns `None` if the optimization fails or does not converge.

    Notes
    -----
    This function solves an optimization problem to find time weights and an
    intercept that best reconstruct the average post-treatment donor outcomes
    using a weighted average of pre-treatment donor outcomes.
    The objective is to minimize the sum of squared differences between
    `mean_donor_outcomes_post_treatment` and
    `intercept + time_weights @ donor_outcomes_pre_treatment`, subject to `sum(time_weights) = 1`
    and `time_weights >= 0`.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 5, 3
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex)
    >>> Y0_post_donors_mean_ex = np.random.rand(N_donors_ex)
    >>> intercept_val, time_w_val = fit_time_weights(Y0_pre_donors_ex, Y0_post_donors_mean_ex)
    >>> if time_w_val is not None:
    ...     print(f"Time weights shape: {time_w_val.shape}")
    ...     print(f"Sum of time weights: {np.sum(time_w_val):.2f}")
    Time weights shape: (5,)
    Sum of time weights: 1.00
    """
    # Input Validation
    # Ensure donor_outcomes_pre_treatment is a 2D NumPy array.
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2:
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    # Ensure mean_donor_outcomes_post_treatment is a 1D NumPy array.
    if not isinstance(mean_donor_outcomes_post_treatment, np.ndarray):
        raise MlsynthDataError("mean_donor_outcomes_post_treatment must be a NumPy array.")
    if mean_donor_outcomes_post_treatment.ndim != 1:
        raise MlsynthDataError("mean_donor_outcomes_post_treatment must be a 1D array (N_donors,).")
    
    # Get dimensions: number of pre-treatment periods and number of donors.
    num_pre_treatment_periods, num_donors_pre = donor_outcomes_pre_treatment.shape
    num_donors_post = mean_donor_outcomes_post_treatment.shape[0]

    # Validate dimensions.
    if num_pre_treatment_periods == 0: # Must have pre-treatment periods to fit weights.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero pre-treatment periods (num_pre_treatment_periods must be > 0).")
    if num_donors_pre == 0: # Must have donors.
        # This case might be implicitly handled by CVXPY if num_donors_post is also 0,
        # but explicit check is better. If num_donors_post > 0, it's a mismatch.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero donors if mean_donor_outcomes_post_treatment has donors.")

    # Number of donors must be consistent between pre-treatment and post-treatment data.
    if num_donors_pre != num_donors_post:
        raise MlsynthDataError(
            f"Shape mismatch: donor_outcomes_pre_treatment has {num_donors_pre} donors, "
            f"but mean_donor_outcomes_post_treatment has {num_donors_post} donors."
        )

    # Define CVXPY optimization variables.
    intercept_variable = cp.Variable() # Intercept term (beta_0).
    time_weights_variable = cp.Variable(num_pre_treatment_periods, nonneg=True) # Time weights (lambda_t), constrained to be non-negative.
    
    # Define the prediction model: intercept + time_weights @ donor_outcomes_pre_treatment.
    # This reconstructs the mean post-treatment donor outcomes using weighted pre-treatment donor outcomes.
    prediction = intercept_variable + (time_weights_variable @ donor_outcomes_pre_treatment)
    # Define constraints: time weights must sum to 1.
    constraints = [cp.sum(time_weights_variable) == 1]
    # Define the objective: minimize sum of squared differences between actual and predicted mean post-treatment donor outcomes.
    objective = cp.Minimize(cp.sum_squares(prediction - mean_donor_outcomes_post_treatment))
    # Create and solve the CVXPY problem.
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.CLARABEL) # Using CLARABEL solver.
    except cp.error.SolverError as e: # Catch solver errors.
        raise MlsynthEstimationError(f"CVXPY solver failed in fit_time_weights: {e}") from e

    # Check problem status and return results if optimal or optimal_inaccurate.
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return intercept_variable.value, time_weights_variable.value
    # If solver finishes but status is not optimal/optimal_inaccurate,
    # it implies a failure to find a good solution. Returning None, None is current behavior.
    # This indicates the optimization did not converge to a satisfactory solution.
    return None, None


def compute_regularization(
    donor_outcomes_pre_treatment: np.ndarray, num_post_treatment_periods: int
) -> float:
    """
    Compute regularization parameter zeta for unit weights.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    num_post_treatment_periods : int
        Number of post-treatment periods (used as a proxy for N_tr_post in original papers).

    Returns
    -------
    float
        The calculated regularization parameter zeta. If `donor_outcomes_pre_treatment` has
        fewer than 2 time periods, a fallback value (currently 1.0, though this
        might indicate insufficient data for robust estimation) is used for
        `std_dev_of_first_differenced_donor_outcomes`, which then influences zeta.

    Notes
    -----
    The regularization parameter `zeta` is calculated as:
    `zeta = (num_post_treatment_periods ** 0.25) * std_dev_of_first_differenced_donor_outcomes`
    where `std_dev_of_first_differenced_donor_outcomes` is the standard deviation of the first-differenced
    outcomes of donor units in the pre-treatment period. This aims to adapt
    the regularization strength based on the variability of donor outcomes and
    the length of the post-treatment period.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 10, 5
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex) * 100
    >>> T_post_ex = 5
    >>> zeta = compute_regularization(Y0_pre_donors_ex, T_post_ex)
    >>> print(f"Zeta: {zeta:.2f}")
    Zeta: ...

    >>> # Example with insufficient pre-treatment periods for diff
    >>> Y0_short_pre_donors_ex = np.random.rand(1, N_donors_ex)
    >>> zeta_short = compute_regularization(Y0_short_pre_donors_ex, T_post_ex)
    >>> # Based on fallback std_dev_of_first_differenced_donor_outcomes = 1.0
    >>> # Expected: (5**0.25) * 1.0 = 1.495...
    >>> print(f"Zeta for short pre-period: {zeta_short:.2f}")
    Zeta for short pre-period: 1.50
    """
    # Input Validation
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2:
        # Allow 0 donors for flexibility, std calculation will handle it or fallback.
        # However, if shape[0] (periods) is 0, diff will fail.
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    if not isinstance(num_post_treatment_periods, int) or num_post_treatment_periods < 0:
        raise MlsynthConfigError("num_post_treatment_periods must be a non-negative integer.")

    # Calculate the standard deviation of the first-differenced donor outcomes in the pre-treatment period.
    # This term captures the volatility of donor outcomes.
    if donor_outcomes_pre_treatment.shape[0] < 2 : # Need at least 2 pre-treatment periods to calculate differences.
        # Fallback value if insufficient pre-treatment periods for differencing.
        # This implies high uncertainty or reliance on the num_post_treatment_periods term.
        std_dev_of_first_differenced_donor_outcomes = 1.0 
    elif donor_outcomes_pre_treatment.shape[1] == 0: # No donors.
        std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if no donors to calculate differences from.
    else:
        # Calculate first differences of donor outcomes along the time axis (axis=0).
        diffs = np.diff(donor_outcomes_pre_treatment, axis=0)
        if diffs.size == 0: # Can happen if T0=1 (already caught by shape[0]<2) or N_donors=0 (caught by shape[1]==0).
             std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if differences result in an empty array.
        else:
             # Calculate standard deviation of these differences. ddof=1 for sample standard deviation.
             std_dev_of_first_differenced_donor_outcomes = np.std(diffs.flatten(), ddof=1)
             if np.isnan(std_dev_of_first_differenced_donor_outcomes): # If all diffs were NaN, leading to NaN std.
                 std_dev_of_first_differenced_donor_outcomes = 1.0 # Fallback if std dev is NaN.

    # Calculate zeta: (num_post_treatment_periods ^ 0.25) * std_dev_of_first_differenced_donor_outcomes.
    # The term (num_post_treatment_periods ^ 0.25) scales regularization by the length of the post-treatment period.
    # A longer post-treatment period might suggest a need for stronger regularization.
    regularization_parameter_zeta: float = (num_post_treatment_periods**0.25) * std_dev_of_first_differenced_donor_outcomes
    return regularization_parameter_zeta


def unit_weights(
    donor_outcomes_pre_treatment: np.ndarray,
    mean_treated_outcome_pre_treatment: np.ndarray,
    regularization_parameter_zeta: float
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Fit unit (donor) weights for SDID.

    Parameters
    ----------
    donor_outcomes_pre_treatment : np.ndarray
        Donor outcomes in pre-treatment period, shape (T0, N_donors).
    mean_treated_outcome_pre_treatment : np.ndarray
        Mean outcome of treated units in pre-treatment period, shape (T0,).
    regularization_parameter_zeta : float
        Regularization parameter.

    Returns
    -------
    Tuple[Optional[float], Optional[np.ndarray]]
        - intercept : Optional[float]
            The estimated intercept term (beta_0 in some notations).
            Returns `None` if the optimization fails or does not converge.
        - unit_weights : Optional[np.ndarray]
            The estimated donor weights (omega_j in some notations).
            Shape (N_donors,). These weights sum to 1 and are non-negative.
            Returns `None` if the optimization fails or does not converge.

    Notes
    -----
    This function solves an optimization problem to find donor weights and an
    intercept that best reconstruct the pre-treatment trajectory of the
    (mean) treated unit using a weighted average of donor unit outcomes.
    The objective is to minimize the sum of squared differences between
    `mean_treated_outcome_pre_treatment` and `intercept + donor_outcomes_pre_treatment @ unit_weights`,
    plus an L2 penalty on the `unit_weights` scaled by `regularization_parameter_zeta`.
    Constraints are `sum(unit_weights) = 1` and `unit_weights >= 0`.

    Examples
    --------
    >>> T0_ex, N_donors_ex = 10, 5
    >>> Y0_pre_donors_ex = np.random.rand(T0_ex, N_donors_ex)
    >>> y_pre_mean_treated_ex = np.random.rand(T0_ex)
    >>> zeta_ex = 0.1
    >>> intercept_val, unit_w_val = unit_weights(
    ...     Y0_pre_donors_ex, y_pre_mean_treated_ex, zeta_ex
    ... )
    >>> if unit_w_val is not None:
    ...     print(f"Unit weights shape: {unit_w_val.shape}")
    ...     print(f"Sum of unit weights: {np.sum(unit_w_val):.2f}")
    Unit weights shape: (5,)
    Sum of unit weights: 1.00
    """
    # Input Validation
    if not isinstance(donor_outcomes_pre_treatment, np.ndarray):
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a NumPy array.")
    if donor_outcomes_pre_treatment.ndim != 2: # Must be 2D: (Time, Donors)
        raise MlsynthDataError("donor_outcomes_pre_treatment must be a 2D array (T0, N_donors).")
    if not isinstance(mean_treated_outcome_pre_treatment, np.ndarray):
        raise MlsynthDataError("mean_treated_outcome_pre_treatment must be a NumPy array.")
    if mean_treated_outcome_pre_treatment.ndim != 1: # Must be 1D: (Time,)
        raise MlsynthDataError("mean_treated_outcome_pre_treatment must be a 1D array (T0,).")
    if not isinstance(regularization_parameter_zeta, (float, int)) or regularization_parameter_zeta < 0:
        raise MlsynthConfigError("regularization_parameter_zeta must be a non-negative float or int.")

    # Get dimensions: number of pre-treatment periods and number of donors.
    num_pre_treatment_periods, num_donors = donor_outcomes_pre_treatment.shape
    
    # Validate dimensions.
    if num_pre_treatment_periods == 0: # Must have pre-treatment periods.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero pre-treatment periods (num_pre_treatment_periods must be > 0).")
    if num_donors == 0: # Must have donors.
        raise MlsynthDataError("donor_outcomes_pre_treatment cannot have zero donors (num_donors must be > 0).")
    # Number of pre-treatment periods must match between donor outcomes and mean treated outcome.
    if mean_treated_outcome_pre_treatment.shape[0] != num_pre_treatment_periods:
        raise MlsynthDataError(
            f"Shape mismatch: donor_outcomes_pre_treatment has {num_pre_treatment_periods} pre-periods, "
            f"but mean_treated_outcome_pre_treatment has {mean_treated_outcome_pre_treatment.shape[0]}."
        )

    # Define CVXPY optimization variables.
    intercept_variable = cp.Variable() # Intercept term (beta_0).
    unit_weights_variable = cp.Variable(num_donors, nonneg=True) # Unit (donor) weights (omega_j), constrained to be non-negative.
    
    # Define the prediction model: intercept + donor_outcomes_pre_treatment @ unit_weights.
    # This reconstructs the mean pre-treatment treated outcome using weighted donor outcomes.
    prediction = intercept_variable + donor_outcomes_pre_treatment @ unit_weights_variable
    
    # Define the L2 regularization penalty on unit weights.
    # Penalty = T0 * zeta^2 * sum_squares(omega).
    # Ensure penalty term is well-defined even if num_pre_treatment_periods is 0 (though caught above by validation).
    penalty_coefficient = num_pre_treatment_periods * (float(regularization_parameter_zeta)**2)
    penalty = penalty_coefficient * cp.sum_squares(unit_weights_variable)
    
    # Define the objective: minimize sum of squared differences between actual and predicted mean treated outcome, plus the L2 penalty.
    objective = cp.Minimize(cp.sum_squares(prediction - mean_treated_outcome_pre_treatment) + penalty)
    # Define constraints: unit weights must sum to 1.
    constraints = [cp.sum(unit_weights_variable) == 1]
    # Create and solve the CVXPY problem.
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.CLARABEL) # Using CLARABEL solver.
    except cp.error.SolverError as e: # Catch solver errors.
        raise MlsynthEstimationError(f"CVXPY solver failed in unit_weights: {e}") from e

    # Check problem status and return results if optimal or optimal_inaccurate.
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return intercept_variable.value, unit_weights_variable.value
    # If solver finishes but status is not optimal/optimal_inaccurate,
    # it implies a failure to find a good solution. Returning None, None indicates this.
    return None, None
