import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import rc_context
import os
from typing import Optional, Dict, List, Tuple, Any

import cvxpy # For cvxpy.error.SolverError
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError


def prenorm(input_array: np.ndarray, target: float = 100) -> np.ndarray:
    """Normalize a vector or matrix based on its last element or last row.

    This function scales the input array `input_array` such that its last element (if `input_array`
    is 1D) or each column's last element (if `input_array` is 2D, using the last row
    `input_array[-1, :]` for normalization factors) is scaled towards a `target` value.
    Specifically, each element `input_array[i]` (for 1D) or `input_array[i, j]` (for 2D) is
    transformed to `(input_array[i] / input_array[-1]) * target` or `(input_array[i, j] / input_array[-1, j]) * target`.

    Parameters
    ----------
    input_array : np.ndarray
        Input array to be normalized. Can be 1-dimensional (vector) or
        2-dimensional (matrix).
        - If 1D, shape (T,): Normalization is based on the last element `input_array[-1]`.
        - If 2D, shape (T, N): Normalization is performed column-wise, based
          on the elements of the last row `input_array[-1, :]`. Each column `input_array[:, j]` is
          normalized using `input_array[-1, j]`.
    target : float, optional
        The target value to which the reference element(s) (last element or
        last row elements) are scaled. Default is 100.

    Returns
    -------
    np.ndarray
        The normalized array, having the same shape as the input `input_array`.

    Raises
    ------
    MlsynthDataError
        If any element used for normalization (i.e., `input_array[-1]` for 1D input,
        or any element in `input_array[-1, :]` for 2D input) is zero.

    Examples
    --------
    >>> # Example with a 1D array
    >>> x_1d = np.array([10, 20, 40, 50], dtype=float)
    >>> prenorm(x_1d, target=100)
    array([ 20.,  40.,  80., 100.])

    >>> # Example with a 2D array (matrix)
    >>> x_2d = np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12]], dtype=float)
    >>> # Last row is [4, 8, 12]. Target is 100.
    >>> # Col 0: [1, 2, 4] -> [1/4*100, 2/4*100, 4/4*100] = [25, 50, 100]
    >>> # Col 1: [2, 4, 8] -> [2/8*100, 4/8*100, 8/8*100] = [25, 50, 100]
    >>> # Col 2: [3, 6, 12] -> [3/12*100, 6/12*100, 12/12*100] = [25, 50, 100]
    >>> prenorm(x_2d, target=100)
    array([[ 25.,  25.,  25.],
           [ 50.,  50.,  50.],
           [100., 100., 100.]])

    >>> # Example that raises MlsynthDataError
    >>> x_zero_1d = np.array([10, 0], dtype=float)
    >>> try:
    ...     prenorm(x_zero_1d)
    ... except MlsynthDataError as e:
    ...     print(e)
    Division by zero: Denominator for normalization is zero.
    """
    input_array_np = np.asarray(input_array)
    # Determine the denominator for normalization:
    # If 1D array, use the last element.
    # If 2D array, use the last row (for column-wise normalization).
    normalization_denominator = input_array_np[-1] if input_array_np.ndim == 1 else input_array_np[-1, :]

    # Check if any element in the normalization_denominator is zero to prevent division by zero.
    if np.any(normalization_denominator == 0):
        raise MlsynthDataError("Division by zero: Denominator for normalization is zero.")

    # Perform the normalization: element-wise division by the denominator, then scale by target.
    return input_array_np / normalization_denominator * target


def ssdid_w(
    treated_unit_outcomes_all_periods: np.ndarray,
    donor_units_outcomes_all_periods: np.ndarray,
    num_matching_pre_periods: int,
    matching_horizon_offset: int,
    l2_penalty_regularization_strength: float,
    donor_prior_weights_for_penalty: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Solve for synthetic control weights (omega) for the SSDID method.

    This function estimates the donor weights (`optimal_donor_weights`) and an intercept term
    (`optimal_intercept`) that best match the pre-treatment trajectory of a treated unit.
    The optimization minimizes the sum of squared residuals between the treated
    unit's outcomes and the weighted combination of donor outcomes, plus an L2
    penalty on the donor weights. The weights are constrained to sum to 1.

    The objective function is:
    `min_{optimal_donor_weights, optimal_intercept} || donor_units_outcomes_matching_period @ optimal_donor_weights - optimal_intercept - treated_unit_outcomes_matching_period ||_2^2 +
    l2_penalty_regularization_strength^2 * sum(donor_prior_weights_for_penalty_j * optimal_donor_weights_j^2)`
    subject to `sum(optimal_donor_weights) == 1`.

    Parameters
    ----------
    treated_unit_outcomes_all_periods : np.ndarray
        Outcome vector for the treated unit across all time periods.
        Shape (T_total,), where T_total is the total number of time periods. (Formerly `treated_y`)
    donor_units_outcomes_all_periods : np.ndarray
        Matrix of outcomes for donor units across all time periods.
        Shape (T_total, J), where J is the number of donor units. (Formerly `donor_matrix`)
    num_matching_pre_periods : int
        Number of pre-treatment periods to use for matching the treated unit's
        trajectory. The estimation uses data up to `matching_period_end_index = num_matching_pre_periods + matching_horizon_offset`. (Formerly `a`)
    matching_horizon_offset : int
        Horizon parameter that defines the end of the matching period (`matching_period_end_index`).
        The outcomes from `treated_unit_outcomes_all_periods[:matching_period_end_index]` and
        `donor_units_outcomes_all_periods[:matching_period_end_index, :]` are used in the optimization. (Formerly `k`)
    l2_penalty_regularization_strength : float
        Regularization parameter (non-negative) for the L2 penalty on the
        donor weights `optimal_donor_weights`. A larger `l2_penalty_regularization_strength` imposes a stronger penalty. (Formerly `eta`)
    donor_prior_weights_for_penalty : Optional[np.ndarray], default None
        Prior weights for donor units, used in the L2 penalty term.
        Shape (J,). If None, uniform prior weights (1/J for each donor)
        are assumed. Default is None. (Formerly `pi`)

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing:
        - optimal_donor_weights : np.ndarray
            The optimal donor weights, shape (J,). These weights sum to 1. (Formerly `omega`)
        - optimal_intercept : float
            The optimal intercept term. (Formerly `omega_0`)

    Notes
    -----
    - The optimization is performed using CVXPY with the default solver.
    - `matching_period_end_index = num_matching_pre_periods + matching_horizon_offset` defines the window of pre-treatment data used for estimation. (Formerly `t_max = a + k`)

    Raises
    ------
    MlsynthDataError
        If input arrays have incorrect types, dimensions, or inconsistent shapes.
        If `donor_prior_weights_for_penalty` is provided and has an incorrect shape.
    MlsynthConfigError
        If `l2_penalty_regularization_strength` is negative.
        If `num_matching_pre_periods` or `matching_horizon_offset` are negative or lead to invalid slice indices.
    MlsynthEstimationError
        If the CVXPY optimization solver fails.

    Examples
    --------
    >>> T_total_ex, J_ex = 20, 3
    >>> num_matching_pre_periods_ex, matching_horizon_offset_ex, l2_penalty_ex = 10, 0, 0.1
    >>> treated_outcomes_ex = np.random.rand(T_total_ex)
    >>> donor_outcomes_ex = np.random.rand(T_total_ex, J_ex)
    >>> opt_omega_vals, opt_omega_0_val = ssdid_w(
    ...     treated_outcomes_ex, donor_outcomes_ex, num_matching_pre_periods_ex, matching_horizon_offset_ex, l2_penalty_ex
    ... )
    >>> print(f"Omega weights shape: {opt_omega_vals.shape}")
    Omega weights shape: (3,)
    >>> print(f"Sum of omega weights: {np.sum(opt_omega_vals):.2f}")
    Sum of omega weights: 1.00
    >>> print(f"Intercept omega_0: {opt_omega_0_val:.2f}")
    Intercept omega_0: ...
    """
    # Input validation
    if not isinstance(treated_unit_outcomes_all_periods, np.ndarray) or treated_unit_outcomes_all_periods.ndim != 1:
        raise MlsynthDataError("`treated_unit_outcomes_all_periods` must be a 1D NumPy array.")
    if not isinstance(donor_units_outcomes_all_periods, np.ndarray) or donor_units_outcomes_all_periods.ndim != 2:
        raise MlsynthDataError("`donor_units_outcomes_all_periods` must be a 2D NumPy array.")
    
    num_total_periods_from_treated = len(treated_unit_outcomes_all_periods)
    num_total_periods_from_donors, num_donor_units = donor_units_outcomes_all_periods.shape

    if num_total_periods_from_treated != num_total_periods_from_donors:
        raise MlsynthDataError(
            f"Mismatch in total periods: treated has {num_total_periods_from_treated}, "
            f"donors have {num_total_periods_from_donors}."
        )

    if l2_penalty_regularization_strength < 0:
        raise MlsynthConfigError("`l2_penalty_regularization_strength` cannot be negative.")
    if num_matching_pre_periods < 0:
        raise MlsynthConfigError("`num_matching_pre_periods` cannot be negative.")
    if matching_horizon_offset < 0:
        raise MlsynthConfigError("`matching_horizon_offset` cannot be negative.")

    donor_prior_weights_for_penalty_internal = donor_prior_weights_for_penalty
    if donor_prior_weights_for_penalty_internal is not None:
        if not isinstance(donor_prior_weights_for_penalty_internal, np.ndarray) or donor_prior_weights_for_penalty_internal.ndim != 1:
            raise MlsynthDataError("`donor_prior_weights_for_penalty` must be a 1D NumPy array if provided.")
        if len(donor_prior_weights_for_penalty_internal) != num_donor_units:
            raise MlsynthDataError(
                f"Length of `donor_prior_weights_for_penalty` ({len(donor_prior_weights_for_penalty_internal)}) "
                f"does not match number of donor units ({num_donor_units})."
            )
    else:
        donor_prior_weights_for_penalty_internal = np.ones(num_donor_units) / num_donor_units
    
    matching_period_end_index = num_matching_pre_periods + matching_horizon_offset
    if not (0 <= matching_period_end_index <= num_total_periods_from_treated):
        raise MlsynthConfigError(
            f"`matching_period_end_index` ({matching_period_end_index}) derived from "
            f"`num_matching_pre_periods` and `matching_horizon_offset` is out of bounds "
            f"for arrays of length {num_total_periods_from_treated}."
        )

    # Extract data for the matching period (up to matching_period_end_index)
    treated_unit_outcomes_matching_period = treated_unit_outcomes_all_periods[:matching_period_end_index]
    donor_units_outcomes_matching_period = donor_units_outcomes_all_periods[:matching_period_end_index, :]

    # Define CVXPY optimization variables
    donor_weights_variable = cp.Variable(num_donor_units)  # omega_hat in paper
    intercept_variable = cp.Variable()  # omega_0_hat in paper

    # Define the objective function components
    # Residuals: (Donor outcomes * weights - intercept) - Treated outcomes
    residuals = donor_units_outcomes_matching_period @ donor_weights_variable - intercept_variable - treated_unit_outcomes_matching_period
    # Sum of squared residuals (matching term)
    sum_squared_residuals = cp.sum_squares(residuals)
    # L2 penalty term on donor weights, incorporating prior weights
    l2_penalty_term = l2_penalty_regularization_strength**2 * cp.sum(
        cp.multiply(donor_prior_weights_for_penalty_internal, cp.square(donor_weights_variable))
    )
    # Full objective function
    objective = sum_squared_residuals + l2_penalty_term
    
    # Define constraints: donor weights must sum to 1
    constraints = [cp.sum(donor_weights_variable) == 1]

    # Set up and solve the CVXPY problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve()
    except cvxpy.error.SolverError as e:
        raise MlsynthEstimationError(f"CVXPY solver failed in ssdid_w: {e}") from e
    
    if donor_weights_variable.value is None: # Check if solver found a solution
        raise MlsynthEstimationError("CVXPY optimization failed to find a solution for donor weights in ssdid_w.")


    optimal_donor_weights = donor_weights_variable.value
    optimal_intercept = float(intercept_variable.value) # Ensure intercept is a float
    
    return optimal_donor_weights, optimal_intercept


def ssdid_lambda(
    treated_unit_outcomes_all_periods: np.ndarray,
    donor_units_outcomes_all_periods: np.ndarray,
    num_pre_treatment_periods_for_lambda: int,
    post_treatment_horizon_offset: int,
    l2_penalty_regularization_strength: float,
) -> Tuple[np.ndarray, float]:
    """Solve for time weights (lambda) and intercept for the SSDID method.

    This function estimates time weights (`optimal_time_weights`) and an intercept
    (`optimal_intercept`) used in the Synthetic Difference-in-Differences (SSDID)
    method. These weights are optimized to make a weighted average of the
    treated unit's pre-treatment outcomes match the average outcome of donor
    units at a specific post-treatment horizon `post_treatment_horizon_offset`.

    The objective function is:
    `min_{lambda, lambda_0} || treated_unit_outcomes_pre_treatment_for_lambda @ lambda - lambda_0 - target_donor_outcomes_at_horizon ||_2^2 +
    l2_penalty_regularization_strength^2 * ||lambda||_2^2`
    subject to `sum(lambda) == 1`.
    Here, `target_donor_outcomes_at_horizon` is `donor_units_outcomes_all_periods[num_pre_treatment_periods_for_lambda + post_treatment_horizon_offset, :]`
    (the average outcome of donor units at time `num_pre_treatment_periods_for_lambda + post_treatment_horizon_offset`).

    Parameters
    ----------
    treated_unit_outcomes_all_periods : np.ndarray
        Outcome vector for the treated unit across all time periods.
        Shape (T_total,), where T_total is the total number of time periods.
        Only the first `num_pre_treatment_periods_for_lambda` periods are used in the optimization. (Formerly `treated_y`)
    donor_units_outcomes_all_periods : np.ndarray
        Matrix of outcomes for donor units across all time periods.
        Shape (T_total, J), where J is the number of donor units.
        The row `donor_units_outcomes_all_periods[num_pre_treatment_periods_for_lambda + post_treatment_horizon_offset, :]` is used as the target for matching. (Formerly `donor_matrix`)
    num_pre_treatment_periods_for_lambda : int
        Number of pre-treatment periods of `treated_unit_outcomes_all_periods` to use for constructing
        the weighted average. The `time_weights_variable` will have length `num_pre_treatment_periods_for_lambda`. (Formerly `a`)
    post_treatment_horizon_offset : int
        Post-treatment horizon offset. The target for matching is the average
        donor outcome at time period `num_pre_treatment_periods_for_lambda + post_treatment_horizon_offset`. (Formerly `k`)
    l2_penalty_regularization_strength : float
        Regularization parameter (non-negative) for the L2 penalty on the
        time weights `time_weights_variable`. A larger `l2_penalty_regularization_strength` imposes a stronger penalty. (Formerly `eta`)

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing:
        - optimal_time_weights : np.ndarray
            The optimal time weights, shape (num_pre_treatment_periods_for_lambda,). These weights sum to 1. (Formerly `lambda_val`)
        - optimal_intercept : float
            The optimal intercept term. (Formerly `lambda_0_val`)

    Notes
    -----
    - The optimization is performed using CVXPY with the default solver.
    - `target_donor_period_index = num_pre_treatment_periods_for_lambda + post_treatment_horizon_offset` is the index into `donor_units_outcomes_all_periods` for the target observation.

    Raises
    ------
    MlsynthDataError
        If input arrays have incorrect types, dimensions, or inconsistent shapes.
    MlsynthConfigError
        If `l2_penalty_regularization_strength` is negative.
        If `num_pre_treatment_periods_for_lambda` or `post_treatment_horizon_offset` are negative or lead to invalid slice indices.
    MlsynthEstimationError
        If the CVXPY optimization solver fails.

    Examples
    --------
    >>> T_total_ex, J_ex = 20, 3
    >>> num_pre_periods_lambda_ex, horizon_offset_ex, l2_penalty_ex = 10, 2, 0.05 # Target donor obs at time a+k = 12
    >>> treated_outcomes_ex = np.random.rand(T_total_ex)
    >>> donor_outcomes_ex = np.random.rand(T_total_ex, J_ex)
    >>> opt_lambda_weights, opt_lambda_intercept = ssdid_lambda(
    ...     treated_outcomes_ex, donor_outcomes_ex, num_pre_periods_lambda_ex, horizon_offset_ex, l2_penalty_ex
    ... )
    >>> print(f"Lambda weights shape: {opt_lambda_weights.shape}")
    Lambda weights shape: (10,)
    >>> print(f"Sum of lambda weights: {np.sum(opt_lambda_weights):.2f}")
    Sum of lambda weights: 1.00
    >>> print(f"Lambda intercept: {opt_lambda_intercept:.2f}")
    Lambda intercept: ...
    """
    # Input validation
    if not isinstance(treated_unit_outcomes_all_periods, np.ndarray) or treated_unit_outcomes_all_periods.ndim != 1:
        raise MlsynthDataError("`treated_unit_outcomes_all_periods` must be a 1D NumPy array.")
    if not isinstance(donor_units_outcomes_all_periods, np.ndarray) or donor_units_outcomes_all_periods.ndim != 2:
        raise MlsynthDataError("`donor_units_outcomes_all_periods` must be a 2D NumPy array.")

    num_total_periods_from_treated = len(treated_unit_outcomes_all_periods)
    num_total_periods_from_donors, _ = donor_units_outcomes_all_periods.shape

    if num_total_periods_from_treated != num_total_periods_from_donors:
        raise MlsynthDataError(
            f"Mismatch in total periods: treated has {num_total_periods_from_treated}, "
            f"donors have {num_total_periods_from_donors}."
        )

    if l2_penalty_regularization_strength < 0:
        raise MlsynthConfigError("`l2_penalty_regularization_strength` cannot be negative.")
    if num_pre_treatment_periods_for_lambda < 0:
        raise MlsynthConfigError("`num_pre_treatment_periods_for_lambda` cannot be negative.")
    if post_treatment_horizon_offset < 0:
        raise MlsynthConfigError("`post_treatment_horizon_offset` cannot be negative.")

    target_donor_period_index = num_pre_treatment_periods_for_lambda + post_treatment_horizon_offset
    if not (0 <= target_donor_period_index < num_total_periods_from_donors):
        raise MlsynthConfigError(
            f"`target_donor_period_index` ({target_donor_period_index}) derived from "
            f"`num_pre_treatment_periods_for_lambda` and `post_treatment_horizon_offset` is out of bounds "
            f"for donor arrays of length {num_total_periods_from_donors}."
        )
    if not (0 <= num_pre_treatment_periods_for_lambda <= num_total_periods_from_treated):
        raise MlsynthConfigError(
            f"`num_pre_treatment_periods_for_lambda` ({num_pre_treatment_periods_for_lambda}) is out of bounds "
            f"for treated array of length {num_total_periods_from_treated}."
        )


    # Extract pre-treatment outcomes for the treated unit
    treated_unit_outcomes_pre_treatment_for_lambda = treated_unit_outcomes_all_periods[:num_pre_treatment_periods_for_lambda]
    # Extract donor outcomes at the specific target horizon period
    target_donor_outcomes_at_horizon = donor_units_outcomes_all_periods[target_donor_period_index, :]

    # Define CVXPY optimization variables
    time_weights_variable = cp.Variable(num_pre_treatment_periods_for_lambda)  # lambda_hat in paper
    intercept_variable = cp.Variable()  # lambda_0_hat in paper

    # Define the objective function components
    # Residuals: (Treated pre-treatment outcomes * time weights - intercept) - Target donor outcomes at horizon
    residuals = (
        treated_unit_outcomes_pre_treatment_for_lambda @ time_weights_variable - intercept_variable - target_donor_outcomes_at_horizon
    )
    # Sum of squared residuals (matching term)
    sum_squared_residuals = cp.sum_squares(residuals)
    # L2 penalty term on time weights
    l2_penalty_term = l2_penalty_regularization_strength**2 * cp.sum_squares(time_weights_variable)
    # Full objective function
    objective = sum_squared_residuals + l2_penalty_term
    
    # Define constraints: time weights must sum to 1
    constraints = [cp.sum(time_weights_variable) == 1]

    # Set up and solve the CVXPY problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve()
    except cvxpy.error.SolverError as e:
        raise MlsynthEstimationError(f"CVXPY solver failed in ssdid_lambda: {e}") from e

    if time_weights_variable.value is None: # Check if solver found a solution
        raise MlsynthEstimationError("CVXPY optimization failed to find a solution for time weights in ssdid_lambda.")

    optimal_time_weights = time_weights_variable.value
    optimal_intercept = float(intercept_variable.value) # Ensure intercept is a float

    return optimal_time_weights, optimal_intercept


def ssdid_est(
    treated_unit_outcomes_all_periods: np.ndarray,
    donor_units_outcomes_all_periods: np.ndarray,
    donor_weights: np.ndarray,
    time_weights_vector: np.ndarray,
    num_pre_treatment_periods: int,
    post_treatment_horizon_offset: int,
) -> float:
    """Compute the Synthetic Difference-in-Differences (SSDID) treatment effect estimate.

    This function calculates the SSDID treatment effect for a specific
    post-treatment horizon `post_treatment_horizon_offset`. The estimate is derived by taking the
    difference between a post-treatment "gap" and a weighted pre-treatment "gap".

    The post-treatment gap is:
    `post_treatment_gap = treated_unit_outcomes_all_periods[num_pre_treatment_periods + post_treatment_horizon_offset] - (donor_units_outcomes_all_periods[num_pre_treatment_periods + post_treatment_horizon_offset, :] @ donor_weights)`

    The weighted pre-treatment gap is:
    `weighted_pre_treatment_gap = time_weights_vector @ (treated_unit_outcomes_all_periods[:num_pre_treatment_periods] - (donor_units_outcomes_all_periods[:num_pre_treatment_periods, :] @ donor_weights))`

    The SSDID estimate is `post_treatment_gap - weighted_pre_treatment_gap`.

    Parameters
    ----------
    treated_unit_outcomes_all_periods : np.ndarray
        Outcome vector for the treated unit across all time periods.
        Shape (T_total,), where T_total is the total number of time periods.
    donor_units_outcomes_all_periods : np.ndarray
        Matrix of outcomes for donor units across all time periods.
        Shape (T_total, J), where J is the number of donor units.
    donor_weights : np.ndarray
        Donor weights used to create the synthetic control unit.
        Shape (J,). These weights typically sum to 1.
    time_weights_vector : np.ndarray
        Time weights used to average the pre-treatment gaps.
        Shape (num_pre_treatment_periods,). These weights typically sum to 1.
    num_pre_treatment_periods : int
        Number of pre-treatment periods. This defines the length of
        `time_weights_vector` and the segment of pre-treatment data used for
        calculating `weighted_pre_treatment_gap`.
    post_treatment_horizon_offset : int
        Post-treatment horizon relative to `num_pre_treatment_periods`. The treatment effect is
        estimated for time period `num_pre_treatment_periods + post_treatment_horizon_offset` (0-indexed).

    Returns
    -------
    float
        The estimated SSDID treatment effect at horizon `post_treatment_horizon_offset`.

    Raises
    ------
    MlsynthDataError
        If input arrays have incorrect types, dimensions, or inconsistent shapes,
        or if period indices are out of bounds.

    Examples
    --------
    >>> T_total_ex, J_ex, num_pre_periods_ex, horizon_offset_ex = 20, 3, 10, 2
    >>> treated_outcomes_ex = np.random.rand(T_total_ex)
    >>> donor_outcomes_ex = np.random.rand(T_total_ex, J_ex)
    >>> donor_weights_ex = np.array([0.5, 0.3, 0.2]) # Example donor weights
    >>> time_weights_ex = np.full(num_pre_periods_ex, 1/num_pre_periods_ex)    # Example time weights (uniform)
    >>> att_k = ssdid_est(treated_outcomes_ex, donor_outcomes_ex, donor_weights_ex, time_weights_ex, num_pre_periods_ex, horizon_offset_ex)
    >>> print(f"SSDID estimate at horizon k={horizon_offset_ex}: {att_k:.3f}")
    SSDID estimate at horizon k=2: ...
    """
    # Input validation
    if not isinstance(treated_unit_outcomes_all_periods, np.ndarray) or treated_unit_outcomes_all_periods.ndim != 1:
        raise MlsynthDataError("`treated_unit_outcomes_all_periods` must be a 1D NumPy array.")
    if not isinstance(donor_units_outcomes_all_periods, np.ndarray) or donor_units_outcomes_all_periods.ndim != 2:
        raise MlsynthDataError("`donor_units_outcomes_all_periods` must be a 2D NumPy array.")
    if not isinstance(donor_weights, np.ndarray) or donor_weights.ndim != 1:
        raise MlsynthDataError("`donor_weights` must be a 1D NumPy array.")
    if not isinstance(time_weights_vector, np.ndarray) or time_weights_vector.ndim != 1:
        raise MlsynthDataError("`time_weights_vector` must be a 1D NumPy array.")

    if num_pre_treatment_periods < 0:
        raise MlsynthDataError("`num_pre_treatment_periods` cannot be negative.")
    if post_treatment_horizon_offset < 0:
        raise MlsynthDataError("`post_treatment_horizon_offset` cannot be negative.")

    total_periods_treated = len(treated_unit_outcomes_all_periods)
    total_periods_donors, num_donors = donor_units_outcomes_all_periods.shape

    if total_periods_treated != total_periods_donors:
        raise MlsynthDataError(
            f"Mismatch in total periods: treated has {total_periods_treated}, donors have {total_periods_donors}."
        )

    post_treatment_period_index = num_pre_treatment_periods + post_treatment_horizon_offset
    if not (0 <= post_treatment_period_index < total_periods_treated):
        raise MlsynthDataError(
            f"Calculated `post_treatment_period_index` ({post_treatment_period_index}) is out of bounds "
            f"for arrays of length {total_periods_treated}."
        )
    if not (0 <= num_pre_treatment_periods <= total_periods_treated):
        raise MlsynthDataError(
            f"`num_pre_treatment_periods` ({num_pre_treatment_periods}) is out of bounds "
            f"for arrays of length {total_periods_treated}."
        )

    if len(time_weights_vector) != num_pre_treatment_periods:
        raise MlsynthDataError(
            f"Length of `time_weights_vector` ({len(time_weights_vector)}) "
            f"does not match `num_pre_treatment_periods` ({num_pre_treatment_periods})."
        )
    if len(donor_weights) != num_donors:
        raise MlsynthDataError(
            f"Length of `donor_weights` ({len(donor_weights)}) "
            f"does not match number of donors ({num_donors})."
        )
    
    # Calculate the outcome of the treated unit at the specified post-treatment horizon
    treated_outcome_at_horizon = treated_unit_outcomes_all_periods[post_treatment_period_index]
    # Calculate the synthetic control outcome at the same horizon using donor weights
    weighted_donor_outcomes_at_horizon = donor_units_outcomes_all_periods[post_treatment_period_index, :] @ donor_weights
    # The "post-treatment gap" is the difference between treated and synthetic control at this horizon
    post_treatment_gap = treated_outcome_at_horizon - weighted_donor_outcomes_at_horizon

    # Extract the pre-treatment outcomes for the treated unit
    treated_outcomes_pre_treatment_segment = treated_unit_outcomes_all_periods[:num_pre_treatment_periods]
    # Calculate the synthetic control outcomes for the pre-treatment period
    weighted_donor_outcomes_pre_treatment_segment = donor_units_outcomes_all_periods[:num_pre_treatment_periods, :] @ donor_weights
    # Calculate the "pre-treatment gaps" (differences between treated and synthetic control for each pre-treatment period)
    pre_treatment_gaps = treated_outcomes_pre_treatment_segment - weighted_donor_outcomes_pre_treatment_segment
    # The "weighted pre-treatment gap" is the average of these pre-treatment gaps, weighted by time_weights_vector
    weighted_pre_treatment_gap = time_weights_vector @ pre_treatment_gaps
    
    # The SSDID estimate is the difference between the post-treatment gap and the weighted pre-treatment gap
    ssdid_estimate = post_treatment_gap - weighted_pre_treatment_gap
    return ssdid_estimate


def sc_diagplot(config_list: List[Dict[str, Any]]) -> None:
    """Generate diagnostic plots for synthetic control analyses.

    For each configuration provided, this function plots the treated unit's
    outcome trajectory against individual donor unit trajectories and the
    mean trajectory of all donor units. A vertical line indicates the
    start of the treatment period.

    Parameters
    ----------
    config_list : List[Dict[str, Any]]
        A list of configuration dictionaries. Each dictionary defines settings
        for a single plot (or a subplot if multiple configs are provided).
        Each dictionary must contain the following keys, which are typically
        passed to an internal data preparation function (like `dataprep`):

        - "df" : pandas.DataFrame
            The input panel data.
        - "unitid" : str
            The name of the column in `df` that identifies the units
            (e.g., countries, firms).
        - "time" : str
            The name of the column in `df` that identifies the time periods
            (e.g., years, quarters).
        - "outcome" : str
            The name of the column in `df` representing the outcome variable
            to be plotted.
        - "treat" : Union[str, int, List[Union[str, int]]]
            Identifier(s) for the treated unit(s). This is used by the
            internal data preparation function to distinguish treated
            units from donor units.
        - "cohort" : Optional[Union[str, int]], default None
            If the data preparation step can result in multiple cohorts
            (e.g., different treatment start times), this key specifies which
            cohort's data to use for the plot. If `None` and multiple cohorts
            are detected by `dataprep`, a `ValueError` is raised.

    Raises
    ------
    MlsynthConfigError
        If `config_list` is not a list.
        If multiple cohorts are detected in the data prepared from a config
        and the "cohort" key is not specified in that config.

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from unittest.mock import patch
    >>> # Create sample data for the plot
    >>> data_dict = {
    ...     'year': [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
    ...     'country': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    ...     'gdp': [1.0, 1.2, 1.0, 2.0, 2.1, 2.2, 1.5, 1.6, 1.7]
    ... }
    >>> df_sample = pd.DataFrame(data_dict)
    >>>
    >>> # Configuration for sc_diagplot
    >>> plot_config = [{
    ...     "df": df_sample,
    ...     "unitid": "country",
    ...     "time": "year",
    ...     "outcome": "gdp",
    ...     "treat": "A"  # Unit 'A' is treated
    ... }]
    >>>
    >>> # Mock the output of dataprep for this example
    >>> # dataprep would normally process df_sample based on the config
    >>> mock_dataprep_output = {
    ...     "y": np.array([1.0, 1.2, 1.0]),  # Treated unit 'A' outcomes
    ...     "donor_matrix": np.array([[2.0, 1.5], [2.1, 1.6], [2.2, 1.7]]), # Donors 'B', 'C'
    ...     "treated_unit_name": "A",
    ...     "pre_periods": 1,  # Indicates treatment starts after the first period (index 0)
    ...     "Ywide": pd.DataFrame(index=[2000, 2001, 2002]) # For time axis labels
    ... }
    >>>
    >>> # Patch dataprep and plt.show to run example non-interactively
    >>> with patch("mlsynth.utils.helperutils.dataprep", return_value=mock_dataprep_output), \
    ...      patch("matplotlib.pyplot.show"):
    ...     sc_diagplot(plot_config)
    >>> # This would typically display a plot showing:
    >>> # - GDP of unit 'A' (black line).
    >>> # - GDP of donor units 'B' and 'C' (gray lines).
    >>> # - Mean GDP of donors 'B' and 'C' (blue line).
    >>> # - A vertical dashed line after year 2000, indicating treatment start.

    Notes
    -----
    This function uses a local import `from .datautils import dataprep`.
    This is done to avoid potential circular import issues if `helperutils.py`
    is imported by modules that `dataprep` might depend on indirectly.
    The plot styling is controlled by a predefined `ubertheme`.
    """
    from .datautils import dataprep  # import inside parent in case of circularity

    if not isinstance(config_list, list):
        raise MlsynthConfigError("Input 'config_list' must be a list of configuration dictionaries.")

    n_plots = len(config_list)

    # Use same theme as before
    ubertheme = {
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "font.size": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "grid.alpha": 0.1,
        "grid.linewidth": 0.5,
        "grid.color": "#000000",
        "legend.fontsize": "small",
    }

    with rc_context(rc=ubertheme):
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)

        if n_plots == 1:
            axes = [axes]  # make iterable for zip

        for ax_plot, config_item_plot in zip(axes, config_list):
            # Prepare data using the provided configuration
            # This step transforms the raw DataFrame into structured arrays needed for plotting.
            # The keys in `config_item_plot` (e.g., "unitid", "time") are passed to `dataprep`.
            prepared_data_for_plot = dataprep(
                df=config_item_plot["df"],
                unit_id_column=config_item_plot["unitid"], 
                time_column_name=config_item_plot["time"],
                outcome_column_name=config_item_plot["outcome"],
                treated_unit_identifier=config_item_plot["treat"],
                # Other dataprep parameters will use their defaults or are assumed not critical for this diagnostic plot.
            )

            cohort_key_from_config = config_item_plot.get("cohort", None)

            # Determine which data to plot based on whether `dataprep` returned single or multiple cohorts.
            if "cohorts" in prepared_data_for_plot: # Multiple cohorts were found
                if cohort_key_from_config is None and len(prepared_data_for_plot["cohorts"]) > 1:
                    # If multiple cohorts exist and no specific cohort is requested, it's ambiguous.
                    raise MlsynthConfigError(
                        "Multiple cohorts found in data. Please specify a 'cohort' in the plot configuration to resolve ambiguity."
                    )
                # Determine the actual cohort identifier to use for plotting.
                # If a cohort is specified in config, use it. Otherwise, if only one cohort, use its key.
                actual_cohort_identifier = (
                    cohort_key_from_config
                    if cohort_key_from_config is not None
                    else list(prepared_data_for_plot["cohorts"].keys())[0]
                )
                plot_specific_data = prepared_data_for_plot["cohorts"][actual_cohort_identifier]
                # If multiple units are treated in this cohort, average their outcomes for the plot.
                treated_unit_trajectory = plot_specific_data["y"].mean(axis=1) 
                treated_unit_names_string = "_".join(map(str, plot_specific_data["treated_units"]))
                plot_treatment_label = (
                    f"Cohort {actual_cohort_identifier} (Treated at {actual_cohort_identifier})"
                )
                plot_title_label = f"Unit(s): {treated_unit_names_string}"
            else: # Single cohort (or no cohort structure from dataprep)
                plot_specific_data = prepared_data_for_plot
                treated_unit_trajectory = plot_specific_data["y"]
                treated_unit_names_string = str(prepared_data_for_plot["treated_unit_name"])
                plot_treatment_label = "Treated Unit"
                plot_title_label = f"Unit: {treated_unit_names_string}"

            donor_outcomes_matrix_for_plot = plot_specific_data["donor_matrix"]
            donor_mean_outcome_trajectory = donor_outcomes_matrix_for_plot.mean(axis=1)
            # Use the index from 'Ywide' (wide format outcome data from dataprep) for the time axis labels.
            # 'Ywide' is expected to be in the outer scope of `prepared_data_for_plot` if cohorts are present.
            time_period_axis_index = prepared_data_for_plot.get("Ywide", plot_specific_data.get("Ywide")).index


            # Plot individual donor trajectories (light gray, low alpha)
            for i in range(donor_outcomes_matrix_for_plot.shape[1]):
                ax_plot.plot(
                    time_period_axis_index,
                    donor_outcomes_matrix_for_plot[:, i],
                    color="gray",
                    linewidth=0.8,
                    alpha=0.3,
                    zorder=1,
                )

            # Plot donor mean trajectory
                ax_plot.plot(
                    time_period_axis_index,
                    donor_mean_outcome_trajectory,
                    label="Donor Mean", # Label for the legend
                    color="blue",
                    linewidth=2,
                    zorder=2, # Ensure donor mean is plotted above individual donors
                )
            # Plot treated unit trajectory (black, prominent)
            ax_plot.plot(
                time_period_axis_index,
                treated_unit_trajectory,
                label=plot_treatment_label, # Label for the legend
                color="black",
                linewidth=2,
                zorder=3, # Ensure treated unit is plotted on top
            )
            # Add a vertical dashed line to indicate the start of the treatment period.
            # `plot_specific_data["pre_periods"]` gives the number of pre-treatment periods,
            # so this index points to the first post-treatment (or treatment start) time point.
            ax_plot.axvline(
                x=time_period_axis_index[plot_specific_data["pre_periods"]],
                color="black",
                linestyle="--",
                linewidth=1,
            )

            ax_plot.set_title(plot_title_label, loc="left") # Set plot title
            ax_plot.set_xlabel(config_item_plot["time"]) # Set x-axis label from config
            ax_plot.set_ylabel(config_item_plot["outcome"])
            ax_plot.legend()

        fig.suptitle("Treated vs Donor Trends", fontsize=16)
        plt.tight_layout()
        plt.show()
