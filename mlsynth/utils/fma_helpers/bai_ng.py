"""Bai & Ng (2002) factor-number selection for the FMA estimator.

Relocated from the former shared ``mlsynth.utils.denoiseutils`` module: the
Bai-Ng information-criterion factor count (:func:`nbpiid`) and its column
preprocessing helpers (:func:`demean_matrix`, :func:`standardize`) are used
only by FMA's factor step, so they live inside that estimator's package.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError


def demean_matrix(input_matrix: np.ndarray) -> np.ndarray:
    """
    Demean a matrix by subtracting the mean of each column from its elements.

    This operation centers the data in each column around zero. It is a common
    preprocessing step in many statistical and machine learning algorithms.

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix to be demeaned. Shape (n_rows, n_cols).
        It can be 1D or 2D. If 1D, it's treated as a single column.

    Returns
    -------
    np.ndarray
        The demeaned matrix, with the same shape as the input `input_matrix`.
        Each column `j` of the output will have `mean(output[:, j])`
        approximately equal to zero (within floating point precision).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.fma_helpers.bai_ng import demean_matrix
    >>> # Example 1: 2D matrix
    >>> X_ex = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=float)
    >>> X_demeaned_ex = demean_matrix(X_ex)
    >>> print("Original matrix X_ex:\\n", X_ex)
    Original matrix X_ex:
     [[1. 4. 7.]
     [2. 5. 8.]
     [3. 6. 9.]]
    >>> print("Demeaned matrix X_demeaned_ex:\\n", X_demeaned_ex)
    Demeaned matrix X_demeaned_ex:
     [[-1. -1. -1.]
     [ 0.  0.  0.]
     [ 1.  1.  1.]]
    >>> print("Column means of X_demeaned_ex (should be ~0):\\n", np.mean(X_demeaned_ex, axis=0))
    Column means of X_demeaned_ex (should be ~0):
     [0. 0. 0.]

    >>> # Example 2: 1D array (treated as a column vector)
    >>> Y_ex = np.array([10, 20, 30], dtype=float)
    >>> Y_demeaned_ex = demean_matrix(Y_ex)
    >>> print("\\nOriginal 1D array Y_ex:", Y_ex)
    Original 1D array Y_ex: [10. 20. 30.]
    >>> print("Demeaned 1D array Y_demeaned_ex:", Y_demeaned_ex)
    Demeaned 1D array Y_demeaned_ex: [-10.   0.  10.]
    >>> print("Mean of Y_demeaned_ex (should be ~0):", np.mean(Y_demeaned_ex))
    Mean of Y_demeaned_ex (should be ~0): 0.0
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim == 0: # Handle scalar array
        raise MlsynthDataError("Input `input_matrix` must be at least 1D.")
    if input_matrix.size == 0: # Check for empty array (0 elements)
        raise MlsynthDataError("Input `input_matrix` cannot be empty.")
    # Check for 2D array with 0 columns, which would cause error with np.mean(axis=0) then subtraction
    if input_matrix.ndim == 2 and input_matrix.shape[1] == 0:
        raise MlsynthDataError("Input `input_matrix` has 0 columns, cannot demean.")
    
    # `np.mean(input_matrix, axis=0)` computes the mean of each column.
    # If input_matrix is 1D, it computes the mean of the array, and broadcasting handles subtraction.
    # If input_matrix is 2D, it returns a 1D array of column means, which is broadcasted
    # during subtraction to apply to each row.
    return input_matrix - np.mean(input_matrix, axis=0)


def standardize(input_matrix: np.ndarray) -> np.ndarray:
    """
    Standardize a matrix by subtracting the mean and dividing by the standard
    deviation of each column.

    This operation transforms each column of the input matrix to have a mean
    of approximately zero and a standard deviation of approximately one.
    It is a common preprocessing step for many statistical and machine
    learning algorithms to ensure that all features (columns) contribute
    equally to the analysis, regardless of their original scale.

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix to be standardized. Shape (n_rows, n_cols).
        It can be 1D or 2D. If 1D, it's treated as a single column.

    Returns
    -------
    np.ndarray
        The standardized matrix, with the same shape as the input `input_matrix`.
        Each column `j` of the output will have `mean(output[:, j])`
        approximately equal to zero and `std(output[:, j])` approximately
        equal to one (within floating point precision), provided the original
        column standard deviation was not zero.

    Raises
    ------
    MlsynthDataError
        If `input_matrix` is not a NumPy array.
        If any column has a standard deviation of zero, to prevent division by zero.

    Notes
    -----
    If a column has a standard deviation of zero (i.e., all elements in that
    column are the same), this function will raise an MlsynthDataError.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.fma_helpers.bai_ng import standardize
    >>> # Example 1: 2D matrix
    >>> X_ex = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]], dtype=float)
    >>> X_standardized_ex = standardize(X_ex)
    >>> print("Original matrix X_ex:\\n", X_ex)
    Original matrix X_ex:
     [[  1.  10. 100.]
     [  2.  20. 200.]
     [  3.  30. 300.]]
    >>> print("Standardized matrix X_standardized_ex:\\n", X_standardized_ex)
    Standardized matrix X_standardized_ex:
     [[-1.22474487 -1.22474487 -1.22474487]
     [ 0.          0.          0.        ]
     [ 1.22474487  1.22474487  1.22474487]]
    >>> print("Column means of X_standardized_ex (should be ~0):\\n",
    ...       np.mean(X_standardized_ex, axis=0))
    Column means of X_standardized_ex (should be ~0):
     [0. 0. 0.]
    >>> print("Column stds of X_standardized_ex (should be ~1):\\n",
    ...       np.std(X_standardized_ex, axis=0))
    Column stds of X_standardized_ex (should be ~1):
     [1. 1. 1.]

    >>> # Example 2: 1D array (treated as a column vector)
    >>> Y_ex = np.array([5, 10, 15, 20, 25], dtype=float)
    >>> Y_standardized_ex = standardize(Y_ex)
    >>> print("\\nOriginal 1D array Y_ex:", Y_ex)
    Original 1D array Y_ex: [ 5. 10. 15. 20. 25.]
    >>> print("Standardized 1D array Y_standardized_ex:", Y_standardized_ex)
    Standardized 1D array Y_standardized_ex: [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
    >>> print("Mean of Y_standardized_ex (should be ~0):", np.mean(Y_standardized_ex))
    Mean of Y_standardized_ex (should be ~0): 0.0
    >>> print("Std of Y_standardized_ex (should be ~1):", np.std(Y_standardized_ex))
    Std of Y_standardized_ex (should be ~1): 1.0
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")

    # Calculate mean and standard deviation for each column (axis=0).
    column_means = np.mean(input_matrix, axis=0)
    column_stds = np.std(input_matrix, axis=0)

    # Check for zero standard deviation to prevent division by zero.
    # If input_matrix is 1D, column_stds will be a scalar.
    if np.isscalar(column_stds):
        if column_stds == 0:
            raise MlsynthDataError(
                "Cannot standardize array with zero standard deviation."
            )
    elif np.any(column_stds == 0): # If input_matrix is 2D, check each column's std.
        zero_std_cols = np.where(column_stds == 0)[0] # Find columns with zero std.
        raise MlsynthDataError(
            f"Cannot standardize columns with zero standard deviation: columns {list(zero_std_cols)}"
        )
    
    # Standardize: (X - mean(X_col)) / std(X_col) for each column.
    # Broadcasting handles the operations correctly for 1D and 2D inputs.
    return (input_matrix - column_means) / column_stds


def nbpiid(
    input_panel_data: np.ndarray,
    max_factors_to_test: int,
    criterion_selector_code: int,
    preprocessing_method_code: int,
    N_series_adjustment: Optional[int] = None,
    T_obs_adjustment: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Determine the number of factors in a panel dataset using Bai and Ng (2002)
    information criteria.

    This function implements several information criteria proposed by Bai and Ng (2002)
    to estimate the optimal number of common factors in a panel dataset `input_panel_data`.
    It preprocesses the data (demeaning or standardizing if specified),
    then for each possible number of factors from 1 to `max_factors_to_test`, it estimates
    the factors and computes the sum of squared residuals. Based on these,
    it calculates the specified information criterion and selects the number
    of factors that minimizes it.

    Parameters
    ----------
    input_panel_data : np.ndarray
        Input data matrix, typically with observations in rows and series
        (variables) in columns. Shape (num_time_periods, num_series).
    max_factors_to_test : int
        Maximum number of factors to consider. Must be less than or equal to
        min(num_time_periods, num_series).
    criterion_selector_code : int
        Specifies the type of information criterion to use. The penalties are:
        - 1: `log(NT/NT1) * k * NT1/NT` (IC_p1 like)
        - 2: `(NT1/NT) * log(min(N,T)) * k` (IC_p2 like)
        - 3: `k * log(min(N,T))/min(N,T)` (IC_p3 like)
        - 4: `2*k/T` (AIC like)
        - 5: `log(T)*k/T` (BIC/SIC like)
        - 6: `2*k*NT1/NT`
        - 7: `log(NT)*k*NT1/NT`
        - 10: Modified version of jj=1, `((N+N_series_adjustment)*(T+T_obs_adjustment)/NT) * log(NT/NT1) * k * NT1/NT`.
              Requires `N_series_adjustment` and `T_obs_adjustment`.
        - 11: Modified version of jj=1, `(T / (4*log(log(T)))) * log(NT/NT1) * k * NT1/NT`.
        Where `k` is the number of factors, `N` is `num_series`, `T` is `num_time_periods`,
        `NT = N*T`, `NT1 = N+T`.
    preprocessing_method_code : int
        Flag for data preprocessing before factor estimation:
        - 0: No preprocessing.
        - 1: Demean columns (subtract column means).
        - 2: Standardize columns (subtract column means and divide by column standard deviations).
    N_series_adjustment : Optional[int], default None
        A parameter required only if `criterion_selector_code=10`. Represents an adjustment factor for num_series.
    T_obs_adjustment : Optional[int], default None
        A parameter required only if `criterion_selector_code=10`. Represents an adjustment factor for num_time_periods.

    Returns
    -------
    selected_number_of_factors : int
        The selected number of factors that minimizes the chosen information criterion.
    estimated_common_component_selected_k : np.ndarray
        The estimated common component matrix, `Fhat @ lambda_hat.T`, using the
        `selected_number_of_factors` number of factors. Shape (num_time_periods, num_series).
    estimated_factors_selected_k : np.ndarray
        The estimated factors (principal components). Shape (num_time_periods, selected_number_of_factors).

    Raises
    ------
    MlsynthDataError
        If input data `input_panel_data` is not a 2D NumPy array, is empty,
        or has dimensions incompatible with `max_factors_to_test`.
    MlsynthConfigError
        If configuration parameters (`max_factors_to_test`, `criterion_selector_code`,
        `preprocessing_method_code`, `N_series_adjustment`, `T_obs_adjustment`)
        are invalid or inconsistent.
    MlsynthEstimationError
        If numerical issues arise during estimation (e.g., SVD/eigen decomposition failure,
        issues with log/division by zero due to data properties).

    References
    ----------
    .. [1] Bai, J., & Ng, S. (2002). Determining the number of factors in approximate
           factor models. Econometrica, 70(1), 191-221.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.fma_helpers.bai_ng import nbpiid
    >>> # Generate synthetic data with 2 factors
    >>> T_ex, N_ex = 100, 50
    >>> F_true_ex = np.random.randn(T_ex, 2)
    >>> Lambda_true_ex = np.random.randn(2, N_ex)
    >>> E_true_ex = np.random.randn(T_ex, N_ex) * 0.5 # Noise
    >>> X_data_ex = F_true_ex @ Lambda_true_ex + E_true_ex
    >>> k_max_factors_ex = 5
    >>> criterion_code_ex = 2 # Use IC_p2 like criterion
    >>> preprocess_code_ex = 1 # Demean the data
    >>> num_factors_res, common_comp_res, factors_est_res = nbpiid(
    ...     input_panel_data=X_data_ex,
    ...     max_factors_to_test=k_max_factors_ex,
    ...     criterion_selector_code=criterion_code_ex,
    ...     preprocessing_method_code=preprocess_code_ex
    ... )
    >>> print(f"Selected number of factors: {num_factors_res}") # doctest: +SKIP
    # Expected: Selected number of factors: 2 (or close, depending on noise)
    >>> print("Shape of common component:", common_comp_res.shape)
    Shape of common component: (100, 50)
    >>> print("Shape of estimated factors:", factors_est_res.shape) # doctest: +SKIP
    # Expected: Shape of estimated factors: (100, 2) (if num_factors_res is 2)
    """
    # Input Validation
    if not isinstance(input_panel_data, np.ndarray):
        raise MlsynthDataError("Input `input_panel_data` must be a NumPy array.")
    if input_panel_data.ndim != 2:
        raise MlsynthDataError("Input `input_panel_data` must be a 2D array.")
    if input_panel_data.size == 0:
        raise MlsynthDataError("Input `input_panel_data` cannot be empty.")

    num_time_periods, num_series = input_panel_data.shape

    if not isinstance(max_factors_to_test, int):
        raise MlsynthConfigError("`max_factors_to_test` must be an integer.")
    if max_factors_to_test <= 0:
        raise MlsynthConfigError("`max_factors_to_test` must be positive.")
    if max_factors_to_test > min(num_time_periods, num_series):
        raise MlsynthConfigError(
            f"`max_factors_to_test` ({max_factors_to_test}) cannot exceed "
            f"min(num_time_periods, num_series) which is {min(num_time_periods, num_series)}."
        )

    if not isinstance(criterion_selector_code, int):
        raise MlsynthConfigError("`criterion_selector_code` must be an integer.")
    supported_criteria = [1, 2, 3, 4, 5, 6, 7, 10, 11]
    if criterion_selector_code not in supported_criteria:
        raise MlsynthConfigError(
            f"`criterion_selector_code` must be one of {supported_criteria}, "
            f"got {criterion_selector_code}."
        )

    if not isinstance(preprocessing_method_code, int):
        raise MlsynthConfigError("`preprocessing_method_code` must be an integer.")
    supported_preprocessing = [0, 1, 2]
    if preprocessing_method_code not in supported_preprocessing:
        raise MlsynthConfigError(
            f"`preprocessing_method_code` must be one of {supported_preprocessing}, "
            f"got {preprocessing_method_code}."
        )

    if criterion_selector_code == 10:
        if N_series_adjustment is None or not isinstance(N_series_adjustment, int):
            raise MlsynthConfigError(
                "`N_series_adjustment` must be an integer if criterion_selector_code is 10."
            )
        if T_obs_adjustment is None or not isinstance(T_obs_adjustment, int):
            raise MlsynthConfigError(
                "`T_obs_adjustment` must be an integer if criterion_selector_code is 10."
            )
    # End Input Validation

    # Define constants used in penalty calculations.
    # N = num_series, T = num_time_periods
    total_elements_N_times_T: int = num_series * num_time_periods
    sum_N_plus_T: int = num_series + num_time_periods
    penalty_values_array: np.ndarray = np.zeros(max_factors_to_test) # Stores penalty g(N,T,k) for k=1..k_max
    factor_count_iterator: np.ndarray = np.arange(1, max_factors_to_test + 1) # k values from 1 to k_max

    # Pre-calculate terms for penalties to avoid redundant computations and validate inputs.
    if num_time_periods <= 0 or num_series <= 0: # Should be caught by .size check earlier
        raise MlsynthDataError("Number of time periods and series must be positive.")
    if sum_N_plus_T == 0 or total_elements_N_times_T == 0 : # Should not happen if N,T > 0
        raise MlsynthEstimationError("Sum or product of N and T is zero, leading to division by zero in penalty.")

    ratio_nt_nt1 = total_elements_N_times_T / sum_N_plus_T # (N*T)/(N+T)
    if ratio_nt_nt1 <= 0: # This implies N*T and N+T have different signs or one is zero, which is not possible if N,T > 0.
        raise MlsynthEstimationError(f"Ratio NT/NT1 ({ratio_nt_nt1}) must be positive for log calculation.")
    log_ratio_nt_nt1 = np.log(ratio_nt_nt1) # log((N*T)/(N+T))

    min_N_T = min(num_series, num_time_periods) # min(N,T)
    # min_N_T > 0 is guaranteed because max_factors_to_test <= min(N,T) and max_factors_to_test > 0.
    log_min_N_T = np.log(min_N_T) # log(min(N,T))
    log_T = np.log(num_time_periods) # log(T) (T > 0 guaranteed)
    log_NT = np.log(total_elements_N_times_T) # log(N*T) (N*T > 0 guaranteed)

    # Specific validation for criterion 11 due to log(log(T)).
    if criterion_selector_code == 11:
        if num_time_periods <= 1: # log(T) would be <=0 or undefined.
            raise MlsynthEstimationError("T must be > 1 for criterion 11 due to log(T) term.")
        log_T_val_for_loglog = np.log(num_time_periods) # This is log(T)
        if log_T_val_for_loglog <= 1: # If log(T) <= 1 (i.e., T <= e), then log(log(T)) is <=0 or undefined.
             raise MlsynthEstimationError("log(T) must be > 1 for criterion 11 (i.e., T > e approx 2.718).")
        if np.log(log_T_val_for_loglog) == 0: # Avoid division by zero if log(log(T)) is zero.
            raise MlsynthEstimationError("log(log(T)) is zero, leading to division by zero in criterion 11.")

    # Calculate penalty terms g(N,T,k) based on the chosen criterion.
    # These penalties are functions of N, T, and k (number of factors).
    if criterion_selector_code == 1: # IC_p1 like
        penalty_values_array = (log_ratio_nt_nt1 * factor_count_iterator * sum_N_plus_T / total_elements_N_times_T)
    elif criterion_selector_code == 10: # Modified IC_p1
        penalty_values_array = (
            ((num_series + N_series_adjustment) * (num_time_periods + T_obs_adjustment) / total_elements_N_times_T) # type: ignore
            * log_ratio_nt_nt1 * factor_count_iterator * sum_N_plus_T / total_elements_N_times_T
        )
    elif criterion_selector_code == 11: # Another modified IC_p1
        penalty_values_array = (
            (num_time_periods / (4 * np.log(np.log(num_time_periods)))) # T / (4*log(log(T)))
            * log_ratio_nt_nt1 * factor_count_iterator * sum_N_plus_T / total_elements_N_times_T
        )
    elif criterion_selector_code == 2: # IC_p2 like
        penalty_values_array = ((sum_N_plus_T / total_elements_N_times_T) * log_min_N_T * factor_count_iterator)
    elif criterion_selector_code == 3: # IC_p3 like
        penalty_values_array = (factor_count_iterator * log_min_N_T / min_N_T)
    elif criterion_selector_code == 4: # AIC like
        penalty_values_array = (2 * factor_count_iterator / num_time_periods)
    elif criterion_selector_code == 5: # BIC/SIC like
        penalty_values_array = (log_T * factor_count_iterator / num_time_periods)
    elif criterion_selector_code == 6: # Another Bai & Ng penalty
        penalty_values_array = (2 * factor_count_iterator * sum_N_plus_T / total_elements_N_times_T)
    elif criterion_selector_code == 7: # Another Bai & Ng penalty
        penalty_values_array = (log_NT * factor_count_iterator * sum_N_plus_T / total_elements_N_times_T)

    # Preprocess the input data (demean or standardize if specified).
    preprocessed_data: np.ndarray
    try:
        if preprocessing_method_code == 2: # Standardize
            preprocessed_data = standardize(input_panel_data)
        elif preprocessing_method_code == 1: # Demean
            preprocessed_data = demean_matrix(input_panel_data)
        else:  # No preprocessing
            preprocessed_data = input_panel_data.copy() # Work on a copy
    except MlsynthDataError as e: # standardize/demean_matrix can raise MlsynthDataError
        raise MlsynthEstimationError(f"Data preprocessing failed: {e}")

    # Calculate sum of squared residuals V(k) for different numbers of factors k.
    # V(k) = sum_i ( (X_i - F_k * Lambda_k_i)^2 ) / (N*T)
    # Here, it's calculated as mean over series of (sum of squared residuals per series / T)
    residual_variance_per_k_factors: np.ndarray = np.zeros(max_factors_to_test + 1) # Index 0 for k=0, up to k_max

    # Factors are estimated using PCA on X*X.T (if T < N) or X.T*X (if N < T).
    # Here, it seems to use PCA on X*X.T, so factors F are eigenvectors.
    try:
        # Eigen decomposition of X @ X.T (where X is preprocessed_data)
        # If X = F @ Lambda.T + E, then X @ X.T approx F @ (Lambda.T @ Lambda) @ F.T
        # Eigenvectors of X @ X.T are F (up to rotation).
        data_product_for_eigen_decomposition: np.ndarray = np.dot(preprocessed_data, preprocessed_data.T) # (T x T) matrix
        eigenvalues_from_decomposition, all_candidate_factors_unscaled = np.linalg.eigh(data_product_for_eigen_decomposition)
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"Eigen decomposition failed: {e}")

    # Sort factors by largest eigenvalues (eigenvectors are columns, sort them).
    all_candidate_factors_unscaled = all_candidate_factors_unscaled[:, ::-1] # F_hat (T x T), columns are factors

    # Calculate V(k) for k = 1 to max_factors_to_test.
    for k_val in factor_count_iterator: # k_val from 1 to max_factors_to_test
        factors_for_current_k: np.ndarray = all_candidate_factors_unscaled[:, :k_val] # F_k (T x k)
        # Loadings Lambda_k = F_k.T @ X (k x N)
        loadings_for_current_k: np.ndarray = np.dot(factors_for_current_k.T, preprocessed_data)
        # Common component C_k = F_k @ Lambda_k (T x N)
        common_component_for_current_k: np.ndarray = np.dot(factors_for_current_k, loadings_for_current_k)
        residuals_for_current_k: np.ndarray = preprocessed_data - common_component_for_current_k # E_k
        # V(k) = sum_{i=1 to N} sum_{t=1 to T} (E_kit)^2 / (N*T)
        # The code calculates it as: mean_over_series( sum_over_time(E_kit^2)/T )
        # This is equivalent to sum_overall_squared_residuals / (N*T) if T is consistent.
        residual_variance_per_k_factors[k_val] = np.mean( # Mean over N series
            np.sum(residuals_for_current_k ** 2 / num_time_periods, axis=0) # Sum over T, scaled by 1/T, for each series
        )
    
    # Calculate V(0) (sum of squared residuals if k=0, i.e., common component is zero).
    # V(0) = sum_{i=1 to N} sum_{t=1 to T} (X_it)^2 / (N*T)
    residual_variance_per_k_factors[0] = np.mean(
        np.sum(preprocessed_data ** 2 / num_time_periods, axis=0)
    )
    
    # Calculate Information Criteria IC(k) = log(V(k)) + penalty(k).
    # Bai & Ng (2002) use V(k) directly, not log(V(k)), for some criteria.
    # The original MATLAB code seems to use: IC(k) = V(k) + penalty(k) * sigma_hat_sq_kmax
    # where sigma_hat_sq_kmax is V(k_max).
    ic_values = np.zeros(max_factors_to_test + 1) # For k=0 to k_max
    sigma_kmax_sq = residual_variance_per_k_factors[max_factors_to_test] # V(k_max)
    if sigma_kmax_sq < 0: sigma_kmax_sq = 0 # Ensure non-negative (should be from sum of squares)

    ic_values[0] = residual_variance_per_k_factors[0] # Penalty for k=0 is effectively 0.
    for k_idx, k_val in enumerate(factor_count_iterator): # k_val from 1 to k_max; k_idx from 0 to k_max-1
        # IC(k) = V(k) + g(N,T,k) * V(k_max)
        ic_values[k_val] = residual_variance_per_k_factors[k_val] + penalty_values_array[k_idx] * sigma_kmax_sq
    
    # Select number of factors k_hat that minimizes IC(k).
    selected_number_of_factors = int(np.argmin(ic_values)) # k_hat (can be 0)

    # Get the estimated factors and common component for the selected k_hat.
    if selected_number_of_factors == 0:
        # If k_hat=0, factors are empty, common component is zero matrix.
        estimated_factors_selected_k = np.zeros((num_time_periods, 0)) # (T x 0)
        estimated_common_component_selected_k = np.zeros_like(preprocessed_data)
    else:
        estimated_factors_selected_k = all_candidate_factors_unscaled[:, :selected_number_of_factors]
        estimated_loadings_selected_k = np.dot(estimated_factors_selected_k.T, preprocessed_data)
        estimated_common_component_selected_k = np.dot(estimated_factors_selected_k, estimated_loadings_selected_k)

    return (
        selected_number_of_factors, # k_hat
        estimated_common_component_selected_k,
        estimated_factors_selected_k,
    )
