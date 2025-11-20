import numpy as np
from scipy.stats import norm
from mlsynth.utils.resultutils import effects
from typing import Any, Tuple, List, Optional, Dict, Union
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError, MlsynthConfigError


def universal_rank(singular_values: np.ndarray, matrix_aspect_ratio: float) -> int:
    """
    Calculate the universal rank based on singular values and matrix aspect ratio.

    Parameters
    ----------
    singular_values : np.ndarray
        A 1D array of singular values, typically sorted in descending order.
    matrix_aspect_ratio : float
        The aspect ratio of the original matrix from which `singular_values`
        were derived, calculated as min(n_rows, n_cols) / max(n_rows, n_cols).
        This ratio should be between 0 and 1.

    Returns
    -------
    int
        The estimated rank of the matrix, determined by thresholding the
        singular values. The threshold is a multiple of the median singular
        value, where the multiple depends on the aspect ratio.
        The rank is at least 1.

    Notes
    -----
    The formula for `omega_factor` is based on empirical results or theoretical
    considerations for optimal singular value thresholding, often attributed
    to Gavish and Donoho (2014) or similar works.
    """
    if not isinstance(singular_values, np.ndarray):
        raise MlsynthDataError("Input `singular_values` must be a NumPy array.")
    if singular_values.ndim != 1:
        raise MlsynthDataError("Input `singular_values` must be a 1D array.")
    if singular_values.size == 0:
        raise MlsynthDataError("Input `singular_values` cannot be empty.")

    if not isinstance(matrix_aspect_ratio, (float, int)): # Allow int for ratio 1 or 0
        raise MlsynthDataError("Input `matrix_aspect_ratio` must be a float.")
    if not (0 <= matrix_aspect_ratio <= 1):
        raise MlsynthDataError(
            "Input `matrix_aspect_ratio` must be between 0 and 1."
        )

    # The omega_factor is a correction factor based on the matrix aspect ratio,
    # used to determine an optimal threshold for singular value truncation.
    # This specific polynomial form is often derived from empirical studies or
    # theoretical results in random matrix theory (e.g., Gavish and Donoho, 2014).
    omega_factor: float = (
        0.56 * matrix_aspect_ratio ** 3
        - 0.95 * matrix_aspect_ratio ** 2
        + 1.43
        + 1.82 * matrix_aspect_ratio
    )
    try:
        median_singular_value = np.median(singular_values)
    except Exception as e: # Catches errors from np.median on problematic inputs, though checks above should prevent most
        raise MlsynthDataError(f"Could not compute median of singular_values: {e}")

    # The threshold is set as omega_factor times the median singular value.
    # Singular values below this threshold are considered noise.
    threshold_value: float = omega_factor * median_singular_value
    # The estimated rank is the number of singular values greater than the threshold.
    # Ensure rank is at least 1, unless singular_values is empty (caught earlier).
    estimated_rank: int = max(len(singular_values[singular_values > threshold_value]), 1)
    return estimated_rank


def spectral_rank(singular_values: np.ndarray, energy_threshold: float = 0.95) -> int:
    """
    Calculate rank based on spectral energy.

    Parameters
    ----------
    singular_values : np.ndarray
        A 1D array of singular values, typically sorted in descending order.
    energy_threshold : float, default 0.95
        The desired proportion of cumulative spectral energy to retain.
        The rank is chosen such that the sum of squares of the top `estimated_rank`
        singular values accounts for at least this proportion of the total
        sum of squares of all singular values. If `energy_threshold` is 1.0,
        the rank will be the total number of singular values.

    Returns
    -------
    int
        The estimated rank, i.e., the smallest number of singular values
        whose squared sum meets or exceeds the `energy_threshold` of the total
        spectral energy.
    """
    if not isinstance(singular_values, np.ndarray):
        raise MlsynthDataError("Input `singular_values` must be a NumPy array.")
    if singular_values.ndim != 1:
        raise MlsynthDataError("Input `singular_values` must be a 1D array.")
    if singular_values.size == 0:
        raise MlsynthDataError("Input `singular_values` cannot be empty.")

    if not isinstance(energy_threshold, float):
        raise MlsynthDataError("Input `energy_threshold` must be a float.")
    if not (0 <= energy_threshold <= 1):
        raise MlsynthDataError(
            "Input `energy_threshold` must be between 0 and 1 (inclusive)."
        )

    if energy_threshold == 1.0:
        # If 100% energy is required, the rank is the total number of singular values.
        estimated_rank: int = len(singular_values)
    else:
        # Spectral energy is proportional to the square of singular values.
        squared_singular_values = singular_values ** 2
        sum_squared_singular_values = squared_singular_values.sum()

        if sum_squared_singular_values == 0:
            # If total spectral energy is zero (all singular values are zero).
            if energy_threshold > 0:
                # If any energy is required but none exists, rank cannot be determined meaningfully.
                 raise MlsynthDataError(
                    "Cannot compute spectral rank: all singular values are zero and energy_threshold > 0."
                )
            return 0 # If threshold is 0 and all SVs are 0, rank is 0.

        # Calculate the cumulative sum of squared singular values, normalized by the total sum.
        # This gives the proportion of total energy captured by the top k singular values.
        cumulative_spectral_energy_ratio: np.ndarray = (
            squared_singular_values.cumsum() / sum_squared_singular_values
        )
        try:
            # Find the first index (0-based) where the cumulative energy ratio
            # meets or exceeds the specified energy_threshold.
            # Add 1 to convert 0-based index to rank (number of singular values).
            estimated_rank = (
                np.where(cumulative_spectral_energy_ratio >= energy_threshold)[0][0] + 1
            )
        except IndexError:
            # This IndexError occurs if no element in cumulative_spectral_energy_ratio
            # is >= energy_threshold. This can happen if energy_threshold is very close to 1.0
            # and floating point inaccuracies prevent exact match, or if all SVs are zero (handled above).
            # If the threshold is effectively 1.0 or less than the max achievable ratio,
            # it implies all singular values are needed to meet the (near) 100% energy.
            if np.isclose(energy_threshold, 1.0) or energy_threshold < np.max(cumulative_spectral_energy_ratio):
                estimated_rank = len(singular_values)
            else:
                # This case should be rare if energy_threshold <= 1.
                raise MlsynthEstimationError(
                    f"Could not determine rank for energy_threshold={energy_threshold}. "
                    f"Max cumulative energy ratio: {np.max(cumulative_spectral_energy_ratio):.4f}"
                )
    return estimated_rank


def svt( # Singular Value Truncation (hard thresholding)
    input_matrix: np.ndarray,
    fixed_rank: Optional[int] = None,
    spectral_energy_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """Perform Singular Value Truncation (hard thresholding) on a matrix.

    This function computes the Singular Value Decomposition (SVD) of the input
    matrix `input_matrix` and then reconstructs a low-rank approximation by
    keeping only a specified number of singular values (and corresponding
    singular vectors).
    The rank for truncation is determined based on the provided parameters:
    1. If `fixed_rank` is given, it is used directly.
    2. Else, if `spectral_energy_threshold` is given, `spectral_rank` is used.
    3. Otherwise, `universal_rank` (based on matrix aspect ratio and median
       singular value) is used by default.

    Parameters
    ----------
    input_matrix : np.ndarray
        Input matrix to be approximated. Shape (num_rows, num_cols).
    fixed_rank : Optional[int], default None
        If provided, the rank of the approximation is set to this value.
    spectral_energy_threshold : Optional[float], default None
        If `fixed_rank` is None and `spectral_energy_threshold` is provided,
        this threshold is used with `spectral_rank` to determine the rank
        based on cumulative spectral energy.

    Returns
    -------
    low_rank_approximation : np.ndarray
        The low-rank approximation of the input matrix `input_matrix`.
        Shape (num_rows, num_cols).
    num_cols_processed : int
        The second dimension (number of columns) of the input matrix `input_matrix`.
        This is returned for convenience but is simply `input_matrix.shape[1]`.
    truncated_left_singular_vectors_out : np.ndarray
        The truncated left singular vectors (U_k). Shape (num_rows, truncation_rank).
    truncated_singular_values_out : np.ndarray
        The truncated singular values (as a 1D array, not diagonal matrix).
        Shape (truncation_rank,).
    truncated_right_singular_vectors_transposed_out : np.ndarray
        The truncated right singular vectors (V_k^T, already transposed).
        Shape (truncation_rank, num_cols).
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim != 2:
        raise MlsynthDataError("Input `input_matrix` must be a 2D array.")

    if fixed_rank is not None:
        if not isinstance(fixed_rank, int):
            raise MlsynthConfigError("Input `fixed_rank` must be an integer.")
        if fixed_rank <= 0:
            raise MlsynthConfigError("Input `fixed_rank` must be positive.")
    
    if spectral_energy_threshold is not None:
        if not isinstance(spectral_energy_threshold, float):
            raise MlsynthConfigError("Input `spectral_energy_threshold` must be a float.")
        if not (0 <= spectral_energy_threshold <= 1):
            raise MlsynthConfigError(
                "Input `spectral_energy_threshold` must be between 0 and 1."
            )

    try:
        (num_rows, num_cols) = input_matrix.shape
        (
            left_singular_vectors,
            singular_values_all,
            right_singular_vectors_transposed,
        ) = np.linalg.svd(input_matrix, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"SVD computation failed: {e}")

    matrix_aspect_ratio: float = min(num_rows, num_cols) / max(num_rows, num_cols)

    truncation_rank: int
    if fixed_rank is not None:
        if fixed_rank > len(singular_values_all):
            raise MlsynthConfigError(
                f"fixed_rank ({fixed_rank}) cannot be greater than the number of singular values ({len(singular_values_all)})."
            )
        truncation_rank = fixed_rank
    elif spectral_energy_threshold is not None:
        truncation_rank = spectral_rank(
            singular_values_all, energy_threshold=spectral_energy_threshold
        )
    else:
        truncation_rank = universal_rank(
            singular_values_all, matrix_aspect_ratio=matrix_aspect_ratio
        )
    
    # Ensure truncation_rank is not greater than available singular values
    truncation_rank = min(truncation_rank, len(singular_values_all))
    # Ensure truncation_rank is at least 0 (it could be 0 if all singular values are zero and fixed_rank is not set)
    truncation_rank = max(truncation_rank, 0)


    # Truncate SVD components to the determined truncation_rank.
    # Keep the top `truncation_rank` singular values.
    truncated_singular_values: np.ndarray = singular_values_all[:truncation_rank]
    # Keep the corresponding `truncation_rank` left singular vectors (columns of U).
    truncated_left_singular_vectors: np.ndarray = left_singular_vectors[
        :, :truncation_rank
    ]
    # Keep the corresponding `truncation_rank` right singular vectors (rows of V^T).
    truncated_right_singular_vectors_transposed: np.ndarray = (
        right_singular_vectors_transposed[:truncation_rank, :]
    )

    # Reconstruct the low-rank approximation of the input matrix using the truncated components:
    # L = U_k * Sigma_k * V_k^T
    low_rank_approximation: np.ndarray = np.dot(
        truncated_left_singular_vectors, # U_k
        np.dot(
            np.diag(truncated_singular_values), # Sigma_k (diagonal matrix)
            truncated_right_singular_vectors_transposed, # V_k^T
        ),
    )

    # The following are projection matrices onto the column space (Hu) and row space (Hv)
    # of the low-rank approximation, and their orthogonal complements (Hu_perp, Hv_perp).
    # These are not returned by this function but are common concepts related to SVD.
    # Hu = truncated_left_singular_vectors @ truncated_left_singular_vectors.T
    # Hv = truncated_right_singular_vectors_transposed.T @ truncated_right_singular_vectors_transposed
    # Hu_perp = np.eye(num_rows) - Hu
    # Hv_perp = np.eye(num_cols) - Hv

    return (
        low_rank_approximation,
        num_cols,
        truncated_left_singular_vectors,
        truncated_singular_values,
        truncated_right_singular_vectors_transposed,
    )


def shrink(input_array: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to array elements.

    This is the 'soft impute' operation, often applied to singular values.

    Parameters
    ----------
    input_array : np.ndarray
        Input array or matrix to which soft thresholding will be applied
        element-wise.
    threshold : float
        The threshold value for shrinkage. Must be non-negative.

    Returns
    -------
    np.ndarray
        The shrunken array or matrix, with the same shape as `input_array`.
        The operation is :math:`\text{sign}(X) \cdot \max(|X| - \text{threshold}, 0)`.
    """
    if not isinstance(input_array, np.ndarray):
        raise MlsynthDataError("Input `input_array` must be a NumPy array.")
    if not isinstance(threshold, (float, int)):
        raise MlsynthConfigError("Input `threshold` must be a float or integer.")
    if threshold < 0:
        raise MlsynthConfigError("Input `threshold` must be non-negative.")

    # Soft thresholding operation:
    # 1. Subtract threshold from the absolute value of each element.
    # 2. Set any resulting negative values to zero (np.maximum(..., 0)).
    # 3. Restore the original sign.
    shrunken_abs_values: np.ndarray = np.abs(input_array) - threshold
    return np.sign(input_array) * np.maximum(
        shrunken_abs_values, np.zeros_like(shrunken_abs_values) # Ensures non-negativity before sign restoration
    )


def SVT(input_matrix: np.ndarray, threshold_value: float) -> np.ndarray: # Singular Value Thresholding (soft)
    """
    Perform Singular Value Thresholding matrix operation.

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix to be processed. Shape (n_rows, n_cols).
    threshold_value : float
        The threshold value used for soft thresholding the singular values.
        This value is passed to the `shrink` function.

    Returns
    -------
    np.ndarray
        The matrix reconstructed after performing SVD, applying soft
        thresholding to its singular values with `threshold_value`, and then
        multiplying the SVD components back. Shape (n_rows, n_cols).

    See Also
    --------
    shrink : The soft thresholding function applied to singular values.
    numpy.linalg.svd : The SVD implementation used.
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim != 2:
        raise MlsynthDataError("Input `input_matrix` must be a 2D array.")

    # threshold_value validation is handled by the `shrink` function.

    try:
        left_singular_vectors, singular_values, right_singular_vectors_transposed = (
            np.linalg.svd(input_matrix, full_matrices=False)
        )
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"SVD computation failed in SVT: {e}")
    
    # Reconstruct the matrix using the original singular vectors (U, V^T)
    # but with the singular values (Sigma) replaced by their soft-thresholded versions.
    # L_reconstructed = U * shrink(Sigma, threshold) * V^T
    reconstructed_matrix: np.ndarray = (
        left_singular_vectors # U
        @ np.diag(shrink(singular_values, threshold_value)) # shrink(Sigma, threshold) as diagonal matrix
        @ right_singular_vectors_transposed # V^T
    )
    return reconstructed_matrix


def RPCA(input_matrix: np.ndarray) -> np.ndarray: # Robust Principal Component Analysis
    r"""
    Robust Principal Component Analysis (RPCA).

    Decomposes the input matrix `input_matrix` into a low-rank matrix L
    and a sparse matrix S. This function returns the low-rank component L.

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix to decompose. Shape (num_rows, num_cols).

    Returns
    -------
    np.ndarray
        The estimated low-rank component (L) of the input matrix `input_matrix`.
        Shape (num_rows, num_cols).

    Notes
    -----
    This function implements the Principal Component Pursuit (PCP) algorithm
    via Alternating Direction Method of Multipliers (ADMM) to solve the RPCA problem:
    :math:`\min_{L,S} ||L||_* + \lambda ||S||_1` subject to :math:`X = L + S`.
    The algorithm iteratively updates the low-rank matrix `L` using Singular
    Value Thresholding (`SVT`) and the sparse matrix `S` using soft
    thresholding (`shrink`).
    The parameters `penalty_parameter_mu` and `sparsity_penalty_lambda` are
    set based on common heuristics.
    The iteration stops when the reconstruction error `||X - L - S||_F` is
    below a threshold or a maximum number of iterations (1000) is reached.
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim != 2:
        raise MlsynthDataError("Input `input_matrix` must be a 2D array.")

    num_rows, num_cols = input_matrix.shape
    sum_abs_input_matrix = np.sum(np.abs(input_matrix))

    if sum_abs_input_matrix == 0: # Handle all-zero matrix
        return np.zeros_like(input_matrix)

    # `mu` is a penalty parameter for the augmented Lagrangian.
    # A common heuristic for setting mu.
    penalty_parameter_mu: float = (num_rows * num_cols) / (4 * sum_abs_input_matrix)
    
    input_norm = np.linalg.norm(input_matrix, 'fro') # Frobenius norm of the input matrix
    if input_norm == 0: # Also handles all-zero matrix, though sum_abs_input_matrix should catch it
        return np.zeros_like(input_matrix)
    # Convergence is checked based on the Frobenius norm of the reconstruction error (X - L - S).
    convergence_threshold: float = 1e-9 * input_norm


    # Initialize components:
    # S: Sparse component (captures outliers/gross errors)
    sparse_component_S: np.ndarray = np.zeros_like(input_matrix)
    # Y: Lagrange multiplier for the constraint X = L + S
    lagrange_multiplier_Y: np.ndarray = np.zeros_like(input_matrix)
    # L: Low-rank component (the desired underlying structure)
    low_rank_component_L: np.ndarray = np.zeros_like(input_matrix)

    iteration_count: int = 0
    # `lambda` (here `sparsity_penalty_lambda`) is the weight for the l1-norm of S.
    # A common heuristic for lambda.
    sparsity_penalty_lambda: float = 1 / np.sqrt(np.maximum(num_rows, num_cols))

    # ADMM iterations:
    while (
        # Check if reconstruction error is above threshold
        np.linalg.norm(input_matrix - low_rank_component_L - sparse_component_S)
        > convergence_threshold
    ) and (iteration_count < 1000): # Max iterations
        # Update L (low-rank component):
        # L_k+1 = argmin_L ||L||_* + (mu/2) * ||L - (X - S_k + Y_k/mu)||_F^2
        # This is solved by Singular Value Thresholding (SVT) on (X - S_k + Y_k/mu)
        # with threshold 1/mu.
        low_rank_component_L = SVT(
            input_matrix - sparse_component_S + (1 / penalty_parameter_mu) * lagrange_multiplier_Y,
            1 / penalty_parameter_mu, # Threshold for SVT
        )
        # Update S (sparse component):
        # S_k+1 = argmin_S lambda*||S||_1 + (mu/2) * ||S - (X - L_k+1 + Y_k/mu)||_F^2
        # This is solved by element-wise soft thresholding (shrinkage) on (X - L_k+1 + Y_k/mu)
        # with threshold lambda/mu.
        sparse_component_S = shrink(
            input_matrix - low_rank_component_L + (1 / penalty_parameter_mu) * lagrange_multiplier_Y,
            sparsity_penalty_lambda / penalty_parameter_mu, # Threshold for shrinkage
        )
        # Update Y (Lagrange multiplier):
        # Y_k+1 = Y_k + mu * (X - L_k+1 - S_k+1)
        lagrange_multiplier_Y = lagrange_multiplier_Y + penalty_parameter_mu * (
            input_matrix - low_rank_component_L - sparse_component_S
        )
        iteration_count += 1
    return low_rank_component_L # Return the estimated low-rank component

def RPCA_HQF(
    observed_matrix_with_noise: np.ndarray,
    target_rank_for_low_rank_component: int,
    max_iterations: int,
    noise_scale_adaptation_factor: float,
    factor_regularization_penalty: float,
) -> np.ndarray:
    """
    Robust Principal Component Analysis via Non-convex Half-quadratic Regularization.

    This function aims to decompose an observed matrix `observed_matrix_with_noise` into a
    low-rank component `estimated_low_rank_matrix` (the "denoised" matrix) and a sparse noise
    component `estimated_sparse_noise_component`. It assumes `observed_matrix_with_noise = estimated_low_rank_matrix + estimated_sparse_noise_component`.
    The low-rank component `estimated_low_rank_matrix` is constrained to have rank `target_rank_for_low_rank_component`.
    The method uses an iterative approach based on half-quadratic regularization,
    alternating between updating `U`, `V` (factors of `estimated_low_rank_matrix = U @ V`), and
    the sparse noise `estimated_sparse_noise_component`.

    Parameters
    ----------
    observed_matrix_with_noise : np.ndarray
        The observed m x n matrix, potentially corrupted by sparse noise/outliers.
        Shape (m, n).
    target_rank_for_low_rank_component : int
        The target rank for the low-rank component `estimated_low_rank_matrix`.
    max_iterations : int
        The maximum number of iterations for the algorithm.
    noise_scale_adaptation_factor : float
        A parameter that controls the sparsity of the noise component `estimated_sparse_noise_component`.
        It influences the `scale` used for thresholding the noise.
    factor_regularization_penalty : float
        Regularization parameter for the factors `U` and `V`.

    Returns
    -------
    estimated_low_rank_matrix : np.ndarray
        The estimated low-rank (denoised) matrix. Shape (m, n).
        This is the primary output of interest.

    Notes
    -----
    - The algorithm initializes `U` and `V` using a Power Factorization (PF) like approach.
    - The noise `estimated_sparse_noise_component` is estimated by thresholding the difference `observed_matrix_with_noise - estimated_low_rank_matrix`.
      The threshold (`scale`) is adaptively updated based on the median absolute
      deviation of the residuals.
    - The updates for `U` and `V` involve solving regularized least squares problems.
    - Convergence is checked based on the change in RMSE of the reconstruction;
      if the improvement is small for two consecutive iterations, the loop terminates.
    - A random seed (42) is set for reproducibility of the initial `U` matrix.

    This implementation is based on a specific variant of RPCA using
    half-quadratic regularization. The exact formulation and parameter choices
    might differ from other RPCA algorithms like Principal Component Pursuit (PCP).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import RPCA_HQF
    >>> # Create a low-rank matrix
    >>> m_ex, n_ex, true_rank_ex = 50, 60, 3
    >>> U_true_ex = np.random.rand(m_ex, true_rank_ex)
    >>> V_true_ex = np.random.rand(true_rank_ex, n_ex)
    >>> M_low_rank_ex = U_true_ex @ V_true_ex
    >>> # Add some sparse noise
    >>> M_observed_ex = M_low_rank_ex.copy()
    >>> n_outliers_ex = int(0.05 * m_ex * n_ex) # 5% outliers
    >>> outlier_indices_rows_ex = np.random.choice(m_ex, n_outliers_ex)
    >>> outlier_indices_cols_ex = np.random.choice(n_ex, n_outliers_ex)
    >>> M_observed_ex[outlier_indices_rows_ex, outlier_indices_cols_ex] += \
    ...     10 * np.random.randn(n_outliers_ex)
    >>> # Apply RPCA_HQF
    >>> target_rank_val_ex = 3
    >>> max_iter_val_ex = 100
    >>> ip_sparsity_control_ex = 1.5
    >>> lambda_reg_ex = 0.1
    >>> M_denoised_ex = RPCA_HQF(
    ...     observed_matrix_with_noise=M_observed_ex,
    ...     target_rank_for_low_rank_component=target_rank_val_ex,
    ...     max_iterations=max_iter_val_ex,
    ...     noise_scale_adaptation_factor=ip_sparsity_control_ex,
    ...     factor_regularization_penalty=lambda_reg_ex
    ... )
    >>> print("Original matrix rank (approx):", np.linalg.matrix_rank(M_low_rank_ex))
    Original matrix rank (approx): 3
    >>> print("Denoised matrix shape:", M_denoised_ex.shape)
    Denoised matrix shape: (50, 60)
    >>> print("Denoised matrix rank (approx):", np.linalg.matrix_rank(M_denoised_ex)) # doctest: +SKIP
    # Expected rank of M_denoised_ex should be close to target_rank_val_ex (e.g., 3)
    # e.g., Denoised matrix rank (approx): 3
    >>> # Check if denoised matrix is closer to original low-rank than observed
    >>> error_observed_ex = np.linalg.norm(M_observed_ex - M_low_rank_ex, 'fro')
    >>> error_denoised_ex = np.linalg.norm(M_denoised_ex - M_low_rank_ex, 'fro')
    >>> print(f"Error of observed vs true low-rank: {error_observed_ex:.2f}") # doctest: +SKIP
    >>> print(f"Error of denoised vs true low-rank: {error_denoised_ex:.2f}") # doctest: +SKIP
    # Expect error_denoised_ex to be smaller than error_observed_ex
    """
    if not isinstance(observed_matrix_with_noise, np.ndarray) or observed_matrix_with_noise.ndim != 2:
        raise MlsynthDataError("`observed_matrix_with_noise` must be a 2D NumPy array.")
    if observed_matrix_with_noise.size == 0:
        raise MlsynthDataError("`observed_matrix_with_noise` cannot be empty.")
    
    m, n = observed_matrix_with_noise.shape

    if not isinstance(target_rank_for_low_rank_component, (int, np.integer)): # Allow NumPy integers
        raise MlsynthConfigError("`target_rank_for_low_rank_component` must be an integer type (Python int or NumPy integer).")
    
    target_rank_for_low_rank_component = int(target_rank_for_low_rank_component) # Convert to Python int

    if target_rank_for_low_rank_component <= 0:
        raise MlsynthConfigError("`target_rank_for_low_rank_component` must be positive after conversion to int.")
    if target_rank_for_low_rank_component > min(m, n):
        raise MlsynthConfigError(
            f"`target_rank_for_low_rank_component` ({target_rank_for_low_rank_component}) "
            f"cannot exceed min(rows, cols) which is {min(m, n)}."
        )

    if not isinstance(max_iterations, int):
        raise MlsynthConfigError("`max_iterations` must be an integer.")
    if max_iterations <= 0:
        raise MlsynthConfigError("`max_iterations` must be positive.")

    if not isinstance(noise_scale_adaptation_factor, (float, int)):
        raise MlsynthConfigError("`noise_scale_adaptation_factor` must be a float or integer.")
    # Assuming noise_scale_adaptation_factor should be positive, common for scaling factors
    if noise_scale_adaptation_factor <= 0:
        raise MlsynthConfigError("`noise_scale_adaptation_factor` must be positive.")

    if not isinstance(factor_regularization_penalty, (float, int)):
        raise MlsynthConfigError("`factor_regularization_penalty` must be a float or integer.")
    if factor_regularization_penalty < 0: # Allow 0 for no regularization
        raise MlsynthConfigError("`factor_regularization_penalty` must be non-negative.")

    # Initialize U and V factors.
    # U: m x r, V: r x n, where r is target_rank_for_low_rank_component.
    np.random.seed(42)  # For reproducibility of random initialization.
    U: np.ndarray = np.random.rand(m, target_rank_for_low_rank_component)
    V: np.ndarray # Declare V, will be assigned in initialization loop.

    # Power Factorization (PF) like initialization for U and V.
    # Iteratively update U and V a few times to get a reasonable starting point.
    try:
        for _ in range(3): # Small number of iterations for initialization
            # V_new = U_pinv @ X_obs
            pseudo_inverse_of_U: np.ndarray = np.linalg.pinv(U)
            V = pseudo_inverse_of_U @ observed_matrix_with_noise
            # U_new = X_obs @ V_pinv
            pseudo_inverse_of_V: np.ndarray = np.linalg.pinv(V)
            U = observed_matrix_with_noise @ pseudo_inverse_of_V
    except np.linalg.LinAlgError as e: # Catch errors from pseudo-inverse
        raise MlsynthEstimationError(f"SVD/pinv failed during RPCA_HQF initialization: {e}")

    # Initial estimate of the low-rank component.
    estimated_low_rank_matrix: np.ndarray = U @ V
    # Initial residual matrix (potential noise).
    residual_matrix_for_noise_estimation: np.ndarray = (
        observed_matrix_with_noise - estimated_low_rank_matrix
    )
    flattened_residual_for_noise_estimation: np.ndarray = (
        residual_matrix_for_noise_estimation.flatten() # For MAD calculation
    )
    
    # Estimate initial scale of the noise using Median Absolute Deviation (MAD).
    # MAD is a robust measure of variability. 1.4815 is a factor to make MAD consistent with std for normal data.
    scale: float
    if flattened_residual_for_noise_estimation.size == 0: # Should not happen if m,n > 0
        scale = 1.0 # Default scale if no residuals (e.g., perfect fit or empty matrix)
    else:
        median_abs_deviation = np.median(
            np.abs( # |residual_i - median(residuals)|
                flattened_residual_for_noise_estimation
                - np.median(flattened_residual_for_noise_estimation)
            )
        )
        scale = 10 * 1.4815 * median_abs_deviation # Initial scale for thresholding noise
        if scale == 0: # If MAD is zero (e.g., all residuals are identical), use a small default.
            scale = 1e-6

    # Adaptive threshold matrix for identifying sparse noise.
    adaptive_noise_threshold_matrix: np.ndarray = np.ones((m, n)) * scale

    # Estimate initial sparse noise component S.
    # S_ij = R_ij if |R_ij - median(R)| >= threshold_ij, else 0.
    estimated_sparse_noise_mask: np.ndarray = np.ones((m, n)) # Mask for non-zero noise entries
    # Calculate median of residuals once for this step.
    median_res_for_mask_init = np.median(flattened_residual_for_noise_estimation) if flattened_residual_for_noise_estimation.size > 0 else 0
    estimated_sparse_noise_mask[
        np.abs(
            residual_matrix_for_noise_estimation - median_res_for_mask_init
        )
        - adaptive_noise_threshold_matrix # Element-wise comparison
        < 0 # If |R_ij - med(R)| < threshold_ij, then it's not considered noise.
    ] = 0 # Set mask to 0 for these entries.
    estimated_sparse_noise_component: np.ndarray = (
        residual_matrix_for_noise_estimation * estimated_sparse_noise_mask # S = R .* Mask
    )

    # Store previous U and V for regularization term in updates.
    U_factor_previous_iteration: np.ndarray = U.copy()
    V_factor_previous_iteration: np.ndarray = V.copy()

    # Iteration control variables.
    rmse_previous_iteration: float = np.inf # Initialize with a large value.
    consecutive_no_improvement_count: int = 0

    # Main iterative loop.
    for iter_count in range(max_iterations):
        # Matrix for updating U and V: X_obs - S_current
        matrix_for_factor_update: np.ndarray = (
            observed_matrix_with_noise - estimated_sparse_noise_component
        )
        
        try:
            # Update U: U_new = ( (X-S)@V.T - lambda*U_prev ) @ inv(V@V.T - lambda*I)
            # This solves a regularized least squares problem for U.
            term_U_inv = V @ V.T + factor_regularization_penalty * np.eye(target_rank_for_low_rank_component) # (V@V.T + lambda*I)
            # Use pseudo-inverse if term_U_inv is singular or ill-conditioned.
            if np.linalg.matrix_rank(term_U_inv) < term_U_inv.shape[0]:
                U = (matrix_for_factor_update @ V.T + factor_regularization_penalty * U_factor_previous_iteration) @ np.linalg.pinv(term_U_inv)
            else:
                U = (matrix_for_factor_update @ V.T + factor_regularization_penalty * U_factor_previous_iteration) @ np.linalg.inv(term_U_inv)

            # Update V: V_new = inv(U.T@U + lambda*I) @ ( U.T@(X-S) + lambda*V_prev )
            # This solves a regularized least squares problem for V.
            term_V_inv = U.T @ U + factor_regularization_penalty * np.eye(target_rank_for_low_rank_component) # (U.T@U + lambda*I)
            if np.linalg.matrix_rank(term_V_inv) < term_V_inv.shape[0]:
                 V = np.linalg.pinv(term_V_inv) @ (U.T @ matrix_for_factor_update + factor_regularization_penalty * V_factor_previous_iteration)
            else:
                V = np.linalg.inv(term_V_inv) @ (U.T @ matrix_for_factor_update + factor_regularization_penalty * V_factor_previous_iteration)
        
        except np.linalg.LinAlgError as e:
            # If matrix inversion fails, it might be due to singularity.
            # This can happen if regularization is too low or data is problematic.
            raise MlsynthEstimationError(f"Linear algebra error during U/V update in RPCA_HQF iteration {iter_count}: {e}")

        # Store current U, V for next iteration's regularization term.
        U_factor_previous_iteration = U.copy()
        V_factor_previous_iteration = V.copy()
        # Update low-rank estimate: M_new = U_new @ V_new
        estimated_low_rank_matrix = U @ V

        # Update sparse noise component S based on new M.
        residual_matrix_for_noise_estimation = (
            observed_matrix_with_noise - estimated_low_rank_matrix # R = X_obs - M_new
        )
        flattened_residual_for_noise_estimation = (
            residual_matrix_for_noise_estimation.flatten()
        )

        # Adaptively update the scale for noise thresholding.
        if flattened_residual_for_noise_estimation.size > 0:
            median_of_residuals = np.median(flattened_residual_for_noise_estimation)
            current_mad = np.median(
                np.abs(flattened_residual_for_noise_estimation - median_of_residuals)
            )
            # Update scale, but don't let it grow indefinitely (min with previous scale).
            scale = min(scale, noise_scale_adaptation_factor * 1.4815 * current_mad)
            if scale == 0: scale = 1e-6 # Ensure scale is not zero.
        else: # Should not happen if m,n > 0
            scale = 1e-6 # Default small scale if no residuals.

        adaptive_noise_threshold_matrix = np.ones((m, n)) * scale
        estimated_sparse_noise_mask = np.ones((m, n))
        
        median_res_for_mask_update = np.median(flattened_residual_for_noise_estimation) if flattened_residual_for_noise_estimation.size > 0 else 0
        estimated_sparse_noise_mask[
            np.abs(residual_matrix_for_noise_estimation - median_res_for_mask_update) - adaptive_noise_threshold_matrix < 0
        ] = 0
        estimated_sparse_noise_component = (
            residual_matrix_for_noise_estimation * estimated_sparse_noise_mask
        )

        # Check for convergence based on RMSE of reconstruction (X_obs vs M_new).
        current_rmse: float
        if m * n == 0: # Avoid division by zero if matrix is empty.
            current_rmse = np.inf 
        else:
            current_rmse = (
                np.linalg.norm(
                    observed_matrix_with_noise - estimated_low_rank_matrix, "fro" # ||X_obs - M_new||_F
                )
                / np.sqrt(m * n) # Normalize by sqrt of number of elements.
            )

        rmse_improvement: float = rmse_previous_iteration - current_rmse
        if rmse_improvement < 1e-6: # If improvement is negligible
            consecutive_no_improvement_count += 1
        else:  # Improvement was significant
            consecutive_no_improvement_count = 0

        # Stop if RMSE hasn't improved much for 2 consecutive iterations.
        if consecutive_no_improvement_count > 1:
            break
        rmse_previous_iteration = current_rmse

    return estimated_low_rank_matrix # Return the final estimated low-rank component.


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
    >>> from mlsynth.utils.denoiseutils import demean_matrix
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
    >>> from mlsynth.utils.denoiseutils import nbpiid
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
    >>> from mlsynth.utils.denoiseutils import standardize
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

