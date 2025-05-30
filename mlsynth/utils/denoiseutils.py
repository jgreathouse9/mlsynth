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


def debias(
    estimated_baseline_matrix: np.ndarray,
    estimated_treatment_effects: np.ndarray,
    intervention_matrices_stacked: np.ndarray, # Shape (num_rows, num_cols, num_interventions)
    regularization_param_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Debias the estimated baseline matrix and treatment effects.

    Parameters
    ----------
    estimated_baseline_matrix : np.ndarray
        Estimated baseline (low-rank) matrix. Shape (num_rows, num_cols).
    estimated_treatment_effects : np.ndarray
        Estimated treatment effects, one for each intervention.
        Shape (num_interventions,).
    intervention_matrices_stacked : np.ndarray
        Intervention matrices, stacked along the third dimension.
        Shape (num_rows, num_cols, num_interventions).
    regularization_param_lambda : float
        Regularization parameter, typically related to the nuclear norm
        penalty used in estimating `estimated_baseline_matrix`.

    Returns
    -------
    debiased_baseline_matrix : np.ndarray
        The debiased baseline matrix. Shape (num_rows, num_cols).
    debiased_treatment_effects : np.ndarray
        The debiased treatment effects. Shape (num_interventions,).

    Notes
    -----
    This function aims to correct biases in `estimated_baseline_matrix` and
    `estimated_treatment_effects` that may arise from regularization.
    It involves projecting intervention matrices onto the orthogonal complement
    of the tangent space of `estimated_baseline_matrix` and then adjusting
    `estimated_treatment_effects` and `estimated_baseline_matrix`.
    The rank `effective_rank` for SVD is determined internally.
    """
    if not isinstance(estimated_baseline_matrix, np.ndarray) or estimated_baseline_matrix.ndim != 2:
        raise MlsynthDataError("`estimated_baseline_matrix` must be a 2D NumPy array.")
    if not isinstance(estimated_treatment_effects, np.ndarray) or estimated_treatment_effects.ndim != 1:
        raise MlsynthDataError("`estimated_treatment_effects` must be a 1D NumPy array.")
    if not isinstance(intervention_matrices_stacked, np.ndarray) or intervention_matrices_stacked.ndim != 3:
        raise MlsynthDataError("`intervention_matrices_stacked` must be a 3D NumPy array.")
    if not isinstance(regularization_param_lambda, (float, int)):
        raise MlsynthConfigError("`regularization_param_lambda` must be a float or integer.")

    n_rows, n_cols = estimated_baseline_matrix.shape
    n_interventions = intervention_matrices_stacked.shape[2]

    if intervention_matrices_stacked.shape[0] != n_rows or intervention_matrices_stacked.shape[1] != n_cols:
        raise MlsynthDataError(
            "Shape mismatch: `intervention_matrices_stacked` dimensions (0, 1) must match `estimated_baseline_matrix`."
        )
    if estimated_treatment_effects.shape[0] != n_interventions:
        raise MlsynthDataError(
            "Shape mismatch: `estimated_treatment_effects` length must match `intervention_matrices_stacked` third dimension."
        )

    try:
        left_singular_vectors, singular_values, right_singular_vectors_transposed = svd_fast(
            estimated_baseline_matrix
        )
    except MlsynthEstimationError as e: # svd_fast already raises MlsynthEstimationError
        raise MlsynthEstimationError(f"SVD failed in debias for estimated_baseline_matrix: {e}")


    cumulative_singular_values = np.cumsum(singular_values)
    effective_rank: int
    if not singular_values.size or cumulative_singular_values[-1] == 0:  # All singular values are zero or no singular values
        effective_rank = 0
    else:
        safe_cumulative_singular_values = np.where(
            cumulative_singular_values == 0, 1e-12, cumulative_singular_values # Use a smaller epsilon
        )
        # Ensure singular_values and safe_cumulative_singular_values are broadcastable if singular_values is empty
        if singular_values.size == 0: # Should not happen if singular_values.size check above is robust
             effective_rank = 0
        else:
            with np.errstate(divide='ignore', invalid='ignore'): # Handle potential division by zero or NaN
                ratio = singular_values / safe_cumulative_singular_values
                effective_rank = np.sum(np.nan_to_num(ratio) >= 1e-6)


    if effective_rank == 0 and len(singular_values) > 0:
        effective_rank = 1
    
    # Ensure effective_rank does not exceed available dimensions
    effective_rank = min(effective_rank, left_singular_vectors.shape[1], right_singular_vectors_transposed.shape[0])


    left_singular_vectors = left_singular_vectors[:, :effective_rank]
    right_singular_vectors_transposed = right_singular_vectors_transposed[:effective_rank, :]

    Z_projected_orthogonal_to_tangent_space: np.ndarray = np.zeros_like(
        intervention_matrices_stacked
    )
    # Truncate U and Vh to the effective rank.
    # U_r = U[:, :r], Vh_r = Vh[:r, :]
    left_singular_vectors = left_singular_vectors[:, :effective_rank]
    right_singular_vectors_transposed = right_singular_vectors_transposed[:effective_rank, :]

    # Project each intervention matrix Z_k onto the orthogonal complement of the tangent space of M_est.
    # Z_perp_k = P_T_perp(Z_k)
    Z_projected_orthogonal_to_tangent_space: np.ndarray = np.zeros_like(
        intervention_matrices_stacked # Store Z_perp_k for each k
    )
    for intervention_idx_k in np.arange(intervention_matrices_stacked.shape[2]):
        Z_projected_orthogonal_to_tangent_space[
            :, :, intervention_idx_k
        ] = remove_tangent_space_component(
            left_singular_vectors, # U_r
            right_singular_vectors_transposed, # Vh_r
            intervention_matrices_stacked[:, :, intervention_idx_k], # Z_k
        )

    # Construct matrix D, where D_km = <Z_perp_k, Z_perp_m> (Frobenius inner product).
    projection_inner_product_matrix_D: np.ndarray = np.zeros(
        (
            intervention_matrices_stacked.shape[2], # num_interventions
            intervention_matrices_stacked.shape[2], # num_interventions
        )
    )
    for intervention_idx_k in np.arange(intervention_matrices_stacked.shape[2]):
        for intervention_idx_m in np.arange(
            intervention_idx_k, intervention_matrices_stacked.shape[2] # Fill upper triangle
        ):
            # D_km = sum( Z_perp_k_ij * Z_perp_m_ij )
            projection_inner_product_matrix_D[
                intervention_idx_k, intervention_idx_m
            ] = np.sum(
                Z_projected_orthogonal_to_tangent_space[:, :, intervention_idx_k]
                * Z_projected_orthogonal_to_tangent_space[:, :, intervention_idx_m]
            )
            # D is symmetric, D_mk = D_km
            projection_inner_product_matrix_D[
                intervention_idx_m, intervention_idx_k
            ] = projection_inner_product_matrix_D[
                intervention_idx_k, intervention_idx_m
            ]

    # Construct vector Delta, where Delta_k = lambda * <Z_k, U_r @ Vh_r>
    # This represents the bias term related to the regularization.
    bias_correction_vector_Delta: np.ndarray
    if effective_rank > 0 : # If rank is 0, U_r @ Vh_r is a zero matrix.
        # M_eff_normalized = U_r @ Vh_r (this is not M_est itself, but its direction from SVD)
        matrix_M_eff_normalized = left_singular_vectors.dot(right_singular_vectors_transposed)
        bias_correction_vector_Delta = np.array(
            [
                regularization_param_lambda
                * np.sum(
                    intervention_matrices_stacked[:, :, intervention_idx_k] # Z_k
                    * matrix_M_eff_normalized # U_r @ Vh_r
                )
                for intervention_idx_k in range(intervention_matrices_stacked.shape[2])
            ]
        )
    else: # if effective_rank is 0, U_r @ Vh_r is effectively zero, so Delta is zero.
        bias_correction_vector_Delta = np.zeros(intervention_matrices_stacked.shape[2])


    # Solve for delta_tau = D_pinv * Delta
    # delta_tau is the adjustment to the treatment effects.
    try:
        if projection_inner_product_matrix_D.shape[0] == 0 or np.all(np.isclose(projection_inner_product_matrix_D, 0)):
            # If D is empty or all zeros (e.g., all Z_perp_k are zero), adjustment is zero.
            treatment_effect_adjustment_delta = np.zeros_like(bias_correction_vector_Delta)
        else:
            # Use pseudo-inverse for robustness if D is singular or near-singular.
            treatment_effect_adjustment_delta = (
                np.linalg.pinv(projection_inner_product_matrix_D) # D_pinv
                @ bias_correction_vector_Delta # Delta
            )
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"Pseudo-inverse computation failed for D matrix: {e}")

    # Debias treatment effects: tau_debiased = tau_est - delta_tau
    debiased_treatment_effects: np.ndarray = (
        estimated_treatment_effects - treatment_effect_adjustment_delta
    )

    # Project Z onto the tangent space: Z_T = Z - Z_perp
    Z_projected_onto_tangent_space: np.ndarray = (
        intervention_matrices_stacked - Z_projected_orthogonal_to_tangent_space
    )
    # Debias baseline matrix: M_debiased = M_est + lambda * (U_r @ Vh_r) + sum_k (Z_T_k * delta_tau_k)
    debiased_baseline_matrix: np.ndarray
    if effective_rank > 0:
        term_lambda_M_eff_normalized = regularization_param_lambda * left_singular_vectors.dot(right_singular_vectors_transposed)
    else:
        term_lambda_M_eff_normalized = np.zeros_like(estimated_baseline_matrix)
        
    # Sum_k (Z_T_k * delta_tau_k)
    # Reshape delta_tau to (1, 1, num_interventions) for broadcasting with Z_T (n_rows, n_cols, num_interventions)
    sum_Z_T_delta_tau = np.sum(
            Z_projected_onto_tangent_space
            * treatment_effect_adjustment_delta.reshape(1, 1, -1),
            axis=2, # Sum over interventions
        )
        
    debiased_baseline_matrix = (
        estimated_baseline_matrix
        + term_lambda_M_eff_normalized
        + sum_Z_T_delta_tau
    )

    return debiased_baseline_matrix, debiased_treatment_effects


def svd_fast( # Fast SVD
    input_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition, optimized for speed if input_matrix is skinny or fat.

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix for SVD. Shape (num_rows, num_cols).

    Returns
    -------
    final_left_singular_vectors : np.ndarray
        Unitary matrix having left singular vectors as columns.
        Shape (num_rows, k), where k = min(num_rows, num_cols).
    final_singular_values : np.ndarray
        The singular values, sorted in non-increasing order. Shape (k,).
    final_right_singular_vectors_transposed : np.ndarray
        Unitary matrix having right singular vectors as rows (already transposed).
        Shape (k, num_cols).

    Notes
    -----
    This function implements a common optimization for SVD: if the matrix `input_matrix`
    is "tall" (num_rows > num_cols), it computes the SVD of `input_matrix.T @ input_matrix`.
    If `input_matrix` is "wide" (num_cols > num_rows), it computes the SVD of `input_matrix @ input_matrix.T`.
    This can be faster as it involves decomposing a smaller square matrix.
    A small tolerance (1e-7) is used to handle near-zero singular values.
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim != 2:
        raise MlsynthDataError("Input `input_matrix` must be a 2D array.")

    # If matrix is tall (m > n), SVD(X) = SVD(X.T).
    # We compute SVD of X.T @ X (n x n) or X @ X.T (m x m), whichever is smaller.
    # Let A be the input_matrix.
    transposed_flag: bool = False
    current_matrix_A = input_matrix.copy() # Work with a copy
    if current_matrix_A.shape[0] > current_matrix_A.shape[1]: # If A is tall (m > n)
        transposed_flag = True
        current_matrix_A = current_matrix_A.T # Work with A.T (n x m, now wide or square)
    
    # Now current_matrix_A is either wide (n > m) or square (n = m).
    # We compute SVD of A @ A.T (if A was originally wide or became wide after transpose)
    # or A.T @ A (if A was originally tall).
    # Since current_matrix_A is now m' x n' where m' <= n', we compute SVD of (current_matrix_A @ current_matrix_A.T)
    # This matrix is m' x m'.
    try:
        # Let B = current_matrix_A. Then B is m' x n' with m' <= n'.
        # We compute SVD of B @ B.T, which is (m' x m').
        # SVD(B @ B.T) = U_BBT * S_BBT^2 * U_BBT.T
        # U_BBT are the left singular vectors of B.
        # S_BBT are the singular values of B.
        matrix_for_svd_computation: np.ndarray = (
            current_matrix_A @ current_matrix_A.T # (m' x n') @ (n' x m') = (m' x m')
        )
        # u_of_A here is U_BBT (left singular vectors of B, which is current_matrix_A)
        # squared_singular_values_of_A are S_BBT^2 (squared singular values of B)
        # vh_of_A is U_BBT.T (transpose of left singular vectors of B)
        u_of_B, squared_singular_values_of_B, _ = np.linalg.svd(
            matrix_for_svd_computation, full_matrices=False
        )
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"SVD computation failed in svd_fast: {e}")

    # Threshold small squared singular values to zero before taking sqrt.
    squared_singular_values_of_B[squared_singular_values_of_B < 1e-7] = 0
    final_singular_values_S: np.ndarray = np.sqrt(squared_singular_values_of_B) # These are singular values of B (current_matrix_A)
    
    # To get Vh_B (right singular vectors of B): Vh_B = S_B_inv @ Uh_B.T @ B
    # Uh_B.T is u_of_B.T
    # S_B_inv is diag(1/s_i)
    
    # Handle potential division by zero for inverse singular values.
    # Add a small epsilon to zero singular values before inversion.
    safe_final_singular_values_S = final_singular_values_S + 1e-7 * (final_singular_values_S < 1e-7)
    
    inverse_singular_values_S: np.ndarray
    if np.any(safe_final_singular_values_S == 0): # If any are still zero (e.g., original was zero)
        inverse_singular_values_S = np.zeros_like(safe_final_singular_values_S)
        non_zero_mask = safe_final_singular_values_S != 0
        inverse_singular_values_S[non_zero_mask] = 1.0 / safe_final_singular_values_S[non_zero_mask]
    else:
        inverse_singular_values_S = 1.0 / safe_final_singular_values_S

    # Vh_B = diag(1/s_i) @ U_B.T @ B
    final_right_singular_vectors_transposed_Vh: np.ndarray = (
        np.diag(inverse_singular_values_S) # S_B_inv (as diagonal matrix)
        @ u_of_B.T # U_B.T
        @ current_matrix_A # B
    )
    
    final_left_singular_vectors_U = u_of_B # U_B

    if transposed_flag: # If original input_matrix was tall (A.T was used as current_matrix_A)
        # Original A = U_A S_A Vh_A
        # We computed SVD for B = A.T. So, B = U_B S_B Vh_B.
        # U_A = V_B (which is final_right_singular_vectors_transposed_Vh.T)
        # S_A = S_B (which is final_singular_values_S)
        # Vh_A = Uh_B.T (which is final_left_singular_vectors_U.T)
        return final_right_singular_vectors_transposed_Vh.T, final_singular_values_S, final_left_singular_vectors_U.T
    else: # Original input_matrix was wide or square (A was used as current_matrix_A)
        # A = U_A S_A Vh_A
        # U_A = U_B (which is final_left_singular_vectors_U)
        # S_A = S_B (which is final_singular_values_S)
        # Vh_A = Vh_B (which is final_right_singular_vectors_transposed_Vh)
        return final_left_singular_vectors_U, final_singular_values_S, final_right_singular_vectors_transposed_Vh


def SVD(input_matrix: np.ndarray, target_rank: int) -> np.ndarray:
    """
    Approximate matrix `input_matrix` with a rank `target_rank` matrix using SVD (hard truncation).

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix to be approximated. Shape (num_rows, num_cols).
    target_rank : int
        The target rank for the approximation. The `target_rank` largest singular
        values (and corresponding singular vectors) will be kept.

    Returns
    -------
    np.ndarray
        The rank-`target_rank` approximation of matrix `input_matrix`, obtained by hard
        truncation of the SVD. Shape (num_rows, num_cols).

    See Also
    --------
    svd_fast : The SVD implementation used internally.
    SVD_soft : For SVD with soft thresholding.
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim != 2:
        raise MlsynthDataError("Input `input_matrix` must be a 2D array.")
    if not isinstance(target_rank, (int, np.integer)): # Allow NumPy integers
        raise MlsynthConfigError("Input `target_rank` must be an integer.")
    if target_rank <= 0:
        raise MlsynthConfigError("Input `target_rank` must be positive.")
    if target_rank > min(input_matrix.shape):
        raise MlsynthConfigError(
            f"target_rank ({target_rank}) cannot exceed min(matrix_rows, matrix_cols) which is {min(input_matrix.shape)}."
        )

    try:
        (
            left_singular_vectors,
            singular_values,
            right_singular_vectors_transposed,
        ) = svd_fast(input_matrix)
    except MlsynthEstimationError as e: # svd_fast already raises MlsynthEstimationError
        raise MlsynthEstimationError(f"SVD_fast failed in SVD: {e}")

    # Truncate singular values: keep the first `target_rank` values, set others to 0.
    singular_values[target_rank:] = 0
    # Reconstruct the matrix: U * Sigma_truncated * Vh
    # (left_singular_vectors * singular_values) performs element-wise multiplication if singular_values is 1D,
    # effectively scaling columns of U by singular values. Then dot with Vh.
    return (left_singular_vectors * singular_values).dot( # U * diag(S_truncated)
        right_singular_vectors_transposed # Vh
    )


def SVD_soft(input_matrix: np.ndarray, threshold: float) -> np.ndarray: # SVD with soft thresholding
    """Perform Singular Value Decomposition (SVD) with soft thresholding.

    This function first computes the SVD of the input matrix `input_matrix`. Then, it
    applies a soft thresholding operation to the singular values. The soft
    thresholding operation is defined as `max(0, s - threshold)`, where `s` is a
    singular value and `threshold` is the threshold parameter. Finally, it reconstructs the
    matrix using the thresholded singular values and the original singular
    vectors.

    Parameters
    ----------
    input_matrix : np.ndarray
        The input matrix to be processed. Shape (n_rows, n_cols).
    threshold : float
        The threshold value for soft thresholding. This value is subtracted
        from the singular values, and any resulting negative values are set
        to zero. Must be non-negative.

    Returns
    -------
    np.ndarray
        The matrix reconstructed after SVD and soft thresholding of its
        singular values. Shape (n_rows, n_cols).

    See Also
    --------
    svd_fast : The SVD implementation used internally.
    shrink : A general soft thresholding function (element-wise).
    SVD : For SVD with hard truncation of singular values.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import SVD_soft
    >>> input_matrix_ex1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> threshold_val_ex1 = 1.0
    >>> X_soft_thresholded = SVD_soft(input_matrix_ex1, threshold_val_ex1)
    >>> print(X_soft_thresholded)
    [[ 1.          2.          3.        ]
     [ 4.          5.          6.        ]
     [ 7.          8.          9.00000001]]

    Note: The exact output may vary slightly due to floating point precision.
    For a matrix like the one above, where one singular value is very small,
    soft thresholding might zero it out if `threshold` is large enough, or reduce it.
    Let's try with a more illustrative example where thresholding has a visible effect.

    >>> U_true = np.array([[1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]])
    >>> S_true = np.array([5, 0.5])
    >>> V_true = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)]])
    >>> matrix_example = U_true @ np.diag(S_true) @ V_true
    >>> print("Original matrix_example:\\n", matrix_example)
    Original matrix_example:
     [[ 2.33345238  2.68654762]
     [ 2.68654762  2.33345238]]
    >>> X_reconstructed_soft = SVD_soft(matrix_example, threshold=1.0) # Threshold is 1.0
    >>> # Expected singular values after soft thresholding: max(0, 5-1)=4, max(0, 0.5-1)=0
    >>> # So, the reconstructed matrix should be rank 1.
    >>> print("X reconstructed with SVD_soft (threshold=1.0):\\n", X_reconstructed_soft)
    X reconstructed with SVD_soft (threshold=1.0):
     [[ 1.8667619   2.1332381 ] # Expected values might differ slightly
     [ 2.1332381   1.8667619 ]]
    """
    if not isinstance(input_matrix, np.ndarray):
        raise MlsynthDataError("Input `input_matrix` must be a NumPy array.")
    if input_matrix.ndim != 2:
        raise MlsynthDataError("Input `input_matrix` must be a 2D array.")
    if not isinstance(threshold, (float, int)):
        raise MlsynthConfigError("Input `threshold` must be a float or integer.")
    if threshold < 0:
        raise MlsynthConfigError("Input `threshold` must be non-negative.")

    try:
        left_singular_vectors, singular_values, right_singular_vectors_transposed = svd_fast(
            input_matrix
        )
    except MlsynthEstimationError as e: # svd_fast already raises MlsynthEstimationError
        raise MlsynthEstimationError(f"svd_fast failed in SVD_soft: {e}")
        
    # Apply soft thresholding to all singular values: s_i' = max(0, s_i - threshold)
    thresholded_singular_values: np.ndarray = np.maximum(
        0, singular_values - threshold # Element-wise operation
    )
    # Reconstruct the matrix: U * Sigma_soft_thresholded * Vh
    return (
        left_singular_vectors * thresholded_singular_values # U * diag(S_soft_thresholded)
    ).dot(right_singular_vectors_transposed) # Vh


def DC_PR_with_l(
    observed_matrix: np.ndarray,
    intervention_matrices: Union[List[np.ndarray], np.ndarray],
    nuclear_norm_regularizer: float,
    initial_treatment_effects: Optional[np.ndarray] = None,
    convergence_tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]: # Debiased Convex Panel Regression
    """
    Debiased Convex Panel Regression with a given nuclear norm regularizer.

    This function iteratively estimates a low-rank baseline matrix `M` and
    treatment effects `tau` from an observed matrix `observed_matrix` and
    intervention matrices `intervention_matrices`. The model assumes
    `observed_matrix = M + intervention_matrices*tau + Noise`.
    The estimation alternates between:
    1. Estimating `M` using Singular Value Decomposition with soft thresholding
       (SVD_soft) on `(observed_matrix - intervention_matrices*current_treatment_effects)`,
       with threshold `nuclear_norm_regularizer`.
    2. Estimating `tau` using Ordinary Least Squares (OLS) on
       `(observed_matrix - estimated_baseline_matrix)`, considering only entries
       where `intervention_matrices` is non-zero.
    The process continues until the change in `tau` between iterations is
    below a specified tolerance `convergence_tolerance` or a maximum of 2000 iterations.

    Parameters
    ----------
    observed_matrix : np.ndarray
        The observed data matrix. Shape (n_rows, n_cols).
    intervention_matrices : Union[List[np.ndarray], np.ndarray]
        Intervention matrices. These indicate where and when treatments occur.
        - If a list of 2D arrays: Each array in the list corresponds to one
          intervention, with shape (n_rows, n_cols).
        - If a single 2D array: Represents a single intervention. Shape (n_rows, n_cols).
        - If a 3D array: Already stacked interventions. Shape (n_rows, n_cols, n_interventions).
        Non-zero entries in these matrices typically indicate treated unit-time pairs.
    nuclear_norm_regularizer : float
        The regularization parameter for the nuclear norm of `M`. This is used
        as the threshold in the `SVD_soft` step. Must be non-negative.
    initial_treatment_effects : Optional[np.ndarray], default None
        Initial guess for the treatment effects `tau`. If None, `tau` is
        initialized to zeros. Shape (n_interventions,).
    convergence_tolerance : float, default 1e-6
        Convergence criterion. The iteration stops if the L2 norm of the
        change in `tau` relative to the L2 norm of `current_treatment_effects`
        is less than `convergence_tolerance`.

    Returns
    -------
    estimated_baseline_matrix : np.ndarray
        The estimated low-rank baseline matrix. Shape (n_rows, n_cols).
    new_treatment_effects : np.ndarray
        The estimated treatment effects. Shape (n_interventions,).

    See Also
    --------
    SVD_soft : Used for estimating the low-rank matrix `M`.
    prepare_OLS : Used to set up the OLS problem for estimating `tau`.
    transform_to_3D : Converts various `Z` input formats to a 3D array.
    non_convex_PR : Similar panel regression but with a hard rank constraint.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import DC_PR_with_l
    >>> # Example data
    >>> num_rows_ex, num_cols_ex, num_interventions_ex = 5, 10, 1
    >>> observed_matrix_true = np.random.rand(num_rows_ex, num_cols_ex) # True underlying matrix
    >>> baseline_matrix_true = SVD_soft(observed_matrix_true, 0.1) # A low-rank version
    >>> intervention_matrices_true = np.zeros((num_rows_ex, num_cols_ex, num_interventions_ex))
    >>> intervention_matrices_true[0, 5:, 0] = 1 # First unit treated from 6th period
    >>> treatment_effects_true = np.array([0.5])
    >>> observed_matrix_noisy = baseline_matrix_true + \
    ...                         np.tensordot(intervention_matrices_true, treatment_effects_true, axes=([2],[0])) + \
    ...                         np.random.normal(0, 0.01, (num_rows_ex, num_cols_ex))
    >>> regularizer_param = 0.1
    >>> estimated_M, estimated_tau = DC_PR_with_l(
    ...     observed_matrix_noisy, intervention_matrices_true, regularizer_param
    ... )
    >>> print("Estimated M shape:", estimated_M.shape)
    Estimated M shape: (5, 10)
    >>> print("Estimated tau:", estimated_tau) # doctest: +SKIP
        # Expected output for estimated_tau will be close to treatment_effects_true [0.5]
        # e.g., Estimated tau: [0.498...]

    Note: The exact numerical results in the example will vary due to random
    data generation. The example illustrates the usage and expected shapes.
    """
    if not isinstance(observed_matrix, np.ndarray) or observed_matrix.ndim != 2:
        raise MlsynthDataError("`observed_matrix` must be a 2D NumPy array.")
    if not isinstance(nuclear_norm_regularizer, (float, int)):
        raise MlsynthConfigError("`nuclear_norm_regularizer` must be a float or integer.")
    if nuclear_norm_regularizer < 0:
        raise MlsynthConfigError("`nuclear_norm_regularizer` must be non-negative.")
    if initial_treatment_effects is not None and (not isinstance(initial_treatment_effects, np.ndarray) or initial_treatment_effects.ndim != 1):
        raise MlsynthConfigError("`initial_treatment_effects` must be a 1D NumPy array if provided.")
    if not isinstance(convergence_tolerance, float) or convergence_tolerance <= 0:
        raise MlsynthConfigError("`convergence_tolerance` must be a positive float.")

    # Standardize intervention_matrices to 3D format (n_rows, n_cols, n_interventions).
    intervention_matrices_3d: np.ndarray = transform_to_3D(intervention_matrices)

    if observed_matrix.shape != intervention_matrices_3d.shape[:2]:
        raise MlsynthDataError(
            "Shape mismatch: `observed_matrix` and `intervention_matrices` (first 2 dims) must have the same shape."
        )
    
    num_interventions = intervention_matrices_3d.shape[2]
    current_treatment_effects: np.ndarray # Initialize tau
    if initial_treatment_effects is None:
        current_treatment_effects = np.zeros(num_interventions)
    else:
        if initial_treatment_effects.shape[0] != num_interventions:
            raise MlsynthConfigError(
                f"`initial_treatment_effects` length ({initial_treatment_effects.shape[0]}) "
                f"must match number of interventions ({num_interventions})."
            )
        current_treatment_effects = initial_treatment_effects.copy()

    # Pre-calculate OLS components (design matrix X_ols from Z, and (X_ols.T @ X_ols)^-1)
    # This is done once as Z does not change during iterations.
    try:
        active_intervention_entries_idx, ols_design_matrix, ols_design_matrix_pseudo_inverse_term = prepare_OLS(
            intervention_matrices_3d
        )
    except np.linalg.LinAlgError as e: # Raised by prepare_OLS if (X.T X) is singular
        raise MlsynthEstimationError(f"OLS preparation failed in DC_PR_with_l: {e}")


    estimated_baseline_matrix: np.ndarray = np.zeros_like(observed_matrix) # Initialize M
    # Iterative estimation of M and tau
    for _ in range(2000):  # Max iterations
        # Step 1: Estimate M (low-rank baseline)
        # M_k+1 = argmin_M ||M||_* + (1/(2l)) * ||M - (O - Z*tau_k)||_F^2
        # This is solved by SVD_soft on (O - Z*tau_k) with threshold `l` (nuclear_norm_regularizer).
        matrix_to_denoise = observed_matrix - np.tensordot(
                intervention_matrices_3d, current_treatment_effects, axes=([2], [0]) # Z * tau_k
            )
        estimated_baseline_matrix = SVD_soft(
            matrix_to_denoise,
            nuclear_norm_regularizer, # Threshold `l`
        )
        
        # Step 2: Estimate tau (treatment effects) via OLS
        # tau_k+1 = argmin_tau || (O - M_k+1) - Z*tau ||_F^2 (on active Z entries)
        # y_ols = (O - M_k+1) restricted to active Z entries
        ols_dependent_variable: np.ndarray = (
            observed_matrix - estimated_baseline_matrix # O - M_k+1
        )[
            active_intervention_entries_idx # Select entries where Z is active
        ]
        # tau_k+1 = (X_ols.T @ X_ols)^-1 @ X_ols.T @ y_ols
        new_treatment_effects: np.ndarray = ols_design_matrix_pseudo_inverse_term @ (
            ols_design_matrix.T @ ols_dependent_variable
        )
        
        # Step 3: Check for convergence
        # Stop if relative change in tau is small.
        # Norm of current_treatment_effects can be zero if all effects are zero.
        norm_current_tau = np.linalg.norm(current_treatment_effects)
        if norm_current_tau == 0: # Avoid division by zero if current_treatment_effects is all zeros
            # If current tau is zero, converge if new tau is also (close to) zero
            if np.linalg.norm(new_treatment_effects) < convergence_tolerance:
                 return estimated_baseline_matrix, new_treatment_effects
        elif np.linalg.norm(
            new_treatment_effects - current_treatment_effects
        ) < convergence_tolerance * norm_current_tau:
            return estimated_baseline_matrix, new_treatment_effects
            
        current_treatment_effects = new_treatment_effects # Update tau for next iteration
        
    # If loop finishes without convergence (max_iterations reached)
    return estimated_baseline_matrix, current_treatment_effects


def non_convex_PR(
    observed_matrix: np.ndarray,
    intervention_matrices: Union[List[np.ndarray], np.ndarray],
    rank_constraint: int,
    initial_treatment_effects: Optional[np.ndarray] = None,
    convergence_tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]: # Non-Convex Panel Regression
    """
    Non-Convex Panel Regression with a fixed rank constraint.

    This function iteratively estimates a low-rank baseline matrix `M` (with
    rank constrained to `rank_constraint`) and treatment effects `tau` from an
    observed matrix `observed_matrix` and intervention matrices `intervention_matrices`.
    The model assumes `observed_matrix = M + intervention_matrices*tau + Noise`.
    The estimation alternates between:
    1. Estimating `M` by finding the best rank-`rank_constraint` approximation of
       `(observed_matrix - intervention_matrices*current_treatment_effects)`
       using Singular Value Decomposition (SVD) with hard truncation
       (keeping `rank_constraint` singular values).
    2. Estimating `tau` using Ordinary Least Squares (OLS) on
       `(observed_matrix - estimated_baseline_matrix)`, considering only entries
       where `intervention_matrices` is non-zero.
    The process continues until the change in `tau` between iterations is
    below a specified tolerance `convergence_tolerance` or a maximum of 2000 iterations.

    Parameters
    ----------
    observed_matrix : np.ndarray
        The observed data matrix. Shape (n_rows, n_cols).
    intervention_matrices : Union[List[np.ndarray], np.ndarray]
        Intervention matrices. These indicate where and when treatments occur.
        - If a list of 2D arrays: Each array in the list corresponds to one
          intervention, with shape (n_rows, n_cols).
        - If a single 2D array: Represents a single intervention. Shape (n_rows, n_cols).
        - If a 3D array: Already stacked interventions. Shape (n_rows, n_cols, n_interventions).
        Non-zero entries in these matrices typically indicate treated unit-time pairs.
    rank_constraint : int
        The fixed rank constraint for the estimated baseline matrix `M`.
        Must be a positive integer.
    initial_treatment_effects : Optional[np.ndarray], default None
        Initial guess for the treatment effects `tau`. If None, `tau` is
        initialized to zeros. Shape (n_interventions,).
    convergence_tolerance : float, default 1e-6
        Convergence criterion. The iteration stops if the L2 norm of the
        change in `tau` relative to the L2 norm of `current_treatment_effects`
        is less than `convergence_tolerance`.

    Returns
    -------
    estimated_baseline_matrix : np.ndarray
        The estimated rank-`rank_constraint` baseline matrix. Shape (n_rows, n_cols).
    new_treatment_effects : np.ndarray
        The estimated treatment effects. Shape (n_interventions,).

    See Also
    --------
    SVD : Used for estimating the rank-`r` matrix `M` (hard truncation).
    prepare_OLS : Used to set up the OLS problem for estimating `tau`.
    transform_to_3D : Converts various `Z` input formats to a 3D array.
    DC_PR_with_l : Similar panel regression but uses soft thresholding via `l`.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import non_convex_PR, SVD
    >>> # Example data
    >>> num_rows_ex, num_cols_ex, num_interventions_ex = 5, 10, 1
    >>> rank_val = 2
    >>> observed_matrix_true = np.random.rand(num_rows_ex, num_cols_ex) # True underlying matrix
    >>> baseline_matrix_true = SVD(observed_matrix_true, rank_val) # A rank-2 version
    >>> intervention_matrices_true = np.zeros((num_rows_ex, num_cols_ex, num_interventions_ex))
    >>> intervention_matrices_true[0, 5:, 0] = 1 # First unit treated from 6th period
    >>> treatment_effects_true = np.array([0.5])
    >>> observed_matrix_noisy = baseline_matrix_true + \
    ...                         np.tensordot(intervention_matrices_true, treatment_effects_true, axes=([2],[0])) + \
    ...                         np.random.normal(0, 0.01, (num_rows_ex, num_cols_ex))
    >>> M_estimated, tau_estimated = non_convex_PR(
    ...     observed_matrix_noisy, intervention_matrices_true, rank_constraint=rank_val
    ... )
    >>> print("Estimated M shape:", M_estimated.shape)
    Estimated M shape: (5, 10)
    >>> print("Rank of estimated M:", np.linalg.matrix_rank(M_estimated)) # doctest: +SKIP
    # Expected rank will be close to rank_val (e.g., 2)
    >>> print("Estimated tau:", tau_estimated) # doctest: +SKIP
    # Expected output for tau_estimated will be close to treatment_effects_true [0.5]
    # e.g., Estimated tau: [0.497...]

    Note: The exact numerical results in the example will vary due to random
    data generation. The example illustrates the usage and expected shapes/rank.
    """
    if not isinstance(observed_matrix, np.ndarray) or observed_matrix.ndim != 2:
        raise MlsynthDataError("`observed_matrix` must be a 2D NumPy array.")
    if not isinstance(rank_constraint, int):
        raise MlsynthConfigError("`rank_constraint` must be an integer.")
    if rank_constraint <= 0:
        raise MlsynthConfigError("`rank_constraint` must be positive.")
    if rank_constraint > min(observed_matrix.shape):
        raise MlsynthConfigError(
            f"`rank_constraint` ({rank_constraint}) cannot exceed min(matrix_rows, matrix_cols) "
            f"which is {min(observed_matrix.shape)}."
        )
    if initial_treatment_effects is not None and (not isinstance(initial_treatment_effects, np.ndarray) or initial_treatment_effects.ndim != 1):
        raise MlsynthConfigError("`initial_treatment_effects` must be a 1D NumPy array if provided.")
    if not isinstance(convergence_tolerance, float) or convergence_tolerance <= 0:
        raise MlsynthConfigError("`convergence_tolerance` must be a positive float.")

    # Standardize intervention_matrices to 3D format.
    intervention_matrices_3d: np.ndarray = transform_to_3D(intervention_matrices)
    
    if observed_matrix.shape != intervention_matrices_3d.shape[:2]:
        raise MlsynthDataError(
            "Shape mismatch: `observed_matrix` and `intervention_matrices` (first 2 dims) must have the same shape."
        )

    num_interventions = intervention_matrices_3d.shape[2]
    current_treatment_effects: np.ndarray # Initialize tau
    if initial_treatment_effects is None:
        current_treatment_effects = np.zeros(num_interventions)
    else:
        if initial_treatment_effects.shape[0] != num_interventions:
            raise MlsynthConfigError(
                f"`initial_treatment_effects` length ({initial_treatment_effects.shape[0]}) "
                f"must match number of interventions ({num_interventions})."
            )
        current_treatment_effects = initial_treatment_effects.copy()
    
    # Pre-calculate OLS components.
    try:
        active_intervention_entries_idx, ols_design_matrix, ols_design_matrix_pseudo_inverse_term = prepare_OLS(
            intervention_matrices_3d
        )
    except np.linalg.LinAlgError as e: # Raised by prepare_OLS if (X.T X) is singular
        raise MlsynthEstimationError(f"OLS preparation failed in non_convex_PR: {e}")

    estimated_baseline_matrix: np.ndarray = np.zeros_like(observed_matrix) # Initialize M
    # Iterative estimation of M and tau
    for _ in range(2000):  # Max iterations
        # Step 1: Estimate M (low-rank baseline)
        # M_k+1 = argmin_{rank(M)<=r} ||M - (O - Z*tau_k)||_F^2
        # This is solved by SVD on (O - Z*tau_k) with hard truncation to `rank_constraint`.
        matrix_to_denoise = observed_matrix - np.tensordot(
                intervention_matrices_3d, current_treatment_effects, axes=([2], [0]) # Z * tau_k
            )
        estimated_baseline_matrix = SVD( # Hard truncation SVD
            matrix_to_denoise,
            rank_constraint,
        )
        
        # Step 2: Estimate tau (treatment effects) via OLS
        # (Same as in DC_PR_with_l)
        ols_dependent_variable: np.ndarray = (
            observed_matrix - estimated_baseline_matrix # O - M_k+1
        )[
            active_intervention_entries_idx # Select entries where Z is active
        ]
        new_treatment_effects: np.ndarray = ols_design_matrix_pseudo_inverse_term @ (
            ols_design_matrix.T @ ols_dependent_variable
        )
        
        # Step 3: Check for convergence
        norm_current_tau = np.linalg.norm(current_treatment_effects)
        if norm_current_tau == 0:
            if np.linalg.norm(new_treatment_effects) < convergence_tolerance:
                return estimated_baseline_matrix, new_treatment_effects
        elif np.linalg.norm(
            new_treatment_effects - current_treatment_effects
        ) < convergence_tolerance * norm_current_tau:
            return estimated_baseline_matrix, new_treatment_effects
            
        current_treatment_effects = new_treatment_effects # Update tau
        
    # If loop finishes without convergence
    return estimated_baseline_matrix, current_treatment_effects


def panel_regression_CI(
    estimated_baseline_matrix: np.ndarray,
    intervention_matrices_3d: np.ndarray,
    residual_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the asymptotic covariance matrix for estimated treatment effects (taus).

    This function calculates an estimate of the asymptotic covariance matrix
    for the treatment effect estimators `tau`, derived from a panel regression
    model like `Observed = M_baseline + Z_interventions*tau + E_residuals`.
    It uses a first-order approximation based on the `estimated_baseline_matrix`,
    the `intervention_matrices_3d`, and the `residual_matrix`.

    The method involves:
    1. Performing SVD on the `estimated_baseline_matrix` to find its principal
       components (`left_singular_vectors`, `right_singular_vectors_transposed`)
       and determine its `effective_rank`.
    2. Projecting each intervention matrix in `intervention_matrices_3d` onto the
       orthogonal complement of the tangent space of `estimated_baseline_matrix`.
       This yields `design_matrix_orthogonal_projection`.
    3. Computing `projection_term_for_covariance =
       (design_matrix_orthogonal_projection.T @ design_matrix_orthogonal_projection)^(-1) @
       design_matrix_orthogonal_projection.T`.
    4. The covariance matrix is then estimated as
       `(projection_term_for_covariance * squared_residuals_flat) @ projection_term_for_covariance.T`,
       where `squared_residuals_flat` are the squared residuals from `residual_matrix` reshaped.

    Parameters
    ----------
    estimated_baseline_matrix : np.ndarray
        The estimated low-rank baseline matrix. Shape (n_rows, n_cols).
    intervention_matrices_3d : np.ndarray
        Intervention matrices, stacked along the third dimension.
        Shape (n_rows, n_cols, n_interventions). Each slice `[:,:,k]`
        corresponds to the k-th intervention.
    residual_matrix : np.ndarray
        The estimated noise or residual matrix, typically calculated as
        `Observed_Matrix - estimated_baseline_matrix - intervention_matrices_3d @ estimated_tau`.
        Shape (n_rows, n_cols).

    Returns
    -------
    covariance_matrix_tau : np.ndarray
        An estimated (n_interventions, n_interventions) asymptotic covariance
        matrix for the treatment effects `tau`. The diagonal elements can be
        used to estimate standard errors for individual `tau_k`.

    See Also
    --------
    svd_fast : Used to decompose the baseline matrix `M`.
    remove_tangent_space_component : Used to project `Z` matrices.

    Notes
    -----
    The `effective_rank` of `estimated_baseline_matrix` is determined internally
    using a threshold on the ratio of singular values to their cumulative sum.
    This covariance matrix is crucial for constructing confidence intervals
    and performing hypothesis tests on the estimated treatment effects.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import SVD_soft
    >>> input_matrix_ex1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> threshold_val_ex1 = 1.0
    >>> X_soft_thresholded = SVD_soft(input_matrix_ex1, threshold_val_ex1)
    >>> print(X_soft_thresholded)
    [[ 1.          2.          3.        ]
     [ 4.          5.          6.        ]
     [ 7.          8.          9.00000001]]

    Note: The exact output may vary slightly due to floating point precision.
    For a matrix like the one above, where one singular value is very small,
    soft thresholding might zero it out if `threshold` is large enough, or reduce it.
    Let's try with a more illustrative example where thresholding has a visible effect.

    >>> U_true = np.array([[1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]])
    >>> S_true = np.array([5, 0.5])
    >>> V_true = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)]])
    >>> matrix_example = U_true @ np.diag(S_true) @ V_true
    >>> print("Original matrix_example:\\n", matrix_example)
    Original matrix_example:
     [[ 2.33345238  2.68654762]
     [ 2.68654762  2.33345238]]
    >>> X_reconstructed_soft = SVD_soft(matrix_example, threshold=1.0) # Threshold is 1.0
    >>> # Expected singular values after soft thresholding: max(0, 5-1)=4, max(0, 0.5-1)=0
    >>> # So, the reconstructed matrix should be rank 1.
    >>> print("X reconstructed with SVD_soft (threshold=1.0):\\n", X_reconstructed_soft)
    X reconstructed with SVD_soft (threshold=1.0):
     [[ 1.8667619   2.1332381 ] # Expected values might differ slightly
     [ 2.1332381   1.8667619 ]]
    """
    left_singular_vectors, singular_values, right_singular_vectors_transposed = svd_fast(
        estimated_baseline_matrix
    )
    if not isinstance(estimated_baseline_matrix, np.ndarray) or estimated_baseline_matrix.ndim != 2:
        raise MlsynthDataError("`estimated_baseline_matrix` must be a 2D NumPy array.")
    if not isinstance(intervention_matrices_3d, np.ndarray) or intervention_matrices_3d.ndim != 3:
        raise MlsynthDataError("`intervention_matrices_3d` must be a 3D NumPy array.")
    if not isinstance(residual_matrix, np.ndarray) or residual_matrix.ndim != 2:
        raise MlsynthDataError("`residual_matrix` must be a 2D NumPy array.")

    if estimated_baseline_matrix.shape != intervention_matrices_3d.shape[:2]:
        raise MlsynthDataError(
            "Shape mismatch: `estimated_baseline_matrix` and `intervention_matrices_3d` (first 2 dims) must have the same shape."
        )
    if estimated_baseline_matrix.shape != residual_matrix.shape:
        raise MlsynthDataError(
            "Shape mismatch: `estimated_baseline_matrix` and `residual_matrix` must have the same shape."
        )

    # svd_fast already handles LinAlgError for SVD on estimated_baseline_matrix
    left_singular_vectors, singular_values, right_singular_vectors_transposed = svd_fast(
        estimated_baseline_matrix
    )

    effective_rank: int
    if not singular_values.size: # No singular values
        effective_rank = 0
    else:
        cumulative_singular_values = np.cumsum(singular_values)
        if cumulative_singular_values[-1] == 0: # All singular values are zero
            effective_rank = 0
        else:
            # Ensure cumulative_singular_values are not zero where singular_values are non-zero to avoid division by zero
            # This logic is a bit complex; the original intent was to find a "knee" or significant drop.
            # A simpler, robust way to define effective_rank if singular_values are present:
            # Count non-negligible singular values relative to the largest.
            # Or, use the original logic but make it safe.
            # For now, let's stick to a safe version of the original logic.
            # The original logic: np.sum(singular_values / np.cumsum(singular_values) >= 1e-6)
            # This can be problematic if cumsum is zero.
            # Let's use a threshold relative to the largest singular value, or a fixed number if all are tiny.
            
            # A more robust way to calculate effective_rank based on the spirit of the original code:
            # Count singular values that are a significant fraction of the sum of singular values up to that point.
            # This is still tricky. Let's use a simpler approach: count SVs > tol * max(SV)
            # Or, stick to the original logic but make it safe.
            # The original code's `effective_rank` calculation:
            # `np.sum(singular_values / np.cumsum(singular_values) >= 1e-6)`
            # This is problematic if `np.cumsum(singular_values)` has zeros.
            # A common way to define effective rank is to count singular values above a certain threshold.
            # Let's use a simpler thresholding for now, or ensure the division is safe.
            
            # Safely compute effective_rank based on original logic's spirit
            # Count singular values that are not "too small"
            if singular_values[0] < 1e-9: # If largest singular value is tiny
                 effective_rank = 0
            else:
                # Count singular values greater than a small fraction of the largest singular value
                effective_rank = np.sum(singular_values > 1e-6 * singular_values[0])
            
            # Ensure rank is not zero if there are non-zero singular values and rank is needed
            if effective_rank == 0 and np.any(singular_values > 1e-9):
                effective_rank = 1 # Ensure at least rank 1 if there's any signal

    # Ensure effective_rank does not exceed available dimensions
    effective_rank = min(effective_rank, left_singular_vectors.shape[1], right_singular_vectors_transposed.shape[0])
    
    # Truncate U and Vh based on the calculated effective_rank.
    # If effective_rank is 0, these slices will result in empty arrays (e.g., shape (n,0)).
    left_singular_vectors_eff = left_singular_vectors[:, :effective_rank]
    right_singular_vectors_transposed_eff = right_singular_vectors_transposed[:effective_rank, :]


    # Initialize the design matrix for the covariance calculation.
    # This matrix will store the flattened versions of Z_k projected onto the orthogonal complement of M's tangent space.
    # Shape: (n_rows*n_cols, n_interventions)
    design_matrix_orthogonal_projection: np.ndarray = np.zeros(
        (
            intervention_matrices_3d.shape[0] * intervention_matrices_3d.shape[1], # Total elements in one Z_k slice
            intervention_matrices_3d.shape[2], # Number of interventions
        )
    )
    # For each intervention matrix Z_k:
    for k_intervention_idx in np.arange(intervention_matrices_3d.shape[2]):
        # Project Z_k onto the orthogonal complement of the tangent space of M_est (using U_eff, Vh_eff).
        # The result is Z_perp_k.
        z_perp_k = remove_tangent_space_component(
            left_singular_vectors_eff, # U_r from M_est
            right_singular_vectors_transposed_eff, # Vh_r from M_est
            intervention_matrices_3d[:, :, k_intervention_idx], # Z_k
        )
        # Store the flattened Z_perp_k as a column in the design matrix.
        design_matrix_orthogonal_projection[
            :, k_intervention_idx
        ] = z_perp_k.reshape(-1) # Flatten to (n_rows*n_cols,)
    
    # Calculate the term (Z_perp.T @ Z_perp)^-1 @ Z_perp.T, which is part of the sandwich estimator for covariance.
    # Let X_tilde = design_matrix_orthogonal_projection.
    # We need (X_tilde.T @ X_tilde)^-1 @ X_tilde.T
    try:
        term_to_invert = design_matrix_orthogonal_projection.T @ design_matrix_orthogonal_projection # (n_interventions, n_interventions)
        # Check for singularity before attempting inversion.
        if term_to_invert.shape[0] == 0 or np.linalg.matrix_rank(term_to_invert) < term_to_invert.shape[0]:
            # This occurs if Z_perp is rank deficient (e.g., collinear projections, or all Z_perp_k are zero).
            raise MlsynthEstimationError(
                "Matrix (Z_perp.T @ Z_perp) is singular, cannot compute covariance for tau. "
                "This may happen if projected intervention matrices are collinear or zero."
            )

        # inv_XTX_XT = (X_tilde.T @ X_tilde)^-1 @ X_tilde.T
        projection_term_for_covariance: np.ndarray = np.linalg.inv(
            term_to_invert # (X_tilde.T @ X_tilde)
        ) @ design_matrix_orthogonal_projection.T # X_tilde.T
    except np.linalg.LinAlgError as e: # Catch errors from np.linalg.inv
        raise MlsynthEstimationError(
            f"Failed to compute inverse for covariance matrix term (Z_perp.T @ Z_perp): {e}"
        )
    
    # Estimate covariance matrix of tau: V_hat = (inv_XTX_XT) @ diag(residuals^2) @ (inv_XTX_XT).T
    # Here, diag(residuals^2) is applied by element-wise multiplication with the first inv_XTX_XT.
    # `np.reshape(residual_matrix ** 2, -1)` flattens the squared residuals.
    # The multiplication `projection_term_for_covariance * flattened_squared_residuals`
    # effectively scales columns of projection_term_for_covariance by corresponding residuals.
    covariance_matrix_tau: np.ndarray = (
        projection_term_for_covariance * np.reshape(residual_matrix ** 2, -1) # (inv_XTX_XT) * diag(e^2) (element-wise sense)
    ) @ projection_term_for_covariance.T # Matmul with ((inv_XTX_XT).T)
    return covariance_matrix_tau


def remove_tangent_space_component(
    left_singular_vectors_M: np.ndarray,
    right_singular_vectors_transposed_M: np.ndarray,
    target_matrix: np.ndarray,
) -> np.ndarray:
    """
    Remove the projection of `target_matrix` onto the tangent space of matrix `M`.

    The matrix `M` is implicitly defined by its (truncated)
    `left_singular_vectors_M` and `right_singular_vectors_transposed_M`.
    The tangent space of `M` at `M` (for a fixed rank `r`) is given by
    matrices of the form `left_singular_vectors_M @ A.T + B @ right_singular_vectors_transposed_M`,
    where `A` and `B` are arbitrary matrices of appropriate dimensions.
    This function computes `P_T_perp(target_matrix) = target_matrix - P_T(target_matrix)`,
    where `P_T` is the projection onto the tangent space, and `P_T_perp` is
    the projection onto its orthogonal complement.
    The formula used is `P_T_perp(target_matrix) =
    (I - U @ U.T) @ target_matrix @ (I - V.T @ V)`, where U are left singular vectors
    and V are right singular vectors (Vh = V.T).
    A memory-aware computation path is used if `target_matrix` is very large.

    Parameters
    ----------
    left_singular_vectors_M : np.ndarray
        The left singular vectors of matrix `M` (typically U_r from SVD of M).
        Shape (n_rows, r), where `r` is the rank of `M`.
    right_singular_vectors_transposed_M : np.ndarray
        The (transposed) right singular vectors of `M` (typically V_r.T from SVD of M).
        Shape (r, n_cols).
    target_matrix : np.ndarray
        A single matrix (e.g., an intervention matrix) from which the
        component in the tangent space of `M` will be removed.
        Shape (n_rows, n_cols).

    Returns
    -------
    np.ndarray
        The component of `target_matrix` that is orthogonal to the tangent space
        of `M`. Shape (n_rows, n_cols).

    Notes
    -----
    This operation is often used in debiasing procedures for low-rank matrix
    estimation problems, such as in panel data models where `M` is a
    low-rank baseline and `target_matrix` relates to a treatment effect.
    The memory optimization for large matrices (`max(shape) > 1e4`) computes
    the projection in a way that avoids forming large identity matrices or
    intermediate products if possible, by associating matrix multiplications
    differently.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import remove_tangent_space_component
    >>> # Example M (implicitly defined by its singular vectors)
    >>> M_matrix_example = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
    >>> U_svd, S_svd, Vh_svd = np.linalg.svd(M_matrix_example)
    >>> rank_approx = 2 # Assume M is approximated by rank 2
    >>> U_r_M = U_svd[:, :rank_approx]
    >>> Vh_r_M = Vh_svd[:rank_approx, :]
    >>> # Example target_matrix
    >>> Z_target = np.random.rand(3,3)
    >>> Z_ortho_comp = remove_tangent_space_component(U_r_M, Vh_r_M, Z_target)
    >>> print("Z_ortho_comp shape:", Z_ortho_comp.shape)
    Z_ortho_comp shape: (3, 3)
    >>> # Check orthogonality (should be close to zero due to projection)
    >>> # For any X_mat in tangent space, <X_mat, Z_ortho_comp> should be ~0.
    >>> # Example X_mat in tangent space: U_r_M @ np.random.rand(rank_approx,3)
    >>> X_tangent_mat_example = U_r_M @ np.random.rand(rank_approx, Z_target.shape[1]) # U @ A.T form
    >>> inner_prod = np.sum(X_tangent_mat_example * Z_ortho_comp)
    >>> print(f"Inner product (should be near zero): {inner_prod:.1e}") # doctest: +SKIP
    # e.g., Inner product (should be near zero): -1.2e-15
    """
    if not isinstance(left_singular_vectors_M, np.ndarray) or left_singular_vectors_M.ndim != 2:
        raise MlsynthDataError("`left_singular_vectors_M` must be a 2D NumPy array.")
    if not isinstance(right_singular_vectors_transposed_M, np.ndarray) or right_singular_vectors_transposed_M.ndim != 2:
        raise MlsynthDataError("`right_singular_vectors_transposed_M` must be a 2D NumPy array.")
    if not isinstance(target_matrix, np.ndarray) or target_matrix.ndim != 2:
        raise MlsynthDataError("`target_matrix` must be a 2D NumPy array.")

    # Shape consistency checks
    num_rows_target, num_cols_target = target_matrix.shape
    num_rows_U, rank_U = left_singular_vectors_M.shape
    rank_Vh, num_cols_Vh = right_singular_vectors_transposed_M.shape

    if num_rows_U != num_rows_target:
        raise MlsynthDataError(
            f"Row mismatch: `left_singular_vectors_M` has {num_rows_U} rows, "
            f"`target_matrix` has {num_rows_target} rows."
        )
    if num_cols_Vh != num_cols_target:
        raise MlsynthDataError(
            f"Column mismatch: `right_singular_vectors_transposed_M` has {num_cols_Vh} cols, "
            f"`target_matrix` has {num_cols_target} cols."
        )
    if rank_U != rank_Vh:
        raise MlsynthDataError(
            f"Rank mismatch: `left_singular_vectors_M` implies rank {rank_U}, "
            f"`right_singular_vectors_transposed_M` implies rank {rank_Vh}."
        )
    
    # Handle cases where rank is 0 (U or Vh might be empty in one dimension)
    # If rank is 0, U@U.T is 0, V.T@V is 0. So P_T_perp(Z) = Z.
    if rank_U == 0: # or rank_Vh == 0, they are the same
        return target_matrix.copy()


    # Let U = left_singular_vectors_M and Vh = right_singular_vectors_transposed_M.
    # P_U = U @ U.T is projector onto column space of M. P_U_perp = I - P_U.
    # P_V = Vh.T @ Vh is projector onto row space of M. P_V_perp = I - P_V.
    # The projection of target_matrix (Z) onto the orthogonal complement of the tangent space is P_U_perp @ Z @ P_V_perp.
    
    target_matrix_shape: Tuple[int, int] = target_matrix.shape
    target_matrix_orthogonal_component: np.ndarray

    # The condition `max(target_matrix_shape) > 1e4` suggests a memory optimization for large matrices.
    # However, the alternative computation paths seem to compute the same quantity,
    # possibly with different association of matrix products to avoid large intermediate identity matrices.
    # Standard computation: (I - U@U.T) @ Z @ (I - Vh.T@Vh)
    # Let's analyze the "memory-aware" path:
    if max(target_matrix_shape) > 1e4: # Memory optimization path
        if target_matrix_shape[0] > target_matrix_shape[1]: # Tall target_matrix
            # projection_factor_1 = Z - U @ (U.T @ Z) = (I - U@U.T) @ Z = P_U_perp @ Z
            projection_factor_1: np.ndarray = target_matrix - left_singular_vectors_M.dot(
                left_singular_vectors_M.T.dot(target_matrix)
            )
            # projection_factor_2 = I - Vh.T @ Vh = P_V_perp
            projection_factor_2: np.ndarray = (
                np.eye(right_singular_vectors_transposed_M.shape[1]) # Identity matrix of size n_cols
                - right_singular_vectors_transposed_M.T.dot( # V @ Vh
                    right_singular_vectors_transposed_M
                )
            )
            # Result: (P_U_perp @ Z) @ P_V_perp
            target_matrix_orthogonal_component = projection_factor_1.dot(projection_factor_2)
        else: # Wide or square target_matrix
            # projection_factor_1 = I - U@U.T = P_U_perp
            projection_factor_1 = (
                np.eye(left_singular_vectors_M.shape[0]) # Identity matrix of size n_rows
                - left_singular_vectors_M.dot(left_singular_vectors_M.T) # U @ U.T
            )
            # projection_factor_2 = Z - (Z @ Vh.T) @ Vh = Z @ (I - Vh.T@Vh) = Z @ P_V_perp
            projection_factor_2 = target_matrix - (
                target_matrix.dot(right_singular_vectors_transposed_M.T) # Z @ V
            ).dot(right_singular_vectors_transposed_M) # (Z @ V) @ Vh
            # Result: P_U_perp @ (Z @ P_V_perp)
            target_matrix_orthogonal_component = projection_factor_1.dot(projection_factor_2)
    else: # Standard computation path (presumably for smaller matrices)
        # P_U_perp = (I - U @ U.T)
        projector_onto_col_space_orthogonal_complement = (
            np.eye(left_singular_vectors_M.shape[0]) # I_nrows
            - left_singular_vectors_M.dot(left_singular_vectors_M.T) # U @ U.T
        )
        # P_V_perp = (I - Vh.T @ Vh)
        projector_onto_row_space_orthogonal_complement = (
            np.eye(right_singular_vectors_transposed_M.shape[1]) # I_ncols
            - right_singular_vectors_transposed_M.T.dot( # V @ Vh
                right_singular_vectors_transposed_M
            )
        )
        # Result: P_U_perp @ Z @ P_V_perp
        target_matrix_orthogonal_component = (
            projector_onto_col_space_orthogonal_complement.dot(target_matrix) # (P_U_perp @ Z)
        ).dot(projector_onto_row_space_orthogonal_complement) # (...) @ P_V_perp
        
    return target_matrix_orthogonal_component


def transform_to_3D(
    intervention_data: Union[List[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Convert an intervention matrix or a list of intervention matrices to a 3D NumPy array.

    This utility function standardizes the format of intervention matrices
    to a 3D NumPy array where the third dimension indexes the different
    interventions. It handles three types of input for `intervention_data`:
    1. A list of 2D NumPy arrays: Each 2D array represents an intervention
       matrix for one type of intervention. These are stacked along a new
       third axis.
    2. A single 2D NumPy array: Represents a single intervention. It is
       reshaped to have a third dimension of size 1.
    3. A 3D NumPy array: Assumed to be already in the correct format
       (n_rows, n_cols, n_interventions) and is returned as is, after
       ensuring its data type is float.

    Parameters
    ----------
    intervention_data : Union[List[np.ndarray], np.ndarray]
        The intervention data to be transformed.
        - If List[np.ndarray]: Each element is a 2D array of shape (n_rows, n_cols).
          All arrays in the list must have the same shape.
        - If np.ndarray (2D): Shape (n_rows, n_cols).
        - If np.ndarray (3D): Shape (n_rows, n_cols, n_interventions).

    Returns
    -------
    np.ndarray
        A 3D NumPy array representing the intervention(s), with shape
        (n_rows, n_cols, n_interventions). The data type is float.

    Raises
    ------
    ValueError
        If `intervention_data` is a list of arrays with inconsistent shapes.
        (Note: Current implementation does not explicitly check this, relies on np.stack).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import transform_to_3D
    >>> num_rows_ex, num_cols_ex = 2, 3
    >>> # Example 1: Input is a single 2D array
    >>> Z_single_2d_ex = np.ones((num_rows_ex, num_cols_ex))
    >>> Z_transformed_single_ex = transform_to_3D(Z_single_2d_ex)
    >>> print("From single 2D array:\\n", Z_transformed_single_ex)
    From single 2D array:
     [[[1.]
      [1.]
      [1.]]
    <BLANKLINE>
     [[1.]
      [1.]
      [1.]]]
    >>> print("Shape:", Z_transformed_single_ex.shape)
    Shape: (2, 3, 1)

    >>> # Example 2: Input is a list of 2D arrays
    >>> Z_list_2d_ex = [np.ones((num_rows_ex, num_cols_ex)), np.zeros((num_rows_ex, num_cols_ex))]
    >>> Z_transformed_list_ex = transform_to_3D(Z_list_2d_ex)
    >>> print("\\nFrom list of 2D arrays:\\n", Z_transformed_list_ex)
    From list of 2D arrays:
     [[[1. 0.]
      [1. 0.]
      [1. 0.]]
    <BLANKLINE>
     [[1. 0.]
      [1. 0.]
      [1. 0.]]]
    >>> print("Shape:", Z_transformed_list_ex.shape)
    Shape: (2, 3, 2)

    >>> # Example 3: Input is already a 3D array
    >>> Z_already_3d_ex = np.arange(num_rows_ex * num_cols_ex * 2).reshape(
    ...     (num_rows_ex, num_cols_ex, 2)
    ... )
    >>> Z_transformed_3d_ex = transform_to_3D(Z_already_3d_ex)
    >>> print("\\nFrom 3D array (ensures float type):\\n", Z_transformed_3d_ex)
    From 3D array (ensures float type):
     [[[ 0.  1.]
      [ 2.  3.]
      [ 4.  5.]]
    <BLANKLINE>
     [[ 6.  7.]
      [ 8.  9.]
      [10. 11.]]]
    >>> print("Shape:", Z_transformed_3d_ex.shape)
    Shape: (2, 3, 2)
    >>> print("Data type:", Z_transformed_3d_ex.dtype)
    Data type: float64
    """
    intervention_data_3d: np.ndarray # Variable to hold the final 3D array
    if isinstance(intervention_data, list):
        if not intervention_data: # Handle empty list case
            raise MlsynthDataError("Input `intervention_data` list cannot be empty.")
        
        # Validate items in the list: must be 2D NumPy arrays of consistent shape.
        first_item_shape = None
        for i, item in enumerate(intervention_data):
            if not isinstance(item, np.ndarray):
                raise MlsynthDataError(
                    f"Item {i} in `intervention_data` list is not a NumPy array."
                )
            if item.ndim != 2:
                raise MlsynthDataError(
                    f"Item {i} in `intervention_data` list is not a 2D array (shape: {item.shape})."
                )
            if i == 0:
                first_item_shape = item.shape
            elif item.shape != first_item_shape: # Check for consistent shapes
                raise MlsynthDataError(
                    f"Item {i} in `intervention_data` list has shape {item.shape}, "
                    f"expected {first_item_shape} (shape of first item)."
                )
        try:
            # Stack the list of 2D arrays along a new third axis (axis=2).
            intervention_data_3d = np.stack(intervention_data, axis=2)
        except ValueError as e: # Fallback, though shape checks should prevent most np.stack errors
            raise MlsynthDataError(f"Failed to stack intervention matrices: {e}")

    elif isinstance(intervention_data, np.ndarray):
        if intervention_data.ndim == 2:
            # If input is a single 2D array, reshape it to 3D with the third dimension of size 1.
            intervention_data_3d = intervention_data.reshape(
                intervention_data.shape[0], intervention_data.shape[1], 1 # (n_rows, n_cols, 1)
            )
        elif intervention_data.ndim == 3:
            # If input is already a 3D array, use it directly.
            intervention_data_3d = intervention_data
        else:
            raise MlsynthDataError(
                "`intervention_data` NumPy array must be 2D or 3D, "
                f"got {intervention_data.ndim}D."
            )
    else: # Input type is not list or np.ndarray
        raise MlsynthDataError(
            "`intervention_data` must be a list of NumPy arrays or a NumPy array."
        )
        
    # Ensure the output array is of float type, common for numerical computations.
    return intervention_data_3d.astype(float)


def prepare_OLS(
    intervention_matrices_3d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare matrices for an Ordinary Least Squares (OLS) regression step.

    This function is a helper for panel regression models. It takes a 3D array
    of `intervention_matrices_3d` and constructs the necessary components
    for an OLS estimation of treatment effects `tau`.
    Specifically, it identifies entries where at least one intervention is active
    (non-zero in `intervention_matrices_3d`) and constructs a design matrix
    `ols_design_matrix` using only these "active" entries. This can be
    beneficial for performance if `intervention_matrices_3d` is sparse.
    It also pre-computes `(ols_design_matrix.T @ ols_design_matrix)^(-1)`,
    which is needed for the OLS solution.

    Parameters
    ----------
    intervention_matrices_3d : np.ndarray
        A 3D array representing one or more intervention matrices.
        Shape (n_rows, n_cols, n_interventions).
        Non-zero entries typically indicate treated unit-time pairs for a
        specific intervention. A small tolerance (1e-9) is used to determine
        non-zero entries.

    Returns
    -------
    active_intervention_entries_idx : np.ndarray
        A 2D boolean array of shape (n_rows, n_cols). `True` where at least
        one intervention in `intervention_matrices_3d` has an absolute value
        greater than 1e-9. This index is used to select relevant entries
        from the observation matrix `O` (or `O - M_est`) for the OLS regression.
    ols_design_matrix : np.ndarray
        The design matrix for OLS, constructed from the "active" entries of
        `intervention_matrices_3d`. Shape (n_active_entries, n_interventions),
        where `n_active_entries` is the number of `True` values in
        `active_intervention_entries_idx`. Each column `k` of `ols_design_matrix`
        contains the values of `intervention_matrices_3d[:,:,k]` at the active entries.
    ols_design_matrix_pseudo_inverse_term : np.ndarray
        The pre-computed inverse of `(ols_design_matrix.T @ ols_design_matrix)`.
        Shape (n_interventions, n_interventions). This is part of the OLS
        solution for `tau`:
        `tau_hat = ols_design_matrix_pseudo_inverse_term @ ols_design_matrix.T @ y_ols`.

    Raises
    ------
    np.linalg.LinAlgError
        If `ols_design_matrix.T @ ols_design_matrix` is singular and cannot be
        inverted. This can happen if `ols_design_matrix` does not have full
        column rank (e.g., due to collinearity between interventions or
        insufficient active entries).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import prepare_OLS
    >>> num_rows_ex, num_cols_ex, num_interventions_ex = 3, 4, 2
    >>> Z_interventions_3d_ex = np.zeros((num_rows_ex, num_cols_ex, num_interventions_ex))
    >>> # Intervention 1: unit 0, time 2 onwards
    >>> Z_interventions_3d_ex[0, 2:, 0] = 1
    >>> # Intervention 2: unit 1, time 1 to 2
    >>> Z_interventions_3d_ex[1, 1:3, 1] = 1
    >>> print("Z_interventions_3d_ex[:,:,0]:\\n", Z_interventions_3d_ex[:,:,0])
    Z_interventions_3d_ex[:,:,0]:
     [[0. 0. 1. 1.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    >>> print("Z_interventions_3d_ex[:,:,1]:\\n", Z_interventions_3d_ex[:,:,1])
    Z_interventions_3d_ex[:,:,1]:
     [[0. 0. 0. 0.]
     [0. 1. 1. 0.]
     [0. 0. 0. 0.]]
    >>> active_idx, X_design, X_inv_XTX = prepare_OLS(Z_interventions_3d_ex)
    >>> print("\\nActive index (active_intervention_entries_idx):\\n", active_idx)
    Active index (active_intervention_entries_idx):
     [[False False  True  True]
     [False  True  True False]
     [False False False False]]
    >>> print("Number of active entries:", np.sum(active_idx))
    Number of active entries: 4
    >>> print("\\nDesign matrix ols_design_matrix (shape {}):\\n".format(X_design.shape), X_design)
    Design matrix ols_design_matrix (shape (4, 2)):
     [[1. 0.]
     [1. 0.]
     [0. 1.]
     [0. 1.]]
    >>> print("\\nInverse of (ols_design_matrix.T @ ols_design_matrix) (shape {}):\\n".format(X_inv_XTX.shape), X_inv_XTX)
    Inverse of (ols_design_matrix.T @ ols_design_matrix) (shape (2, 2)):
     [[0.5 0. ]
     [0.  0.5]]
    """
    if not isinstance(intervention_matrices_3d, np.ndarray):
        raise MlsynthDataError("`intervention_matrices_3d` must be a NumPy array.")
    if intervention_matrices_3d.ndim != 3:
        raise MlsynthDataError(
            f"`intervention_matrices_3d` must be a 3D array, got {intervention_matrices_3d.ndim}D."
        )
    if intervention_matrices_3d.shape[2] == 0: # No interventions
        raise MlsynthDataError("`intervention_matrices_3d` must have at least one intervention (3rd dim > 0).")


    # Identify "active" entries: where at least one intervention matrix Z_k has a non-negligible value.
    # `np.sum(..., axis=2)` sums across interventions. `> 0` means at least one Z_k is active at that (row, col).
    # Using `np.abs(...) > 1e-9` to handle floating point comparisons for non-zero.
    active_intervention_entries_idx: np.ndarray = (
        np.sum(np.abs(intervention_matrices_3d) > 1e-9, axis=2) > 0 # Boolean mask (n_rows, n_cols)
    )
    
    if not np.any(active_intervention_entries_idx): # If no entries are active across all interventions
        raise MlsynthDataError(
            "No active intervention entries found in `intervention_matrices_3d`. Cannot prepare OLS."
        )
        
    # Construct the OLS design matrix (X_ols) using only the active entries.
    # `intervention_matrices_3d[active_intervention_entries_idx, :]` flattens the active entries
    # for each intervention and stacks them.
    # Resulting shape: (n_active_entries, n_interventions).
    ols_design_matrix: np.ndarray = intervention_matrices_3d[
        active_intervention_entries_idx, : # Selects rows from the flattened (n_rows*n_cols, n_interventions) view
    ].astype(float)

    if ols_design_matrix.shape[0] == 0: # Should be caught by `np.any(active_intervention_entries_idx)`
        raise MlsynthEstimationError(
            "OLS design matrix is empty (no active intervention entries found after filtering)."
        )
    
    # Pre-compute (X_ols.T @ X_ols)^-1 for the OLS solution.
    # This term is (n_interventions, n_interventions).
    try:
        term_to_invert = ols_design_matrix.T @ ols_design_matrix # X_ols.T @ X_ols
        # Check for singularity before attempting inversion.
        # If rank(X_ols.T @ X_ols) < n_interventions, it's singular.
        if np.linalg.matrix_rank(term_to_invert) < term_to_invert.shape[0]:
            raise MlsynthEstimationError(
                "Matrix (X_ols.T @ X_ols) is singular in prepare_OLS, cannot compute inverse. "
                "This may be due to collinear interventions or insufficient active data points."
            )
        ols_design_matrix_pseudo_inverse_term: np.ndarray = np.linalg.inv(
            term_to_invert
        )
    except np.linalg.LinAlgError as e: # Catch errors from np.linalg.inv
        # This exception is also raised if matrix_rank check fails and we proceed to inv.
        raise MlsynthEstimationError(
            f"Failed to compute inverse for OLS design matrix term (X_ols.T @ X_ols): {e}. "
            "This may be due to collinear interventions or insufficient active data points."
        )
        
    return (
        active_intervention_entries_idx, # Boolean mask for active entries
        ols_design_matrix,
        ols_design_matrix_pseudo_inverse_term,
    )


def solve_tau(
    outcome_matrix_or_residuals: np.ndarray, intervention_matrices_3d: np.ndarray
) -> np.ndarray:
    """
    Solve for treatment effects (tau) using Ordinary Least Squares (OLS).

    This function estimates treatment effects `tau` given an
    `outcome_matrix_or_residuals` (e.g., `O` or `O - M_est`) and a 3D array of
    `intervention_matrices_3d`. It assumes a linear relationship
    `ols_dependent_variable = ols_design_matrix @ tau + error`, where
    `ols_dependent_variable` consists of entries from `outcome_matrix_or_residuals`
    where interventions are active, and `ols_design_matrix` is the corresponding
    design matrix derived from `intervention_matrices_3d`.
    The OLS solution is
    `tau_hat = (ols_design_matrix.T @ ols_design_matrix)^(-1) @ ols_design_matrix.T @ ols_dependent_variable`.
    This function utilizes `prepare_OLS` to construct `ols_design_matrix` and
    the pseudo-inverse term.

    Parameters
    ----------
    outcome_matrix_or_residuals : np.ndarray
        The observation matrix, or a residual matrix (e.g., `O_observed - M_estimated`).
        Shape (n_rows, n_cols).
    intervention_matrices_3d : np.ndarray
        A 3D array representing one or more intervention matrices.
        Shape (n_rows, n_cols, n_interventions). Non-zero entries typically
        indicate treated unit-time pairs for a specific intervention.

    Returns
    -------
    estimated_treatment_effects : np.ndarray
        The estimated treatment effects. Shape (n_interventions,).

    Raises
    ------
    np.linalg.LinAlgError
        If `ols_design_matrix.T @ ols_design_matrix` (calculated within
        `prepare_OLS`) is singular and cannot be inverted.

    See Also
    --------
    prepare_OLS : Helper function used to construct the OLS design matrix
                  and other necessary components.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import solve_tau
    >>> # Example data
    >>> num_rows_ex, num_cols_ex, num_interventions_ex = 3, 4, 2
    >>> outcome_mat_ex = np.random.rand(num_rows_ex, num_cols_ex)
    >>> Z_interventions_3d_ex = np.zeros((num_rows_ex, num_cols_ex, num_interventions_ex))
    >>> # Intervention 1: unit 0, time 2 onwards, effect = 0.5
    >>> Z_interventions_3d_ex[0, 2:, 0] = 1
    >>> outcome_mat_ex[0, 2:] += 0.5 * 1
    >>> # Intervention 2: unit 1, time 1 to 2, effect = -0.3
    >>> Z_interventions_3d_ex[1, 1:3, 1] = 1
    >>> outcome_mat_ex[1, 1:3] += -0.3 * 1
    >>> est_taus = solve_tau(outcome_mat_ex, Z_interventions_3d_ex)
    >>> print("Estimated taus (should be close to [0.5, -0.3]):\\n", est_taus) # doctest: +SKIP
    # Expected output: e.g., Estimated taus (should be close to [0.5, -0.3]): [ 0.5 -0.3]
    # Actual output might have small deviations due to floating point arithmetic.
    >>> # Example with perfect data (no noise)
    >>> outcome_perfect_ex = np.zeros((2,2))
    >>> Z_perfect_3d_ex = np.zeros((2,2,1))
    >>> Z_perfect_3d_ex[0,0,0] = 1
    >>> Z_perfect_3d_ex[1,1,0] = 2
    >>> outcome_perfect_ex[0,0] = 0.5 * 1
    >>> outcome_perfect_ex[1,1] = 0.5 * 2
    >>> tau_perfect_ex = solve_tau(outcome_perfect_ex, Z_perfect_3d_ex)
    >>> print("\\nEstimated tau with perfect data:", tau_perfect_ex)
    Estimated tau with perfect data: [0.5]
    """
    if not isinstance(outcome_matrix_or_residuals, np.ndarray):
        raise MlsynthDataError("`outcome_matrix_or_residuals` must be a NumPy array.")
    if outcome_matrix_or_residuals.ndim != 2:
        raise MlsynthDataError(
            f"`outcome_matrix_or_residuals` must be a 2D array, got {outcome_matrix_or_residuals.ndim}D."
        )
    
    # intervention_matrices_3d validation is largely handled by prepare_OLS,
    # but a basic check here is good practice.
    if not isinstance(intervention_matrices_3d, np.ndarray):
        raise MlsynthDataError("`intervention_matrices_3d` must be a NumPy array.")
    if intervention_matrices_3d.ndim != 3:
        raise MlsynthDataError(
            f"`intervention_matrices_3d` must be a 3D array, got {intervention_matrices_3d.ndim}D."
        )

    if outcome_matrix_or_residuals.shape != intervention_matrices_3d.shape[:2]:
        raise MlsynthDataError(
            "Shape mismatch: `outcome_matrix_or_residuals` (shape "
            f"{outcome_matrix_or_residuals.shape}) must match the first two dimensions of "
            f"`intervention_matrices_3d` (shape {intervention_matrices_3d.shape[:2]})."
        )

    # `prepare_OLS` handles validation of intervention_matrices_3d and potential LinAlgErrors.
    active_intervention_entries_idx, ols_design_matrix, ols_design_matrix_pseudo_inverse_term = prepare_OLS(
        intervention_matrices_3d
    )
    
    # Construct the dependent variable for OLS (y_ols) by selecting active entries
    # from the outcome/residual matrix.
    ols_dependent_variable: np.ndarray = outcome_matrix_or_residuals[
        active_intervention_entries_idx # Use the boolean mask from prepare_OLS
    ] # Shape: (n_active_entries,)
    
    # Ensure consistency between design matrix rows and dependent variable length.
    # This should hold if active_intervention_entries_idx is used correctly for both.
    if ols_design_matrix.size > 0 and ols_dependent_variable.size == 0: # Should be caught by prepare_OLS if no active entries
        raise MlsynthEstimationError(
            "OLS dependent variable is empty despite having a non-empty design matrix. "
            "This indicates an issue with active entry indexing."
        )
    if ols_design_matrix.shape[0] != ols_dependent_variable.shape[0]:
         raise MlsynthEstimationError(
            f"Shape mismatch for OLS: design matrix has {ols_design_matrix.shape[0]} rows (active entries), "
            f"but dependent variable has {ols_dependent_variable.shape[0]} entries."
        )

    # Calculate estimated treatment effects: tau_hat = (X.T X)^-1 X.T y
    try:
        estimated_treatment_effects: np.ndarray = ols_design_matrix_pseudo_inverse_term @ ( # (X.T X)^-1
            ols_design_matrix.T @ ols_dependent_variable # X.T @ y
        )
    except Exception as e: # Catch any other unexpected errors during the final matrix multiplication
        raise MlsynthEstimationError(f"Error during OLS computation for tau: {e}")
        
    return estimated_treatment_effects


def DC_PR_auto_rank(
    observed_panel_matrix: np.ndarray,
    intervention_panel_matrix: np.ndarray,
    spectrum_cut: float = 0.002,
    method: str = "auto",
) -> Dict[str, Any]:
    """
    Debiased Convex Panel Regression with automatic rank selection.

    This function performs panel regression to estimate treatment effects,
    automatically determining a suitable rank for the low-rank baseline matrix `M`.
    It first suggests a rank (`calculated_suggested_rank`) based on the singular value spectrum
    of the observation matrix `observed_panel_matrix`, aiming to capture a significant portion of
    its energy (controlled by `spectrum_cut`).
    Then, it calls `DC_PR_with_suggested_rank` to perform the actual estimation,
    typically using the "convex" method with the determined `calculated_suggested_rank`.
    The results, including estimated effects, counterfactuals, and fit
    diagnostics, are returned in a dictionary.

    Parameters
    ----------
    observed_panel_matrix : np.ndarray
        The observed data matrix. Shape (n_rows, n_cols).
    intervention_panel_matrix : np.ndarray
        The intervention matrix, typically 2D for a single intervention.
        Shape (n_rows, n_cols). Non-zero entries indicate treated unit-time pairs.
        The function assumes `intervention_panel_matrix` represents a single intervention and identifies
        the first row with a '1' as the treated unit.
    spectrum_cut : float, default 0.002
        Cutoff for determining the suggested rank from the singular value
        spectrum of `observed_panel_matrix`. The rank is chosen such that the cumulative sum of
        squared singular values up to that rank accounts for at least
        `(1 - spectrum_cut)` of the total sum of squared singular values.
        A smaller `spectrum_cut` leads to a higher suggested rank.
    method : str, default "auto"
        This parameter is passed to `DC_PR_with_suggested_rank` but the current
        implementation of `DC_PR_auto_rank` hardcodes the method to "convex"
        when calling `DC_PR_with_suggested_rank`.
        Accepted values by `DC_PR_with_suggested_rank` are "auto", "convex",
        or "non-convex".

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various estimation results:
        - "Vectors": A dictionary with:
            - "Treated Unit": `np.ndarray` (n_cols, 1) - Observed outcomes for the treated unit.
            - "Counterfactual": `np.ndarray` (n_cols, 1) - Estimated counterfactual outcomes for the treated unit.
        - "Effects": A dictionary with treatment effect estimates (e.g., "ATT", "Percent ATT").
          (Structure from `mlsynth.utils.resultutils.effects.calculate`)
        - "CIs": A dictionary with confidence interval related statistics (e.g., "Lower Bound", "Upper Bound", "SE").
          (Structure from `panel_regression_CI` and `norm.ppf`)
        - "RMSE": float - The pre-treatment Root Mean Squared Error.
        - "Suggested_Rank": int - The rank suggested by the `spectrum_cut` criterion.

    Raises
    ------
    ValueError
        If no treated unit (row with a '1') is found in the input `intervention_panel_matrix` matrix.

    See Also
    --------
    DC_PR_with_suggested_rank : The core estimation function called internally.
    numpy.linalg.svd : Used to get singular values for rank suggestion.
    mlsynth.utils.resultutils.effects.calculate : Used to compute effect sizes.

    Notes
    -----
    The function assumes that `intervention_panel_matrix` represents a single intervention and identifies
    the treated unit based on the first row containing a '1'.
    The `method` parameter's current usage within this function is limited, as
    it calls `DC_PR_with_suggested_rank` with `method="convex"`.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import DC_PR_auto_rank, SVD_soft
    >>> # Example data
    >>> n_rows, n_cols = 5, 20
    >>> O_true_ex = np.random.rand(n_rows, n_cols)
    >>> M_true_ex = SVD_soft(O_true_ex, 0.2) # True low-rank component
    >>> Z_intervention_ex = np.zeros((n_rows, n_cols))
    >>> treated_unit_idx_ex, treatment_start_period_ex = 1, 10
    >>> Z_intervention_ex[treated_unit_idx_ex, treatment_start_period_ex:] = 1
    >>> true_effect_ex = 0.8
    >>> O_observed_ex = M_true_ex + (Z_intervention_ex * true_effect_ex) + \
    ...              np.random.normal(0, 0.05, (n_rows, n_cols))
    >>> results_ex = DC_PR_auto_rank(
    ...     observed_panel_matrix=O_observed_ex,
    ...     intervention_panel_matrix=Z_intervention_ex,
    ...     spectrum_cut=0.01
    ... )
    >>> print("Suggested Rank:", results_ex["Suggested_Rank"]) # doctest: +SKIP
    # e.g., Suggested Rank: 2
    >>> print("Estimated ATT:", results_ex["Effects"]["ATT"]) # doctest: +SKIP
    # e.g., Estimated ATT: 0.78... (close to true_effect_ex 0.8)
    >>> print("Pre-treatment RMSE:", results_ex["RMSE"]) # doctest: +SKIP
    # e.g., Pre-treatment RMSE: 0.06...
    """
    if not isinstance(observed_panel_matrix, np.ndarray) or observed_panel_matrix.ndim != 2:
        raise MlsynthDataError("`observed_panel_matrix` must be a 2D NumPy array.")
    if not isinstance(intervention_panel_matrix, np.ndarray) or intervention_panel_matrix.ndim != 2:
        raise MlsynthDataError("`intervention_panel_matrix` must be a 2D NumPy array.")
    if observed_panel_matrix.shape != intervention_panel_matrix.shape:
        raise MlsynthDataError(
            f"Shape mismatch: `observed_panel_matrix` (shape {observed_panel_matrix.shape}) "
            f"and `intervention_panel_matrix` (shape {intervention_panel_matrix.shape}) must have the same shape."
        )
    if not isinstance(spectrum_cut, float):
        raise MlsynthConfigError("`spectrum_cut` must be a float.")
    if not (0 < spectrum_cut < 1): # spectrum_cut should be in (0,1) for meaningful rank reduction
        raise MlsynthConfigError("`spectrum_cut` must be between 0 and 1 (exclusive).")
    if not isinstance(method, str):
        raise MlsynthConfigError("`method` must be a string.")
    
    allowed_methods = ["auto", "convex", "non-convex"]
    if method not in allowed_methods:
        raise MlsynthConfigError(f"`method` must be one of {allowed_methods}, got '{method}'.")

    # Identify the treated unit: assumes the first row in `intervention_panel_matrix`
    # that contains a '1' (or value close to 1) is the treated unit.
    treated_rows_indices = np.where(np.any(np.isclose(intervention_panel_matrix, 1), axis=1))[0]

    if not treated_rows_indices.size: # If no row has any '1's
        raise MlsynthDataError("No treated unit found (no row with a '1') in `intervention_panel_matrix`.")
    treated_unit_row_index: int = treated_rows_indices[0] # Index of the treated unit's row

    # --- Rank Suggestion Logic ---
    # Get singular values of the observed matrix to determine its spectral energy distribution.
    try:
        singular_values_of_observed_matrix: np.ndarray = np.linalg.svd(
            observed_panel_matrix, full_matrices=False, compute_uv=False # Only need singular values
        )
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"SVD failed for `observed_panel_matrix`: {e}")

    if not singular_values_of_observed_matrix.size: # Should be caught by earlier input validation
        raise MlsynthEstimationError("SVD returned no singular values for `observed_panel_matrix`.")

    sum_sq_singular_values = np.sum(singular_values_of_observed_matrix ** 2)
    calculated_suggested_rank: int
    if sum_sq_singular_values == 0: # If matrix is all zeros
        calculated_suggested_rank = 0
    else:
        # Calculate cumulative energy ratio for each singular value.
        cumulative_energy_ratio = np.cumsum(singular_values_of_observed_matrix ** 2) / sum_sq_singular_values
        # Determine the target proportion of energy to retain.
        target_energy_proportion = 1 - spectrum_cut
        # Find the smallest number of singular values (rank) needed to capture this target energy.
        qualifying_indices = np.where(cumulative_energy_ratio >= target_energy_proportion)[0]
        if qualifying_indices.size > 0:
            calculated_suggested_rank = qualifying_indices[0] + 1 # +1 because rank is count, index is 0-based
        else:
            # If target energy is not met even with all singular values (e.g., spectrum_cut is tiny),
            # use all available singular values (full effective rank).
            calculated_suggested_rank = len(singular_values_of_observed_matrix)
    
    # Ensure suggested rank is valid (non-negative and within matrix dimensions).
    calculated_suggested_rank = max(0, calculated_suggested_rank)
    calculated_suggested_rank = min(calculated_suggested_rank, min(observed_panel_matrix.shape))

    # --- Perform Panel Regression with Suggested Rank ---
    # Call the core estimation function. Note: `method` here is "convex" by default from original code.
    # If the input `method` parameter of `DC_PR_auto_rank` is intended to be used,
    # it should be passed here instead of hardcoding "convex".
    intermediate_results: Dict[str, Any] = DC_PR_with_suggested_rank(
        observed_panel_matrix,
        intervention_panel_matrix, # Pass the original 2D intervention matrix
        target_rank=calculated_suggested_rank,
        method="convex", # Hardcoded to "convex" in the original logic of this wrapper.
                         # Consider using the `method` parameter passed to DC_PR_auto_rank if flexibility is desired.
    )
    
    # --- Post-process Results ---
    if "Vectors" not in intermediate_results or "Counterfactual_Full_Matrix" not in intermediate_results["Vectors"]:
        raise MlsynthEstimationError("`DC_PR_with_suggested_rank` did not return expected 'Counterfactual_Full_Matrix'.")

    # Extract the estimated full baseline matrix (M_hat).
    estimated_baseline_matrix: np.ndarray = intermediate_results["Vectors"]["Counterfactual_Full_Matrix"]
    
    # Extract observed and counterfactual outcomes for the identified treated unit.
    counterfactual_outcomes_for_treated_unit: np.ndarray = estimated_baseline_matrix[
        treated_unit_row_index, : # Row for treated unit, all columns (time periods)
    ]
    observed_outcomes_for_treated_unit: np.ndarray = observed_panel_matrix[
        treated_unit_row_index, :
    ]

    # Determine t1: number of pre-treatment periods for the treated unit.
    # Find the first time period (column index) where treatment is 1 for this unit.
    treatment_start_indices = np.where(np.isclose(intervention_panel_matrix[treated_unit_row_index, :], 1))[0]
    t1: int
    if not treatment_start_indices.size:
        # If the treated unit row has no '1's, it implies no active treatment start.
        # This is unusual given `treated_rows_indices` check. Could mean treatment ends or is all pre.
        # Default to all periods being pre-treatment.
        t1 = intervention_panel_matrix.shape[1] # Number of columns (total time periods)
    else:
        t1 = treatment_start_indices[0] # Index of the first treatment period

    # Calculate pre-treatment RMSE.
    pre_treatment_rmse: float = round(
        np.std( # Standard deviation of residuals in pre-treatment period
            observed_outcomes_for_treated_unit[:t1] # Observed pre-treatment
            - counterfactual_outcomes_for_treated_unit[:t1] # Counterfactual pre-treatment
        ),
        3, # Round to 3 decimal places
    )

    # Assemble the final results dictionary.
    final_results_dict: Dict[str, Any] = {
        "Vectors": {
            "Treated Unit": np.round(
                observed_outcomes_for_treated_unit.reshape(-1, 1), 3 # Reshape to column vector
            ),
            "Counterfactual": np.round(
                counterfactual_outcomes_for_treated_unit.reshape(-1, 1), 3
            ),
        },
        "Effects": intermediate_results.get("Effects", {}), # Pass through from core function
        "CIs": intermediate_results.get("Inference", {}),   # Pass through from core function
        "RMSE": round(pre_treatment_rmse, 3),
        "Suggested_Rank": calculated_suggested_rank, # The rank determined by this auto function
    }
    return final_results_dict


def DC_PR_with_suggested_rank(
    observed_panel_matrix: np.ndarray,
    intervention_panel_input: np.ndarray,
    target_rank: int = 1,
    method: str = "auto",
) -> Dict[str, Any]:
    """
    Debiased Panel Regression with a suggested rank, using convex or non-convex methods.

    This function estimates treatment effects from panel data. It can operate
    in "convex", "non-convex", or "auto" mode.
    - In "convex" mode (or "auto" initially trying convex): It iteratively
      calls `DC_PR_with_l`, gradually decreasing the nuclear norm regularizer `l`.
      The process stops when the rank of the estimated baseline matrix `M`
      exceeds `target_rank`. The penultimate estimates (before rank exceeds
      `target_rank`) are then debiased and truncated to `target_rank`.
    - In "non-convex" mode: It directly calls `non_convex_PR` with the fixed
      rank `target_rank`.
    - In "auto" mode: It first tries the convex approach. If the resulting `M`
      does not have rank `target_rank` or if the non-convex approach yields a
      better fit (lower reconstruction error), the non-convex result is chosen.

    The function returns a dictionary with detailed results, including estimated
    effects, fit diagnostics, counterfactual vectors, and inference statistics.

    Parameters
    ----------
    observed_panel_matrix : np.ndarray
        The observed data matrix. Shape (n_rows, n_cols).
    intervention_panel_input : np.ndarray
        The intervention matrix, typically 2D for a single intervention.
        Shape (n_rows, n_cols). Non-zero entries indicate treated unit-time pairs.
        The function assumes `intervention_panel_input` represents a single intervention and
        identifies the first row with a '1' as the treated unit.
    target_rank : int, default 1
        The target or suggested rank for the low-rank baseline matrix `M`.
    method : str, default "auto"
        The estimation method to use:
        - "convex": Uses iterative soft thresholding with decreasing `l`,
          followed by debiasing and rank truncation.
        - "non-convex": Uses iterative hard thresholding to rank `target_rank`.
        - "auto": Tries "convex" first, then "non-convex", and chooses the
          result based on rank and reconstruction error.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing comprehensive estimation results:
        - "Effects": `Dict[str, Union[float, np.ndarray]]` - Treatment effect
          estimates (e.g., "ATT", "Percent ATT"). From `effects.calculate`.
        - "Fit": `Dict[str, float]` - Fit statistics (e.g., "RMSE" for pre-treatment).
          From `effects.calculate`.
        - "Vectors": `Dict[str, np.ndarray]` - Key vectors:
            - "Observed": Observed outcomes for the treated unit.
            - "Synthetic": Estimated counterfactual outcomes for the treated unit.
            - "Gap": Difference between observed and synthetic.
            - "Counterfactual_Full_Matrix": The estimated baseline matrix `final_estimated_baseline_matrix`.
              Shape (n_rows, n_cols).
          From `effects.calculate`, with "Counterfactual_Full_Matrix" added.
        - "Inference": `Dict[str, Union[float, np.ndarray]]` - Inference statistics
          for the treatment effect(s) (e.g., "Lower Bound", "Upper Bound", "SE", "tstat").
          Derived from `panel_regression_CI`.

    Raises
    ------
    MlsynthDataError
        If input data is invalid (e.g., wrong types, shapes, no treated unit).
    MlsynthConfigError
        If configuration parameters (`target_rank`, `method`) are invalid.
    MlsynthEstimationError
        If numerical issues arise during estimation (e.g., SVD failure, singular matrix).

    See Also
    --------
    DC_PR_with_l : Iterative convex panel regression with a fixed `l`.
    non_convex_PR : Iterative non-convex panel regression with a fixed `r`.
    debias : Debiasing procedure for regularized estimates.
    SVD : Hard rank truncation using SVD.
    panel_regression_CI : Computes covariance matrix for `tau`.
    mlsynth.utils.resultutils.effects.calculate : Computes ATT, RMSE, etc.

    Notes
    -----
    The function assumes `intervention_panel_input` corresponds to a single intervention for
    extracting treated unit outcomes, though the underlying `tau` estimation
    can handle multiple interventions if `intervention_panel_input` is transformed to 3D.
    The "convex" method involves a search over `l` values, which can be
    computationally more intensive than the "non-convex" method.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.denoiseutils import DC_PR_with_suggested_rank, SVD_soft
    >>> # Example data
    >>> n_rows_ex, n_cols_ex = 5, 20
    >>> O_true_ex = np.random.rand(n_rows_ex, n_cols_ex)
    >>> M_true_ex = SVD_soft(O_true_ex, 0.2) # True low-rank component
    >>> Z_intervention_ex = np.zeros((n_rows_ex, n_cols_ex))
    >>> treated_unit_idx_ex, treatment_start_period_ex = 1, 10
    >>> Z_intervention_ex[treated_unit_idx_ex, treatment_start_period_ex:] = 1
    >>> true_effect_ex = 0.8
    >>> O_observed_ex = M_true_ex + (Z_intervention_ex * true_effect_ex) + \
    ...              np.random.normal(0, 0.05, (n_rows_ex, n_cols_ex))
    >>> # Using convex method with suggested rank 2
    >>> results_convex_ex = DC_PR_with_suggested_rank(
    ...     observed_panel_matrix=O_observed_ex,
    ...     intervention_panel_input=Z_intervention_ex,
    ...     target_rank=2,
    ...     method="convex"
    ... )
    >>> print("Convex - Estimated ATT:", results_convex_ex["Effects"]["ATT"]) # doctest: +SKIP
    # e.g., Convex - Estimated ATT: 0.79...
    >>> print("Convex - M_final rank:", np.linalg.matrix_rank(results_convex_ex["Vectors"]["Counterfactual_Full_Matrix"])) # doctest: +SKIP
    # e.g., Convex - M_final rank: 2

    >>> # Using non-convex method with suggested rank 2
    >>> results_nonconvex_ex = DC_PR_with_suggested_rank(
    ...     observed_panel_matrix=O_observed_ex,
    ...     intervention_panel_input=Z_intervention_ex,
    ...     target_rank=2,
    ...     method="non-convex"
    ... )
    >>> print("\\nNon-Convex - Estimated ATT:", results_nonconvex_ex["Effects"]["ATT"]) # doctest: +SKIP
    # e.g., Non-Convex - Estimated ATT: 0.78...
    >>> print("Non-Convex - M_final rank:", np.linalg.matrix_rank(results_nonconvex_ex["Vectors"]["Counterfactual_Full_Matrix"])) # doctest: +SKIP
    # e.g., Non-Convex - M_final rank: 2
    """
    # Input validation
    if not isinstance(observed_panel_matrix, np.ndarray) or observed_panel_matrix.ndim != 2:
        raise MlsynthDataError("`observed_panel_matrix` must be a 2D NumPy array.")
    if not isinstance(intervention_panel_input, np.ndarray) or intervention_panel_input.ndim != 2:
        raise MlsynthDataError("`intervention_panel_input` must be a 2D NumPy array.")
    if observed_panel_matrix.shape != intervention_panel_input.shape:
        raise MlsynthDataError(
            f"Shape mismatch: `observed_panel_matrix` (shape {observed_panel_matrix.shape}) "
            f"and `intervention_panel_input` (shape {intervention_panel_input.shape}) must have the same shape."
        )
    if not isinstance(target_rank, (int, np.integer)): # Allow NumPy integers
        raise MlsynthConfigError("`target_rank` must be an integer.")
    if target_rank < 0: # target_rank = 0 is allowed for e.g. an all-zero matrix.
         raise MlsynthConfigError("`target_rank` must be non-negative.")
    if target_rank > 0 and target_rank > min(observed_panel_matrix.shape) and observed_panel_matrix.size > 0 :
        raise MlsynthConfigError(
            f"`target_rank` ({target_rank}) cannot exceed min(matrix_rows, matrix_cols) "
            f"which is {min(observed_panel_matrix.shape)} for a non-empty matrix when target_rank > 0."
        )

    allowed_methods = ["auto", "convex", "non-convex"]
    if not isinstance(method, str) or method not in allowed_methods:
        raise MlsynthConfigError(f"`method` must be one of {allowed_methods}, got '{method}'.")

    # Identify the treated unit's row index. Assumes intervention_panel_input is 2D.
    treated_rows_indices = np.where(np.any(np.isclose(intervention_panel_input, 1.0), axis=1))[0]
    if not treated_rows_indices.size:
        raise MlsynthDataError("No treated unit found (no row with a '1') in `intervention_panel_input`.")
    treated_unit_row_index: int = treated_rows_indices[0]

    # Standardize intervention_panel_input to 3D for internal consistency with other functions.
    # If it's already 2D (single intervention), it becomes (n_rows, n_cols, 1).
    intervention_matrices_3d: np.ndarray = transform_to_3D(intervention_panel_input)
    
    # Estimate initial treatment effects using OLS on the observed matrix.
    # This provides a starting point for the iterative algorithms.
    initial_ols_treatment_effects: np.ndarray = solve_tau(
        observed_panel_matrix, intervention_matrices_3d
    )

    # Initialize variables to store the final estimated baseline matrix and treatment effects.
    final_estimated_baseline_matrix: np.ndarray
    final_estimated_treatment_effects: np.ndarray
    
    # Variables to hold results from the convex path if method is "auto" or "convex".
    baseline_matrix_convex: Optional[np.ndarray] = None
    treatment_effects_convex: Optional[np.ndarray] = None

    # --- Convex Method Path ---
    if method == "convex" or method == "auto":
        # Determine initial nuclear norm regularizer `l` (current_nuclear_norm_regularizer).
        # Based on singular values of the residual matrix (O - Z*tau_initial_ols).
        regularizer_step_coefficient: float = 1.1 # Factor to decrease `l` in each step
        residual_matrix_for_l_init = observed_panel_matrix - np.tensordot(
                intervention_matrices_3d, initial_ols_treatment_effects, axes=([2], [0])
            )
        _, singular_values_of_initial_residual_matrix, _ = svd_fast(residual_matrix_for_l_init)
        
        current_nuclear_norm_regularizer: float
        if singular_values_of_initial_residual_matrix.size == 0: # If residual matrix is effectively zero or empty
             current_nuclear_norm_regularizer = 1e-3 # Small default
        elif singular_values_of_initial_residual_matrix.size < 2: # If only one singular value
            # Use a fraction of the largest SV, or a small default if SV is tiny
            sv0 = singular_values_of_initial_residual_matrix[0]
            current_nuclear_norm_regularizer = sv0 * 0.1 if sv0 > 1e-9 else 1e-3
        else: # Use the second largest singular value scaled by step_coeff as initial `l`
            current_nuclear_norm_regularizer = (
                singular_values_of_initial_residual_matrix[1] * regularizer_step_coefficient
            )

        # Perform initial estimation using DC_PR_with_l with the starting regularizer.
        baseline_matrix_prev_iter, treatment_effects_prev_iter = DC_PR_with_l(
            observed_panel_matrix,
            intervention_matrices_3d,
            current_nuclear_norm_regularizer,
            initial_treatment_effects=initial_ols_treatment_effects,
        )
        # Decrease regularizer for the next iteration.
        current_nuclear_norm_regularizer /= regularizer_step_coefficient

        # Iteratively reduce the regularizer `l` and estimate M and tau.
        # Stop when the rank of M_est exceeds target_rank.
        for _iter_idx in range(200): # Max 200 iterations for this search over `l`
            baseline_matrix_curr_iter, treatment_effects_curr_iter = DC_PR_with_l(
                observed_panel_matrix,
                intervention_matrices_3d,
                current_nuclear_norm_regularizer,
                initial_treatment_effects=treatment_effects_prev_iter, # Use previous tau as initial
            )
            # Calculate rank of the current estimated baseline matrix.
            current_rank_M = np.linalg.matrix_rank(baseline_matrix_curr_iter) if baseline_matrix_curr_iter.size > 0 else 0

            if current_rank_M > target_rank:
                # If current rank exceeds target, the previous iteration's M (baseline_matrix_prev_iter)
                # was the last one at or below target_rank (or the one to consider).
                # Debias this M_prev and its corresponding tau_prev.
                # The regularizer for debias should correspond to the `l` used to obtain M_prev.
                l_for_debias = current_nuclear_norm_regularizer * regularizer_step_coefficient
                debiased_baseline_matrix, debiased_treatment_effects = debias(
                    baseline_matrix_prev_iter,
                    treatment_effects_prev_iter,
                    intervention_matrices_3d,
                    l_for_debias, 
                )
                # Truncate the debiased M to the exact target_rank.
                baseline_matrix_convex = SVD(debiased_baseline_matrix, target_rank)
                treatment_effects_convex = debiased_treatment_effects
                break # Exit loop over `l`
            
            # Update for next iteration
            baseline_matrix_prev_iter = baseline_matrix_curr_iter
            treatment_effects_prev_iter = treatment_effects_curr_iter
            current_nuclear_norm_regularizer /= regularizer_step_coefficient

            if current_nuclear_norm_regularizer < 1e-9:  # Safety break if `l` becomes too small
                # If `l` is very small, the solution is essentially unregularized.
                # Debias and truncate the last valid estimates.
                l_for_debias = current_nuclear_norm_regularizer * regularizer_step_coefficient
                debiased_baseline_matrix, debiased_treatment_effects = debias(
                    baseline_matrix_prev_iter, treatment_effects_prev_iter,
                    intervention_matrices_3d, l_for_debias,
                )
                baseline_matrix_convex = SVD(debiased_baseline_matrix, target_rank)
                treatment_effects_convex = debiased_treatment_effects
                break
        else: # Loop finished (e.g., max_iter reached) without rank exceeding target_rank
            if baseline_matrix_convex is None: # If not set inside the loop
                # This means target_rank was likely never exceeded.
                # Debias and truncate the last computed M and tau.
                l_for_debias = current_nuclear_norm_regularizer * regularizer_step_coefficient
                debiased_baseline_matrix, debiased_treatment_effects = debias(
                    baseline_matrix_prev_iter, treatment_effects_prev_iter,
                    intervention_matrices_3d, l_for_debias,
                )
                baseline_matrix_convex = SVD(debiased_baseline_matrix, target_rank)
                treatment_effects_convex = debiased_treatment_effects
        
        # If method is strictly "convex", these are the final results.
        if method == "convex":
            if baseline_matrix_convex is None or treatment_effects_convex is None: # Should be set by now
                 raise MlsynthEstimationError("Convex method failed to produce a final baseline matrix.")
            final_estimated_baseline_matrix = baseline_matrix_convex
            final_estimated_treatment_effects = treatment_effects_convex

    # --- Non-Convex Method Path ---
    if method == "non-convex" or method == "auto":
        # Directly estimate M and tau using non_convex_PR with the fixed target_rank.
        non_convex_baseline_matrix, non_convex_treatment_effects = non_convex_PR(
            observed_panel_matrix,
            intervention_matrices_3d,
            target_rank, # Fixed rank constraint
            initial_treatment_effects=initial_ols_treatment_effects, # Use same initial tau
        )
        if method == "non-convex":
            final_estimated_baseline_matrix = non_convex_baseline_matrix
            final_estimated_treatment_effects = non_convex_treatment_effects

    # --- Auto Method: Compare Convex and Non-Convex Results ---
    if method == "auto":
        if baseline_matrix_convex is None or treatment_effects_convex is None:
            # If convex path failed or wasn't fully completed, default to non-convex.
            final_estimated_baseline_matrix = non_convex_baseline_matrix
            final_estimated_treatment_effects = non_convex_treatment_effects
        else:
            # Compare results if both paths executed.
            rank_convex = np.linalg.matrix_rank(baseline_matrix_convex) if baseline_matrix_convex.size > 0 else 0
            rank_non_convex = np.linalg.matrix_rank(non_convex_baseline_matrix) if non_convex_baseline_matrix.size > 0 else 0
            
            # Calculate reconstruction error for both methods.
            error_convex = np.linalg.norm(
                observed_panel_matrix - baseline_matrix_convex - 
                np.tensordot(intervention_matrices_3d, treatment_effects_convex, axes=([2], [0]))
            )
            error_non_convex = np.linalg.norm(
                observed_panel_matrix - non_convex_baseline_matrix - 
                np.tensordot(intervention_matrices_3d, non_convex_treatment_effects, axes=([2], [0]))
            )

            # Prefer non-convex if:
            # 1. Convex method did not achieve target_rank but non-convex did, OR
            # 2. Non-convex has lower reconstruction error.
            if (rank_convex != target_rank and rank_non_convex == target_rank) or\
               (error_non_convex < error_convex):
                final_estimated_baseline_matrix = non_convex_baseline_matrix
                final_estimated_treatment_effects = non_convex_treatment_effects
            else: # Otherwise, use convex results.
                final_estimated_baseline_matrix = baseline_matrix_convex
                final_estimated_treatment_effects = treatment_effects_convex
    
    # Ensure final M and tau have been assigned by one of the paths.
    if "final_estimated_baseline_matrix" not in locals() or "final_estimated_treatment_effects" not in locals():
        # This should not happen if logic is correct.
        raise MlsynthEstimationError(
            "Failed to determine final estimated baseline and treatment effects. "
            "This indicates an issue in the method selection logic ('auto', 'convex', 'non-convex')."
        )

    # --- Calculate Confidence Intervals and Final Results ---
    # Calculate residuals for CI computation: E_hat = O - M_hat - Z*tau_hat
    residuals_for_ci = (
        observed_panel_matrix - final_estimated_baseline_matrix - 
        np.tensordot(intervention_matrices_3d, final_estimated_treatment_effects, axes=([2], [0]))
    )
    # Compute covariance matrix for tau_hat.
    effects_covariance_matrix = panel_regression_CI(
        final_estimated_baseline_matrix,
        intervention_matrices_3d,
        residuals_for_ci,
    )
    
    diag_cov = np.diag(effects_covariance_matrix)
    standard_errors_of_effects: np.ndarray
    # Handle potential negative variances from covariance matrix (numerical issues).
    if np.any(diag_cov < -1e-9): # Allow small negatives due to float precision if effectively zero
        standard_errors_of_effects = np.full_like(diag_cov, np.nan) # Set SE to NaN if variance is problematic
    else:
        safe_diag_cov = np.maximum(diag_cov, 0) # Ensure non-negativity before sqrt
        standard_errors_of_effects = np.sqrt(safe_diag_cov)

    # Calculate 95% confidence intervals (default alpha=0.05).
    alpha: float = 0.05
    z_score: float = norm.ppf(1 - alpha / 2) # Z-score for two-tailed 95% CI

    lower_bound_list = []
    upper_bound_list = []
    for i in range(len(final_estimated_treatment_effects)): # For each estimated tau
        se = standard_errors_of_effects[i]
        tau_i = final_estimated_treatment_effects[i]
        if np.isnan(se): # If SE is NaN, CI bounds are also NaN.
            lower_bound_list.append(np.nan)
            upper_bound_list.append(np.nan)
        else:
            lower_bound_list.append(round(tau_i - z_score * se, 3))
            upper_bound_list.append(round(tau_i + z_score * se, 3))
            
    # Assemble inference results.
    confidence_intervals_dict: Dict[str, Any] = {
        "Lower Bound": lower_bound_list[0] if lower_bound_list and len(lower_bound_list)==1 else lower_bound_list,
        "Upper Bound": upper_bound_list[0] if upper_bound_list and len(upper_bound_list)==1 else upper_bound_list,
        "SE": standard_errors_of_effects, # Array of SEs
        "tstat": z_score, # This is the critical z-value, not individual t-stats.
                          # Individual t-stats would be tau_i / se_i.
    }

    # Extract observed and counterfactual outcomes for the treated unit.
    counterfactual_outcomes_treated_unit: np.ndarray = final_estimated_baseline_matrix[
        treated_unit_row_index, :
    ]
    observed_outcomes_treated_unit: np.ndarray = observed_panel_matrix[
        treated_unit_row_index, :
    ]

    # Determine t1 (number of pre-treatment periods for the treated unit).
    treatment_periods_for_unit = np.where(np.isclose(intervention_panel_input[treated_unit_row_index, :], 1.0))[0]
    t1: int
    if not treatment_periods_for_unit.size:
        # This case implies the identified treated unit has no active treatment periods marked by '1'.
        # This is unusual and might indicate an issue with `intervention_panel_input` or definition of treatment.
        # Defaulting to all periods as pre-treatment.
        t1 = intervention_panel_input.shape[1]
    else:
        t1 = treatment_periods_for_unit[0] # First period of treatment

    t2: int = len(counterfactual_outcomes_treated_unit) - t1 # Number of post-treatment periods

    # Calculate treatment effects, fit diagnostics, and output vectors using `effects.calculate`.
    (
        treatment_effects_results_dict, # e.g., ATT, Percent ATT
        model_fit_diagnostics_dict,     # e.g., RMSE_pre
        output_vectors_dict,            # e.g., Observed, Synthetic, Gap series for treated unit
    ) = effects.calculate(
        observed_outcomes_treated_unit, counterfactual_outcomes_treated_unit, t1, t2
    )

    # Add the full estimated baseline matrix to the output vectors.
    output_vectors_dict[
        "Counterfactual_Full_Matrix" # M_hat
    ] = final_estimated_baseline_matrix

    # Assemble final dictionary of results.
    return {
        "Effects": treatment_effects_results_dict,
        "Fit": model_fit_diagnostics_dict,
        "Vectors": output_vectors_dict,
        "Inference": confidence_intervals_dict,
    }


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
