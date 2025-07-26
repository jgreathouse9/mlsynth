from sklearn.decomposition import PCA
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd # For PDAfs doctest and potentially internal DataFrame use
from mlsynth.utils.denoiseutils import svt, spectral_rank # svt for SVDCluster, spectral_rank for fpca
from typing import Tuple, List, Any, Dict
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError


def determine_optimal_clusters(X: np.ndarray) -> int:
    """Determine the optimal number of clusters using Silhouette score.

    Iterates through a range of possible cluster numbers (from 2 up to
    `min(10, n_samples - 1)`) and selects the number of clusters that
    yields the highest average silhouette score. `KMeans` is used for
    clustering at each step.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix, shape (n_samples, n_features). `n_samples` is
        the number of data points to cluster, and `n_features` is the number
        of features for each point.

    Returns
    -------
    int
        The number of optimal clusters found. Returns 1 if `n_samples` is
        less than 2 or if no valid range of clusters can be tested (e.g.,
        if `n_samples - 1 < 2`).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_blobs
    >>> # from mlsynth.utils.selectorsutils import determine_optimal_clusters # Original
    >>> from mlsynth.utils.selector_helpers import determine_optimal_clusters # Corrected for this file
    >>> # Generate sample data with 3 distinct clusters
    >>> X_data, _ = make_blobs(n_samples=150, centers=3, n_features=2,
    ...                        cluster_std=0.8, random_state=42)
    >>> optimal_k = determine_optimal_clusters(X_data)
    >>> print(f"Optimal number of clusters: {optimal_k}")
    Optimal number of clusters: 3
    >>>
    >>> # Example with fewer samples
    >>> X_few_samples = np.array([[1,2], [1.1,2.1], [10,11], [10.1,11.1]])
    >>> optimal_k_few = determine_optimal_clusters(X_few_samples)
    >>> print(f"Optimal for few samples: {optimal_k_few}")
    Optimal for few samples: 2
    >>>
    >>> # Example with too few samples to form multiple clusters
    >>> X_too_few = np.array([[1,2]])
    >>> optimal_k_too_few = determine_optimal_clusters(X_too_few)
    >>> print(f"Optimal for too few samples: {optimal_k_too_few}")
    Optimal for too few samples: 1
    """
    if not isinstance(X, np.ndarray):
        raise MlsynthDataError("Input `X` must be a NumPy array.")
    if X.ndim != 2:
        raise MlsynthDataError("Input `X` must be a 2D array (n_samples, n_features).")
    if X.size == 0: # Handles both (0, M) and (N, 0) and (0,0)
        if X.shape[0] > 0 and X.shape[1] == 0: # Case (N, 0) - N samples, 0 features
             raise MlsynthDataError("Input `X` has 0 features, cannot compute silhouette scores.")
        # For (0, M) or (0,0), it implies 0 samples.
        # If X.shape[0] == 0, it will be caught by n_samples < 2 check later and return 1.

    silhouette_scores_list = []
    n_samples = X.shape[0]

    if n_samples < 2:
        return 1

    max_clusters_to_evaluate = min(10, n_samples - 1)

    if max_clusters_to_evaluate < 2: # Not enough samples to form at least 2 clusters
        return 1

    cluster_num_range = range(2, max_clusters_to_evaluate + 1)

    for current_num_clusters in cluster_num_range:
        try:
            # Initialize KMeans. 'k-means++' for smart initialization, 'n_init="auto"' uses a dynamic number of initializations.
            kmeans = KMeans(
                n_clusters=current_num_clusters, random_state=0, init="k-means++", n_init="auto"
            )
            cluster_labels = kmeans.fit_predict(X)
            
            # Silhouette score requires at least 2 distinct clusters.
            if len(np.unique(cluster_labels)) < 2:
                average_silhouette_score = -1.0 # Assign a low score if not enough clusters formed
            else:
                 average_silhouette_score = silhouette_score(X, cluster_labels)
            silhouette_scores_list.append(average_silhouette_score)
        except ValueError as e: # Catches errors from silhouette_score if labels are problematic
            raise MlsynthEstimationError(
                f"Error during KMeans fitting or silhouette score calculation for {current_num_clusters} clusters: {e}"
            )
        except Exception as e: 
            raise MlsynthEstimationError(
                f"Unexpected error for {current_num_clusters} clusters: {e}"
            )

    if not silhouette_scores_list: # Should not happen if cluster_num_range is valid
        return 1 # Default to 1 cluster if no scores were computed

    # Select the number of clusters that yielded the highest silhouette score.
    num_optimal_clusters = cluster_num_range[np.argmax(silhouette_scores_list)]
    return num_optimal_clusters


def SVDCluster(
    X: np.ndarray, y: np.ndarray, donor_names: List[Any]
) -> Tuple[np.ndarray, List[Any], np.ndarray]:
    """Cluster donor units based on SVD embeddings.

    Identifies donors belonging to the same cluster as the treated unit.
    Clustering is performed on SVD embeddings of the combined treated and
    donor unit data (units as rows, time periods as columns after transpose).
    The treated unit is assumed to be the first unit when `y` is prepended to `X`.
    The optimal number of clusters is determined using the silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Donor matrix, shape (n_time_periods, n_donors). Each column is a
        time series for a donor unit.
    y : np.ndarray
        Treated unit outcome vector, shape (n_time_periods,).
    donor_names : List[Any]
        List of donor names, corresponding to columns of `X`.
        Length must be `X.shape[1]`.

    Returns
    -------
    Tuple[np.ndarray, List[Any], np.ndarray]
        A tuple containing:
        - selected_donors_matrix (np.ndarray): Subset of the donor matrix `X`
          corresponding to donors in the same cluster as the treated unit,
          shape (n_time_periods, n_selected_donors).
        - selected_donor_names (List[Any]): Names of the selected donors.
        - selected_donor_original_indices (np.ndarray): 1D array of integer
          indices of the selected donors within the original `X` matrix
          (i.e., column indices).

    Examples
    --------
    >>> import numpy as np
    >>> # from mlsynth.utils.selectorsutils import SVDCluster # Original
    >>> from mlsynth.utils.selector_helpers import SVDCluster # Corrected for this file
    >>> # Mock svt as it's a complex dependency from denoiseutils
    >>> class MockSVT:
    ...     def __call__(self, data_matrix, fixed_rank=None, spectral_energy_threshold=None): # Added params to match denoiseutils.svt
    ...         U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    ...         # Return structure: low_rank_matrix (not used here), original_n_cols, U, S_diag, V_transpose
    ...         # For SVDCluster, it uses u_left_singular_vectors, s_singular_values_diag from svt output
    ...         # So, we need to return something that matches that structure.
    ...         # svt returns: low_rank_approximation, num_cols, U_k, S_k_diag, Vh_k
    ...         # SVDCluster uses the 3rd and 4th elements: U_k, S_k_diag
    ...         return data_matrix, data_matrix.shape[1], U, S, Vt
    >>>
    >>> # Replace actual svt with mock for doctest
    >>> import mlsynth.utils.denoiseutils as du_actual # Keep actual denoiseutils
    >>> original_svt_func = du_actual.svt # Store original
    >>> du_actual.svt = MockSVT() # Replace with mock
    >>>
    >>> T, n_donors = 50, 5
    >>> X_data = np.random.rand(T, n_donors)
    >>> y_data = np.random.rand(T)
    >>> # Make two donors similar to y_data, and one very different
    >>> X_data[:, 0] = y_data * 0.8 + 0.1 * np.random.randn(T)
    >>> X_data[:, 1] = y_data * 0.9 + 0.05 * np.random.randn(T)
    >>> X_data[:, 3] += 10 # Dissimilar donor
    >>> donor_names_list = [f"Donor_{i}" for i in range(n_donors)]
    >>>
    >>> X_s, names_s, indices_s = SVDCluster(X_data, y_data, donor_names_list) # doctest: +SKIP
    >>> print(f"Selected donor names: {names_s}") # doctest: +SKIP
    >>> # Expected: Donor_0, Donor_1 might be selected (probabilistic) # doctest: +SKIP
    >>> print(f"Shape of selected X: {X_s.shape}") # doctest: +SKIP
    >>> print(f"Selected indices: {indices_s}") # doctest: +SKIP
    >>>
    >>> # Restore original svt
    >>> du_actual.svt = original_svt_func
    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise MlsynthDataError("Input `X` (donor matrix) must be a NumPy array.")
    if X.ndim != 2:
        raise MlsynthDataError("Input `X` (donor matrix) must be a 2D array.")

    if not isinstance(y, np.ndarray):
        raise MlsynthDataError("Input `y` (treated unit outcome vector) must be a NumPy array.")
    if y.ndim == 1:
        y_reshaped_for_check = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] == 1:
        y_reshaped_for_check = y
    else:
        raise MlsynthDataError("Input `y` must be 1D or have shape (n_time_periods, 1).")

    if y_reshaped_for_check.size == 0:
        raise MlsynthDataError("Input `y` cannot be empty.")
    
    if X.size > 0 and y_reshaped_for_check.shape[0] != X.shape[0]: # X.size check for X with 0 columns
        raise MlsynthDataError(
            f"Shape mismatch: `y` has {y_reshaped_for_check.shape[0]} time periods, "
            f"`X` has {X.shape[0]} time periods."
        )

    if not isinstance(donor_names, list):
        raise MlsynthConfigError("Input `donor_names` must be a list.")
    if X.shape[1] != len(donor_names): # This is fine even if X.shape[1] is 0
        raise MlsynthConfigError(
            f"Mismatch: `X` has {X.shape[1]} donors, but `donor_names` has {len(donor_names)} entries."
        )

    # Ensure y_reshaped matches the number of time periods in X if X has data,
    # otherwise use y's own length. This handles X being (T, 0).
    y_reshaped = y_reshaped_for_check[:X.shape[0] if X.shape[0] > 0 and X.shape[1] > 0 else y_reshaped_for_check.shape[0]]


    # Combine treated unit (y) and donor units (X) into a single matrix.
    # Each row represents a unit, each column a time point (after transpose).
    # The treated unit is the first row.
    if X.shape[1] > 0: # If there are donors
        unit_time_matrix = np.hstack((y_reshaped, X)).T
    else: # Only treated unit, no donors
        unit_time_matrix = y_reshaped.T # Transpose to (1, n_time_periods)

    if unit_time_matrix.size == 0: # Should be caught by earlier checks on y and X
        raise MlsynthDataError("Combined unit-time matrix is empty.")

    try:
        # Perform Singular Value Thresholding (SVT) which includes SVD.
        # svt returns: low_rank_approximation, num_cols, U_k, S_k_diag, Vh_k
        # U_k (u_left_singular_vectors) are the left singular vectors.
        # S_k_diag (s_singular_values_diag) are the singular values (as a diagonal matrix or 1D array).
        _, _, u_left_singular_vectors, s_singular_values_diag, _ = svt(unit_time_matrix)
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"SVD computation failed in SVDCluster: {e}")
    except Exception as e: # Catch other errors from svt
        raise MlsynthEstimationError(f"Error during svt call in SVDCluster: {e}")

    if u_left_singular_vectors.size == 0 or s_singular_values_diag.size == 0:
        # If SVD results are empty, no meaningful embeddings can be formed.
        return np.array([]).reshape(X.shape[0],0), [], np.array([]) # Return empty selections

    # Create unit embeddings by scaling left singular vectors by singular values.
    # This gives a representation of each unit in the SVD-derived feature space.
    unit_embeddings = u_left_singular_vectors * s_singular_values_diag

    num_total_units = unit_embeddings.shape[0]
    # If fewer than 3 units, clustering is not meaningful or silhouette score is problematic.
    # Return all available donors in such cases.
    if num_total_units < 3: # Typically need at least 2 clusters for silhouette, and 1 unit per cluster.
        if X.shape[1] > 0: # If there are donors
             return X, donor_names, np.arange(X.shape[1]) # Return all original donors
        else: # No donors, only treated unit
             return np.array([]).reshape(X.shape[0],0), [], np.array([]) # Return empty selections

    # Determine the range of cluster numbers to evaluate.
    # Silhouette score requires at least 2 clusters.
    cluster_num_range = range(2, num_total_units) # Max clusters can be num_total_units - 1

    if not list(cluster_num_range): # If num_total_units is 2, range(2,2) is empty.
        # This implies only 2 units total (treated + 1 donor). They are trivially in the "same" group.
        if X.shape[1] > 0: # If there is that one donor
            return X, donor_names, np.arange(X.shape[1])
        else: # Should not happen if num_total_units is 2
            return np.array([]).reshape(X.shape[0],0), [], np.array([])

    try:
        # Calculate silhouette scores for different numbers of clusters.
        silhouette_scores_values = [
            silhouette_score(
                unit_embeddings,
                KMeans(
                    n_clusters=k, random_state=42, n_init='auto', init="k-means++"
                ).fit_predict(unit_embeddings),
            )
            for k in cluster_num_range
        ]
        if not silhouette_scores_values: # Should be caught by "if not list(cluster_num_range)"
            num_optimal_clusters = 1 # Default if somehow list is empty (e.g. all silhouette scores failed)
        else:
            # Choose the number of clusters that maximizes the silhouette score.
            num_optimal_clusters = cluster_num_range[np.argmax(silhouette_scores_values)]

        # Perform KMeans clustering with the optimal number of clusters.
        kmeans = KMeans(
            n_clusters=num_optimal_clusters, random_state=42, n_init='auto', init="k-means++"
        ).fit(unit_embeddings)
        cluster_labels = kmeans.labels_
    except ValueError as e: # Catches errors from silhouette_score or KMeans
        raise MlsynthEstimationError(f"Clustering failed in SVDCluster: {e}")
    except Exception as e: # Catch other unexpected errors
        raise MlsynthEstimationError(f"Unexpected error during clustering in SVDCluster: {e}")


    # Identify the cluster label of the treated unit (which is the first unit in unit_embeddings).
    treated_unit_cluster_label = cluster_labels[0]
    # Find indices of donor units (from cluster_labels[1:]) that share the same cluster label.
    selected_donor_indices_in_cluster_array = np.where(
        cluster_labels[1:] == treated_unit_cluster_label
    )[0]

    # These indices are relative to the original donor matrix X.
    selected_donor_original_indices = selected_donor_indices_in_cluster_array
    selected_donor_names = [donor_names[i] for i in selected_donor_original_indices]
    selected_donors_matrix = X[:, selected_donor_original_indices]

    return selected_donors_matrix, selected_donor_names, selected_donor_original_indices


def fpca(X: np.ndarray) -> Tuple[int, np.ndarray, int]:
    """Perform Functional Principal Component Analysis (FPCA).

    This function smooths input time series data using B-splines, performs
    PCA on the smoothed data, determines a spectral rank for truncation,
    scales the FPC scores, and then determines the optimal number of
    clusters from these scaled scores using the silhouette method.

    A fallback to standard PCA is implemented if B-spline interpolation
    is not possible (e.g., too few time points).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix, shape (n_units, n_time_points). Each row is a
        time series for a unit. Must have at least one time point.
        If `n_units` is 0, returns (0, empty 2D array, 0).

    Returns
    -------
    Tuple[int, np.ndarray, int]
        A tuple containing:
        - num_optimal_clusters (int): The optimal number of clusters
          determined from the scaled FPC scores. Can be 0 or 1 in edge cases.
        - scaled_fpc_scores (np.ndarray): The scaled Functional Principal
          Component scores used for clustering, shape (n_units, specrank) or
          (n_units, 0) if specrank is 0. Returns an empty 2D array if
          `n_units` is 0.
        - spectral_rank_value (int): The spectral rank (number of components)
          used for truncating PCA components. Can be 0.

    Raises
    ------
    ValueError
        If `X` has zero time points (columns).

    Examples
    --------
    >>> import numpy as np
    >>> # from mlsynth.utils.selectorsutils import fpca # Original
    >>> from mlsynth.utils.selector_helpers import fpca, determine_optimal_clusters # Corrected for this file
    >>> from mlsynth.utils.denoiseutils import spectral_rank # Corrected for this file
    >>>
    >>> # Mock spectral_rank and determine_optimal_clusters for stable doctest
    >>> import mlsynth.utils.selector_helpers as sh_helpers # Mock functions in this module
    >>> import mlsynth.utils.denoiseutils as du_helpers # Mock functions in denoiseutils
    >>>
    >>> original_spectral_rank_func = du_helpers.spectral_rank
    >>> original_det_opt_clusters_func = sh_helpers.determine_optimal_clusters
    >>>
    >>> du_helpers.spectral_rank = lambda s, energy_threshold: max(1, int(len(s) * 0.5)) # Mock: take 50% of components or 1
    >>> sh_helpers.determine_optimal_clusters = lambda x_input: 2 if x_input.shape[0] > 1 and x_input.shape[1] > 0 else 1 # Mock: return 2 clusters if possible
    >>>
    >>> n_units, n_time = 20, 30
    >>> X_data = np.random.rand(n_units, n_time)
    >>> # Add some structure: first half of units are sine waves, second half are cosine
    >>> time_pts = np.linspace(0, 2 * np.pi, n_time)
    >>> X_data[:n_units//2, :] = np.sin(time_pts) + 0.1 * np.random.randn(n_units//2, n_time)
    >>> X_data[n_units//2:, :] = np.cos(time_pts) + 0.1 * np.random.randn(n_units - n_units//2, n_time)
    >>>
    >>> opt_k, scores, rank = fpca(X_data) # doctest: +SKIP
    >>> print(f"Optimal clusters: {opt_k}") # doctest: +SKIP
    >>> # Expected: Optimal clusters: 2 (due to mock) # doctest: +SKIP
    >>> print(f"Shape of FPC scores: {scores.shape}") # doctest: +SKIP
    >>> # Expected: Shape of FPC scores: (20, rank_determined_by_mock_spectral_rank) # doctest: +SKIP
    >>> print(f"Spectral rank: {rank}") # doctest: +SKIP
    >>>
    >>> # Test edge case: few time points (fallback to direct PCA)
    >>> X_few_time = np.random.rand(5, 2) # 5 units, 2 time points (k_spline=3)
    >>> opt_k_ft, scores_ft, rank_ft = fpca(X_few_time) # doctest: +SKIP
    >>> print(f"Optimal clusters (few time): {opt_k_ft}") # doctest: +SKIP
    >>> print(f"Scores shape (few time): {scores_ft.shape}") # doctest: +SKIP
    >>>
    >>> # Restore original functions
    >>> du_helpers.spectral_rank = original_spectral_rank_func
    >>> sh_helpers.determine_optimal_clusters = original_det_opt_clusters_func
    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise MlsynthDataError("Input `X` must be a NumPy array.")
    if X.ndim != 2:
        raise MlsynthDataError("Input `X` must be a 2D array (n_units, n_time_points).")

    if X.shape[0] == 0: # No units
        # If no units, return 0 clusters, empty scores, and 0 rank.
        return 0, np.array([]).reshape(0,0), 0 # Return shape (0,0) for scores

    n_time_points = X.shape[1]
    if n_time_points == 0:
        raise MlsynthDataError("Input matrix X must have at least one time point (column) for FPCA.")

    time_points_normalized = np.linspace(0, 1, num=n_time_points) # Normalized time for spline fitting
    spline_degree = 3  # Degree of the B-spline

    try:
        # Fallback to simple PCA if spline interpolation is not possible (e.g., too few time points for spline degree).
        # make_interp_spline requires N > k (number of time points > spline degree).
        if X.T.shape[0] <= spline_degree :
            if X.shape[0] == 0: # Should be caught by earlier check, but defensive
                return 0, np.array([]).reshape(0,0), 0

            # Perform standard PCA directly on the original data.
            pca_direct = PCA()
            X_pca_direct = pca_direct.fit_transform(X)
            
            try:
                # Get singular values for spectral rank determination.
                _, S_direct, _ = np.linalg.svd(X)
            except np.linalg.LinAlgError as e:
                raise MlsynthEstimationError(f"SVD failed during PCA fallback in fpca: {e}")
            except Exception as e: # Catch other SVD errors
                raise MlsynthEstimationError(f"Unexpected SVD error during PCA fallback in fpca: {e}")

            # Determine spectral rank based on energy threshold.
            specrank_direct = spectral_rank(S_direct, energy_threshold=0.95) if len(S_direct) > 0 else 0
            X_pca_direct_truncated = X_pca_direct[:, :specrank_direct]

            if X_pca_direct_truncated.shape[1] == 0: # if specrank is 0 (no components selected)
                # If no components selected, implies all units are similar enough to be one cluster.
                # Return scores shape (n_units, 0)
                return 1, np.array([]).reshape(X_pca_direct_truncated.shape[0],0), 0

            # Scale PCA scores (standardize: subtract mean, divide by std dev).
            std_dev_direct = np.std(X_pca_direct_truncated, axis=0)
            scaled_pca_scores_direct = np.divide(
                X_pca_direct_truncated - np.mean(X_pca_direct_truncated, axis=0),
                np.where(std_dev_direct == 0, 1, std_dev_direct), # Avoid division by zero if a component has zero variance
                out=np.zeros_like(X_pca_direct_truncated), 
                where=np.where(std_dev_direct == 0, 1, std_dev_direct) != 0 # type: ignore[arg-type]
            )
            # Determine optimal clusters from these scaled PCA scores.
            num_optimal_clusters_direct = determine_optimal_clusters(scaled_pca_scores_direct)
            return num_optimal_clusters_direct, scaled_pca_scores_direct, specrank_direct

        # Proceed with B-spline smoothing and FPCA if enough time points.
        # Smooth each unit's time series using B-splines. X.T makes time series the rows for make_interp_spline.
        spline_interpolator = make_interp_spline(time_points_normalized, X.T, k=spline_degree)
        smoothed_data_transposed = spline_interpolator(time_points_normalized) # Evaluate spline at original time points
        
        # Perform PCA on the smoothed data (transpose back so units are rows).
        pca_model = PCA()
        pca_transformed_data = pca_model.fit_transform(smoothed_data_transposed.T)

        try:
            # Get singular values of the smoothed data for spectral rank.
            _, S_singular_values, _ = np.linalg.svd(smoothed_data_transposed.T) 
        except np.linalg.LinAlgError as e:
            raise MlsynthEstimationError(f"SVD failed on smoothed data in fpca: {e}")
        except Exception as e: # Catch other SVD errors
            raise MlsynthEstimationError(f"Unexpected SVD error on smoothed data in fpca: {e}")

        # Determine spectral rank (number of components to keep).
        spectral_rank_value = spectral_rank(S_singular_values, energy_threshold=0.95) if len(S_singular_values) > 0 else 0
        # Truncate PCA components to the determined spectral rank.
        pca_transformed_data_truncated = pca_transformed_data[:, :spectral_rank_value]

        if pca_transformed_data_truncated.shape[1] == 0: # If spectral_rank_value is 0
            # No components selected, implies 1 cluster. Return scores shape (n_units, 0).
            return 1, np.array([]).reshape(pca_transformed_data_truncated.shape[0],0), 0
        
        # Scale the truncated FPC scores (standardize).
        pca_scores_std_dev = np.std(pca_transformed_data_truncated, axis=0)
        scaled_fpc_scores = np.divide(
            pca_transformed_data_truncated - np.mean(pca_transformed_data_truncated, axis=0),
            np.where(pca_scores_std_dev == 0, 1, pca_scores_std_dev), # Avoid division by zero
            out=np.zeros_like(pca_transformed_data_truncated),
            where=np.where(pca_scores_std_dev == 0, 1, pca_scores_std_dev) != 0 # type: ignore[arg-type]
        )

    except ValueError as e: # Catch specific errors from sklearn (PCA) or scipy (spline)
        raise MlsynthEstimationError(f"Error during FPCA processing (e.g., spline, PCA): {e}")
    except Exception as e: # Catch any other unexpected errors during the main try block
        raise MlsynthEstimationError(f"Unexpected error in fpca main processing: {e}")

    # Determine the optimal number of clusters from the scaled FPC scores.
    num_optimal_clusters = determine_optimal_clusters(scaled_fpc_scores)

    return num_optimal_clusters, scaled_fpc_scores, spectral_rank_value


def PDAfs(
    y: np.ndarray,
    donormatrix: np.ndarray,
    num_pre_treatment_periods: int,
    total_time_periods: int,
    total_num_donors: int,
) -> Dict[str, Any]:
    """Perform Forward Selection Panel Data Approach (PDAfs).

    Selects donor units iteratively based on minimizing an information
    criterion (current_information_criterion). The criterion balances model fit
    (residual_variance_pre_treatment) and a penalty for model complexity
    (complexity_penalty_factor), which depends on the number of selected donors,
    total donors (total_num_donors), and pre-treatment periods (num_pre_treatment_periods).
    The selection process starts with the donor that maximizes SSE (Sum of
    Squared Explained variance) and iteratively adds donors that improve Q_new.

    Parameters
    ----------
    y : np.ndarray
        Outcome vector for the treated unit, shape (total_time_periods,).
    donormatrix : np.ndarray
        Matrix of outcomes for donor units,
        shape (total_time_periods, total_num_donors).
    num_pre_treatment_periods : int
        Number of pre-treatment periods. Data is sliced
        `[:num_pre_treatment_periods]` for model fitting.
    total_time_periods : int
        Total number of time periods (length of `y` and rows of `donormatrix`).
        This parameter seems unused in the current implementation logic beyond
        potentially defining `treated_outcomes_post_treatment`, but
        `treated_outcomes_post_treatment` itself is not used.
    total_num_donors : int
        Total number of available donor units (columns in `donormatrix`).
        Must match `donormatrix.shape[1]`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results of the forward selection:
        - "selected_donor_indices" (np.ndarray): 1D array of integer indices
          of the selected donor units from `donormatrix`. Shape (k_selected,).
        - "final_model_coefficients" (np.ndarray): Coefficients of the final
          model, shape (k_selected + 1,). Intercept is the first element.
        - "final_information_criterion" (float): The final value of the
          information criterion for the selected model.
        - "final_residual_variance" (float): Estimated residual variance of
          the final model in the pre-treatment period.
        - "predicted_treated_outcomes_pre_treatment" (np.ndarray): Predicted
          pre-treatment outcomes for the treated unit using the final model,
          shape (num_pre_treatment_periods,).

    Examples
    --------
    >>> import numpy as np
    >>> # from mlsynth.utils.selectorsutils import PDAfs # Original
    >>> from mlsynth.utils.selector_helpers import PDAfs # Corrected for this file
    >>>
    >>> t_total, n_donors_total, t_pre = 100, 10, 70
    >>> y_target = np.random.rand(t_total)
    >>> donor_data = np.random.rand(t_total, n_donors_total)
    >>> # Make some donors more correlated with y_target in pre-treatment
    >>> for i in range(n_donors_total // 2):
    ...     donor_data[:t_pre, i] = y_target[:t_pre] * (0.5 + 0.1*i) + \
    ...                               0.2 * np.random.randn(t_pre)
    >>>
    >>> results = PDAfs(y_target, donor_data, t_pre, t_total, n_donors_total) # doctest: +SKIP
    >>> print(f"Selected donors: {results['selected_donor_indices']}") # doctest: +SKIP
    >>> print(f"Model coefficients: {results['final_model_coefficients']}") # doctest: +SKIP
    >>> print(f"Q_new: {results['final_information_criterion']:.4f}") # doctest: +SKIP
    >>> print(f"Sigma squared: {results['final_residual_variance']:.4f}") # doctest: +SKIP
    >>> print(f"Shape of y_hat: {results['predicted_treated_outcomes_pre_treatment'].shape}") # doctest: +SKIP
    """
    # Input validation
    if not isinstance(y, np.ndarray):
        raise MlsynthDataError("Input `y` must be a NumPy array.")
    if y.ndim != 1:
        raise MlsynthDataError("Input `y` must be a 1D array.")
    if y.size == 0:
        raise MlsynthDataError("Input `y` cannot be empty.")

    if not isinstance(donormatrix, np.ndarray):
        raise MlsynthDataError("Input `donormatrix` must be a NumPy array.")
    if donormatrix.ndim != 2:
        raise MlsynthDataError("Input `donormatrix` must be a 2D array.")

    if y.shape[0] != donormatrix.shape[0]:
        raise MlsynthDataError(
            f"Shape mismatch: `y` has {y.shape[0]} time periods, "
            f"`donormatrix` has {donormatrix.shape[0]} time periods."
        )

    if not isinstance(num_pre_treatment_periods, int):
        raise MlsynthConfigError("Input `num_pre_treatment_periods` must be an integer.")
    if not (0 < num_pre_treatment_periods <= y.shape[0]):
        raise MlsynthConfigError(
            f"`num_pre_treatment_periods` ({num_pre_treatment_periods}) must be positive "
            f"and not exceed total time periods in `y` ({y.shape[0]})."
        )
    
    # total_time_periods is validated by its use in slicing y and donormatrix if y.shape[0] is not used directly
    # For now, ensure it's an int and consistent if used.
    if not isinstance(total_time_periods, int): # Parameter is unused, but validate if kept
        raise MlsynthConfigError("Input `total_time_periods` must be an integer.")
    # if total_time_periods != y.shape[0]:
    #     raise MlsynthConfigError(
    #         f"`total_time_periods` ({total_time_periods}) does not match "
    #         f"the number of periods in `y` ({y.shape[0]})."
    #     )


    if not isinstance(total_num_donors, int):
        raise MlsynthConfigError("Input `total_num_donors` must be an integer.")
    if total_num_donors != donormatrix.shape[1]:
        raise MlsynthConfigError(
            f"`total_num_donors` ({total_num_donors}) does not match "
            f"the number of columns in `donormatrix` ({donormatrix.shape[1]})."
        )
    if total_num_donors == 0: # This implies donormatrix.shape[1] is 0
        raise MlsynthDataError("`donormatrix` must have at least one donor column (`total_num_donors` > 0).")
    
    # If total_num_donors > 0 (i.e., has columns), then check if it has rows.
    # A zero size at this point means it must have 0 rows (e.g. shape (0, M) for M > 0).
    if donormatrix.size == 0:
        raise MlsynthDataError("Input `donormatrix` cannot be empty (likely 0 rows).")


    treated_outcomes_pre_treatment = y[:num_pre_treatment_periods]
    donor_outcomes_pre_treatment = donormatrix[:num_pre_treatment_periods, :]
    
    if treated_outcomes_pre_treatment.size == 0:
        raise MlsynthDataError("`treated_outcomes_pre_treatment` is empty after slicing. Check `num_pre_treatment_periods`.")
    if donor_outcomes_pre_treatment.size == 0:
         raise MlsynthDataError("`donor_outcomes_pre_treatment` is empty after slicing. Check `num_pre_treatment_periods` or `donormatrix` content.")


    sse_per_donor = np.zeros(total_num_donors)

    # Step 1: Select initial donor unit.
    # Iterate through each donor, fit a simple model (intercept + donor),
    # and calculate Sum of Squared Explained variance (SSE).
    for donor_idx in range(total_num_donors):
        # Design matrix for OLS: column of ones (intercept) and the current donor's pre-treatment outcomes.
        design_matrix = np.column_stack(
            (np.ones(num_pre_treatment_periods), donor_outcomes_pre_treatment[:, donor_idx])
        )
        try:
            # Solve (X'X)b = X'y for b (coefficients).
            coefficients = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ treated_outcomes_pre_treatment)
        except np.linalg.LinAlgError as e:
            # If matrix is singular or other issue, this donor cannot be reliably evaluated.
            raise MlsynthEstimationError(f"Linear algebra error in initial donor selection (donor {donor_idx}): {e}")
        # SSE = y_hat' * y_hat, where y_hat = Xb.
        sum_squared_explained = (design_matrix @ coefficients).T @ (design_matrix @ coefficients)
        sse_per_donor[donor_idx] = sum_squared_explained
    
    if not np.any(np.isfinite(sse_per_donor)): # Check if all SSEs are non-finite (e.g. NaN, inf)
        raise MlsynthEstimationError("All SSE calculations for initial donor selection resulted in non-finite values.")
    # Select the donor that maximizes SSE. nanargmax handles potential NaNs from LinAlgErrors if not caught.
    selected_donor_indices_iterative = np.nanargmax(sse_per_donor)

    # Step 2: Fit model using the initially selected donor unit.
    current_selection_for_fit = np.array([selected_donor_indices_iterative]) # Ensure it's an array for consistent indexing
    design_matrix = np.column_stack(
         (np.ones(num_pre_treatment_periods), donor_outcomes_pre_treatment[:, current_selection_for_fit])
    )
    try:
        coefficients = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ treated_outcomes_pre_treatment)
    except np.linalg.LinAlgError as e:
        raise MlsynthEstimationError(f"Linear algebra error in initial model fit: {e}")
    
    predicted_outcomes_pre_treatment = np.dot(design_matrix, coefficients)
    residuals_pre_treatment = treated_outcomes_pre_treatment - predicted_outcomes_pre_treatment
    # Estimate residual variance (sigma^2).
    residual_variance_pre_treatment = np.dot(residuals_pre_treatment.T, residuals_pre_treatment) / num_pre_treatment_periods
    # Calculate complexity penalty factor for the information criterion.
    complexity_penalty_factor = (
        np.log(np.log(total_num_donors)) * np.log(num_pre_treatment_periods) / num_pre_treatment_periods
    )
    # Calculate the information criterion (Q_new).
    current_information_criterion = np.log(residual_variance_pre_treatment) + complexity_penalty_factor
    previous_information_criterion = current_information_criterion + 1  # Initialize Q_old to be worse than Q_new.

    # Step 3: Iterative process to refine donor unit selection.
    # Continue adding donors as long as the information criterion improves.
    while current_information_criterion < previous_information_criterion:
        # Identify donors not yet selected.
        remaining_donor_indices = np.setdiff1d(
            np.arange(total_num_donors), selected_donor_indices_iterative
        )
        # If no more donors to consider, stop.
        if len(remaining_donor_indices) == 0: 
            break
        
        remaining_donor_outcomes_pre_treatment = donor_outcomes_pre_treatment[:, remaining_donor_indices]
        previous_information_criterion = current_information_criterion # Store current Q as Q_old for next comparison.
        sse_per_remaining_donor = np.zeros(len(remaining_donor_indices))

        # Step 4: Calculate SSE for each remaining donor unit if added to the current selection.
        for idx, current_remaining_donor_original_idx in enumerate(remaining_donor_indices):
            # Tentatively add the current remaining donor to the selected set.
            current_selection_for_sse = np.append(selected_donor_indices_iterative, current_remaining_donor_original_idx)
            design_matrix = np.column_stack(
                (np.ones(num_pre_treatment_periods), donor_outcomes_pre_treatment[:, current_selection_for_sse])
            )
            try:
                coefficients_sse = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ treated_outcomes_pre_treatment)
                sum_squared_explained_sse = (design_matrix @ coefficients_sse).T @ (design_matrix @ coefficients_sse)
                sse_per_remaining_donor[idx] = sum_squared_explained_sse
            except np.linalg.LinAlgError:
                 # If model fitting fails (e.g., singular matrix), assign a very low SSE.
                 sse_per_remaining_donor[idx] = -np.inf 

        if not np.any(np.isfinite(sse_per_remaining_donor)):
             # If all remaining donor SSEs are non-finite, cannot proceed with selection.
             break 

        # Step 5: Select the donor unit that maximizes SSE among the remaining ones.
        next_selected_donor_index_in_remaining = np.nanargmax(sse_per_remaining_donor)
        next_selected_donor_original_index = remaining_donor_indices[next_selected_donor_index_in_remaining]
        
        # Tentatively add this newly selected donor unit to the current set.
        tentative_selected_indices = np.append(selected_donor_indices_iterative, next_selected_donor_original_index)
        num_selected_donors_current_iteration = len(tentative_selected_indices)

        # Step 6: Refit the model with the new (tentative) set of donor units.
        design_matrix = np.column_stack(
            (np.ones(num_pre_treatment_periods), donor_outcomes_pre_treatment[:, tentative_selected_indices])
        )
        try:
            coefficients = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ treated_outcomes_pre_treatment)
        except np.linalg.LinAlgError: 
            # If refitting fails, stop the iteration and revert to the previous selection.
            break 
            
        predicted_outcomes_pre_treatment = np.dot(design_matrix, coefficients)
        residuals_pre_treatment = treated_outcomes_pre_treatment - predicted_outcomes_pre_treatment
        
        if num_pre_treatment_periods == 0: # Avoid division by zero, should be caught by validation
            residual_variance_pre_treatment = np.inf
        else:
            residual_variance_pre_treatment = np.dot(residuals_pre_treatment.T, residuals_pre_treatment) / num_pre_treatment_periods
        
        # Handle potential log(0) or log(negative) if residual_variance is not positive.
        if residual_variance_pre_treatment <= 0:
            log_residual_variance = -np.inf # Assign a very large negative number for Q calculation
        else:
            log_residual_variance = np.log(residual_variance_pre_treatment)

        # Recalculate complexity penalty and information criterion with the new set of donors.
        complexity_penalty_factor = (
            np.log(np.log(total_num_donors))
            * num_selected_donors_current_iteration # Penalty increases with more donors
            * np.log(num_pre_treatment_periods)
            / num_pre_treatment_periods
        )
        current_information_criterion = log_residual_variance + complexity_penalty_factor

        # If the new Q is better (lower) than the previous Q, accept the new donor.
        if current_information_criterion < previous_information_criterion:
            selected_donor_indices_iterative = tentative_selected_indices
            final_coefficients = coefficients
            final_y_hat = predicted_outcomes_pre_treatment
            final_sigma_sq = residual_variance_pre_treatment
            final_Q = current_information_criterion
        else: 
            # If Q did not improve, revert Q_new to Q_old and stop.
            current_information_criterion = previous_information_criterion 
            break 
            
    # This block executes if the while loop condition (Q_new < Q_old) was false from the start,
    # or if the loop broke due to LinAlgError, no more remaining donors, or Q not improving.
    # We need to ensure final results are from the last successful iteration (or initial fit).
    if 'final_coefficients' not in locals(): 
        # This means the loop was never entered or broke on the first attempt to add a second donor.
        # The `coefficients`, `predicted_outcomes_pre_treatment`, `residual_variance_pre_treatment`,
        # and `current_information_criterion` hold values from the initial single-donor model fit.
        final_coefficients = coefficients 
        final_y_hat = predicted_outcomes_pre_treatment
        final_sigma_sq = residual_variance_pre_treatment
        final_Q = current_information_criterion # This was Q for the first selected donor
        # Ensure selected_donor_indices_iterative is an array for consistency, holding the single best initial donor.
        if np.ndim(selected_donor_indices_iterative) == 0:
            selected_donor_indices_iterative = np.array([selected_donor_indices_iterative])


    return {
        "selected_donor_indices": selected_donor_indices_iterative if np.ndim(selected_donor_indices_iterative)>0 else np.array([selected_donor_indices_iterative]),
        "final_model_coefficients": final_coefficients,
        "final_information_criterion": final_Q,
        "final_residual_variance": final_sigma_sq,
        "predicted_treated_outcomes_pre_treatment": final_y_hat,
    }


def stepwise_donor_selection(L_full, L_post, ell_eval, m, varsigma=1e-6, tol=1e-8):
    from mlsynth.utils.estutils import _solve_SHC_QP
    """
    Performs stepwise donor selection for synthetic control estimation with BIC-style early stopping.

    This function iteratively selects donor candidates by minimizing the mean squared error (MSE)
    between the latent trend evaluation vector `ell_eval` and a weighted combination of donor segments.
    At each step, it solves a constrained quadratic programming problem to find the optimal weights.
    The process uses a Bayesian Information Criterion (BIC)-style penalty to determine when to stop
    adding donors to prevent overfitting.

    Parameters
    ----------
    L_full : np.ndarray, shape (m, N)
        Donor matrix for the pre-treatment window.
        Each column corresponds to a donor candidate's latent trend segment of length m.

    L_post : np.ndarray, shape (n, N)
        Donor matrix for the post-treatment window.
        Each column corresponds to the donor candidate's latent trend segment of length n
        immediately following the pre-treatment window.

    ell_eval : np.ndarray, shape (m,)
        The evaluation vector representing the latent trend for the most recent donor window,
        which serves as the target for donor matching.

    m : int
        Length of the donor window (number of time points per donor).

    varsigma : float, optional, default=1e-6
        Regularization parameter applied to the penalty term in the quadratic programming problem.
        Helps stabilize optimization when the donor matrix is nearly rank deficient.

    tol : float, optional, default=1e-8
        Tolerance threshold for eigenvalue cutoff when constructing the penalty term.

    Returns
    -------
    dict
        A dictionary containing:
        - "best_donors": list of int
            Indices of the selected donors providing the best fit.
        - "best_weights": np.ndarray
            Optimal weights corresponding to the selected donors.
        - "best_mse": float
            Mean squared error of the best donor combination on the evaluation vector.
        - "mse_path": list of float
            MSE values tracked at each donor addition step.
        - "bic_path": list of float
            BIC values tracked at each donor addition step.

    Notes
    -----
    - The selection proceeds by adding one donor at a time from the remaining candidates,
      choosing the donor that minimizes the MSE when combined with the already selected donors.
    - At each step, weights are obtained by solving a quadratic program with non-negativity and sum-to-one constraints.
    - A BIC-style criterion is computed to penalize model complexity (number of donors).
    - Early stopping occurs if the BIC increases for two consecutive steps after at least three donors,
      which prevents overfitting by halting the addition of less informative donors.
    - If no donor improves the MSE at a given step, the selection terminates early.

    """
    T0, N = L_full.shape
    n = L_post.shape[1]

    varsigma = 1e-6
    tol = 1e-8

    donor_indices = []
    mse_list = []
    weight_list = []
    yhat_list = []
    bic_list = []

    remaining = list(range(N))
    current_donors = []

    lambda_penalty = np.log(m)  # BIC-style penalty

    for j in range(1, N + 1):
        best_mse = np.inf
        best_idx = None
        best_w = None
        best_yhat = None

        for idx in remaining:
            candidate = current_donors + [idx]
            L_j = L_full[:, candidate]

            w, _= _solve_SHC_QP(L_j, ell_eval, use_augmented=False, varsigma=varsigma, tol=tol)

            if w is not None:
                yhat_j = L_j @ w
                mse = np.mean((ell_eval - yhat_j) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_idx = idx
                    best_w = w
                    best_yhat = yhat_j

        if best_idx is None:
            break  # no improvement, stop early

        current_donors.append(best_idx)
        remaining.remove(best_idx)

        bic_j = m * np.log(best_mse) + lambda_penalty * j
        bic_list.append(bic_j)

        # Early stopping if BIC increases for 2 consecutive steps after at least 3 donors
        if j > 3 and bic_list[-1] > bic_list[-2] and bic_list[-2] > bic_list[-3]:
            break

        donor_indices.append(current_donors.copy())
        mse_list.append(best_mse)
        weight_list.append(best_w)
        yhat_list.append(best_yhat)

    best_j = np.argmin(mse_list)

    return {
        "best_donors": donor_indices[best_j],
        "best_weights": weight_list[best_j],
        "best_mse": mse_list[best_j],
        "mse_path": mse_list,
        "bic_path": bic_list
    }
