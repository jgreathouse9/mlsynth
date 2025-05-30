import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import chi2
from typing import Tuple
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError # Import custom exception


def normalize(Y: np.ndarray) -> np.ndarray:
    """Column-wise mean normalization of a matrix.

    Subtracts the column mean from each element in that column.

    Parameters
    ----------
    Y : np.ndarray
        The input matrix to be normalized, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The column-wise mean-normalized matrix, shape (n_samples, n_features).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.selectorsutils import normalize
    >>> Y = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> normalize(Y)
    array([[-1., -1., -1.],
           [ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    """
    # Input validation: Ensure Y is a NumPy array.
    if not isinstance(Y, np.ndarray):
        raise MlsynthDataError("Input `Y` must be a NumPy array.")
    # Ensure Y is at least 1D.
    if Y.ndim == 0: # Catches 0D arrays
        raise MlsynthDataError("Input `Y` must be at least 1D.")
    # Specific check for 2D array with 0 columns, as mean calculation would be problematic.
    if Y.ndim == 2 and Y.shape[1] == 0:
        raise MlsynthDataError("Input `Y` has 0 columns, cannot normalize.")
    # General check for empty array (e.g., 1D empty, or 2D empty in rows if not caught by 0 columns).
    if Y.size == 0:
        raise MlsynthDataError("Input `Y` cannot be empty.")
    # Ensure Y is not more than 2D.
    if Y.ndim > 2:
        raise MlsynthDataError("Input `Y` must be 1D or 2D.")

    # Perform column-wise mean normalization.
    # Y.mean(axis=0, keepdims=True) calculates the mean of each column and keeps the dimension
    # for broadcasting, so it subtracts the respective column mean from each element.
    # For 1D array, Y.mean(axis=0) works correctly.
    # If a column has all same values, mean is that value, Y - mean results in zeros for that column.
    # This operation does not involve division, so no division by zero risk.
    return Y - Y.mean(axis=0, keepdims=True)


def granger_mask( # noqa: C901
    y: np.ndarray, Y0: np.ndarray, T0: int, alpha: float = 0.05, maxlag: int = 1
) -> np.ndarray:
    """Create a boolean mask indicating if columns of Y0 Granger-cause y.

    Parameters
    ----------
    y : np.ndarray
        The target time series, shape (T,). Should be 1D or (T,1).
    Y0 : np.ndarray
        The matrix of potential predictor time series (donors), shape (T, J).
        Each column `j` is a time series for a donor unit.
    T0 : int
        Number of pre-treatment periods to use for the Granger causality test.
        `y` and `Y0` will be sliced as `[:T0]`.
    alpha : float, optional
        Significance level for the Granger causality test. Default is 0.05.
    maxlag : int, optional
        The maximum lag order for the Granger causality test. Default is 1.

    Returns
    -------
    np.ndarray
        A boolean array of shape (J,), where `True` indicates that the
        corresponding column in `Y0` Granger-causes `y` at the given
        significance level `alpha`.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.selectorsutils import granger_mask
    >>> T, J, T0 = 100, 3, 80
    >>> y_data = np.random.rand(T)
    >>> Y0_data = np.random.rand(T, J)
    >>> # Example: Y0_data[:, 0] is made to be correlated with y_data with a lag
    >>> for i in range(1, T):
    ...     Y0_data[i, 0] = 0.5 * y_data[i-1] + 0.5 * Y0_data[i-1, 0] + 0.1 * np.random.randn()
    >>> mask = granger_mask(y_data, Y0_data, T0, alpha=0.05, maxlag=1) # doctest: +SKIP
    >>> print(mask.shape) # doctest: +SKIP
    (3,)
    >>> # Expected: mask[0] might be True, others likely False (probabilistic) # doctest: +SKIP
    """
    # Input validation
    # Validate target time series `y`.
    if not isinstance(y, np.ndarray):
        raise MlsynthDataError("Input `y` must be a NumPy array.")
    if y.ndim == 2 and y.shape[1] != 1: # If 2D, must be a column vector.
        raise MlsynthDataError("Input `y` must be 1D or have shape (T, 1).")
    if y.ndim > 2: # Must be 1D or 2D.
        raise MlsynthDataError("Input `y` must be 1D or 2D.")
    if y.size == 0: # Cannot be empty.
        raise MlsynthDataError("Input `y` cannot be empty.")

    # Validate predictor time series matrix `Y0`.
    if not isinstance(Y0, np.ndarray):
        raise MlsynthDataError("Input `Y0` must be a NumPy array.")
    if Y0.ndim != 2: # Must be 2D.
        raise MlsynthDataError("Input `Y0` must be a 2D array.")
    if Y0.size == 0: # Cannot be empty.
        raise MlsynthDataError("Input `Y0` cannot be empty.")

    # Ensure `y` and `Y0` have the same number of time periods (rows).
    if y.shape[0] != Y0.shape[0]:
        raise MlsynthDataError(
            f"Shape mismatch: `y` has {y.shape[0]} time periods, "
            f"`Y0` has {Y0.shape[0]} time periods."
        )

    # Validate `T0` (number of pre-treatment periods for the test).
    if not isinstance(T0, int):
        raise MlsynthConfigError("Input `T0` must be an integer.")
    if not (0 < T0 <= y.shape[0]): # Must be positive and within bounds.
        raise MlsynthConfigError(
            f"`T0` ({T0}) must be positive and not exceed total time periods ({y.shape[0]})."
        )

    # Validate `alpha` (significance level).
    if not isinstance(alpha, float):
        raise MlsynthConfigError("Input `alpha` must be a float.")
    if not (0 < alpha < 1): # Must be between 0 and 1.
        raise MlsynthConfigError("Input `alpha` must be between 0 and 1 (exclusive).")

    # Validate `maxlag` (maximum lag order for the test).
    if not isinstance(maxlag, int):
        raise MlsynthConfigError("Input `maxlag` must be an integer.")
    if maxlag <= 0: # Must be positive.
        raise MlsynthConfigError("Input `maxlag` must be positive.")
    
    # Check if T0 is sufficient for the given maxlag.
    # statsmodels grangercausalitytests requires enough observations for the regressions.
    # A common rule of thumb or requirement for stability.
    if T0 <= 4 * maxlag + 2 : 
        raise MlsynthConfigError(
            f"`T0` ({T0}) may be too small for `maxlag` ({maxlag}). "
            f"Ensure T0 > 4 * maxlag + 2. (e.g., T0={4*maxlag+3})"
        )

    mask = [] # Initialize a list to store boolean mask results.
    # Prepare the target series `y` for the test, using only pre-treatment periods and flattening if 2D.
    y_series_for_test = y[:T0].flatten() 

    # Iterate over each column (donor unit) in Y0.
    for j in range(Y0.shape[1]):
        try:
            # Create a DataFrame with the target series `y` and the current predictor series `x` (donor j).
            df = pd.DataFrame({"y": y_series_for_test, "x": Y0[:T0, j]})
            # Perform the Granger causality test. `verbose=False` suppresses printed output.
            result = grangercausalitytests(df[["y", "x"]], maxlag=maxlag, verbose=False)
            # Extract the p-value for the F-test from the results at the specified maxlag.
            # result[maxlag] accesses results for that lag.
            # [0] accesses the dictionary of test statistics.
            # ['ssr_ftest'] accesses the F-test results (sum of squared residuals F-test).
            # [1] is the p-value.
            pval = result[maxlag][0]["ssr_ftest"][1]
            # Append True to mask if p-value is less than alpha (significant Granger causality).
            mask.append(pval < alpha)
        except Exception as e: # Catch any errors during the test.
            raise MlsynthEstimationError(f"Granger causality test failed for donor column {j}: {e}")
    return np.array(mask) # Convert list of booleans to a NumPy array.


def proximity_mask( # noqa: C901
    Y0: np.ndarray, T0: int, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a mask based on proximity of donor units to the average of other donors.

    Calculates a distance for each donor unit based on its squared deviation
    from the average of all other donor units in the pre-treatment period.
    A unit is masked if this distance is below a threshold derived from the
    Chi-squared distribution with `T0` degrees of freedom.

    Parameters
    ----------
    Y0 : np.ndarray
        The matrix of donor unit outcomes, shape (T, num_donors). Each column
        is a time series for a donor unit.
    T0 : int
        Number of pre-treatment periods to use for calculating distances.
        `Y0` will be sliced as `Y0[:T0, :]`.
    alpha : float, optional
        Significance level for the Chi-squared threshold. Default is 0.05.
        A lower `alpha` makes the threshold higher, thus selecting fewer donors
        (those that are "closer" or less deviant).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - mask (np.ndarray): Boolean array of shape (num_donors,). `True` if a donor
          is considered "proximal" (its distance is below the threshold).
        - dists (np.ndarray): Array of calculated distances for each donor,
          shape (num_donors,).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.selectorsutils import proximity_mask
    >>> T, J, T0 = 50, 4, 40
    >>> Y0_data = np.random.rand(T, J)
    >>> # Make one donor very different
    >>> Y0_data[:, 2] += 10
    >>> mask, dists = proximity_mask(Y0_data, T0, alpha=0.05)
    >>> print(mask.shape)
    (4,)
    >>> print(dists.shape)
    (4,)
    >>> # Expected: mask[2] might be False if it's too distant, others True (probabilistic)
    >>> # print(mask) # e.g. [ True  True False  True]
    """
    # Input validation
    # Validate donor matrix Y0.
    if not isinstance(Y0, np.ndarray):
        raise MlsynthDataError("Input `Y0` must be a NumPy array.")
    if Y0.ndim != 2: # Must be 2D.
        raise MlsynthDataError("Input `Y0` must be a 2D array.")
    if Y0.shape[1] == 0: # Must have at least one donor.
        raise MlsynthDataError("Input `Y0` must have at least one donor column.")
    if Y0.size == 0: # Cannot be empty.
        raise MlsynthDataError("Input `Y0` cannot be empty.")

    # Validate T0 (number of pre-treatment periods).
    if not isinstance(T0, int):
        raise MlsynthConfigError("Input `T0` must be an integer.")
    if not (0 < T0 <= Y0.shape[0]): # Must be positive and within bounds.
        raise MlsynthConfigError(
            f"`T0` ({T0}) must be positive and not exceed total time periods ({Y0.shape[0]})."
        )

    # Validate alpha (significance level).
    if not isinstance(alpha, float):
        raise MlsynthConfigError("Input `alpha` must be a float.")
    if not (0 < alpha < 1): # Must be between 0 and 1.
        raise MlsynthConfigError("Input `alpha` must be between 0 and 1 (exclusive).")

    num_donors = Y0.shape[1]
    dists = np.zeros(num_donors) # Initialize array to store distances for each donor.
    # Slice Y0 to include only pre-treatment periods.
    Y0_pre_treatment = Y0[:T0, :]

    # Iterate over each donor unit to calculate its distance.
    for donor_idx in range(num_donors):
        # Create a mask to select all other donor units.
        others_mask = np.ones(num_donors, dtype=bool)
        others_mask[donor_idx] = False
        others = Y0_pre_treatment[:, others_mask] # Matrix of other donor units' pre-treatment outcomes.

        # If there are no other donors (e.g., only one donor in Y0), distance is 0.
        if others.shape[1] == 0:
            dists[donor_idx] = 0.0
            continue
        
        # Calculate the average outcome trajectory of all other donor units.
        avg_others = others.mean(axis=1)
        # Calculate the squared deviation of the current donor from this average, summed over pre-treatment periods, then normalized by T0.
        dists[donor_idx] = np.sum((Y0_pre_treatment[:, donor_idx] - avg_others) ** 2) / T0
    
    try:
        # Calculate the threshold using the percent point function (inverse CDF) of the Chi-squared distribution.
        # `1 - alpha` is used because we are interested in the upper tail (large distances are outliers).
        # Degrees of freedom `df` is T0.
        threshold = chi2.ppf(1 - alpha, df=T0)
    except Exception as e: # Catch errors during threshold calculation.
        raise MlsynthEstimationError(f"Failed to compute chi-squared threshold: {e}")
        
    # Return a boolean mask (True if distance < threshold) and the calculated distances.
    return dists < threshold, dists


def rbf_scores(dists: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Calculate Radial Basis Function (RBF) scores from distances.

    The RBF score is calculated as `exp(-dists^2 / (2 * sigma^2))`.
    Higher scores indicate closer proximity (smaller distances).

    Parameters
    ----------
    dists : np.ndarray
        An array of distances, shape (N,).
    sigma : float, optional
        The width parameter (standard deviation) of the RBF kernel.
        Controls the "reach" of the kernel. Default is 1.0.

    Returns
    -------
    np.ndarray
        An array of RBF scores, shape (N,). Scores are between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.selectorsutils import rbf_scores
    >>> distances = np.array([0, 1, 2, 3])
    >>> scores = rbf_scores(distances, sigma=1.0)
    >>> print(scores)
    [1.         0.60653066 0.13533528 0.011109  ]
    >>> scores_wide_sigma = rbf_scores(distances, sigma=2.0)
    >>> print(scores_wide_sigma)
    [1.         0.8824969  0.60653066 0.32465247]
    """
    # Input validation
    # Ensure `dists` is a 1D NumPy array.
    if not isinstance(dists, np.ndarray):
        raise MlsynthDataError("Input `dists` must be a NumPy array.")
    if dists.ndim != 1:
        raise MlsynthDataError("Input `dists` must be a 1D array.")
    
    # Ensure `sigma` is a positive number.
    if not isinstance(sigma, (float, int)):
        raise MlsynthConfigError("Input `sigma` must be a float or integer.")
    if sigma <= 0:
        raise MlsynthConfigError("Input `sigma` must be positive.")
        
    # Calculate RBF scores using the formula: exp(-dists^2 / (2 * sigma^2)).
    # This transforms distances into similarity scores, where smaller distances yield scores closer to 1.
    return np.exp(-(dists**2) / (2 * sigma**2))


def ansynth_select_donors(
    y: np.ndarray, Y0: np.ndarray, T0: int, alpha: float = 0.1, sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select donor units using a hybrid anomaly-based approach.

    This method combines Granger causality, proximity masking, and RBF scores
    to select a relevant subset of donor units from `Y0` for modeling `y`.
    It first normalizes the pre-treatment data, then applies Granger and
    proximity masks. RBF scores are calculated from proximity distances,
    and donors are selected if they pass both masks and have positive RBF scores.

    Parameters
    ----------
    y : np.ndarray
        The target time series (treated unit outcomes), shape (T,) or (T, 1).
    Y0 : np.ndarray
        The matrix of potential donor unit outcomes, shape (T, J). Each column
        `j` is a time series for a donor unit.
    T0 : int
        Number of pre-treatment periods. Data is sliced `[:T0]` for mask
        calculations, but `Y0_filtered` returns full-period data for selected donors.
    alpha : float, optional
        Significance level used for Granger causality and proximity masks.
        Default is 0.1.
    sigma : float, optional
        Width parameter for the RBF kernel used in scoring. Default is 1.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - selected_donor_indices (np.ndarray): 1D array of integer indices of
          the selected donor units from the original `Y0` matrix,
          shape (num_selected,).
        - filtered_donor_outcomes (np.ndarray): The outcome matrix of selected
          donor units, shape (T, num_selected_donors).
        - filtered_donor_scores (np.ndarray): 1D array of the RBF scores for
          the selected donors, shape (num_selected,).

    Examples
    --------
    >>> import numpy as np
    >>> from mlsynth.utils.selectorsutils import ansynth_select_donors
    >>> T, J, T0 = 100, 5, 80
    >>> y_data = np.random.rand(T)
    >>> Y0_data = np.random.rand(T, J)
    >>> # Make one donor Granger-cause y and be proximal
    >>> for i in range(1, T0):
    ...     Y0_data[i, 0] = 0.6 * y_data[i-1] + 0.4 * Y0_data[i-1, 0] + 0.05 * np.random.randn()
    ...     Y0_data[i, 1] = y_data[i] * 0.8 + 0.1 * np.random.randn() # Proximal
    >>> # Make another donor distant
    >>> Y0_data[:T0, 3] += 5
    >>>
    >>> keep_indices, Y0_f, S_diag_f = ansynth_select_donors(
    ...     y_data, Y0_data, T0, alpha=0.1, sigma=1.0
    ... ) # doctest: +SKIP
    >>> print(f"Selected donor indices: {keep_indices}") # doctest: +SKIP
    >>> print(f"Shape of filtered Y0: {Y0_f.shape}") # doctest: +SKIP
    >>> print(f"RBF scores of selected: {S_diag_f}") # doctest: +SKIP
    """
    # Input validation (similar to granger_mask and proximity_mask)
    # Validate target series `y`.
    if not isinstance(y, np.ndarray):
        raise MlsynthDataError("Input `y` must be a NumPy array.")
    if y.ndim == 2 and y.shape[1] != 1:
        raise MlsynthDataError("Input `y` must be 1D or have shape (T, 1).")
    if y.ndim > 2:
        raise MlsynthDataError("Input `y` must be 1D or 2D.")
    if y.size == 0:
        raise MlsynthDataError("Input `y` cannot be empty.")

    # Validate donor matrix `Y0`.
    if not isinstance(Y0, np.ndarray):
        raise MlsynthDataError("Input `Y0` must be a NumPy array.")
    if Y0.ndim != 2:
        raise MlsynthDataError("Input `Y0` must be a 2D array.")
    if Y0.shape[1] == 0: 
        raise MlsynthDataError("Input `Y0` must have at least one donor column.")
    if Y0.size == 0: 
        raise MlsynthDataError("Input `Y0` cannot be empty.")

    # Ensure consistent time periods.
    if y.shape[0] != Y0.shape[0]:
        raise MlsynthDataError(
            f"Shape mismatch: `y` has {y.shape[0]} time periods, "
            f"`Y0` has {Y0.shape[0]} time periods."
        )

    # Validate `T0`.
    if not isinstance(T0, int):
        raise MlsynthConfigError("Input `T0` must be an integer.")
    if not (0 < T0 <= y.shape[0]):
        raise MlsynthConfigError(
            f"`T0` ({T0}) must be positive and not exceed total time periods ({y.shape[0]})."
        )

    # Validate `alpha`.
    if not isinstance(alpha, float):
        raise MlsynthConfigError("Input `alpha` must be a float.")
    if not (0 < alpha < 1):
        raise MlsynthConfigError("Input `alpha` must be between 0 and 1 (exclusive).")

    # Validate `sigma`.
    if not isinstance(sigma, (float, int)): 
        raise MlsynthConfigError("Input `sigma` must be a float or integer.")
    if sigma <= 0:
        raise MlsynthConfigError("Input `sigma` must be positive.")

    # Slice data to the pre-treatment period defined by T0.
    y_pre_treatment = y[:T0]
    donor_outcomes_pre_treatment = Y0[:T0, :]

    # Ensure pre-treatment target series is not empty after slicing.
    if y_pre_treatment.size == 0:
        raise MlsynthDataError("Pre-treatment period for `y` is empty after slicing with T0.")
    
    # Normalize the pre-treatment target series (mean subtraction).
    y_norm_pre_treatment = y_pre_treatment - y_pre_treatment.mean()
    # Normalize the pre-treatment donor outcomes (column-wise mean subtraction).
    normalized_donor_outcomes_pre_treatment = normalize(donor_outcomes_pre_treatment)

    # Step 1: Apply Granger causality mask.
    # Identifies donors whose past values significantly predict the target series' past values.
    granger_causality_mask_vals = granger_mask(
        y_norm_pre_treatment, normalized_donor_outcomes_pre_treatment, T0, alpha=alpha
    )
    
    # Step 2: Apply proximity mask and get distances.
    # Identifies donors that are "close" to the average of other donors in the pre-treatment period.
    proximity_mask_values, proximity_distances = proximity_mask(
        normalized_donor_outcomes_pre_treatment, T0, alpha=alpha
    )
    
    # Combine Granger and proximity masks: a donor must pass both.
    hybrid_mask = granger_causality_mask_vals & proximity_mask_values

    # Step 3: Calculate RBF scores from proximity distances.
    # Converts distances to similarity scores, where smaller distances (closer proximity) get higher scores.
    rbf_proximity_scores = rbf_scores(proximity_distances, sigma=sigma)
    
    # Calculate final scores for donors: RBF score if they pass the hybrid mask, 0 otherwise.
    # This effectively weights the RBF scores by the hybrid (Granger & proximity) selection.
    final_donor_scores = hybrid_mask * rbf_proximity_scores

    # Select donors that have a positive final score.
    # This means they passed both Granger and proximity tests, and their RBF score (derived from proximity) is positive.
    selected_donor_indices = np.where(final_donor_scores > 0)[0]
    
    # Filter the original full-period donor outcomes matrix Y0 to include only selected donors.
    filtered_donor_outcomes = Y0[:, selected_donor_indices]
    # Get the final scores for the selected donors.
    filtered_donor_scores = final_donor_scores[selected_donor_indices]

    return selected_donor_indices, filtered_donor_outcomes, filtered_donor_scores

# SVDCluster MOVED to selector_helpers.py

# determine_optimal_clusters MOVED to selector_helpers.py

# fpca MOVED to selector_helpers.py

# PDAfs MOVED to selector_helpers.py
