from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from typing import Optional, Tuple, Dict, List
import pandas as pd

def _prepare_working_df(
    df: pd.DataFrame, 
    post_col: Optional[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input DataFrame into pre- and post-treatment subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Assumed to be pre-validated.
    post_col : str or None
        Column indicating post-treatment status (0 = pre, nonzero = post).
        If None, the entire dataset is treated as pre-treatment.

    Returns
    -------
    pre_df : pd.DataFrame
        Subset of rows corresponding to the pre-treatment period.
    post_df : pd.DataFrame
        Subset of rows corresponding to the post-treatment period.
        Empty if `post_col` is None.

    Raises
    ------
    MlsynthDataError
        If `post_col` is provided but no pre-treatment rows are found.

    Notes
    -----
    - This function assumes prior validation (column existence, types, etc.).
    - Copies are returned to avoid mutating the original DataFrame.
    - Prints a summary of the split for transparency/debugging.
    """
    if post_col is not None:
        pre_mask = df[post_col] == 0
        pre_df = df[pre_mask].copy()
        post_df = df[~pre_mask].copy()

        if len(pre_df) == 0:
            raise MlsynthDataError("No pre-period data found using post_col.")

        print(f"Using post_col='{post_col}': "
              f"{len(pre_df):,} pre rows | {len(post_df):,} post rows")
    else:
        pre_df = df.copy()
        post_df = pd.DataFrame()
        print("No post_col provided → treating entire dataset as pre-treatment (design mode).")

    return pre_df, post_df



# In mlsynth/utils/fast_scm_setup.py  (or wherever your helpers live)

def build_candidate_mask(
    working_df: pd.DataFrame,
    candidate_col: str,
    unitid: str
) -> tuple[np.ndarray, set]:
    """
    Construct a boolean candidate mask aligned with sorted unit labels.

    Parameters
    ----------
    working_df : pd.DataFrame
        Pre-treatment (or working) dataset.
    candidate_col : str
        Column indicating whether a unit is a candidate (truthy = candidate).
    unitid : str
        Column identifying units.

    Returns
    -------
    candidate_mask : np.ndarray of bool, shape (J,)
        Boolean mask aligned with `unit_labels`, where True indicates a candidate unit.
    candidate_unit_set : set
        Set of unit identifiers corresponding to candidate units.
    unit_labels : np.ndarray, shape (J,)
        Sorted unique unit identifiers defining column order.

    Notes
    -----
    - Candidate status is determined at the unit level using the first observed value.
    - Alignment between mask and matrices (e.g., Y) is guaranteed via `unit_labels`.
    """
    candidate_per_unit = working_df.groupby(unitid)[candidate_col].first().astype(bool)
    unit_labels = np.sort(working_df[unitid].unique())

    candidate_mask = np.isin(unit_labels, candidate_per_unit[candidate_per_unit].index)
    candidate_unit_set = set(unit_labels[candidate_mask])

    return candidate_mask, candidate_unit_set, unit_labels


def build_Y_matrix(
    working_df: pd.DataFrame,
    outcome: str,
    time: str,
    unitid: str,
    unit_labels: np.ndarray
) -> np.ndarray:
    """
    Construct the outcome matrix Y with consistent unit column ordering.

    Parameters
    ----------
    working_df : pd.DataFrame
        Input dataset (typically pre-treatment).
    outcome : str
        Column name for the outcome variable.
    time : str
        Column name for the time index.
    unitid : str
        Column name identifying units.
    unit_labels : np.ndarray
        Ordered unit identifiers defining column alignment.

    Returns
    -------
    Y : np.ndarray, shape (T, J)
        Outcome matrix with rows as time and columns as units.

    Notes
    -----
    - Missing unit-time combinations will result in NaNs.
    - Column order strictly follows `unit_labels` to ensure alignment with other matrices.
    """
    Y_wide = working_df.pivot(
        index=time,
        columns=unitid,
        values=outcome
    ).reindex(columns=unit_labels)

    return Y_wide.to_numpy().astype(float)


def build_Z_matrix(
    working_df: pd.DataFrame,
    covariates: Optional[List[str]],
    time: str,
    unitid: str,
    unit_labels: np.ndarray
) -> Optional[np.ndarray]:
    """
    Construct the covariate matrix Z by stacking covariates vertically.

    Parameters
    ----------
    working_df : pd.DataFrame
        Input dataset.
    covariates : list of str or None
        List of covariate column names. If None or empty, no matrix is built.
    time : str
        Column name for the time index.
    unitid : str
        Column name identifying units.
    unit_labels : np.ndarray
        Ordered unit identifiers defining column alignment.

    Returns
    -------
    Z : np.ndarray, shape (T * K, J), or None
        Stacked covariate matrix, where K is the number of covariates.
        Returns None if no covariates are provided.

    Notes
    -----
    - Each covariate is pivoted into (T, J) and stacked vertically.
    - Column alignment matches Y via `unit_labels`.
    - Prints a summary of the resulting shape.
    """
    if not covariates:
        return None

    covariate_list = []
    for col in covariates:
        cov_wide = working_df.pivot(
            index=time,
            columns=unitid,
            values=col
        ).reindex(columns=unit_labels)
        covariate_list.append(cov_wide.to_numpy().astype(float))

    Z = np.vstack(covariate_list)
    print(f"Built Z matrix with {len(covariates)} covariates (stacked below Y, shape: {Z.shape})")
    return Z


def build_f_vector(
    working_df: pd.DataFrame,
    weight_col: Optional[str],
    unitid: str,
    unit_labels: np.ndarray
) -> np.ndarray:
    """
    Construct a normalized unit weight vector.

    Parameters
    ----------
    working_df : pd.DataFrame
        Input dataset.
    weight_col : str or None
        Column providing per-unit weights. If None, uniform weights are used.
    unitid : str
        Column identifying units.
    unit_labels : np.ndarray
        Ordered unit identifiers defining alignment.

    Returns
    -------
    f : np.ndarray, shape (J,)
        Normalized weight vector over units (sums to 1).

    Notes
    -----
    - If `weight_col` is provided, the first value per unit is used.
    - Weights are normalized to sum to 1.
    - Defaults to uniform weights if no column is provided.
    """
    J = len(unit_labels)

    if weight_col is not None:
        weights_per_unit = working_df.groupby(unitid)[weight_col].first().reindex(unit_labels)
        f = weights_per_unit.to_numpy().astype(float)
        f = f / f.sum()  # normalize
        print(f"Using weights from column '{weight_col}' (normalized).")
    else:
        f = np.full(J, 1.0 / J, dtype=float)
        print(f"Using uniform weights f = 1/{J} for all units.")

    return f

def prepare_experiment_inputs(
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
    m: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Combine inputs into a unified feature matrix and aligned selection structures.

    Parameters
    ----------
    Y : np.ndarray, shape (T, J)
        Outcome matrix.
    Z : np.ndarray, optional
        Covariate matrix. If provided, concatenated with Y along columns.
    f : np.ndarray, optional
        Weight vector of length N. Defaults to uniform weights if None.
    candidate_mask : np.ndarray of bool, optional
        Boolean mask indicating candidate columns. If shorter than N, it will
        be extended with False values. If None, defaults to selecting only Y columns.
    m : int, default=5
        Minimum required number of candidate units.

    Returns
    -------
    X : np.ndarray, shape (T, N)
        Combined feature matrix (Y and optionally Z).
    f : np.ndarray, shape (N,)
        Weight vector aligned with X.
    candidate_idx : np.ndarray
        Indices of candidate columns in X.
    T : int
        Number of time periods.
    N : int
        Total number of columns in X.

    Raises
    ------
    ValueError
        If candidate_mask is longer than N or if fewer than `m` candidates are available.

    Notes
    -----
    - Candidate mask is automatically extended to match the full feature dimension.
    - By default, only outcome columns (Y) are considered candidates.
    """
    X = np.concatenate([Y, Z], axis=1) if Z is not None else Y.copy()
    T, N = X.shape
    J = Y.shape[1]

    # Default weights
    if f is None:
        f = np.ones(N) / N

    # Handle candidate_mask
    if candidate_mask is None:
        candidate_mask = np.zeros(N, dtype=bool)
        candidate_mask[:J] = True  # only outcome units
    else:
        # Extend user-provided mask to match N
        if len(candidate_mask) < N:
            candidate_mask = np.concatenate([
                candidate_mask,
                np.zeros(N - len(candidate_mask), dtype=bool)
            ])
        elif len(candidate_mask) > N:
            raise ValueError(f"user-provided candidate_mask length ({len(candidate_mask)}) > N ({N})")

    candidate_idx = np.where(candidate_mask)[0]

    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    return X, f, candidate_idx, T, N




def split_periods(T0: int, T: int, frac_E: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the time series into estimation, backcast (baseline), and post-treatment periods.

    Parameters
    ----------
    T0 : int
        Number of pre-treatment time points.
    T : int
        Total number of time points in the series.
    frac_E : float, default=0.7
        Fraction of pre-treatment period to allocate for estimation (E). The remainder is used for
        baseline/backcast (B).

    Returns
    -------
    E_idx : np.ndarray, shape (int( T0*frac_E ),)
        Indices of the estimation period (pre-treatment).
    B_idx : np.ndarray, shape (T0 - int(T0*frac_E),)
        Indices of the backcast/baseline period used for evaluation.
    post_idx : np.ndarray, shape (T - T0,)
        Indices of the post-treatment period.

    Notes
    -----
    - E_idx and B_idx together span the pre-treatment period [0, T0).
    - post_idx spans the post-treatment period [T0, T).
    """
    TE = int(T0 * frac_E)                    # Estimation period length
    E_idx = np.arange(TE)
    B_idx = np.arange(TE, T0)                # Backcast / validation period
    post_idx = np.arange(T0, T)              # Post-treatment period
    return E_idx, B_idx, post_idx


def build_X_tilde(X: np.ndarray, f: np.ndarray, idx: np.ndarray, J: int):
    """
    Standardize the feature matrix X over the estimation period, returning a normalized matrix
    and its Gram matrix.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix including outcomes and covariates.
    f : np.ndarray, shape (N,)
        Weight vector (typically group means) used for centering.
    idx : np.ndarray, shape (T_E,)
        Indices corresponding to the estimation period.
    J : int
        Number of outcome columns in X (the first J columns).

    Returns
    -------
    X_E : np.ndarray, shape (T_E, N)
        Standardized X over the estimation period. Each row is centered by the weighted mean
        mu = X[:, :J] @ f[:J] and scaled by per-time standard deviation across all columns.
    G : np.ndarray, shape (N, N)
        Gram matrix of the standardized X: G = X_E.T @ X_E.

    Notes
    -----
    - Standardization is performed only over the estimation period rows (idx).
    - Weighted mean is applied to the first J columns (outcome variables) only.
    - Rows with near-zero standard deviation are scaled with sigma=1.0 to avoid division by zero.
    - This function is fully vectorized using NumPy for efficiency.
    """
    X_sub = X[idx, :]
    
    # Weighted mean over the first J columns only (outcomes)
    mu = X_sub[:, :J] @ f[:J].reshape(-1, 1)      # FIXED: use f[:J]
    
    # Per-time standard deviation (across all units)
    sigma = np.std(X_sub, axis=1, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    
    XE = (X_sub - mu) / sigma
    G = XE.T @ XE
    
    return XE, G
