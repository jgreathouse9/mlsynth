from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from typing import Optional, Tuple, Dict, List
import pandas as pd
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class IndexSet:
    labels: np.ndarray
    label_to_idx: Dict[Any, int]

    @classmethod
    def from_labels(cls, labels: Iterable[Any]) -> "IndexSet":
        labels = np.asarray(list(labels))
        return cls(
            labels=labels,
            label_to_idx={label: i for i, label in enumerate(labels)}
        )

    # -----------------------------
    # Core mapping utilities
    # -----------------------------
    def get_labels(self, indices):
        return self.labels[np.asarray(indices)]

    def get_index(self, labels):
        return np.array([self.label_to_idx[l] for l in labels])

    # -----------------------------
    # Python / NumPy interoperability
    # -----------------------------
    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __array__(self):
        return self.labels

    def __repr__(self):
        return f"IndexSet(n={len(self.labels)})"

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
def build_candidate_mask(working_df, candidate_col, unit_index, unitid):
    return (
        working_df.groupby(unitid)[candidate_col]
        .first()
        .reindex(unit_index)
        .fillna(False)
        .astype(bool)
        .values
    )



def build_Y_matrix(working_df, outcome, time, unitid, unit_index):
    Y_wide = (
        working_df.pivot(index=time, columns=unitid, values=outcome)
        .reindex(columns=unit_index)
    )
    return Y_wide.to_numpy(dtype=float)



def build_Z_matrix(working_df, covariates, time, unitid, unit_index):
    if not covariates:
        return None

    covariate_list = []
    for col in covariates:
        cov_wide = (
            working_df.pivot(index=time, columns=unitid, values=col)
            .reindex(columns=unit_index)
        )
        covariate_list.append(cov_wide.to_numpy(dtype=float))

    return np.vstack(covariate_list)


def build_f_vector(working_df, weight_col, unitid, unit_index):
    J = len(unit_index)

    if weight_col is not None:
        weights = (
            working_df.groupby(unitid)[weight_col]
            .first()
            .reindex(unit_index)
            .to_numpy(dtype=float)
        )
        return weights / weights.sum()
    else:
        return np.full(J, 1.0 / J)





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





def split_periods(
    T0: int, 
    frac_E: float = 0.7, 
    post_df: Optional[pd.DataFrame] = None,
    time_col: str = "time"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # 1. Calculate Pre-period splits
    TE = int(T0 * frac_E)
    E_idx = np.arange(TE)
    B_idx = np.arange(TE, T0)

    # 2. Determine Post-period length
    # If post_df is provided, we count unique time periods in it.
    # Otherwise, we assume a Design-only mode with 0 post-periods.
    if post_df is not None and not post_df.empty:
        n_post = post_df[time_col].nunique()
    else:
        n_post = 0

    # 3. Create indices relative to the full Y matrix (size T0 + n_post)
    post_idx = np.arange(T0, T0 + n_post)

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
