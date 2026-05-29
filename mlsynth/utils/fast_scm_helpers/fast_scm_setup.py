from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from typing import Optional, Tuple, Dict, List
import pandas as pd
from typing import Any, Dict, Iterable
from ...exceptions import MlsynthDataError
from .inference import compute_moving_block_conformal_ci
from.structure import IndexSet

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
    """
    Construct boolean mask identifying candidate units.

    Parameters
    ----------
    working_df : pd.DataFrame
        Preprocessed panel data.
    candidate_col : str
        Column indicating eligibility for treatment.
    unit_index : IndexSet
        Full set of unit labels.
    unitid : str
        Unit identifier column name.

    Returns
    -------
    np.ndarray
        Boolean mask of candidate units aligned with `unit_index`.
    """
    return (
        working_df.groupby(unitid)[candidate_col]
        .first()
        .reindex(unit_index)
        .fillna(False)
        .astype(bool)
        .values
    )



def build_Y_matrix(working_df, outcome, time, unitid, unit_index):
    """
    Construct outcome matrix in (time × unit) format.

    Parameters
    ----------
    working_df : pd.DataFrame
        Input panel data.
    outcome : str
        Outcome variable name.
    time : str
        Time index column.
    unitid : str
        Unit identifier column.
    unit_index : IndexSet
        Ordered unit index.

    Returns
    -------
    np.ndarray, shape (T, J)
        Wide-format outcome matrix.
    """
    Y_wide = (
        working_df.pivot(index=time, columns=unitid, values=outcome)
        .reindex(columns=unit_index)
    )
    return Y_wide.to_numpy(dtype=float)


def build_Z_matrix(working_df, covariates, time, unitid, unit_index):
    """
    Construct stacked covariate matrix, collapsing time-invariant
    variables to a single row to prevent unintended over-weighting.

    Parameters
    ----------
    working_df : pd.DataFrame
        Input panel data.
    covariates : list of str
        List of covariate column names.
    time : str
        Time column.
    unitid : str
        Unit identifier column.
    unit_index : IndexSet
        Unit ordering.

    Returns
    -------
    np.ndarray or None
        Stacked covariate matrix (N rows x Units).
    """
    if not covariates:
        return None

    covariate_list = []
    for col in covariates:
        # Pivot to get (Time x Units)
        cov_wide = (
            working_df.pivot(index=time, columns=unitid, values=col)
            .reindex(columns=unit_index)
        )

        # Check if the covariate is time-invariant for each unit.
        # If the number of unique values in every column is 1, it's invariant.
        is_invariant = cov_wide.nunique(axis=0).max() <= 1

        if is_invariant:
            # Collapse to a single row (1 x Units)
            # This ensures this covariate has the same 'weight' as one year of outcomes
            cov_data = cov_wide.iloc[0:1, :].to_numpy(dtype=float)
        else:
            # Keep the full time-series (Time x Units)
            # Use this for variables like annual unemployment rates
            cov_data = cov_wide.to_numpy(dtype=float)

        covariate_list.append(cov_data)

    # Stack vertically: total rows will be (Num_Invariant + (Num_Variant * T))
    return np.vstack(covariate_list)

def build_f_vector(working_df, weight_col, unitid, unit_index):
    """
    Construct unit weighting vector.

    Parameters
    ----------
    working_df : pd.DataFrame
        Input dataset.
    weight_col : str or None
        Optional column defining unit weights.
    unitid : str
        Unit identifier column.
    unit_index : IndexSet
        Unit ordering.

    Returns
    -------
    np.ndarray
        Normalized weight vector over units.
    """
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
    # STAGE 1: Vertical stacking of outcomes (T x J) and covariates (K x J)
    # Resulting X shape: (T + K, J)
    X = np.concatenate([Y, Z], axis=0) if Z is not None else Y.copy()

    num_features, J = X.shape

    # Default weights (f) should be length J (number of units)
    if f is None:
        f = np.ones(J) / J

    # Handle candidate_mask (who can be in the treated m-tuple)
    # This must be length J, as it refers to units, not features.
    if candidate_mask is None:
        candidate_idx = np.arange(J)
    else:
        candidate_idx = np.where(candidate_mask)[0]

    if len(candidate_idx) < m:
        raise ValueError(f"Not enough candidate units: {len(candidate_idx)} < m={m}")

    return X, f, candidate_idx, num_features, J


def split_periods(
        T0: int,
        n_covariates: int,
        frac_E: float = 0.7,
        post_df: Optional[pd.DataFrame] = None,
        time_col: str = "time"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Returns indices and time-period counts."""
    # 1. Split the Time-Series (the visible X-axis)
    n_fit_time = int(T0 * frac_E)
    n_blank_time = T0 - n_fit_time

    E_time_idx = np.arange(n_fit_time)
    B_idx = np.arange(n_fit_time, T0)

    # 2. Add Covariates to the Estimation Index (The math/QP side)
    cov_idx = np.arange(T0, T0 + n_covariates)
    E_idx = np.concatenate([E_time_idx, cov_idx]).astype(int)

    # 3. Post-period
    if post_df is not None and not post_df.empty:
        n_post = post_df[time_col].nunique()
    else:
        n_post = 0
    post_idx = np.arange(T0, T0 + n_post)

    # Return indices PLUS the time-counts for the plotter
    return E_idx, B_idx, post_idx, n_fit_time, n_blank_time


def build_X_tilde(X: np.ndarray, f: np.ndarray, idx: np.ndarray):
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

    # Calculate the target population mean for each feature (row)
    # mu shape: (len(idx), 1)
    mu = (X_sub @ f).reshape(-1, 1)

    # Per-row standard deviation for scaling
    # This is critical when mixing outcomes (e.g., USD) with covariates (e.g., Percentages)
    sigma = np.std(X_sub, axis=1, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    # Standardize: (Unit_Value - Population_Mean) / Sigma
    XE = (X_sub - mu) / sigma

    # The Gram matrix G (J x J) represents the feature-weighted distance between units.
    # Minimizing w'Gw minimizes the MSE across all included outcomes and covariates.
    G = XE.T @ XE

    return XE, G



def _run_post_intervention_updates(
    candidate_results,
    Y_pre,
    post_df,
    post_idx,
    unit_index,
    unitid,
    time,
    outcome,
    n_sims,
    alpha,
    seed
):
    """
    Extend all candidate results into the post-intervention period and compute
    final inference statistics.

    This includes:
        - constructing full outcome matrix (pre + post)
        - recomputing synthetic treated/control paths
        - computing effects
        - estimating ATE
        - running inference (p-values + conformal CI)

    Returns
    -------
    y_pop_mean_t : np.ndarray
        Population mean over full time horizon.
    candidate_results : list
        Updated candidates with post-intervention quantities.
    """

    # =========================================================
    # 1. Default baseline (pre-period only)
    # =========================================================
    y_pop_mean_t = Y_pre.mean(axis=1)

    # No post period → nothing to update
    if len(post_idx) == 0 or post_df is None or post_df.empty:
        return y_pop_mean_t, candidate_results

    # =========================================================
    # 2. Build full outcome matrix
    # =========================================================
    Y_post = build_Y_matrix(
        working_df=post_df,
        outcome=outcome,
        time=time,
        unitid=unitid,
        unit_index=unit_index
    )

    Y_full = np.vstack([Y_pre, Y_post])
    y_pop_mean_t = Y_full.mean(axis=1)

    # =========================================================
    # 3. Update each candidate
    # =========================================================
    for cand in candidate_results:

        treated_col_idx = np.asarray(cand.identification.treated_idx, dtype=int)

        w_t = cand.weights.treated
        w_c = cand.weights.control

        # Synthetic paths
        synth_treated = Y_full[:, treated_col_idx] @ w_t
        synth_control = Y_full @ w_c

        effects = synth_treated - synth_control

        cand.predictions.synthetic_treated = synth_treated
        cand.predictions.synthetic_control = synth_control
        cand.predictions.effects = effects

        # ATE (post period only)
        post_effects = effects[post_idx]
        cand.inference.ate = float(np.mean(post_effects))

        cand.inference.treated_col_idx = treated_col_idx.tolist()

        # conformal CI
        cand2 = compute_moving_block_conformal_ci(
            candidate=cand,
            post_idx=post_idx,
            alpha=alpha,
            seed=seed
        )

    return y_pop_mean_t, candidate_results
