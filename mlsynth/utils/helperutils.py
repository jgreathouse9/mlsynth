import numpy as np
from typing import List, Dict, Any, Union, Optional
from mlsynth.mlsynth import CLUSTERSC, NSC, PDA  # SCM classes from mlsynth.mlsynth
from mlsynth.utils.datautils import dataprep  # Data preparation utilities
from mlsynth.utils.estutils import pcr, RPCASYNTH, NSCcv, NSC_opt, pda  # Estimation utilities
from mlsynth.utils.resultutils import effects  # Result utilities

def _get_data(scm: Any) -> Dict[str, Any]:
    """
    Extract data from an SCM instance using dataprep.

    Args:
        scm: Initialized mlsynth SCM instance (e.g., CLUSTERSC, NSC, PDA).

    Returns:
        Dict containing donor_matrix, y, pre_periods, post_periods, donor_names, and T0.

    Raises:
        ValueError: If dataprep fails or returns multiple treated units (cohorts).
    """
    # Run dataprep to process DataFrame into SCM-ready format
    try:
        prepped = dataprep(scm.df, scm.unitid, scm.time, scm.outcome, scm.treat)
        # Ensure single treated unit case (no cohorts for multiple treated units)
        if "cohorts" in prepped:
            raise ValueError("iterative_scm supports only single treated unit case")
        # Return dictionary with required data
        return {
            "donor_matrix": prepped["donor_matrix"],  # Shape (T, N), control unit outcomes
            "y": prepped["y"],  # Shape (T,), treated unit outcome
            "pre_periods": prepped["pre_periods"],  # int, pre-treatment periods
            "post_periods": prepped["post_periods"],  # int, post-treatment periods
            "donor_names": prepped["donor_names"],  # List of control unit IDs
            "T0": prepped["pre_periods"]  # Number of pre-treatment periods
        }
    except Exception as e:
        raise ValueError(f"Failed to extract data from SCM: {str(e)}")

def _estimate_counterfactual(
    scm: Any,
    X_donors: np.ndarray,
    Y_target: np.ndarray,
    donor_names_subset: List[str],
    pre_periods: int,
    idx: int,
    spillover_indices: List[int],
    method: str = None
) -> np.ndarray:
    """
    Estimate counterfactual for a spillover donor using the SCM's method.

    Args:
        scm: Initialized mlsynth SCM instance.
        X_donors: Array of clean donor outcomes, shape (T, N_clean).
        Y_target: Outcome for spillover donor, shape (T,).
        donor_names_subset: List of clean donor unit IDs.
        pre_periods: Number of pre-treatment periods.
        idx: Index of spillover donor in donor_matrix.
        spillover_indices: List of spillover donor indices.
        method: Estimation method (e.g., 'PCR', 'LASSO'). Defaults to scm.method.

    Returns:
        Counterfactual array, shape (T,).

    Raises:
        ValueError: If method is invalid.
        NotImplementedError: If SCM class is unsupported.
    """
    # Handle CLUSTERSC (PCR, RPCA, or BOTH)
    if isinstance(scm, CLUSTERSC):
        # Normalize method to uppercase, default to scm.method
        method = method.upper() if method else scm.method.upper()
        if method not in ["PCR", "RPCA", "BOTH"]:
            raise ValueError("method must be 'PCR', 'RPCA', or 'BOTH' for CLUSTERSC")
        # Use PCR for first spillover donor in BOTH or if PCR specified
        if method == "PCR" or (method == "BOTH" and idx == spillover_indices[0]):
            result = pcr(
                X_donors,
                Y_target,
                scm.objective,
                donor_names_subset,
                X_donors,
                pre=range(pre_periods),  # Pass pre-treatment periods as range
                cluster=scm.cluster,
                Frequentist=scm.Frequentist
            )
            return result["cf_mean"]
        # Use RPCA for other cases
        else:
            # Create temporary DataFrame with spillover donor as treated
            temp_df = scm.df.copy()
            temp_df[scm.treat] = 0
            temp_df.loc[temp_df[scm.unitid] == donor_names_subset[0], scm.treat] = 1
            temp_prepped = dataprep(temp_df, scm.unitid, scm.time, scm.outcome, scm.treat)
            result = RPCASYNTH(temp_df, scm.__dict__, temp_prepped)
            return result["Vectors"]["Counterfactual"]
    
    # Handle NSC (nonlinear SCM)
    elif isinstance(scm, NSC):
        # Cross-validate to find best parameters
        best_a, best_b = NSCcv(Y_target[:pre_periods], X_donors[:pre_periods])
        # Optimize weights
        weights = NSC_opt(Y_target[:pre_periods], X_donors[:pre_periods], best_a, best_b)
        # Compute counterfactual as weighted combination of donors
        return np.dot(X_donors, weights)
    
    # Handle PDA (panel data approach)
    elif isinstance(scm, PDA):
        # Normalize method to uppercase, default to scm.method
        method = method.upper() if method else scm.method.upper()
        if method not in ["LASSO", "L2", "FS"]:
            raise ValueError("method must be 'LASSO', 'L2', or 'FS' for PDA")
        # Create temporary DataFrame with spillover donor as treated
        temp_df = scm.df.copy()
        temp_df[scm.treat] = 0
        temp_df.loc[temp_df[scm.unitid] == donor_names_subset[0], scm.treat] = 1
        temp_prepped = dataprep(temp_df, scm.unitid, scm.time, scm.outcome, scm.treat)
        # Run PDA estimation
        result = pda(temp_prepped, len(temp_prepped["donor_names"]), method=method, tau=scm.tau)
        return result["Vectors"]["Counterfactual"]
    
    # Raise error for unsupported SCM classes
    else:
        raise NotImplementedError(
            f"Iterative SCM not implemented for {type(scm).__name__}. "
            "Supported classes: CLUSTERSC, NSC, PDA. Please provide estimation logic."
        )

def iterative_scm(
    scm: Any,
    spillover_unit_ids: List[str],
    method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply Iterative Synthetic Control Method to handle spillover effects in mlsynth SCM classes.

    This function cleans spillover effects from specified donor units by estimating counterfactual
    outcomes for the provided spillover units and updating the DataFrame before running the SCM's
    fit method. Spillover units must be manually specified via `spillover_unit_ids`, which are
    mapped to columns in the donor matrix using donor names from `dataprep`. It supports single
    treated unit cases, as processed by `dataprep`, and is designed for `CLUSTERSC`, `NSC`, and
    `PDA` classes.

    Parameters
    ----------
    scm : Any
        Initialized mlsynth SCM instance (e.g., CLUSTERSC, NSC, PDA). Must have attributes
        `df`, `unitid`, `time`, `outcome`, `treat`, and a `fit` method.
    spillover_unit_ids : List[str]
        List of unit IDs (from `df[unitid]`) for donors with spillover effects. IDs must match
        `donor_names` from `dataprep` (control unit IDs). Example: `["Florida", "Nevada"]`.
    method : Optional[str], default=None
        Estimation method for the SCM (e.g., 'PCR'/'RPCA'/'BOTH' for CLUSTERSC, 'LASSO'/'L2'/'FS'
        for PDA). Defaults to `scm.method` if applicable. Case-insensitive.

    Returns
    -------
    Dict[str, Any]
        Dictionary matching the SCM's `fit` output, typically containing:
        - 'Effects': Treatment effect estimates.
        - 'Fit': Fit metrics.
        - 'Vectors': Observed, predicted, and counterfactual outcomes.
        - 'Weights': Donor weights and metadata.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., invalid unit IDs, insufficient clean donors, empty
        spillover_unit_ids, or multiple treated units).
    RuntimeError
        If SCM estimation fails during spillover cleaning or final fit.
    NotImplementedError
        If the SCM class is unsupported (only CLUSTERSC, NSC, PDA are supported).

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth.mlsynth import CLUSTERSC
    >>> from mlsynth.utils.iterative_scm import iterative_scm
    >>> # Load smoking data
    >>> url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/smoking_data.csv"
    >>> df = pd.read_csv(url)
    >>> # Configure CLUSTERSC
    >>> config = {
    ...     "df": df,
    ...     "outcome": df.columns[2],
    ...     "treat": df.columns[-1],
    ...     "unitid": df.columns[0],
    ...     "time": df.columns[1],
    ...     "display_graphs": True,
    ...     "save": False,
    ...     "counterfactual_color": "red",
    ...     "method": "PCR",
    ...     "Frequentist": True
    ... }
    >>> cluster_sc = CLUSTERSC(config)
    >>> # Run iterative_scm with manual spillover identification
    >>> results = iterative_scm(
    ...     cluster_sc,
    ...     spillover_unit_ids=["Florida", "Nevada"],
    ...     method="PCR"
    ... )
    """
    # Extract data from SCM using dataprep
    data = _get_data(scm)
    donor_matrix = data["donor_matrix"]  # Shape (T, N), control unit outcomes
    y = data["y"]  # Shape (T,), treated unit outcome
    pre_periods = data["pre_periods"]  # int, pre-treatment periods
    post_periods = data["post_periods"]  # int, post-treatment periods
    donor_names = data["donor_names"]  # List mapping donor_matrix[:, i] to unit IDs
    T0 = data["T0"]  # Pre-treatment period count

    # Validate spillover_unit_ids
    if not spillover_unit_ids:
        raise ValueError("spillover_unit_ids must be a non-empty list of unit IDs")

    # Map spillover unit IDs to donor_matrix column indices
    spillover_indices = []
    for unit_id in spillover_unit_ids:
        try:
            # Find column index in donor_matrix corresponding to unit_id
            idx = donor_names.index(unit_id)
            spillover_indices.append(idx)
        except ValueError:
            raise ValueError(f"Unit ID {unit_id} not found in donor names")
    # Ensure unique spillover IDs
    if len(set(spillover_indices)) != len(spillover_indices):
        raise ValueError("Spillover unit IDs must be unique")
    # Ensure at least 2 clean donors remain
    if len([i for i in range(len(donor_names)) if i not in spillover_indices]) < 2:
        raise ValueError("At least 2 clean donors are required")

    # Clean spillover donors by replacing their outcomes with counterfactuals
    cleaned_donor_matrix = donor_matrix.copy()  # Copy to avoid modifying original
    available_donors = [i for i in range(len(donor_names)) if i not in spillover_indices]  # Clean donor indices

    for idx in spillover_indices:
        # Extract outcome for spillover donor
        Y_target = donor_matrix[:, idx]
        # Use clean donors for counterfactual estimation
        X_donors = cleaned_donor_matrix[:, available_donors]
        # Subset donor_names for clean donors
        donor_names_subset = [donor_names[i] for i in available_donors]
        
        try:
            # Estimate counterfactual for spillover donor
            counterfactual = _estimate_counterfactual(
                scm, X_donors, Y_target, donor_names_subset, pre_periods, idx, spillover_indices, method
            )
        except Exception as e:
            raise RuntimeError(f"SCM failed for donor {donor_names[idx]}: {str(e)}")
        
        # Replace spillover donor's outcome with counterfactual
        cleaned_donor_matrix[:, idx] = counterfactual
        # Add cleaned donor back to available pool
        available_donors.append(idx)

    # Update DataFrame with cleaned outcomes
    cleaned_df = scm.df.copy()
    for idx, unit_id in enumerate(donor_names):
        if idx in spillover_indices:
            # Update outcome column for spillover unit
            unit_mask = cleaned_df[scm.unitid] == unit_id
            cleaned_df.loc[unit_mask, scm.outcome] = cleaned_donor_matrix[:, idx]

    # Run SCM with cleaned data
    try:
        # Create new SCM instance with updated DataFrame
        cleaned_scm = type(scm)({
            **scm.__dict__,
            "df": cleaned_df,
            "method": method if method and hasattr(scm, "method") else getattr(scm, "method", None)
        })
        # Run fit to get final results
        results = cleaned_scm.fit()
    except Exception as e:
        raise RuntimeError(f"Final SCM failed: {str(e)}")

    return results

def prenorm(X, target=100):
    """
    Normalize a vector or matrix so that the last row is equal to `target` (default: 100).

    Parameters:
        X (np.ndarray): Vector (1D) or matrix (2D) to normalize.
        target (float): Value to normalize the last row to (default is 100).

    Returns:
        np.ndarray: Normalized array.
    """
    X = np.asarray(X)

    denom = X[-1] if X.ndim == 1 else X[-1, :]

    # Check for division by zero
    if np.any(denom == 0):
        raise ZeroDivisionError("Division by zero in prenorm function.")

    return X / denom * target
