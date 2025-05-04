import numpy as np
import pandas as pd


def treatlogic(treatment_matrix: np.ndarray):
    if not isinstance(treatment_matrix, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    if treatment_matrix.ndim != 2:
        raise ValueError("Treatment matrix must be 2D")

    treated_mask = treatment_matrix.sum(axis=1) > 0
    treated_indices = list(np.where(treated_mask)[0])
    num_treated_units = len(treated_indices)

    assert num_treated_units > 0, "There must be at least one treated unit"

    num_periods = treatment_matrix.shape[1]

    # === SINGLE TREATED UNIT CASE ===
    if num_treated_units == 1:
        first_treatment_time = np.argmax(treatment_matrix[treated_indices[0]] == 1)
        total_periods = treatment_matrix.shape[1]
        pre_periods = first_treatment_time
        post_periods = total_periods - pre_periods

        return {
            "Num Treated Units": num_treated_units,
            "Treated Index": treated_indices,
            "Pre Periods": pre_periods,
            "Post Periods": post_periods,
            "Total Periods": total_periods,
        }

    # === MULTIPLE TREATED UNITS CASE ===
    else:
        first_treat_periods = np.full(treatment_matrix.shape[0], fill_value=np.nan)
        for unit_idx in treated_indices:
            treat_vector = treatment_matrix[unit_idx]
            treat_times = np.where(treat_vector == 1)[0]
            assert len(treat_times) > 0, f"Unit {unit_idx} has no post-treatment period"
            first_treat_periods[unit_idx] = treat_times[0]
            assert np.all(treat_vector[treat_times[0]:] == 1), f"Treatment is not sustained for unit {unit_idx}"

        first_treat_periods_clean = first_treat_periods[treated_indices].astype(int)
        pre_periods = first_treat_periods_clean
        post_periods = num_periods - pre_periods

        return {
            "Num Treated Units": num_treated_units,
            "Treated Index": treated_indices,
            "First Treat Periods": first_treat_periods_clean,
            "Pre Periods by Unit": pre_periods,
            "Post Periods by Unit": post_periods,
            "Total Periods": num_periods,
        }



def balance(df, unit_col, time_col):
    """
    Check if the panel is strongly balanced.

    Parameters:
    - df: DataFrame
    - unit_col: str, column name representing the units/groups
    - time_col: str, column name representing the time variable

    Raises:
    - ValueError: If the panel is not strongly balanced

    Returns:
    - None if the panel is strongly balanced
    """
    # Check for unique observations
    if df.duplicated([unit_col, time_col]).any():
        raise ValueError(
            "Duplicate observations found. Ensure each combination of unit and time is unique."
        )

    # Group by unit and count the number of observations for each unit
    counts = df.groupby([unit_col, time_col]).size().unstack(fill_value=0)

    # Check if all units have the same number of observations
    is_balanced = (counts.nunique(axis=1) == 1).all()

    if not is_balanced:
        raise ValueError("The panel is not strongly balanced.")



def clean_surrogates2(X, Z0, W, T0, Cy=None):
    """
    Cleans surrogate variables using the provided inputs and returns the updated X.

    Parameters:
    X (ndarray): Matrix of surrogate variables.
    Z0 (ndarray): Matrix of pre-treatment covariates for the donor pool.
    W (ndarray): Matrix of pre-treatment covariates for the treated unit.
    T0 (int): Time point before treatment.
    Cy (ndarray, optional): Additional covariates (default is None).

    Returns:
    ndarray: Updated surrogate variable matrix.
    """
    tauts = []
    for i in range(X.shape[1]):
        X1 = np.copy(X[:, i])
        if Cy is not None:
            Z0_aug = np.column_stack((Z0, Cy))
            W_aug = np.column_stack((W, Cy))
        else:
            Z0_aug = Z0
            W_aug = W
        Y = X1
        Z0W = Z0_aug[:T0].T @ W_aug[:T0]
        Z0Y = Z0_aug[:T0].T @ Y[:T0]
        alpha = np.linalg.solve(Z0W, Z0Y)
        taut = Y - W_aug.dot(alpha)
        tauts.append(taut)

    X_cleaned = np.column_stack(tauts)
    return X_cleaned


def proxy_dataprep(df, surrogate_units, proxy_vars, id_col='Artist', time_col='Date', T=None):
    """
    Efficiently constructs donor, surrogate, and proxy matrices using vectorized operations.

    Args:
    df (pd.DataFrame): The dataset containing relevant columns.
    surrogate_units (list): List of surrogate unit IDs.
    proxy_vars (list): List of proxy variable names.
    id_col (str): Column name representing the unit ID.
    time_col (str): Column name representing the time variable.

    Returns:
    surrogate_matrix (np.ndarray): Surrogate matrix (X).
    surrogate_proxy_matrix (np.ndarray): Surrogate proxy matrix (Z1).
    """

    # Surrogate matrix: Filter for surrogate units and pivot using the first proxy variable

    surrogate_df = df[df[id_col].isin(surrogate_units)].pivot(index=time_col, columns=id_col, values=proxy_vars['donorproxies'][0])
    surrogate_matrix = surrogate_df.to_numpy()

    # Surrogate proxy matrix: Use the second proxy variable for the same surrogate units
    surrogate_proxy_df = df[df[id_col].isin(surrogate_units)].pivot(index=time_col, columns=id_col, values=proxy_vars['surrogatevars'][0])
    surrogate_proxy_matrix = surrogate_proxy_df.to_numpy()

    return surrogate_matrix, surrogate_proxy_matrix
