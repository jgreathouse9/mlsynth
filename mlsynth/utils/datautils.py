import numpy as np
import pandas as pd


def test_treat(treatment_matrix: np.ndarray):
    """
    Test the treatment matrix for validity.

    Parameters:
        treatment_matrix (numpy.ndarray): The treatment matrix to be tested.

    Raises:
        TypeError: If the treatment matrix is not a NumPy array.
        AssertionError: If any of the tests fail.

    Returns:
        dict: A dictionary containing the results of the tests.
    """
    # Check if treatment_matrix is a numpy array
    if not isinstance(treatment_matrix, np.ndarray):
        raise TypeError("treatment_matrix must be a NumPy array")

    # Test 1: Calculate the l0 norm of the matrix and assert that it's greater than 0
    l0_norm = np.count_nonzero(treatment_matrix)
    assert (
        l0_norm > 0
    ), f"The l0 norm is {l0_norm}, which is not greater than 0. No treated units found."

    # Test 2: Calculate the l1 norm of the treated column(s) to check that the treatment values are integers
    columns_with_ones = np.where(np.any(treatment_matrix == 1, axis=0))[0]
    treated_column = treatment_matrix[:, columns_with_ones]
    t2 = int(
        np.linalg.norm(treated_column, ord=1)
    )  # Explicitly convert to integer
    assert t2 == np.linalg.norm(
        treated_column, ord=1
    ), f"The l1 norm of the treated column is not an integer."

    # Test 3: Use the infinite norm of the treated column(s) to ensure that the maximum value is 1
    inf_norm = np.linalg.norm(treated_column, ord=np.inf)
    assert (
        inf_norm == 1
    ), f"The maximum value of the treated column is {inf_norm}, which is not equal to 1."

    t1 = int(
        np.size(treated_column) - np.count_nonzero(treated_column)
    )  # Explicitly convert to integer
    total_periods = t2 + t1

    return {
        "Num Treated Units": l0_norm,
        "Post Periods": t2,  # Ensured integer
        "Treated Index": columns_with_ones,
        "Pre Periods": t1,  # Ensured integer
        "Total Periods": total_periods,  # Ensured integer
    }


def dataprep(df, unitid, time, outcome, treat):
    test_results = test_treat(
        df.pivot(index=time, columns=unitid, values=treat).to_numpy()
    )
    t2 = test_results["Post Periods"]
    t1 = test_results["Pre Periods"]
    trcolnum = test_results["Treated Index"]
    t = test_results["Total Periods"]

    Ywide = df.pivot(index=time, columns=unitid, values=outcome)
    treated_unit_name = Ywide.columns[test_results["Treated Index"]][0]
    y = Ywide[treated_unit_name].to_numpy()
    donor_df = Ywide.drop(
        Ywide.columns[test_results["Treated Index"][0]], axis=1
    )
    donor_names = donor_df.columns

    return {
        "treated_unit_name": treated_unit_name,
        "Ywide": Ywide,
        "y": y,
        "donor_names": donor_names,
        "donor_matrix": donor_df.to_numpy(),
        "total_periods": t,
        "pre_periods": t1,
        "post_periods": t2,
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
