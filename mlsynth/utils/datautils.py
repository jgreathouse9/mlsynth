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
