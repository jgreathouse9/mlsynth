import numpy as np

def prepare_data(df, unitid, time, outcome, treat):
    treated_unit_name = df.loc[df[treat] == 1, unitid].values[0]
    Ywide = df.pivot(index=time, columns=unitid, values=outcome)
    y = Ywide[treated_unit_name].values
    donor_df = df[df[unitid] != treated_unit_name]
    donor_names = donor_df[unitid].unique()
    Xbar = Ywide[donor_names].values
    t = y.shape[0]
    t1 = len(df[(df[unitid] == treated_unit_name) & (df[treat] == 0)])
    t2 = t - t1
    return treated_unit_name, Ywide, y, donor_names, Xbar, t, t1, t2


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
            "Duplicate observations found. Ensure each combination of unit and time is unique.")

    # Group by unit and count the number of observations for each unit
    counts = df.groupby([unit_col, time_col]).size().unstack(fill_value=0)

    # Check if all units have the same number of observations
    is_balanced = (counts.nunique(axis=1) == 1).all()

    if not is_balanced:
        raise ValueError("The panel is not strongly balanced.")