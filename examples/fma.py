import pandas as pd
from mlsynth.mlsynth import FMA
import os
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)


def load_and_process_data():
    """
    Loads the GDP data, processes it, and returns the DataFrame with additional columns.

    Returns:
        pd.DataFrame: Processed DataFrame with columns 'Country', 'GDP', 'Time', and 'Integration'.
    """
    # Define column names
    column_names = [
        "Hong Kong",
        "Australia",
        "Austria",
        "Canada",
        "Denmark",
        "Finland",
        "France",
        "Germany",
        "Italy",
        "Japan",
        "Korea",
        "Mexico",
        "Netherlands",
        "New Zealand",
        "Norway",
        "Switzerland",
        "United Kingdom",
        "United States",
        "Singapore",
        "Philippines",
        "Indonesia",
        "Malaysia",
        "Thailand",
        "Taiwan",
        "China",
    ]

    # Load the dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/leoyyang/rhcw/master/other/hcw-data.txt",
        header=None,
        delim_whitespace=True,
    )

    # Assign column names
    df.columns = column_names

    # Melt the dataframe
    df = pd.melt(df, var_name="Country", value_name="GDP", ignore_index=False)

    # Add 'Time' column ranging from 0 to 60
    df["Time"] = df.index

    # Create 'Integration' column based on conditions
    df["Integration"] = (
        df["Country"].str.contains("Hong") & (df["Time"] >= 44)
    ).astype(int)

    return df


df = load_and_process_data()

treat = "Integration"
outcome = "GDP"
unitid = "Country"
time = "Time"


new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'FMA' directory
save_directory = os.path.join(os.getcwd(), "FMA")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save = {
    "filename": "HK_Integration",
    "extension": "png",
    "directory": save_directory,
}

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": "blue",
    "treated_color": "black",
    "display_graphs": True,
    "save": save,
    "criti": 10,  # Assuming Stationary
    "DEMEAN": 1,  # Demeans the donor pool
}

model = FMA(config)

FMAest = model.fit()
