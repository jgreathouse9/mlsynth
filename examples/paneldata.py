import pandas as pd
from mlsynth.mlsynth import PDA
import matplotlib
import os

jared_theme = {'axes.grid': True,
              'grid.linestyle': '-',
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 12,
              'legend.title_fontsize': 14,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'axes.labelsize': 12,
              'axes.titlesize': 20,
              'figure.dpi': 100,
               'axes.facecolor': 'white',
               'figure.figsize': (11, 6)}

matplotlib.rcParams.update(jared_theme)


def load_and_process_data():
    """
    Loads the GDP data, processes it, and returns the DataFrame with additional columns.

    Returns:
        pd.DataFrame: Processed DataFrame with columns 'Country', 'GDP', 'Time', and 'Integration'.
    """
    # Define column names
    column_names = [
        "Hong Kong", "Australia", "Austria", "Canada", "Denmark", "Finland",
        "France", "Germany", "Italy", "Japan", "Korea", "Mexico", "Netherlands",
        "New Zealand", "Norway", "Switzerland", "United Kingdom", "United States",
        "Singapore", "Philippines", "Indonesia", "Malaysia", "Thailand", "Taiwan", "China"
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
    df["Integration"] = (df["Country"].str.contains("Hong") & (df["Time"] >= 44)).astype(int)

    return df

df = load_and_process_data()

treat = "Integration"
outcome = "GDP"
unitid = "Country"
time = "Time"


new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'l2relax' directory
save_directory = os.path.join(os.getcwd(), "l2relax")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save={
        "filename": "HK_Integreation",
        "extension": "png",
        "directory": save_directory
}

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": "blue",  # Optional, defaults to "red"
    "treated_color": "black",  # Optional, defaults to "black"
    "display_graphs": True,  # Optional, defaults to True
    "save": save,  # Optional, defaults to False
    "method": "l2"
}

model = PDA(config)

autores = model.fit()
