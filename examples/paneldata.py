import pandas as pd
from mlsynth.mlsynth import PDA
import os
import jared_theme

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

df = df[df["Country"] != "China"].reset_index(drop=True)

new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'l2relax' directory
save_directory = os.path.join(os.getcwd(), "l2relax")

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
    "method": "l2",
}

model = PDA(config)

l2est = model.fit()

# Update the method to "fs" for forward selection
config["method"] = "fs"

# Create the 'fsPDA' directory for saving results
save_directory = os.path.join(os.getcwd(), "fsPDA")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Update the save configuration
config["save"] = {
    "filename": "HK_Integration_fs",
    "extension": "png",
    "directory": save_directory,
}

# Initialize the model with forward selection
model_fs = PDA(config)

# Fit the model using forward selection
fs_est = model_fs.fit()


# Update the method to "LASSO" for LASSO based PDA
config["method"] = "LASSO"

# Create the 'fsPDA' directory for saving results
save_directory = os.path.join(os.getcwd(), "L1PDA")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Update the save configuration
config["save"] = {
    "filename": "HK_Integration",
    "extension": "png",
    "directory": save_directory,
}

# Initialize the model with forward selection
model_fs = PDA(config)

# Fit the model using forward selection
LASSO_est = model_fs.fit()
