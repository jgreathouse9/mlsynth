import pandas as pd
from mlsynth.mlsynth import FMA
import os
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)


file_path = os.path.join(os.path.dirname(__file__), '..', 'basedata', 'HongKong.csv')

# Load the CSV file using pandas
df = pd.read_csv(file_path)

treat = "Integration"
outcome = "GDP"
unitid = "Country"
time = "Time"

df = df[df["Country"] != "China"].reset_index(drop=True)

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
