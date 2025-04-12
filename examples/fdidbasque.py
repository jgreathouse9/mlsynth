from mlsynth.mlsynth import FDID
import matplotlib.pyplot as plt
import pandas as pd
import os
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)

# Access the corresponding dictionary
file_path = os.path.join(os.path.dirname(__file__), '..', 'basedata', 'basque_data.csv')

# Load the CSV file using pandas
df = pd.read_csv(file_path)

# Example usage
unitid = df.columns[0]
time = df.columns[1]
outcome = df.columns[2]
treat = "Terrorism"

new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'FDID' directory
save_directory = os.path.join(os.getcwd(), "fdid")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save = {
    "filename": "FDID_Basque",
    "extension": "png",
    "directory": save_directory,
}

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": ["red"],
    "treated_color": "black",
    "display_graphs": True,
    "save": save,
}

model = FDID(config)

# Run the FDID analysis
autores = model.fit()
