import pandas as pd
from mlsynth.mlsynth import PDA
import os
import matplotlib
import matplotlib.pyplot as plt
from theme import jared_theme

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
    "counterfactual_color": ["blue"],
    "treated_color": "black",
    "display_graphs": True,
    "save": save,
    "method": "l2",
}

model = PDA(config)

l2est = model.fit()
plt.clf()
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
plt.clf()

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
plt.clf()
