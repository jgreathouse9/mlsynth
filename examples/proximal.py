import pandas as pd
import numpy as np
from mlsynth.mlsynth import PROXIMAL
import matplotlib
import os
from theme import jared_theme

matplotlib.rcParams.update(jared_theme)

file_path = os.path.join(os.path.dirname(__file__), '..', 'basedata', 'trust.dta')

# Load the CSV file using pandas
df = pd.read_stata(file_path)

df = df[df["ID"] != 1]  # Dropping the unbalanced unit

surrogates = df[df['introuble'] == 1]['ID'].unique().tolist()  # Our list of surrogates
donors = df[df['type'] == "normal"]['ID'].unique().tolist()  # Our pure controls

vars = ["bid_itp", "ask_itp"]

df[vars] = df[vars].apply(np.log)  # We take the log of these, per the paper.
df['Panic'] = np.where((df['time'] > 229) & (df['ID'] == 34), 1, 0)

# Here is when our treatment began, on the 229th tri-week.
treat = "Panic"
outcome = "prc_log"
unitid = "ID"
time = "date"

var_dict = {
    "donorproxies": ["bid_itp"],
    "surrogatevars": ["ask_itp"]
}

new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'PROXIMAL' directory
save_directory = os.path.join(os.getcwd(), "PROXIMAL")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# First run
save_1 = {
    "filename": "PanicProx",
    "extension": "png",
    "directory": save_directory,
}

config_1 = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "treated_color": "black",
    "counterfactual_color": ["blue"],
    "display_graphs": True,
    "vars": var_dict,
    "donors": donors,
    "save": save_1
}

model_1 = PROXIMAL(config_1)
SC_1 = model_1.fit()

plt.clf()

# Second run with surrogates and new filename
save_2 = {
    "filename": "PanicSurrogates",
    "extension": "png",
    "directory": save_directory,
}

config_2 = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "treated_color": "black",
    "counterfactual_color": ["blue", "red", "lime"],
    "display_graphs": True,
    "vars": var_dict,
    "donors": donors,
    "surrogates": surrogates,  # Added surrogates
    "save": save_2
}

model_2 = PROXIMAL(config_2)
SC_2 = model_2.fit()
