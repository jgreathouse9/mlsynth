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

df = df[df["ID"] != 1] # Dropping the unbalanced unit

surrogates = df[df['introuble'] == 1]['ID'].unique().tolist() # Our list of surrogates

donors = df[df['type'] == "normal"]['ID'].unique().tolist() # Our pure controls

vars = ["bid_itp", "ask_itp"]

df[vars] = df[vars].apply(np.log) # We take the log of these, per the paper.

df['Panic'] = np.where((df['time'] > 229) & (df['ID'] == 34), 1, 0)

# Here is when our treatment began, on the 229th tri-week.

treat = "Panic"
outcome = "prc_log"
unitid = "ID"
time = "date"

var_dict = {
    "surrogatevars": ["bid_itp"],  # Surrogate variable
    "proxyvars": ["ask_itp"]                 # Proxy variable
}


new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'PROXIMAL' directory
save_directory = os.path.join(os.getcwd(), "PROXIMAL")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save = {
    "filename": "PanicProx",
    "extension": "png",
    "directory": save_directory,
}


config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "treated_color": "black",
    "counterfactual_color": ['blue', 'green', 'red'],
    "display_graphs": True,
    "surrogates": surrogates,
    "vars": var_dict,
    "donors": donors,
    "save": save
}

model = PROXIMAL(config)

SC = model.fit()
