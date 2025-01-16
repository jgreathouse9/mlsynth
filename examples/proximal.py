import pandas as pd
import numpy as np
from mlsynth.mlsynth import PROXIMAL
import matplotlib
import os

jared_theme = {
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.color": "black",
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.shadow": True,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "figure.dpi": 100,
    "axes.facecolor": "white",
    "figure.figsize": (10, 5.5),
}

matplotlib.rcParams.update(jared_theme)

df = pd.read_stata("data/trust.dta")

df = df[df["ID"] != 1]

surrogates = df[df['introuble'] == 1]['ID'].unique().tolist()
donors = df[df['type'] == "normal"]['ID'].unique().tolist()

vars = ["bid_itp", "ask_itp"]

df[vars] = df[vars].apply(np.log)

df['Panic'] = np.where((df['time'] > 229) & (df['ID'] == 34), 1, 0) # 229

treat = "Panic"
outcome = "prc_log"
unitid = "ID"
time = "date"

var_dict = {
    "surrogatevars": ["bid_itp"],  # Surrogate variables
    "proxyvars": ["ask_itp"]                 # Proxy variables
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
    "display_graphs": False,
    "surrogates": surrogates,
    "vars": var_dict,
    "donors": donors,
    "save": save
}

model = PROXIMAL(config)

SC = model.fit()
