"""
FDID Estimation
=================
FDID Plot
"""

import pandas as pd
from mlsynth.mlsynth import FDID
import os
import matplotlib

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
    "axes.facecolor": "#c7d0c9",
    "figure.figsize": (10, 5.5),
}

matplotlib.rcParams.update(jared_theme)

# Load the CSV file using pandas
df = pd.read_csv('https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/HongKong.csv')

treat = "Integration"
outcome = "GDP"
unitid = "Country"
time = "Time"

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": "blue",
    "treated_color": "black",
    "display_graphs": True,
    "criti": 10,  # Assuming Stationary
    "DEMEAN": 1,  # Demeans the donor pool
}

model = FDID(config)

FMAest = model.fit()
