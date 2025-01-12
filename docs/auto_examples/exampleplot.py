"""
Example Plot
=============
This is a simple example of how to create a plot using matplotlib.
"""

import pandas as pd
from mlsynth.mlsynth import FDID
import os
from theme import jared_theme

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
