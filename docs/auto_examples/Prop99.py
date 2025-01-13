"""
L2 Relaxer
==========
L2PDA
"""

from mlsynth.mlsynth import PDA
import matplotlib.pyplot as plt
import pandas as pd
import os
from theme import jared_theme
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


# Access the corresponding dictionary
file_path = r'https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/smoking_data.csv'

# Load the CSV file using pandas
df = pd.read_csv(file_path)

# Example usage
unitid = df.columns[0]
time = df.columns[1]
outcome = df.columns[2]
treat = "Proposition 99"

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": "pink",
    "treated_color": "black",
    "display_graphs": True,
    "method": "l2",
}

model = PDA(config)

# Run the FDID analysis
autores = model.fit()
