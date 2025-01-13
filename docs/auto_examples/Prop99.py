from mlsynth.mlsynth import PDA
import matplotlib.pyplot as plt
import pandas as pd
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)

# Access the corresponding dictionary
file_path = 'https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/smoking_data.csv')

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
