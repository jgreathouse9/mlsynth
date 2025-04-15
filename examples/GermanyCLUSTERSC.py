from mlsynth.mlsynth import CLUSTERSC
import pandas as pd
import os
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)

# Access the corresponding dictionary
file_path = os.path.join(os.path.dirname(__file__), '..', 'basedata', 'german_reunification.csv')

# Load the CSV file using pandas
df = pd.read_csv(file_path)

treat = "Reunification"
outcome = "gdp"
unitid = "country"
time = "year"

new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'FMA' directory
save_directory = os.path.join(os.getcwd(), "clustersc")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save = {"filename": "German", "extension": "png", "directory": save_directory}

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": ["red", blue]
    "treated_color": "black",
    "display_graphs": True,
    "save": save,
    "cluster": False, "method": "both"
}

model = CLUSTERSC(config)

asc = model.fit()

keys = ["Effects", "Fit", "Weights"]

save["filename"] = "Cluster_Germany"
config["cluster"] = True
clustmodel = CLUSTERSC(config)
wclust = clustmodel.fit()
