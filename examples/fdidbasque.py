import pandas as pd
import matplotlib.pyplot as plt
from mlsynth.mlsynth import FDID

# URL for the dataset
url = 'https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/basque_data.csv'

# Load the dataset directly from the URL
df = pd.read_csv(url)

# Configure the FDID model
config = {
    "df": df,
    "treat": "Terrorism",
    "time": "year",
    "outcome": "gdpcap",
    "unitid": "regionname",
    "counterfactual_color": "#7DF9FF",
    "treated_color": "red",
    "display_graphs": False,  # Set to False to prevent displaying the graph
    "save": True  # Save the plot instead of displaying
}

model = FDID(config)
autores = model.fit()

# Save the plot if the "save" parameter is set to True
if config["save"]:
    plt.savefig("fdid_analysis_plot.png")
    print("Plot saved as fdid_analysis_plot.png")
