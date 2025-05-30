import pandas as pd
from mlsynth.mlsynth import TSSC
import matplotlib.pyplot as plt
import os
import numpy as np
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)


new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'TSSC' directory
save_directory = os.path.join(os.getcwd(), "TSSC")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save = {
    "filename": "Showroom",
    "extension": "png",
    "directory": save_directory,
}

df = pd.read_csv(
    r"https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/Data.csv",
    sep=",",
).reset_index()
# Rename the 'index' column to 'Time'
df.rename(columns={"index": "Time"}, inplace=True)
df = pd.melt(
    df,
    id_vars=["Time"],
    value_vars=df.columns[1:],
    var_name="Unit",
    value_name="Outcome",
)
df["Showroom"] = (df["Unit"].str.contains("Tr") & (df["Time"] >= 76)).astype(
    int
)

treat = "Showroom"
outcome = "Outcome"
unitid = "Unit"
time = "Time"

treated_unit_name = df.loc[df[treat] == 1, unitid].values[0]

# Pivot the DataFrame
Ywide = df.pivot(index=time, columns=unitid, values=outcome)

t1 = len(df[(df[unitid] == treated_unit_name) & (df[treat] == 0)])

y = Ywide[treated_unit_name].values
t = y.shape[0]
donor_df = df[df[unitid] != treated_unit_name]
donor_names = donor_df[unitid].unique()
Xbar = Ywide[donor_names].values

plt.figure()
plt.axvline(x=t1, color="red", linestyle="--", label="Showroom", linewidth=2)
# Plot donor units as vectors
for donor_name in donor_names:
    plt.plot(Xbar, color="#7DF9FF", linewidth=2.5)

# Plot treated unit (Brooklyn) as a vector
plt.plot(y, color="black", label="Brooklyn", linewidth=2)
# Add labels and legend
plt.xlabel("Time")
plt.ylabel("Sales")
plt.title("Treated Unit vs. Donors")
plt.legend()

# Save the plot
output_path = os.path.join(save_directory, "treatedvsdonors.png")
plt.savefig(output_path, bbox_inches="tight")
plt.close()


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
}

model = TSSC(config)

sutff = model.fit()
