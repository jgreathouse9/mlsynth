# Note that you need my Forward DID method (linked in the tutorial) to run this. I will not post the full mlsynth code until it is ready.

import pandas as pd
from mlsynth import TSSC, FDID
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# matplotlib theme
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
    "figure.figsize": (10, 8),
}

matplotlib.rcParams.update(jared_theme)

df = pd.read_csv(r"Data.csv", sep=",").reset_index()
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

# Show plot
plt.savefig("TrvDonors.png")
plt.show()

model = FDID(
    df=df,
    treat=treat,
    time=time,
    outcome=outcome,
    unitid=unitid,
    figsize=(10, 8),
    treated_color="black",
    counterfactual_color="blue",
    display_graphs=False,
)

RESDICT = model.fit()
observed_unit = RESDICT[0]["FDID"]["Vectors"]["Observed Unit"]
FDID_unit = RESDICT[0]["FDID"]["Vectors"]["Counterfactual"]

DID_unit = RESDICT[1]["DID"]["Vectors"]["Counterfactual"]
treatdate = RESDICT[2]["AUGDID"]["Fit"]["T0"]

model2 = TSSC(
    df=df,
    treat=treat,
    time=time,
    outcome=outcome,
    unitid=unitid,
    figsize=(10, 8),
    treated_color="black",
    counterfactual_color="blue",
    display_graphs=False,
)

SCMDICT = model2.fit()

MSCB = SCMDICT[1]["MSC_b"]["Vectors"]["Counterfactual"]
SC = SCMDICT[0]["SC"]["Vectors"]["Counterfactual"]
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(
    DID_unit, label="DID Brooklyn", linestyle="-", color="lime", linewidth=2
)

plt.plot(
    SC, label="Convex Brooklyn", linestyle="--", color="#7DF9FF", linewidth=2
)

plt.plot(
    FDID_unit,
    label="FDID Brooklyn",
    linestyle="-",
    color="blue",
    linewidth=2.5,
    alpha=0.8,
)
plt.plot(MSCB, label="MSC Brooklyn", linestyle="-", color="red", linewidth=2.5)

plt.plot(
    observed_unit, label="Brooklyn", linestyle="-", color="black", linewidth=2
)

plt.xlabel(time)
plt.ylabel("Sales")
plt.title("Brooklyn versus Counterfactual Brooklyn")

plt.axvline(
    x=treatdate,
    color="black",
    linestyle="--",
    linewidth=3,
    label=f"{treat}, {treatdate}",
)


plt.legend()

plt.savefig("MSCTEs.png")
