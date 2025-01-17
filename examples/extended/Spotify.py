import pandas as pd
from mlsynth.mlsynth import PDA
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



url = "https://raw.githubusercontent.com/jgreathouse9/jgreathouse9.github.io/refs/heads/master/Spotify/Merged_Spotify_Data.csv"
df = pd.read_csv(url)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df = df[(df['Date'] >= '2022-04-21') & (df['Date'] <= '2024-06-01')]


# Function to normalize at the end of August 2023
def normalize(group):
    # Get the value of prc_log at 2023-08-31
    reference_value = group.loc[group['Date'] == pd.Timestamp('2023-08-31'), 'Monthly Listeners']

    if not reference_value.empty:
        # Normalize prc_log column
        group['Monthly Listeners'] = group['Monthly Listeners'] / reference_value.values[0] * 100

    return group


# Normalize prc_log for each artist (grouped by Artist)
df = df.groupby('Artist', group_keys=False).apply(normalize)

# Reset index after normalization
df = df.reset_index(drop=True)

df['Water'] = df.apply(lambda row: 1 if row['Artist'] == 'Tyla' and row['Date'] > pd.to_datetime('2023-08-31') else 0, axis=1)

treat = "Water"
outcome = 'Monthly Listeners'
unitid = "Artist"
time = "Date"

artist_counts = df['Artist'].value_counts()

artists_to_keep = artist_counts[artist_counts == 773].index


df = df[df['Artist'].isin(artists_to_keep)]


config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": "blue",
    "treated_color": "black",
    "display_graphs": True,
    "method": "l2"
}

model = PDA(config)

SC = model.fit()

