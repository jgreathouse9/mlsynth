"""
L2 Relaxer
==========
Applies the l2 relaxer to some Apple Music data. Here we estimate the causal impact of Tyla going viral on the numer of playlists she's on.
"""

import pandas as pd
from mlsynth import PDA

url = "https://raw.githubusercontent.com/jgreathouse9/jgreathouse9.github.io/refs/heads/master/Apple%20Music/AppleMusic.csv"
df = pd.read_csv(url)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2024-12-31')]

df['Water'] = df.apply(lambda row: 1 if row['Artist'] == 'Tyla' and row['Date'] > pd.to_datetime('2023-08-31') else 0, axis=1)

treat = "Water"
outcome = "Playlists"
unitid = "Artist"
time = "Date"

# Define the list of artists to exclude
excluded_artists = ["Moonchild Sanelly", "Tems", "Ayra Starr", "Tyla"]

# Get the list of artists excluding "Tyla"
other_artists = df[~df['Artist'].isin(excluded_artists)]['Artist'].unique().tolist()

# Group by 'Artist' and count the number of observations
artist_counts = df['Artist'].value_counts()

# Filter the artists with exactly 1096 observations
artists_to_keep = artist_counts[artist_counts == 1096].index

# Filter the dataframe to keep only those artists
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


