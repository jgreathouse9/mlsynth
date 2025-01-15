import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from proxfunctions import pi, pi_surrogate, pi_surrogate_post, clean_surrogates2

def proxy_dataprep(df, surrogate_units, proxy_vars):
    """
    Efficiently constructs donor, surrogate, and proxy matrices using vectorized operations.

    Args:
    df (pd.DataFrame): The dataset containing columns ['ID', 'time', 'type', 'prc_log', 'bid_itp', 'ask_itp'].
    surrogate_units (list): List of surrogate unit IDs.
    proxy_vars (list): List of proxy variable names (e.g., ['bid_itp', 'ask_itp']).

    Returns:
    donor_matrix (np.ndarray): Donor matrix (Y0).
    surrogate_matrix (np.ndarray): Surrogate matrix (X).
    surrogate_proxy_matrix (np.ndarray): Surrogate proxy matrix (Z1).
    """
    T = len(df['time'].unique())  # Number of time periods

    # Surrogate matrix: Filter for surrogate units and pivot using bid price
    surrogate_df = df[df['ID'].isin(surrogate_units)].pivot(index='time', columns='ID', values=proxy_vars[0])
    surrogate_matrix = np.log(surrogate_df.to_numpy())

    # Surrogate proxy matrix: Use ask price for the same surrogate units
    surrogate_proxy_df = df[df['ID'].isin(surrogate_units)].pivot(index='time', columns='ID', values=proxy_vars[1])
    surrogate_proxy_matrix = np.log(surrogate_proxy_df.to_numpy())

    return surrogate_matrix, surrogate_proxy_matrix


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
    "axes.facecolor": "white",
    "figure.figsize": (10, 5.5)
}

matplotlib.rcParams.update(jared_theme)

# Imports our dataset from the directory
data = pd.read_stata("data/trust.dta")
data = data[data['ID'] != 1]
T = 411

# The authors specify that only the normal/purely unaffected
# trusts are the ones that will constritute the donor pool. Specifically,
# they write

"""
These trust companies are divided into three groups based on their connections
and status during the crisis: the “troubled” group, the “connected” group, and
the “independent” group. The troubled group comprises three trusts that experienced
severe runs during the panic, while the connected group consists of seven
trusts linked to four major “money trusts”. The remaining 49 trusts are classified
as the independent group.
"""
donor = data[data.type == 'normal']

# The mlsynth analog for this would be Y0, or the donor matrix of all units
# returned by the dataprep function.
"""
To adapt to mlsynth, the wisest strategy of action would be to have users
supply a list of donors that they believe will be good surrogates, and then
later on subtract them from the Ywide frame and add then to, say,
Y0_surrogates
"""
donor_IDs = donor.ID.unique()
W = np.zeros((T, len(donor_IDs)))

# Use vectorized operation to populate W
for i, id in enumerate(donor_IDs):
    W[:, i] = donor.loc[donor.ID == id, 'prc_log'].to_numpy()

idnums = data.loc[data.type == 'normal', 'ID'].unique()

surframe = donor.pivot(index='time', columns='ID', values='bid_itp')

# Here we have our proxies for our  independent donors, or, the constructs that we think will be informative about
# the time variant factors which generate the outcome of our treated unit, absent the panic.

ids = data[data.introuble == 1].ID.unique()
Z0 = surframe[idnums].to_numpy()

'''
Now, we construct the surrogates. The authors say, quote

we consider the logarithm of the stock mid-price of Knickerbocker Trust 
Company as the target outcome, with the three troubled trusts serving as
surrogates. Specifically, we use the bid price (logarithm) of these trusts, including
the bid price (logarithm) of Knickerbocker as surrogates

Again, these have to be odd typos- the code used here is clearly the interpolated
bid price, not mid price.
'''



X, Z1 = proxy_dataprep(data, surrogate_units=ids, proxy_vars=['bid_itp', 'ask_itp'])

Y = data[data.ID == 34].prc_log.to_numpy() # outcome vector for treated unit
# Create figure and axes objects
fig, ax = plt.subplots(figsize=(12, 6))

# Plot original data
ax.plot(data[data.ID==34].date, Y, label='Knickerbocker', linestyle='-', linewidth=2, color='black')

lag = 2

tau, taut, alpha, se_tau = pi(Y, W, Z0, 229, 182, T, lag)
ax.plot(data[data.ID==34].date, Y-taut, label='Proximal Inference', linestyle='--', linewidth=1.5, color='black')

tau, taut, alpha, se_tau = pi_surrogate(Y, W, Z0, Z1, clean_surrogates2(X, Z0, W, 229), 229, 182, T, lag)
ax.plot(data[data.ID==34].date, Y-taut, label='PI-Surrogates', linestyle='--', linewidth=1, color='red')

tau, taut, alpha, se_tau = pi_surrogate_post(Y, W, Z0, Z1, clean_surrogates2(X, Z0, W, 229), 229, 182, T, lag)
ax.plot(data[data.ID==34].date, Y-taut, label='PI-Post Intervention', linestyle='--', linewidth=1, color='blue')

ax.set_xlabel('Date')
ax.set_ylabel('Log Price')
plt.legend()
# Show the plot
plt.show()
