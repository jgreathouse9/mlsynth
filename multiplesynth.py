import pandas as pd

from mlsynth.utils.datautils import dataprep

# Data loading and preparation
url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
data = pd.read_csv(url)
data['Terrorism'] = (
    ((data['regionname'].str.contains("Basque")) & (data['year'] >= 1975)) |
    ((data['regionname'].str.contains("Canarias")) & (data['year'] >= 1976))  # Uniform adoption at 1975
).astype(int)

prepped = dataprep(data, "regionname", "year", "gdpcap", "Terrorism")

# Automate extraction of pre-treatment data for all cohorts
cohort_years = list(prepped['cohorts'].keys())  # Should be [1975] with current Terrorism definition
Y_treated_pre_list = [prepped['cohorts'][year]['y'][:prepped['cohorts'][year]['pre_periods']]
                     for year in cohort_years]
L_list = [prepped['cohorts'][year]['pre_periods'] for year in cohort_years]
Y_donor_pre = prepped['cohorts'][cohort_years[0]]['donor_matrix'][:max(L_list)]  # Use max pre_periods

# Fit partially pooled SCM
nu = 1  # Tradeoff parameter
lambda_reg = 0  # Regularization parameter
result = partial_scm_weights(Y_treated_pre_list, L_list, Y_donor_pre, nu=nu, lambda_reg=lambda_reg)


Gamma_hat, kept, dropped = result

Y_treated_full_list = [
    np.array(prepped['cohorts'][year]['y'])  # shape: (T_total, N_1c)
    for year in cohort_years
]

# Compute ATT for each cohort
ATT_cohorts = []

start_idx = 0
post_lengths = []

for i, year in enumerate(cohort_years):
    Y_full = np.array(prepped['cohorts'][year]['y'])  # full outcomes (pre + post)
    L_c = prepped['cohorts'][year]['pre_periods']    # number of pre-treatment periods
    N_1c = Y_full.shape[1]                           # number of treated units in cohort

    # Extract weights for this cohort from Gamma_hat
    weights_i = Gamma_hat[:, start_idx:start_idx + N_1c]  # shape: (N_donor, N_1c)

    # Donor outcomes for post-treatment periods
    Y_donor_full = np.array(prepped['cohorts'][year]['donor_matrix'])  # shape: (T_total, N_donor)
    Y_post = Y_full[L_c:, :]            # post-treatment outcomes of treated units
    Y_donor_post = Y_donor_full[L_c:, :]  # post-treatment donor outcomes

    # Synthetic control predictions
    Y_synth = Y_donor_post @ weights_i  # shape: (T_post, N_1c)

    # Treatment effect
    tau = Y_post - Y_synth              # shape: (T_post, N_1c)

    ATT_cohorts.append(tau)
    post_lengths.append(tau.shape[0])
    start_idx += N_1c

# Truncate all cohorts to the shortest post-treatment period
T_post_min = min(post_lengths)
ATT_truncated = [tau[:T_post_min, :] for tau in ATT_cohorts]

# Combine cohorts
ATT_matrix = np.hstack(ATT_truncated)  # shape: (T_post_min, sum_c N_1c)

# Average across time for each treated unit
ATT_units = ATT_matrix.mean(axis=0)

# Overall pooled ATT across all treated units
ATT_pooled = ATT_units.mean()

print("Unit-level ATT:", ATT_units)
print("Pooled ATT:", ATT_pooled)
