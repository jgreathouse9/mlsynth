from mlsynth import LEXSCM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_sales_panel(
        n_units: int = 50,
        n_time_periods: int = 100,
        seed: int = 42,
        n_candidates: int = 10,
        treatment_start: int = 80
) -> pd.DataFrame:
    """
    Generates a panel dataset for Design Mode.
    There is a 'post' period defined, but NO treatment has been applied yet.
    """
    np.random.seed(seed)

    # 1. Unit & Time Baselines (Common to all units)
    unit_fe = np.random.normal(0, 10, size=n_units)
    unit_trend = np.random.normal(1.0, 0.2, size=n_units)

    t = np.arange(n_time_periods)
    # Common factors: Trend + Seasonality + Random Macro Shocks
    common_factor = (1.8 * t + 50) + (6 * np.sin(2 * np.pi * t / 52)) + np.random.normal(0, 2.5, size=n_time_periods)

    # 2. Generate Base Sales
    sales = (
            common_factor[:, None]
            + unit_fe[None, :]
            + unit_trend[None, :] * t[:, None] * 0.25
    )
    sales += np.random.normal(0, 4.0, size=(n_time_periods, n_units))

    # 3. Identify Candidate Pool (Units eligible for the 'Treatment' lottery)
    candidate_units = np.random.choice(n_units, size=n_candidates, replace=False)
    candidate_mask = np.isin(np.arange(n_units), candidate_units)

    # 4. Long Format Conversion
    unit_ids = np.repeat(np.arange(n_units), n_time_periods)
    time_ids = np.tile(np.arange(n_time_periods), n_units)
    sales_flat = sales.ravel(order='F')

    df = pd.DataFrame({
        "unitid": unit_ids,
        "time": time_ids,
        "sales": sales_flat
    })

    # Add the configuration columns
    df["candidate"] = np.repeat(candidate_mask, n_time_periods)

    # post=1 indicates the 'future' where we PLAN to treat, but haven't yet.
    df["post"] = (df["time"] >= treatment_start).astype(int)

    return df

# ==================== Generate the dataset ====================
#df = generate_synthetic_sales_panel(n_units=50,n_time_periods=100, seed=875, treatment_start=80)

#df.to_csv("lexsc.csv", index=False)
df = pd.read_csv("lexsc.csv")

base_config = {
    "df": df,
    "outcome": "sales",
    "unitid": "unitid",
    "time": "time", "m": 3, "top_K": 15, "candidate_col": "candidate", "lambda_penalty": .2, "post_col": "post"
}

arco = LEXSCM(base_config).fit()

# -----------------------------
# Extract objects
# -----------------------------
best = arco.best_candidate
pred = best.predictions

synthetic_treated = pred.synthetic_treated
synthetic_control = pred.synthetic_control

# population mean across units at each time t
pop_mean = arco.y_pop_mean_t

t = np.arange(len(pop_mean))

# -----------------------------
# Plot
# -----------------------------
plt.figure()

plt.plot(t, synthetic_treated, label="Synthetic Treated")
plt.plot(t, synthetic_control, label="Synthetic Control")
plt.plot(t, pop_mean, label="Population Mean")

plt.axvline(x=arco.n_periods-arco.n_blank_periods, linestyle="--")

plt.title("Synthetic Treated vs Control vs Population Mean")
plt.xlabel("Time")
plt.ylabel("Outcome")
plt.legend()

plt.show()
