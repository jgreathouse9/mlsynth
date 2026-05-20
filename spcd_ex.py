from __future__ import annotations

import numpy as np
import pandas as pd

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


T = 90
T0 = 75
# ==================== Generate the dataset ====================
df = generate_synthetic_sales_panel(n_units=25,n_time_periods=T, seed=568, treatment_start=T0)

from mlsynth import SPCD

config = {
    "df": df,
    "outcome": "sales",
    "unitid": "unitid",
    "time": "time",
    "T0": T0,
    "variant": "spcd",   # or "spcd"
    "weights": "empirical",    # paper's experimental default
    "display_graph": True,
}

results = SPCD(config).fit()

print("Treated units:", results.selected_unit_labels.tolist())
print("Iterations:   ", results.design.n_iterations,
      "converged:", results.design.converged)
print("alpha/lam/beta used:",
      results.design.alpha_ridge,
      results.design.lam_balance,
      results.design.beta)
