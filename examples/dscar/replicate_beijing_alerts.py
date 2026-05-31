"""Path-A replication driver for Zheng & Chen (2024) Section 5.

Reproduces the two Beijing air-pollution-alert empirical examples
through the public ``DSCAR(config).fit()`` API:

* **Orange alert** (eg1, panel ``beijing_pm25_orange_alert.csv``,
  starting 17 Nov 2016, T=72 h, 94 monitoring stations, 20 treated):
  paper reports an ATT of -33.8 μg/m³ (SE 4.4), relative reduction
  24.3%, mu_0 = 139.0 μg/m³, mu_1 = 105.3 μg/m³.
* **Red alert** (eg2, panel ``beijing_pm25_red_alert.csv``,
  starting 16 Dec 2016, T=72 h, 66 stations, 20 treated): paper
  reports an ATT of -70.4 μg/m³ (SE 9.6), relative reduction 26.2%,
  mu_0 = 269.2 μg/m³, mu_1 = 198.8 μg/m³.

The orange alert reproduces to one decimal point of the paper's
reported numbers. The red alert reproduces the qualitative finding
(significant negative ATT, ~22% reduction) but its magnitude differs
from the paper's by about 20% -- the released reference R script
``eg2/Eg_Air_Pollution_eps_201616_12_16_final.R`` contains a
commented-out per-unit pressure / humidity de-meaning block that
is **not** applied in the released pipeline, suggesting the paper's
red-alert numbers were produced with preprocessing the released code
doesn't perform.

Usage::

    python -m examples.dscar.replicate_beijing_alerts

Reference
---------
Zheng, X., & Chen, S. X. (2024). *Dynamic synthetic control method
for evaluating treatment effects in auto-regressive processes.*
Journal of the Royal Statistical Society Series B, 86(1):155-176.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from mlsynth import DSCAR


def fit_alert(
    csv_path: Path, *, placebo_reps: int = 0,
) -> "tuple[float, float, float, float, Optional[float]]":
    df = pd.read_csv(csv_path)
    df["treat_indicator"] = (
        (df["alert_if"] == 1) & (df["hour_eps"] > 48)
    ).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSCAR({
            "df": df,
            "outcome": "pm25",
            "treat": "treat_indicator",
            "unitid": "id_eps",
            "time": "hour_eps",
            "exog_covariates": ["WSPM", "humi", "dewp", "pres"],
            "lagged_outcome": "pm25_lag1",
            "placebo_reps": placebo_reps,
            "display_graphs": False,
        }).fit()
    mu1 = float(res.fit.Y_treated_mean[48:].mean())
    mu0 = float(res.fit.Y0_hat[48:].mean())
    return res.att, res.att_relative, mu0, mu1, res.se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--placebo-reps", type=int, default=0,
        help="Number of normalised-placebo replications for the SE on the ATT.",
    )
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2] / "basedata"

    print(f"{'Alert':<10} {'ATT':>10} {'paper':>8}    "
          f"{'rel. %':>8} {'paper':>8}    "
          f"{'mu_0':>9} {'paper':>8}    "
          f"{'mu_1':>9} {'paper':>8}    "
          f"{'SE':>8}")
    print("-" * 110)
    for label, fname, paper in [
        ("Orange", "beijing_pm25_orange_alert.csv",
         (-33.8, 24.3, 139.0, 105.3)),
        ("Red", "beijing_pm25_red_alert.csv",
         (-70.4, 26.2, 269.2, 198.8)),
    ]:
        att, rel, mu0, mu1, se = fit_alert(
            base / fname, placebo_reps=args.placebo_reps,
        )
        se_str = f"{se:.3f}" if se is not None else " -"
        print(
            f"{label:<10} "
            f"{att:>+10.4f} {paper[0]:>+8.1f}    "
            f"{100 * rel:>+8.2f} {paper[1]:>+8.1f}    "
            f"{mu0:>9.2f} {paper[2]:>8.1f}    "
            f"{mu1:>9.2f} {paper[3]:>8.1f}    "
            f"{se_str:>8}"
        )


if __name__ == "__main__":
    main()
