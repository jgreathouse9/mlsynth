"""Path A benchmark: Wang (2024) Brexit -> UK foreign direct investment.

Reproduces the empirical result CSC-IPCA is built to demonstrate, on the
author's own data (``basedata/fdi_oecd_brexit.csv``, processed from the
``CongWang141/JMP`` replication package exactly as ``test7_empirical_study``
does): the causal effect of the 2016 Brexit referendum on UK foreign-direct-
investment net inflows, estimated with the public ``CSCIPCA`` API at the
paper's settings (K = 2 factors, nine covariates instrumenting the loadings,
UK treated from 2017).

The paper reports the per-year ATT for the first three post-treatment years:

  ====  ==================  =============
  Year  CSC-IPCA (mlsynth)  Wang (2024)
  ====  ==================  =============
  2017  -7.76               -7.8
  2018  -12.90              -12.9
  2019  -18.34              -18.3
  ====  ==================  =============

These are the reported figures the estimator must land on -- a cell-by-cell
match to the paper's headline numbers, not a summary statistic. The tight
tolerances make this a genuine reproduction check, sensitive to the covariate
cube, the ALS fit, and the treated-mapping re-estimation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_COVS = ["log_gdp", "log_gdp_percap", "import_to_gdp", "export_to_gdp",
         "inflation_gdp_deflator", "gross_capital_forma_gdp", "unemployment",
         "employment_15", "log_population"]


def run() -> dict:
    from mlsynth import CSCIPCA

    df = pd.read_csv(_BASE / "fdi_oecd_brexit.csv")
    res = CSCIPCA({
        "df": df, "outcome": "fdi", "treat": "treated",
        "unitid": "country", "time": "year", "covariates": _COVS,
        "n_factors": 2, "inference": False,
    }).fit()
    years = res.time_series.time_periods
    gap = res.time_series.estimated_gap
    att = {int(y): float(gap[years == y][0]) for y in (2017, 2018, 2019)}
    return {
        "n_countries": float(df["country"].nunique()),
        "att_2017": att[2017],
        "att_2018": att[2018],
        "att_2019": att[2019],
        "converged": float(res.metadata["converged"]),
    }


# Deterministic (the fit is a fixed-point ALS on a fixed panel; the counterfactual
# is rotation-invariant and stable across max_iter). Targets are the paper's
# reported per-year ATT; tolerances are tight because this is a Path-A cell match,
# widened only enough to absorb platform/BLAS drift in the ALS.
EXPECTED = {
    "n_countries": (30.0, 0.0),
    "att_2017": (-7.76, 0.3),      # Wang (2024): -7.8
    "att_2018": (-12.90, 0.3),     # Wang (2024): -12.9
    "att_2019": (-18.34, 0.3),     # Wang (2024): -18.3
    "converged": (1.0, 0.0),
}
