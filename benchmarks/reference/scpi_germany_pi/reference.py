"""Capture the scpi_pkg single-unit prediction-interval reference for
``benchmarks/cases/scpi_germany_pi.py`` (German reunification, CFT 2021).

scpi is GPL and mlsynth is MIT, so we record scpi's numbers once here rather than
importing it in the benchmark. Run in an environment with ``scpi_pkg`` installed
(``pip install scpi_pkg``):

    python benchmarks/reference/scpi_germany_pi/reference.py

Emits the CI_all_gaussian band (levels and cointegrated) and the simplex weights,
which are pasted into ``reference.json``. e_method="gaussian", seed 8894, 2000
sims, matching the tutorial (https://carlos-mendez.org/post/python_scpi/).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "basedata")


def main() -> None:
    from scpi_pkg.scdata import scdata
    from scpi_pkg.scest import scest
    from scpi_pkg.scpi import scpi

    d = pd.read_csv(os.path.join(_BASE, "scpi_germany.csv"))[
        ["country", "year", "gdp"]].dropna()
    unit_co = [c for c in sorted(d.country.unique()) if c != "West Germany"]
    common = dict(df=d, id_var="country", time_var="year", outcome_var="gdp",
                  period_pre=np.arange(1960, 1991), period_post=np.arange(1991, 2004),
                  unit_tr="West Germany", unit_co=unit_co, features=None,
                  cov_adj=None, constant=False)

    dp0 = scdata(cointegrated_data=False, **common)
    est = scest(dp0, w_constr={"name": "simplex"})
    weights = {c: float(w) for c, w in zip(unit_co, est.w.values.flatten())
               if abs(w) > 1e-4}

    bands = {}
    for tag, coint in (("levels", False), ("coint", True)):
        dp = scdata(cointegrated_data=coint, **common)
        np.random.seed(8894)
        pi = scpi(dp, sims=2000, w_constr={"name": "simplex", "Q": 1},
                  u_order=1, u_lags=0, e_order=1, e_lags=0, e_method="gaussian",
                  u_missp=True, u_sigma="HC1", cores=1, e_alpha=0.05, u_alpha=0.05)
        ci = pi.CI_all_gaussian
        bands[tag] = (ci.iloc[:, 0].values, ci.iloc[:, 1].values)
        years = [int(y) for y in ci.index.get_level_values(1)]

    out = {"values": {
        "years": years,
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "fit": [round(float(v), 4) for v in est.Y_post_fit.values.flatten()],
        "obs": [round(float(v), 4) for v in est.Y_post.values.flatten()],
        "coint_lo": [round(float(v), 4) for v in bands["coint"][0]],
        "coint_hi": [round(float(v), 4) for v in bands["coint"][1]],
        "levels_lo": [round(float(v), 4) for v in bands["levels"][0]],
        "levels_hi": [round(float(v), 4) for v in bands["levels"][1]],
    }}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
