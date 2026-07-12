"""Capture the scpi_pkg ridge-constraint reference for
``benchmarks/cases/scpi_ridge_germany.py`` (German reunification).

scpi (Cattaneo, Feng, Palomba & Titiunik 2025, JSS) Table 3 assigns the ridge
weight constraint to Amjad, Kim, Shah & Shen (2018) Robust Synthetic Control --
the family CLUSTERSC's PCR / RSC path implements. This records the ridge
budget ``Q``, penalty ``lambda`` and effective degrees of freedom ``df`` that
``scest`` / ``df_EST`` produce on the raw German-reunification donor design.
These are functions of the panel only (the OLS shrinkage rule-of-thumb for
``Q`` / ``lambda`` and the SVD effective-dof for ``df``), so they are
weight-independent: mlsynth's generalized ``scpi_intervals(w_constr="ridge")``
must reproduce them exactly, and CLUSTERSC's full-rank PCR surfaces them via
``.fit()``.

scpi is GPL and mlsynth is MIT, so the numbers are recorded once here rather
than imported in the benchmark. Run in an environment with ``scpi_pkg``:

    python benchmarks/reference/scpi_ridge_germany/reference.py
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
    from scpi_pkg import funs

    d = pd.read_csv(os.path.join(_BASE, "scpi_germany.csv"))[
        ["country", "year", "gdp"]].dropna()
    unit_co = [c for c in sorted(d.country.unique()) if c != "West Germany"]
    dp = scdata(df=d, id_var="country", time_var="year", outcome_var="gdp",
                period_pre=np.arange(1960, 1991), period_post=np.arange(1991, 2004),
                unit_tr="West Germany", unit_co=unit_co, features=None,
                cov_adj=None, constant=False, cointegrated_data=False)
    est = scest(dp, w_constr={"name": "ridge"})
    wc = est.w_constr["West Germany"]
    J = int(np.asarray(est.B).shape[1])
    w_df = pd.DataFrame(est.w.values, index=est.B.columns)
    df = funs.df_EST(w_constr=dict(wc), w=w_df, B=est.B, J=J, KM=0)
    weights = {c: round(float(v), 6)
               for c, v in zip(unit_co, np.asarray(est.w).flatten())}

    out = {"values": {
        "Q": round(float(wc["Q"]), 8),
        "lambda": round(float(wc["lambda"]), 8),
        "df": round(float(df), 8),
        "J": J,
        "ridge_weights": weights,
    }}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
