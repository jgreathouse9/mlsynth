#!/usr/bin/env python3
"""Reference runner: the authors' ``scmrelax`` L2 relaxation on the balanced-GDP
panel (Liao-Shi-Zheng's Brexit application, treated = United Kingdom, 2016Q3).

This reproduces the authors' own GDP application notebook spec
(``GDP_application_2016.ipynb`` from github.com/YapengZheng/Relaxed_SC, packaged
as github.com/metricshilab/scmrelax): the outcome is the year-over-year quarterly
GDP growth rate ``100 * GDP.pct_change(4)``, the treated unit is the United
Kingdom, and the pre-treatment window ends 2016-06-30 (treatment 2016Q3, the
Brexit referendum quarter). The reference ``scmrelax.L2RelaxationCV`` first
cross-validates ``tau`` over the authors' grid (``cv=3, n_taus=10, nonneg=True``,
exactly as ``scmrelax.fit`` does), then refits at the selected ``tau`` on the full
pre-period -- the unique L2-relaxation QP whose weights the Python case pins.

``scmrelax`` hardcodes the commercial MOSEK solver; the L2 relaxation is a plain
convex QP whose optimum is solver-invariant, so we transparently route its solves
to the open CLARABEL solver for the duration of the run. The selected ``tau`` and
the verbatim weights are emitted under ``== REFERENCE VALUES ==`` so
``benchmarks/reference/generate.py`` can capture them into the bundle.

Run via the bundle's manifest::

    python benchmarks/reference/generate.py rescm_balanced_gdp
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The vendored authors' panel (wide: Region + 58 economies, quarterly real GDP).
_DATA = Path(__file__).resolve().parents[3] / "basedata" / "balanced_gdp.csv"

TARGET = "United Kingdom"
PRE_TREAT_Q = "2016-06-30"     # last pre-treatment quarter; treatment is 2016Q3


def main() -> int:
    import cvxpy as cp
    from scmrelax.scmrelax import L2RelaxationCV

    GDP = pd.read_csv(_DATA, index_col=0, parse_dates=True)
    # Authors' outcome: year-over-year (4-quarter) GDP growth rate, in percent.
    GDP_growth = 100.0 * GDP.pct_change(4).dropna()
    y = GDP_growth[TARGET]
    X = GDP_growth.drop(TARGET, axis=1)
    y_pre = y.loc[:PRE_TREAT_Q].to_numpy()
    X_pre = X.loc[:PRE_TREAT_Q].to_numpy()
    donors = list(X.columns)

    # Route scmrelax's hardcoded MOSEK to the open CLARABEL solver. The L2
    # relaxation is convex with a unique optimum, so this is solver-invariant.
    _orig = cp.Problem.solve

    def _patched(self, *a, **k):
        if k.get("solver") == cp.MOSEK:
            k["solver"] = cp.CLARABEL
            k.pop("mosek_params", None)
        return _orig(self, *a, **k)

    try:
        cp.Problem.solve = _patched
        model = L2RelaxationCV(cv=3, n_taus=10, nonneg=True).fit(X_pre, y_pre)
    finally:
        cp.Problem.solve = _orig

    w = np.asarray(model.coef_, dtype=float)
    tau = float(model.tau_)

    print("== REFERENCE VALUES ==")
    print(f"tau\t{tau:.10f}")
    print(f"weight_sum\t{float(w.sum()):.10f}")
    print(f"weight_min\t{float(w.min()):.10f}")
    print(f"n_nonzero\t{float((w > 1e-4).sum()):.10f}")
    # Headline weights the case cross-validates donor by donor.
    for name in donors:
        j = donors.index(name)
        print(f"weight\t{name}\t{float(w[j]):.10f}")
    print("== END ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
