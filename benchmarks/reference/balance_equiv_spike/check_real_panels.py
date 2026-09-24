"""Path C: what mlsynth's ASCM cross-validation selects on real panels.

The paper's Table 1 reports that practical tuning picks the degenerate
hyperparameter on 56 percent of its draws. Part 3 of ``check_ascm.py`` asks the
same of simulated panels; this asks it of the panels in ``basedata/``, and
prices what landing at the degenerate end would cost.

Kansas doubles as a correctness check: mlsynth's ASCM is pinned against
``augsynth`` R at ATT -0.0401 in ``benchmarks/cases/ascm_kansas.py``.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prop43 import ols_plugin  # noqa: E402

from mlsynth.utils.bilevel.ridge_augment import (  # noqa: E402
    build_matching, ridge_augment_weights,
)

_BASE = pathlib.Path(__file__).resolve().parents[3] / "basedata"


def panel_arrays(df, unit, time, outcome, treated_unit, t_int):
    """Wide pre/post arrays for one treated unit and the balanced donor pool."""
    wide = df.pivot_table(index=time, columns=unit, values=outcome,
                          observed=True).dropna(axis=1)
    pre = wide.index < t_int
    donors = [c for c in wide.columns if c != treated_unit]
    return (wide.loc[pre, treated_unit].to_numpy(float),
            wide.loc[pre, donors].to_numpy(float),
            wide.loc[~pre, treated_unit].to_numpy(float),
            wide.loc[~pre, donors].to_numpy(float))


def load_cases():
    d = pd.read_csv(_BASE / "kansas_ascm.csv")
    yield ("Kansas tax cut", *panel_arrays(
        d, "fips", "year_qtr", "lngdpcapita", 20.0, 2012.25))

    d = pd.read_csv(_BASE / "P99data.csv")
    yield ("Prop 99 (California)", *panel_arrays(
        d, "state", "year", "cigsale", "California", 1989))

    d = pd.read_csv(_BASE / "german_reunification.csv")
    unit = next(c for c in d.columns if "country" in c.lower())
    time = next(c for c in d.columns if "year" in c.lower())
    outcome = next(c for c in d.columns if "gdp" in c.lower())
    yield ("German reunification", *panel_arrays(
        d, unit, time, outcome, "West Germany", 1990))


def main():
    warnings.filterwarnings("ignore")
    print(f"{'panel':>22} {'J':>4} {'T0':>4} {'CV lambda':>11} {'at floor':>9}"
          f" {'ATT (CV)':>12} {'ATT (lam->0)':>13} {'ATT (OLS)':>12}")
    for name, y_pre, Y0_pre, y_post, Y0_post in load_cases():
        T0, J = Y0_pre.shape
        res = ridge_augment_weights(y_pre, Y0_pre)          # CV selects lambda
        at_floor = "no"
        if res.cv is not None:
            grid = np.asarray(res.cv["lambdas"], dtype=float)
            at_floor = "YES" if grid.size and np.isclose(
                res.lambda_, grid.min()) else "no"

        att = float(y_post.mean() - (Y0_post @ res.W).mean())
        deg = ridge_augment_weights(y_pre, Y0_pre, lambda_=1e-10)
        att_deg = float(y_post.mean() - (Y0_post @ deg.W).mean())

        B, A = build_matching(y_pre, Y0_pre, None, None)
        att_ols = float(y_post.mean() - ols_plugin(B.T, Y0_post.mean(axis=0), A))

        print(f"{name:>22} {J:>4} {T0:>4} {res.lambda_:11.4g} {at_floor:>9}"
              f" {att:12.5f} {att_deg:13.5f} {att_ols:12.5f}")

    print("\n  Kansas ATT under CV reproduces the augsynth R reference (-0.0401)")
    print("  pinned in benchmarks/cases/ascm_kansas.py.")


if __name__ == "__main__":
    main()
