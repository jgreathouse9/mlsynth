"""Cross-validation benchmark: SI vs the authors' code (Agarwal-Shah-Shen 2026).

Cross-validation against the reference implementation. ``SI`` is mlsynth's port of
Synthetic Interventions (Agarwal, Shah & Shen 2026, *Synthetic Interventions:
Extending Synthetic Controls to Multiple Treatments*, Oper. Res. 74(2)). This case
runs the **authors' own** estimation code -- vendored verbatim under
:mod:`benchmarks.reference.synth_iv_OR25` from the INFORMS supplement
``opre.2025.1590.cd`` -- side by side with mlsynth's public :class:`mlsynth.SI`
API, and checks they agree to machine precision.

It reproduces the Section 6 case study (Table, ``case_study.ipynb`` "Counterfactual
Estimates"): for each of the five anti-tobacco *program* states, California's and
its peers' 1999-2002 cigarette-consumption counterfactual **under the control and
tax interventions**, with the bias-corrected SI-PCR prediction interval. Fit window
1970-1988, prediction window 1999-2002, donor pool = the units assigned each
intervention (target excluded), Gavish-Donoho rank, ``variance="double"``.

Provenance
----------
* Data: ``basedata/prop99_packsales.csv`` -- the per-capita pack-sales panel (50
  states, 1970-2015) the authors pivot from ``prop99_raw_data.csv``.
* Reference: the authors' ``HSVT`` -> ``qr_column_pivoting_selection`` -> ``OLS``
  with ``variance_estimation`` + ``predictionInterval`` (run live).
* Published anchors (notebook): California under control ``[70.93, 80.63]`` and
  under taxes ``[47.99, 67.06]`` packs/capita.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_REF = Path(__file__).resolve().parents[1] / "reference" / "synth_iv_OR25"

TAX = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey", "New York", "Washington"]
PROGRAM = ["California", "Arizona", "Massachusetts", "Oregon", "Florida"]
FIT_YEARS = list(range(1970, 1989))
PRED_YEARS = list(range(1999, 2003))            # T1 = 4, the case study's window


def _reference_ci(wide, state, donors):
    """The authors' bias-corrected SI-PCR prediction interval (their code)."""
    if str(_REF) not in sys.path:
        sys.path.insert(0, str(_REF))
    from estimation import HSVT, OLS, qr_column_pivoting_selection
    from inference import variance_estimation, predictionInterval

    y_pre_target = wide.loc[state, FIT_YEARS].to_numpy(dtype=float)
    y_pre_donors = wide.loc[donors, FIT_YEARS].to_numpy(dtype=float).T
    y_post_donors = wide.loc[donors, PRED_YEARS].to_numpy(dtype=float).T
    H, U, V = HSVT(y_pre_donors)
    k = U.shape[1]
    omega = qr_column_pivoting_selection(H, k)
    w = OLS(H[:, omega], y_pre_target)
    theta = float(np.mean(y_post_donors[:, omega] @ w))
    sigma = variance_estimation(U, V, y_pre_target, y_post_donors)[0]
    lb, ub = predictionInterval(theta, w, sigma, len(PRED_YEARS), 0.05)
    return float(lb), float(ub)


def run() -> dict:
    from mlsynth import SI

    panel = pd.read_csv(_BASE / "prop99_packsales.csv")
    wide = panel.pivot_table(index="state", columns="year", values="cigsale")
    states = list(wide.index)
    iv_states = {
        "control": [s for s in states if s not in set(TAX) | set(PROGRAM)],
        "taxes": TAX,
        "program": PROGRAM,
    }

    # restrict to the fit + prediction windows (drops 1989-1998, as the paper does)
    d = panel[panel.year.isin(FIT_YEARS + PRED_YEARS)].copy()
    for iv, members in iv_states.items():
        d[iv] = d.state.isin(members).astype(int)

    max_diff = 0.0
    ca = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for target in PROGRAM:
            dd = d.copy()
            dd["treat"] = ((dd.state == target) & (dd.year >= 1999)).astype(int)
            res = SI({
                "df": dd, "outcome": "cigsale", "treat": "treat", "unitid": "state",
                "time": "year", "inters": ["control", "taxes", "program"],
                "bias_correct": True, "variance": "double", "interval": "prediction",
                "rank_method": "donoho", "display_graphs": False,
            }).fit()
            for iv in ("control", "taxes"):
                donors = [s for s in iv_states[iv] if s != target]
                ref_lo, ref_hi = _reference_ci(wide, target, donors)
                mls_lo, mls_hi = res.arms[iv].cf_mean_ci
                max_diff = max(max_diff, abs(mls_lo - ref_lo), abs(mls_hi - ref_hi))
                if target == "California":
                    ca[iv] = (mls_lo, mls_hi)

    return {
        "si_vs_reference_max_abs_diff": float(max_diff),
        "ca_control_lb": ca["control"][0], "ca_control_ub": ca["control"][1],
        "ca_taxes_lb": ca["taxes"][0], "ca_taxes_ub": ca["taxes"][1],
    }


# mlsynth's public SI API matches the authors' own code to machine precision; the
# California interval anchors are the published Section 6 values.
EXPECTED = {
    "si_vs_reference_max_abs_diff": (0.0, 1e-8),
    "ca_control_lb": (70.93, 1e-2),
    "ca_control_ub": (80.63, 1e-2),
    "ca_taxes_lb": (47.99, 1e-2),
    "ca_taxes_ub": (67.06, 1e-2),
}
