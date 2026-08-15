"""Proximal Path-A: the Panic of 1907 (proximal-inference synthetic control).

Cross-validates mlsynth's ``PROXIMAL`` against the authors' Python reference
(``freshtaste/proximal``, pinned commit ``a67d81e``) and the proximal-SC papers'
Table 3 on the Knickerbocker Trust panic of 1907 (``basedata/trust.dta``): the
effect of the panic on the Trust Company of America's stock price, using donor
trusts as outcome proxies and the *affected* trusts (repurposed as surrogates)
to instrument them. Following the paper, prices are logged, the bid price is the
donor proxy, the ask price the surrogate proxy, and the intervention is the
panic (after period 229).

mlsynth reproduces the reference / Table-3 full-window ATTs closely:

  ========  ===============  =====================
  Method    mlsynth ATT      reference (Table 3)
  ========  ===============  =====================
  PI        -1.148           -1.138
  PI-S      -1.148           -1.134
  PI-Post   -1.220           -1.220
  ========  ===============  =====================

(The PI/PI-S gaps are ~0.01 implementation slack; PI-Post matches to the printed
digits.) The case also pins the no-proxy SPSC estimator as a regression guard.
Path A (scenario 3): the data and reference are the authors'; cross-validation
is mandatory and done here. The case **skips gracefully** when the reference
clone is unavailable.

DR and PIPW are not estimated on this panel. Both route through
``fit_treatment_bridge``, which solves the square moment system

    E_pre[exp((1,Z) beta) (1,W)] = E_post[(1,W)]

and here it has no solution: the largest absolute residual is 0.401 against a
tolerance of 1e-06. The system is 49-dimensional -- an intercept and 48
donor-trust proxies -- and asks for the post-period mean of ``(1,W)`` to be
reproducible by an exponential tilt of the pre-period, which at T0 = 229 against
T1 = 182 it is not. That is a property of the design, so the estimator reports
it instead of retrying.

Both cells were pinned here (``dr_att`` -1.194, ``pipw_att`` -0.854) until #420
made the stalled solve raise. Before that the root finder returned its
non-converged iterate, the ATT built from it was an ordinary-looking number, and
those numbers became the regression guards -- so the two cells recorded the
solver's failure, not the estimator's answer. They are removed rather than
re-pinned or given a wider tolerance: the quantities are not identified on this
panel, and a tolerance cannot supply an identification the design withholds.
``benchmarks/tests/test_panic1907_bridge_infeasible.py`` holds the diagnosis so
they cannot come back without the panel or the estimator having changed.
"""
from __future__ import annotations

import os
import warnings

_VARS = {"donorproxies": ["bid_itp"], "surrogatevars": ["ask_itp"]}
_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "trust.dta")


def _panel():
    import numpy as np
    import pandas as pd

    df = pd.read_stata(os.path.abspath(_DATA))
    df = df[df["ID"] != 1]                                   # drop unbalanced unit
    surrogates = df[df["introuble"] == 1]["ID"].unique().tolist()
    donors = df[df["type"] == "normal"]["ID"].unique().tolist()
    df[["bid_itp", "ask_itp"]] = df[["bid_itp", "ask_itp"]].apply(np.log)
    df["Panic"] = np.where((df["time"] > 229) & (df["ID"] == 34), 1, 0)
    return df, donors, surrogates


def _att(df, donors, surrogates, methods):
    from mlsynth import PROXIMAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PROXIMAL({
            "df": df, "treat": "Panic", "time": "date", "outcome": "prc_log",
            "unitid": "ID", "vars": _VARS, "donors": donors,
            "surrogates": surrogates, "methods": methods, "display_graphs": False,
        }).fit()
    return {k: float(v) for k, v in res.att_by_method().items()}


def run() -> dict:
    from benchmarks.reference.clone_proximal import reference_table3

    ref = reference_table3()                                # skips if unavailable
    df, donors, surrogates = _panel()
    pi = _att(df, donors, surrogates, ["PI", "PIS", "PIPost"])
    spsc = _att(df, donors, surrogates, ["SPSC"])["SPSC"]
    # DR and PIPW are not requested: both route through the treatment bridge,
    # which this panel does not identify. See the module docstring.
    return {
        "pi_att": pi["PI"],
        "pis_att": pi["PIS"],
        "pipost_att": pi["PIPost"],
        "pi_vs_ref": abs(pi["PI"] - ref["PI"]),
        "pis_vs_ref": abs(pi["PIS"] - ref["PIS"]),
        "pipost_vs_ref": abs(pi["PIPost"] - ref["PIPost"]),
        "spsc_att": spsc,
        "n_donors": float(len(donors)),
        "n_surrogates": float(len(surrogates)),
    }


def comparison() -> dict:
    """mlsynth PROXIMAL vs the authors' freshtaste reference, ATT by method.

    Pairs mlsynth's two-proxy PI / PI-surrogate / PI-surrogate-post ATTs against
    the reference's committed Table-3 row, so the exporter can lay them side by
    side. Skips (``BenchmarkSkipped``) when the reference clone is unavailable.
    """
    from benchmarks.reference.clone_proximal import reference_table3

    ref = reference_table3()                                # skips if unavailable
    df, donors, surrogates = _panel()
    pi = _att(df, donors, surrogates, ["PI", "PIS", "PIPost"])
    cfg = {"treat": "Panic", "time": "date", "outcome": "prc_log", "unitid": "ID",
           "vars": _VARS, "donors": donors, "surrogates": surrogates,
           "methods": ["PI", "PIS", "PIPost"]}
    rows = [
        {"quantity": "PI ATT", "mlsynth": round(pi["PI"], 6),
         "reference": round(ref["PI"], 6)},
        {"quantity": "PI-surrogate ATT", "mlsynth": round(pi["PIS"], 6),
         "reference": round(ref["PIS"], 6)},
        {"quantity": "PI-surrogate-post ATT", "mlsynth": round(pi["PIPost"], 6),
         "reference": round(ref["PIPost"], 6)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PROXIMAL", "config": cfg},
        "reference": {"impl": "authors' proximal code (freshtaste/proximal, cloned)",
                      "version": "cloned, pinned commit a67d81e"},
    }


# Deterministic (closed-form GMM / weights, no RNG). The ``*_vs_ref`` cells pin
# mlsynth to the committed freshtaste reference (PI-Post to the digit; PI / PI-S
# within ~0.015 implementation slack); the ``*_att`` cells pin each method as a
# regression guard against the paper's Table-3 values.
EXPECTED = {
    "pi_att": (-1.148, 0.05),            # Table 3 PI -1.138
    "pis_att": (-1.148, 0.05),           # Table 3 PI-S -1.134
    "pipost_att": (-1.220, 0.05),        # Table 3 PI-P -1.220
    "pi_vs_ref": (0.010, 0.03),          # reproduces freshtaste PI
    "pis_vs_ref": (0.014, 0.03),         # reproduces freshtaste PI-S
    "pipost_vs_ref": (0.0, 0.01),        # reproduces freshtaste PI-P exactly
    "spsc_att": (-0.892, 0.12),          # single-proxy SC (no surrogates needed)
    "n_donors": (48.0, 0.0),
    "n_surrogates": (3.0, 0.0),
}
