"""Path A + cross-validation: CMBSTS on the Menchetti-Bojinov supermarket study.

Reproduces the empirical Table 3 of Menchetti and Bojinov (2022) -- the effect
of a permanent store-brand price cut on cookie sales, one store-competitor pair
at a time, at the one-month horizon -- and cross-checks the mlsynth port against
the authors' ``CausalMBSTS`` R package on identical inputs.

Pipeline per pair: the hourly-sales transform and frozen-price construction of
the authors' replication code (verified to machine precision against the R
preprocessing), a trend-plus-weekly-seasonal model, a regression block of
calendar dummies, the frozen store price, the competitor price, and ten wine
control series, and the one-month horizon (effect averaged to 2018-11-04).

Control selection. The paper screens wine controls by dynamic time warping
(``MarketMatching``). CMBSTS offers a DTW screen via the optional ``fastdtw``
package; to keep this benchmark deterministic and dependency-free the ten
DTW-selected wine columns per pair are hardcoded (``fastdtw`` chooses exactly
these). They differ from the authors' ``MarketMatching`` set, so the cells are
not identical to the printed Table 3, but the substantive finding reproduces.

What is checked:

* Cross-validation -- the mlsynth store-brand temporal-average effect matches
  the R ``CausalMBSTS`` package run on the same controls, prior, and horizon
  (pinned reference numbers below), within Monte-Carlo error.
* Table 3 finding -- the store-brand effect is large and positive on pairs 4,
  7 and 10, strictly significant (95% credible interval excludes zero) on pair
  10, and no competitor effect is significant. On pairs 4 and 7 the lower
  credible bound sits on the zero boundary, where Monte-Carlo noise alone moves
  it across (the R package and the port land on opposite sides: R gives pair-4
  ``[-0.95, 95.83]`` and pair-7 ``[3.05, 157.81]``, the port the reverse), so
  only their large positive point estimate is asserted. The published strict
  significance on all three uses the authors' ``MarketMatching`` controls and
  ``niter=2200``.

Data ship in ``basedata/cmbsts_supermarket/`` (the authors' AOAS Supplement B).

Provenance: Menchetti & Bojinov (2022), AOAS 16(1): 414-435, Table 3; the
``CausalMBSTS`` R package (Bojinov & Menchetti 2020); reference numbers produced
by that package on the hardcoded controls at ``niter=1000, burn=200``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "cmbsts_supermarket")
_INT = pd.Timestamp("2018-10-04")
_HZ = pd.Timestamp("2018-11-04")          # one-month horizon
# fastdtw-selected wine columns (0-based) per store-competitor pair.
_CTRL = {
    4: [108, 115, 134, 143, 148, 210, 244, 41, 64, 88],
    7: [115, 134, 148, 15, 177, 210, 34, 39, 41, 88],
    10: [0, 129, 14, 173, 182, 230, 253, 33, 54, 86],
}


def _load():
    d = os.path.abspath(_DIR)
    dv = pd.read_csv(os.path.join(d, "dummy_var.csv"), sep=r"\s+")
    dates = pd.to_datetime(dv["dates"])
    store = pd.read_csv(os.path.join(d, "store_sales.csv"), sep=";")
    comp = pd.read_csv(os.path.join(d, "competitor_sales.csv"), sep=";")
    sp = pd.read_csv(os.path.join(d, "store_price.csv"), sep=";").to_numpy(float)
    cp = pd.read_csv(os.path.join(d, "competitor_price.csv"), sep=";").to_numpy(float)
    wines = pd.read_csv(os.path.join(d, "wines.csv"), sep=";")
    # Hourly transform: Sundays (5h) except December and two dates; else 13h.
    sun = ((dates.dt.day_name() == "Sunday") & (dates.dt.month != 12)
           & (~dates.isin([pd.Timestamp("2017-11-26"), pd.Timestamp("2018-11-25")])))
    div = np.where(sun, 5.0, 13.0)
    hstore = store.div(div, axis=0).to_numpy()
    hcomp = comp.div(div, axis=0).to_numpy()
    hwines = wines.div(div, axis=0).to_numpy()
    ii = (dates >= _INT).to_numpy()
    spm = sp.copy(); spm[ii] = spm[~ii][-1]               # freeze store price post-intervention
    return dv, dates, hstore, hcomp, spm, cp, hwines


def _fit_pair(P, dv, dates, hstore, hcomp, spm, cp, hwines):
    from mlsynth import CMBSTS
    excl = dv["excl.dates"].to_numpy()
    sat, sunv, hol = dv["sat"].to_numpy(), dv["sun"].to_numpy(), dv["hol"].to_numpy()
    ii = (dates >= _INT).to_numpy()
    hcount = int(((dates >= _INT) & (dates <= _HZ) & (excl == 0)).sum())
    ctrl = _CTRL[P]
    names = [f"w{j}" for j in ctrl]
    rows = []
    for t in range(len(dates)):
        rows.append({"item": "store", "week": t, "sales": hstore[t, P - 1], "treated": int(ii[t]),
                     "excl": excl[t], "sat": sat[t], "sun": sunv[t], "hol": hol[t],
                     "sprice": spm[t, P - 1], "cprice": cp[t, P - 1]})
        rows.append({"item": "comp", "week": t, "sales": hcomp[t, P - 1], "treated": 0, "excl": excl[t],
                     "sat": np.nan, "sun": np.nan, "hol": np.nan, "sprice": np.nan, "cprice": np.nan})
        for j, n in zip(ctrl, names):
            rows.append({"item": n, "week": t, "sales": hwines[t, j], "treated": 0, "excl": excl[t],
                         "sat": np.nan, "sun": np.nan, "hol": np.nan, "sprice": np.nan, "cprice": np.nan})
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CMBSTS({
            "df": df, "outcome": "sales", "unitid": "item", "time": "week", "treat": "treated",
            "group_units": ["comp"], "control_units": names,
            "covariates": ["sat", "sun", "hol", "sprice", "cprice"],
            "components": ["trend", "seasonal"], "seas_period": 7, "excl_dates": "excl",
            "horizon": hcount, "prior_scale": 1.0, "prior_rho": -0.8,
            "niter": 1000, "burn": 200, "seed": 1, "display_graphs": False,
        }).fit()
    return res.inference_detail


def run() -> dict:
    data = _load()
    det = {P: _fit_pair(P, *data) for P in (4, 7, 10)}
    out = {}
    for P, d in det.items():
        out[f"store_att_{P}"] = float(d.att_mean[0])
        out[f"store_sig_{P}"] = float(d.att_lower[0] > 0.0)
        out[f"store_positive_{P}"] = float(d.att_mean[0] > 0.0)
        out[f"comp_sig_{P}"] = float(d.att_lower[1] > 0.0 or d.att_upper[1] < 0.0)
    return out


# Deterministic (fixed seed). Reference store-brand effects from R CausalMBSTS on
# the hardcoded DTW controls at niter=1000; tolerances absorb the cross-RNG
# Monte-Carlo difference between the two Gibbs samplers.
EXPECTED = {
    # Cross-validation: store-brand effect vs R CausalMBSTS on the same controls.
    "store_att_4": (47.44, 6.0),
    "store_att_7": (78.08, 9.0),
    "store_att_10": (12.27, 3.0),
    # Table 3 finding: large positive store effects on all three pairs ...
    "store_positive_4": (1.0, 0.0),
    "store_positive_7": (1.0, 0.0),
    "store_positive_10": (1.0, 0.0),
    # ... strictly significant where it is robust (pair 10; both R and the port) ...
    "store_sig_10": (1.0, 0.0),
    # ... and no competitor effect is significant (the paper's null on the rivals).
    "comp_sig_4": (0.0, 0.0),
    "comp_sig_7": (0.0, 0.0),
    "comp_sig_10": (0.0, 0.0),
}
