"""Benchmark: SPOTSYNTH on the Panic of 1907 (cross-method, novel application).

Applies the spillover-detection donor screen of

    O'Riordan & Gilligan-Lee (2025). "Spillover detection for donor selection in
    synthetic control models." J. Causal Inference 13:20240036.

to the Knickerbocker / Trust Company of America panic panel (``basedata/trust.dta``;
treated unit ID 34, panic after period 229). This is *not* in the SPOTSYNTH paper;
it is a cross-method check that ties three independent proximal estimators in the
library together on one real dataset, and it documents a regime limit.

Two things are pinned.

* **Cross-validation of the debiased ATT.** SPOTSYNTH's section-3.3 proximal
  debias -- which feeds the screen-*excluded* donors back as proxies -- recovers
  the same effect as the two-proxy ``PROXIMAL`` estimator's PI/PI-S on this panel
  (both ~ -1.15, the Park-Tchetgen / Liu et al. Table-3 magnitude). Two estimators
  built on different proximal assumptions agree to ~0.01.

* **The screen flags Trust Company of America.** TCA (ID 57) -- one of the two
  trusts the proximal papers treat as panic-affected -- is the single most
  anomalous donor (largest leave-one-out forecast error, rank 1 of 57), and is
  excluded. The screen recovers the famously-run-on trust without being told.

It also records an honest *regime limit*: the Panic of 1907 was a systemic shock
(every trust's latent shifted at the intervention), the failure mode the paper
diagrams in Figure 5. So the screen does **not** cleanly separate the spillover
labels here -- ``normal`` trusts are excluded at a rate no lower than the
panic-``connected`` ones. The detector is built for *localized* spillovers (a few
contaminated donors against a stable latent), not a market-wide panic; the
``spotsynth_real_data`` case validates it in that intended regime.

Deterministic: the frequentist (simplex least-squares) SC and the closed-form
screen carry no RNG, so the ATTs and the screen ranking are reproducible.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

TREATED = 34
PANIC_T = 229
_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "trust.dta")


def _panel():
    df = pd.read_stata(os.path.abspath(_DATA))
    df = df[df["ID"] != 1].copy()                            # drop the unbalanced unit
    df["Panic"] = np.where((df["time"] > PANIC_T) & (df["ID"] == TREATED), 1, 0)
    return df


def _spotsynth(df):
    from mlsynth import SPOTSYNTH

    res = SPOTSYNTH({
        "df": df, "outcome": "prc_log", "treat": "Panic", "time": "time",
        "unitid": "ID", "display_graphs": False, "selection": "S1",
        "forecast": "loo", "inference": "frequentist", "debias": True,
    }).fit()
    s = res.screen
    names = [int(n) for n in s.donor_names]
    fe = dict(zip(names, np.asarray(s.forecast_error)))
    excluded = {int(names[i]) for i in np.asarray(s.excluded_idx)}
    order = sorted(fe, key=lambda k: -fe[k])
    tca_rank = order.index(57) + 1                           # rank of TCA by anomaly

    meta = df.groupby("ID").agg(type=("type", "first"),
                                introuble=("introuble", "max"))
    meta = meta[meta.index != TREATED]
    grp = np.where(meta.introuble == 1, "introuble",
                   np.where(meta.type == "connected", "connected", "normal"))
    excl = np.array([1.0 if i in excluded else 0.0 for i in meta.index])
    rate = {g: float(excl[grp == g].mean()) for g in ("normal", "connected", "introuble")}
    return res, tca_rank, (57 in excluded), len(s.selected_idx), rate


def _proximal_pi(df):
    """The two-proxy PROXIMAL PI ATT (donors = normal trusts, surrogates = affected)."""
    from mlsynth import PROXIMAL

    donors = df[df["type"] == "normal"]["ID"].unique().tolist()
    surrogates = df[df["introuble"] == 1]["ID"].unique().tolist()
    d = df.copy()
    d[["bid_itp", "ask_itp"]] = d[["bid_itp", "ask_itp"]].apply(np.log)
    res = PROXIMAL({
        "df": d, "treat": "Panic", "time": "date", "outcome": "prc_log",
        "unitid": "ID", "vars": {"donorproxies": ["bid_itp"], "surrogatevars": ["ask_itp"]},
        "donors": donors, "surrogates": surrogates, "methods": ["PI"],
        "display_graphs": False,
    }).fit()
    return float(res.att_by_method()["PI"])


def run() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _panel()
        res, tca_rank, tca_excluded, n_sel, rate = _spotsynth(df)
        pi = _proximal_pi(df)
    return {
        "att_screened": float(res.att),
        "att_unscreened": float(res.att_unscreened),
        "att_debiased": float(res.att_debiased),
        "proximal_pi_att": pi,
        "debiased_vs_proximal": abs(float(res.att_debiased) - pi),
        "tca_top_anomaly": float(tca_rank == 1),
        "tca_excluded": float(tca_excluded),
        "n_selected": float(n_sel),
        # Systemic-shock signature: the screen does not separate the spillover
        # labels (normal exclusion rate is not below the connected one).
        "normal_excl_rate": rate["normal"],
        "connected_excl_rate": rate["connected"],
        "systemic_no_separation": float(rate["normal"] >= rate["connected"]),
    }


# The debiased SPOTSYNTH ATT cross-validates the PROXIMAL PI to ~0.01 (both the
# Table-3 magnitude); the screen ranks Trust Company of America as the single most
# anomalous donor and excludes it; and on this systemic shock the screen does not
# separate the spillover labels (the paper's Figure-5 regime).
EXPECTED = {
    "att_debiased": (-1.146, 0.05),
    "proximal_pi_att": (-1.148, 0.05),
    "debiased_vs_proximal": (0.002, 0.03),     # two proximal estimators agree
    "att_screened": (-0.960, 0.08),
    "att_unscreened": (-0.920, 0.08),
    "tca_top_anomaly": (1.0, 0.0),             # TCA is the most anomalous donor
    "tca_excluded": (1.0, 0.0),
    "n_selected": (28.0, 3.0),
    "systemic_no_separation": (1.0, 0.0),      # Figure-5 regime: no clean separation
}
