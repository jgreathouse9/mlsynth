"""Cross-validation benchmark: CLUSTERSC RPCA-SC vs Bayani's code (West Germany).

Cross-validates mlsynth's ``CLUSTERSC`` RPCA-SC family -- robust low-rank donor
denoising via Principal Component Pursuit plus a non-negative fit (Bayani 2021,
*Robust PCA Synthetic Control*) -- against the author's *own* dissertation
code, vendored verbatim under ``benchmarks/reference/vendor/bayani_rpca_synth/``
(his ``FPCA.R`` cluster selection and ``RPCA_2.py`` PCP + NNLS; see the
``NOTICE.md`` there) and driven by :mod:`benchmarks.reference.rpca_sc_reference`,
which loads his verbatim ``RPCA`` routine and runs it on his own panel.

The canonical German-reunification panel (``basedata/german_reunification.csv``:
annual GDP per capita for 17 countries, 1960-2003; West Germany treated from
1990) feeds both. On the West-Germany donor cluster, mlsynth reproduces the
author's RPCA-SC value-for-value: the donor weights (Norway ~0.485, France
~0.354, New Zealand ~0.296, Austria ~0.023) to ~2e-7, the full counterfactual
path to ~1e-5 (per-capita GDP in the tens of thousands), and the reunification
effect (a ~1500 USD decline in West German GDP) to ~2e-7. Cross-validation
(scenario 3): a faithful reproduction of the author's own code.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def _mlsynth_and_reference():
    from mlsynth import CLUSTERSC
    from benchmarks.reference.rpca_sc_reference import rpca_sc_west_germany

    df = pd.read_csv(_BASE / "german_reunification.csv")
    df["treat"] = df["Reunification"].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC({
            "df": df, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "method": "rpca", "rpca_method": "PCP", "display_graphs": False,
        }).fit()
    ref = rpca_sc_west_germany()               # runs the author's vendored RPCA
    return res, ref


def run() -> dict:
    res, ref = _mlsynth_and_reference()
    mw = res.donor_weights or {}
    weights_delta = max(abs(mw.get(k, 0.0) - v) for k, v in ref["weights"].items())
    ml_cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    return {
        "norway_weight": float(mw.get("Norway", 0.0)),
        "france_weight": float(mw.get("France", 0.0)),
        "newzealand_weight": float(mw.get("New Zealand", 0.0)),
        "weights_vs_ref": float(weights_delta),
        "counterfactual_vs_ref": float(np.max(np.abs(ml_cf - ref["counterfactual"]))),
        "pre_rmse_vs_ref": float(abs(float(res.pre_rmse) - ref["pre_rmse"])),
        "att_vs_ref": float(abs(float(res.att) - ref["att"])),
        "att_negative": float(res.att < 0.0),
    }


def comparison() -> dict:
    """mlsynth CLUSTERSC RPCA-SC vs the author's RPCA-SC, West Germany, side by side."""
    res, ref = _mlsynth_and_reference()
    mw = res.donor_weights or {}
    ml_cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    years = list(ref["years"])
    ref_cf = ref["counterfactual"]

    def q(name, m, r):
        return {"quantity": name, "mlsynth": round(float(m), 4),
                "reference": round(float(r), 4)}

    rows = [q(f"weight[{d}]", mw.get(d, 0.0), ref["weights"][d])
            for d in ("Norway", "France", "New Zealand", "Austria")]
    rows.append(q("pre-period RMSE", res.pre_rmse, ref["pre_rmse"]))
    rows.append(q("ATT (1990-2003)", res.att, ref["att"]))
    for yr in (1990, 1997, 2003):
        i = years.index(yr)
        rows.append(q(f"counterfactual[{yr}]", ml_cf[i], ref_cf[i]))
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "CLUSTERSC",
                         "config": {"method": "rpca", "rpca_method": "PCP"}},
        "reference": {"impl": "Bayani RPCA-SC -- the author's own code, vendored "
                              "verbatim (vendor/bayani_rpca_synth: FPCA.R + RPCA_2.py)",
                      "version": "Bayani, Robust PCA Synthetic Control (dissertation)"},
    }


# Deterministic on both sides (PCP is a fixed-point ADMM; the NNLS is convex).
# mlsynth must reproduce the author's RPCA-SC value-for-value; tolerances bracket
# solver drift (CLARABEL vs the reference's default, ADMM stopping) -- observed
# ~2e-7 on the weights, ~1e-5 on the counterfactual path.
EXPECTED = {
    "norway_weight": (0.4854, 0.01),
    "france_weight": (0.3540, 0.01),
    "newzealand_weight": (0.2964, 0.01),
    "weights_vs_ref": (0.0, 1e-4),
    "counterfactual_vs_ref": (0.0, 1e-2),
    "pre_rmse_vs_ref": (0.0, 1e-4),
    "att_vs_ref": (0.0, 1e-3),
    "att_negative": (1.0, 0.0),
}
