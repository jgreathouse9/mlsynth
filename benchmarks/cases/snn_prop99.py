"""Cross-validation benchmark: SNN vs ``deshen24/syntheticNN`` (Prop 99).

Cross-validation against the reference implementation. ``SNN`` is mlsynth's port
of Synthetic Nearest Neighbors (Agarwal, Dwivedi, Shah & Shen, *Causal Matrix
Completion*, arXiv:2109.15154), whose canonical implementation is
`deshen24/syntheticNN <https://github.com/deshen24/syntheticNN>`_. On block
missingness -- the synthetic-control setting, where the treated unit's
post-period is the only missing block -- the reference's NetworkX maximum-
biclique anchor search and mlsynth's dependency-free greedy search both return
the full *control x pre-period* block. With the same Donoho-Gavish (2014) rank
and principal-component regression, the two implementations therefore impute the
**same** counterfactual.

Reference (live captured run)
-----------------------------
The reference side is a live captured run of ``deshen24/syntheticNN``, not
numbers transcribed from a paper. ``benchmarks/reference/snn_prop99/reference.py``
fetches the authors' ``snn.py`` at a pinned commit
(``a95b511``, into the gitignored ``benchmarks/reference/.cache``; git clone is
proxy-blocked here, so it falls back to the codeload tarball -- see the bundle
``NOTICE`` and ``benchmarks/reference/clone_syntheticnn.py``) and runs
``SyntheticNearestNeighbors(n_neighbors=1)`` (universal/Donoho-Gavish rank) on the
Prop-99 matrix with California's 1989-2000 entries set to ``NaN``. Its imputed
counterfactual, ATT, and 2000 gap are captured under
``benchmarks/reference/snn_prop99/`` with full provenance, and this case pins
them by reading the captured ``reference.json`` via :func:`reference_value` --
so the constants in ``EXPECTED`` and the captured run are the same object and
cannot silently drift. Regenerate with
``python benchmarks/reference/generate.py snn_prop99``.

Provenance
----------
* Data: ``basedata/smoking_data.csv`` -- the Abadie, Diamond & Hainmueller (2010)
  Prop 99 panel (39 states, 1970-2000; California treated from 1989). Outcome
  ``cigsale`` (per-capita cigarette packs).
* mlsynth reproduces the reference counterfactual to machine precision
  (max abs diff ~4e-7 against the full-precision captured values).

This case also guards the Donoho-Gavish ``omega`` coefficients: an earlier
mlsynth bug had ``1.43`` and ``1.82`` swapped, which mis-selected the rank and
shifted the ATT by ~1 pack/capita. Matched to the reference formula, the gap
vanishes.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# California untreated counterfactual (cigsale) and ATT from the live captured
# deshen24/syntheticNN run in benchmarks/reference/snn_prop99/ (read, not
# transcribed).
_REF_VALUES = load_reference("snn_prop99")["values"]
_POST_YEARS = list(range(1989, 2001))
REF_CF = {yr: _REF_VALUES[f"cf_{yr}"] for yr in _POST_YEARS}
REF_ATT = reference_value("snn_prop99", "snn_att")
REF_GAP_2000 = reference_value("snn_prop99", "snn_gap_2000")


def run() -> dict:
    from mlsynth import SNN

    df = pd.read_csv(_BASE / "smoking_data.csv")
    obs = df.pivot(index="state", columns="year", values="cigsale").loc["California"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SNN({"df": df, "outcome": "cigsale", "treat": "Proposition 99",
                   "unitid": "state", "time": "year", "display_graphs": False}).fit()

    years = sorted(REF_CF)
    cf_mls = np.array([float(obs[y]) - res.att_by_period[y] for y in years])
    cf_ref = np.array([REF_CF[y] for y in years])

    return {
        "snn_att": float(res.att),
        "snn_counterfactual_max_abs_diff": float(np.max(np.abs(cf_mls - cf_ref))),
        "snn_gap_2000": float(res.att_by_period[2000]),
        "n_states": int(df.state.nunique()),
        "n_pre_periods": int((df[df.state == "California"].year < 1989).sum()),
    }


def comparison() -> dict:
    """mlsynth ``SNN`` vs ``deshen24/syntheticNN``, quantity by quantity.

    Lays the mlsynth SNN fit against the canonical syntheticNN run on the same
    Prop-99 matrix (same treated unit, same 1989 pre/post split): the ATT, the
    2000 gap, and the imputed counterfactual at each post-year. The reference
    side is a live captured ``SyntheticNearestNeighbors`` run in
    ``benchmarks/reference/snn_prop99/`` (commit ``a95b511``), not transcribed.
    Returns ``{"rows": [...], "mlsynth_call": {...}, "reference": {...}}`` with
    rows ``{quantity, mlsynth, reference}``.
    """
    from mlsynth import SNN

    cfg = {"outcome": "cigsale", "treat": "Proposition 99", "unitid": "state",
           "time": "year"}
    df = pd.read_csv(_BASE / "smoking_data.csv")
    obs = df.pivot(index="state", columns="year", values="cigsale").loc["California"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SNN({**cfg, "df": df, "display_graphs": False}).fit()

    rows = [
        {"quantity": "ATT", "mlsynth": round(float(res.att), 6),
         "reference": round(REF_ATT, 6)},
        {"quantity": "gap[2000]", "mlsynth": round(float(res.att_by_period[2000]), 6),
         "reference": round(REF_GAP_2000, 6)},
    ]
    for yr in sorted(REF_CF):
        cf_ml = float(obs[yr]) - res.att_by_period[yr]
        rows.append({"quantity": f"counterfactual[{yr}]",
                     "mlsynth": round(cf_ml, 6),
                     "reference": round(REF_CF[yr], 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SNN", "config": cfg},
        "reference": {"impl": "deshen24/syntheticNN (live run, captured), "
                              "SyntheticNearestNeighbors(n_neighbors=1)",
                      "version": "deshen24/syntheticNN @ a95b511 "
                                 "(benchmarks/reference/snn_prop99/)"},
    }


# Deterministic matrix completion: on block missingness both implementations
# recover the same anchor block, rank, and PCR weights, so they impute the same
# counterfactual. Targets are pinned from the live captured syntheticNN run
# (benchmarks/reference/snn_prop99/) via reference_value/load_reference, not
# transcribed; tolerances are the actual mlsynth-vs-syntheticNN gap (~4e-7 on the
# counterfactual, ~3e-7 on the ATT and the 2000 gap -- machine precision).
EXPECTED = {
    "snn_att": (REF_ATT, 1e-5),
    "snn_counterfactual_max_abs_diff": (0.0, 1e-5),
    "snn_gap_2000": (REF_GAP_2000, 1e-5),
    "n_states": (39, 0),
    "n_pre_periods": (19, 0),
}
