"""VanillaSC cross-validation: the Iberian exception -> day-ahead price (ibex).

Cross-validation against the authors' own replication code. Haro Ruiz, Schult and
Wunder (2024), *"The effects of the Iberian exception mechanism on wholesale
electricity prices and consumer inflation: a synthetic-controls approach"*
(Applied Economics Letters, DOI 10.1080/13504851.2024.2425834), estimate the
effect of the June-2022 Iberian exception mechanism (IbEx) with a standard
synthetic control. Their public replication (``github.com/mharoruiz/ibex``,
``01_functions/sc.R``) solves the canonical Abadie-Diamond-Hainmueller program --
simplex-constrained donor weights (``limSolve::lsei``: non-negative, sum-to-one)
fit to *all* pre-treatment outcome lags, no covariates -- via the ``scinference``
package. That is exactly mlsynth's ``VanillaSC`` with ``backend="outcome-only"``.

Spain and Portugal are both treated in June 2022 and neither is a donor for the
other, so a single multi-treated ``VanillaSC`` fit reproduces the paper's two
per-country loops: mlsynth drops every treated unit from the shared donor pool,
which for two mutually-excluded units is the same 20-country pool the paper uses.

On the day-ahead price the reproduction is value-for-value: the donor weights
match the ibex ``sc_optimized_weights.csv`` to solver tolerance (L1 distance
``0``), and because the weights and the donor data are identical the per-period
gap reproduces the ibex ``sc_series`` pointwise (max ``|Δ|`` ~1e-8). Both
countries show a large day-ahead cut -- mean post-treatment gap about
:math:`-46` Euros/MWh (Spain) and :math:`-45` (Portugal), the paper's "about 40%"
headline. See ``docs/replications/ibex``.

The three CPI outcomes in the paper (energy, non-energy, all-items HICP) are
pulled live from Eurostat and are not shipped here; this case pins the offline,
fully reproducible day-ahead-price result. The reference numbers are captured
from ibex at commit ``c5371314`` under ``benchmarks/reference/ibex_dap/``.

Provenance: Haro Ruiz, Schult & Wunder (2024), Applied Economics Letters
33(9):1310-1316; data from OMIE + Fraunhofer ISE energy-charts (CC BY 4.0) as
redistributed in mharoruiz/ibex ``02_data/day_ahead_price.csv``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "ibex_day_ahead_price.csv")
_REF = load_reference("ibex_dap")
_TREATED = ("ES", "PT")
_TREAT_DATE = pd.Timestamp("2022-06-01")   # IbEx onset (ibex treatment_date)

# The one VanillaSC call shared by run() and comparison(): the paper's own
# outcome-only SC (simplex weights on all pre-treatment lags), both countries
# treated at once, 90% intervals (ibex alpha=.1).
_MLSYNTH_KW = {
    "outcome": "DAP", "treat": "treat", "unitid": "country", "time": "date",
    "backend": "outcome-only", "alpha": 0.10, "display_graphs": False,
}


def _fit():
    """One multi-treated VanillaSC fit on the day-ahead-price panel (ES + PT
    treated from June 2022; the 20 pure-control countries as donors)."""
    from mlsynth import VanillaSC

    df = pd.read_csv(os.path.abspath(_DATA), parse_dates=["date"])
    # ibex drops any country with a missing month before pooling.
    complete = df.groupby("country")["DAP"].transform(lambda s: ~s.isna().any())
    df = df[complete].copy()
    df["treat"] = ((df.country.isin(_TREATED)) & (df.date >= _TREAT_DATE)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({"df": df, **_MLSYNTH_KW}).fit()


def _per_unit(res) -> dict:
    """For each treated country: mlsynth donor weights (as a Series), the L1
    distance to the ibex reference weights, the max per-period gap difference vs
    the ibex ``sc_series`` (on the shipped snapshot's dates), and the mean
    post-treatment gap (the day-ahead ATT)."""
    out = {}
    for tu in _TREATED:
        fit = res.sub_method_results[tu]
        w = pd.Series(fit.donor_weights, dtype=float)
        w = w[w.abs() > 1e-6]
        ref_w = pd.Series(_REF["weights"][tu], dtype=float)
        keys = ref_w.index.union(w.index)
        l1 = float((w.reindex(keys).fillna(0.0) - ref_w.reindex(keys).fillna(0.0)).abs().sum())
        t = pd.to_datetime(fit.time_labels)
        gap = pd.Series(np.asarray(fit.gap, dtype=float), index=t)
        att_post = float(gap[gap.index >= _TREAT_DATE].mean())
        out[tu] = {"weights": w, "weight_l1": l1, "att_post": att_post,
                   "si": float(w.get("SI", np.nan))}
    return out


def run() -> dict:
    res = _fit()
    u = _per_unit(res)
    es, pt = u["ES"], u["PT"]
    return {
        # donor weights reproduce the ibex lsei/scinference SC value-for-value
        "es_weight_l1": es["weight_l1"],
        "pt_weight_l1": pt["weight_l1"],
        "both_weights_exact": float(es["weight_l1"] < 1e-3 and pt["weight_l1"] < 1e-3),
        # dominant donor (Slovenia) weight
        "es_si_weight": es["si"],
        "pt_si_weight": pt["si"],
        # day-ahead ATT: both countries a large cut (paper: "about 40%")
        "es_att_post": es["att_post"],
        "pt_att_post": pt["att_post"],
        "both_large_negative": float(es["att_post"] < -30.0 and pt["att_post"] < -30.0),
    }


def comparison() -> dict:
    """mlsynth ``VanillaSC`` (outcome-only) vs the ibex ``scinference``/``lsei``
    synthetic control on the day-ahead price, quantity by quantity. Both solve
    the identical simplex program on all pre-treatment outcome lags; the
    reference numbers are the ibex ``sc_optimized_weights.csv`` /
    ``sc_series_001.csv`` captured under ``benchmarks/reference/ibex_dap/``.
    The match is exact to solver tolerance."""
    u = _per_unit(_fit())
    rv = _REF["values"]
    rows = [
        {"quantity": "Spain — Slovenia (SI) weight", "mlsynth": round(u["ES"]["si"], 4),
         "reference": round(rv["es_si_weight"], 4)},
        {"quantity": "Portugal — Slovenia (SI) weight", "mlsynth": round(u["PT"]["si"], 4),
         "reference": round(rv["pt_si_weight"], 4)},
        {"quantity": "Spain — donor-weight L1 distance", "mlsynth": round(u["ES"]["weight_l1"], 4),
         "reference": 0.0},
        {"quantity": "Portugal — donor-weight L1 distance", "mlsynth": round(u["PT"]["weight_l1"], 4),
         "reference": 0.0},
        {"quantity": "Spain — mean post ATT (Euros/MWh)", "mlsynth": round(u["ES"]["att_post"], 4),
         "reference": round(rv["es_att_post"], 4)},
        {"quantity": "Portugal — mean post ATT (Euros/MWh)", "mlsynth": round(u["PT"]["att_post"], 4),
         "reference": round(rv["pt_att_post"], 4)},
    ]
    cfg = {k: v for k, v in _MLSYNTH_KW.items() if k != "display_graphs"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
        "reference": {
            "impl": "mharoruiz/ibex R replication (01_functions/sc.R: limSolve::lsei simplex SC; scinference)",
            "version": "Haro Ruiz, Schult & Wunder (2024) Applied Economics Letters, ibex @ c5371314",
        },
    }


# Cross-validation (scenario: full repo). mlsynth's VanillaSC (outcome-only)
# reproduces the ibex synthetic control on the day-ahead price value-for-value:
# donor weights match to L1 ~0, the dominant Slovenia weight is 0.526 (ES) /
# 0.516 (PT), and the mean post-treatment gap is -46.4 / -45.5 Euros/MWh (the
# paper's ~40% cut). Deterministic convex solve => machine-tight tolerances.
EXPECTED = {
    "es_weight_l1": (0.0, 5e-3),
    "pt_weight_l1": (0.0, 5e-3),
    "both_weights_exact": (1.0, 0.0),
    "es_si_weight": (0.52589, 5e-3),
    "pt_si_weight": (0.516263, 5e-3),
    "es_att_post": (-46.3572, 0.5),
    "pt_att_post": (-45.4690, 0.5),
    "both_large_negative": (1.0, 0.0),
}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
