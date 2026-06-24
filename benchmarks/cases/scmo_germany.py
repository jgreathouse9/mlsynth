"""Path A benchmark: SCMO German reunification (Tian, Lee & Panchenko 2026, Sec. 4).

Reproduces the concatenated multiple-outcomes synthetic control of West Germany.
Instead of matching on 30 years of GDP, SCMO matches West Germany to the OECD
donors on **nine economic indicators measured in the single year 1989** (private
social expenditure, energy-per-GDP, electricity and patents per capita, real GDP
growth, CPI, trade openness, total tax revenue, GDP per capita). The benchmark
checks that the fitted *synthetic* West Germany reproduces the paper's printed
1989 balance table cell-by-cell, for both the concatenated (multiple-outcomes)
and the separate (single-outcome) synthetic control.

Provenance
----------
* Data: ``basedata/germany_augmented.csv`` (West Germany + 16 OECD donors; GDP
  per capita 1960-2003 plus the OECD predictor columns).
* Headline: Tian-Lee-Panchenko (2026, Econometrics Journal) Table 2 ("Balance on
  economic outcomes in 1989"):

      Outcome (1989)        West Germany | Synth (multi) | Synth (single)
      GDP per capita            18994.0  |    19029.8    |    19075.9
      CPI                          2.8   |       3.1     |       4.0
      Trade openness              57.7   |      59.1     |      59.3
      Total tax revenue           36.2   |      34.1     |      32.9
      Real GDP growth              3.9   |       4.1     |       3.5

  The reference side is a live captured run of the authors' own ``Germany.R``
  (their ``fn_W`` ``quadprog::solve.QP`` synthetic-control program), captured
  under ``benchmarks/reference/scmo_germany/`` with its provenance pinned -- not
  numbers transcribed from the printed ``Output/Ger_tab.txt``. The synthetic
  1989 balance on the mlsynth side is reconstructed from ``res.donor_weights``
  applied to the donors' 1989 indicator values (a pure read through the
  standardized weights accessor). The deterministic ATTs / pre-fit RMSEs that
  mlsynth's SCMO reports for this panel are pinned as regression guards (the
  reference script reports the post-1990 effect only graphically -- Tian et al.
  Figure 1 -- so no ATT number is cross-validated).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"

_SPEC = {"year": 1989, "vars": {
    "private_social_exp": "Private social expenditure",
    "energy_gdp": "Total primary energy supply per unit of GDP",
    "electricity_pc": ("Electricity generation", "per_capita"),
    "patents_pc": ("Triadic patent families", "per_capita"),
    "gdp_growth": "Real GDP growth", "cpi": "CPI: all items",
    "trade": "trade", "tax": "Total tax revenue", "gdp_pc": "gdp"}}


def run() -> dict:
    from mlsynth import SCMO

    df = pd.read_csv(_BASE / "germany_augmented.csv")
    df["Reunification"] = ((df["country"] == "West Germany") & (df["year"] >= 1990)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SCMO({"df": df, "outcome": "gdp", "treat": "Reunification",
                    "unitid": "country", "time": "year", "spec": _SPEC,
                    "schemes": ["separate", "concatenated", "averaged"],
                    "conformal_alpha": 0.1, "display_graphs": False}).fit()

    d89 = df[df["year"] == 1989].set_index("country")
    con, sep = res.fits["concatenated"], res.fits["separate"]
    years = sorted(df["year"].unique())
    i89 = years.index(1989)

    def balance(fit, column: str) -> float:
        # Synthetic 1989 value = donor weights . donor 1989 indicator values.
        return float(sum(w * d89.loc[k, column] for k, w in fit.donor_weights.items()))

    def synth_gdp_1989(fit) -> float:
        # The GDP cell uses the full synthetic path (all weights, no small-weight
        # truncation), i.e. the pre-period counterfactual at 1989.
        return float(np.asarray(fit.counterfactual)[i89])

    return {
        # Tian Table 2, "Synthetic (multiple outcomes)" column.
        "multi_gdp_pc_1989": synth_gdp_1989(con),
        "multi_cpi_1989": balance(con, "CPI: all items"),
        "multi_trade_1989": balance(con, "trade"),
        "multi_tax_1989": balance(con, "Total tax revenue"),
        "multi_gdp_growth_1989": balance(con, "Real GDP growth"),
        # Tian Table 2, "Synthetic (single outcome)" column.
        "single_gdp_pc_1989": synth_gdp_1989(sep),
        # Deterministic mlsynth regression guards (not paper numbers).
        "concatenated_att": float(res.fits["concatenated"].att),
        "averaged_att": float(res.fits["averaged"].att),
        "concatenated_pre_rmse": float(res.fits["concatenated"].pre_rmse),
        "separate_pre_rmse": float(res.fits["separate"].pre_rmse),
    }


def comparison() -> dict:
    """mlsynth SCMO vs the authors' ``Germany.R``, 1989 balance quantity by
    quantity. Pairs the synthetic West Germany's 1989 indicator values (both the
    multiple-outcome and single-outcome synthetic controls) against the live
    captured ``fn_W`` run (``benchmarks/reference/scmo_germany/``).
    """
    m = run()
    pairs = [
        ("multi/GDP_pc", "multi_gdp_pc_1989"),
        ("multi/CPI", "multi_cpi_1989"),
        ("multi/trade_openness", "multi_trade_1989"),
        ("multi/total_tax", "multi_tax_1989"),
        ("multi/real_GDP_growth", "multi_gdp_growth_1989"),
        ("single/GDP_pc", "single_gdp_pc_1989"),
    ]
    rows = [{"quantity": q, "mlsynth": round(float(m[k]), 4),
             "reference": round(reference_value("scmo_germany", k), 4)}
            for q, k in pairs]
    cfg = {"outcome": "gdp", "treat": "Reunification", "unitid": "country",
           "time": "year", "spec": "9 indicators in 1989",
           "schemes": ["separate", "concatenated", "averaged"]}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SCMO", "config": cfg},
        "reference": {"impl": "Tian-Lee-Panchenko Germany.R (fn_W solve.QP, live run, captured)",
                      "version": "Tian, Lee & Panchenko (2026), Econometrics Journal"},
    }


# Deterministic (no resampling). The six balance targets are pinned from the live
# captured Germany.R run (benchmarks/reference/scmo_germany/) via reference_value;
# tolerances cover the small gap between mlsynth's SCMO and the authors' fn_W
# solve.QP. The four ATT / pre-RMSE figures are mlsynth regression guards (the
# reference script has no ATT to cross-validate against).
_sg = lambda k: reference_value("scmo_germany", k)
EXPECTED = {
    "multi_gdp_pc_1989": (_sg("multi_gdp_pc_1989"), 0.6),
    "multi_cpi_1989": (_sg("multi_cpi_1989"), 0.06),
    "multi_trade_1989": (_sg("multi_trade_1989"), 0.1),
    "multi_tax_1989": (_sg("multi_tax_1989"), 0.1),
    "multi_gdp_growth_1989": (_sg("multi_gdp_growth_1989"), 0.06),
    "single_gdp_pc_1989": (_sg("single_gdp_pc_1989"), 0.6),
    "concatenated_att": (-1462.8, 5.0),
    "averaged_att": (-1720.4, 8.0),
    "concatenated_pre_rmse": (110.0, 3.0),
    "separate_pre_rmse": (74.3, 3.0),
}
