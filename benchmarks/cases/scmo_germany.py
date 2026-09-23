"""Path A benchmark: SCMO German reunification (Tian, Lee & Panchenko 2026, Sec. 4).

Reproduces the concatenated multiple-outcomes synthetic control of West Germany.
Instead of matching on 30 years of GDP, SCMO matches West Germany to the OECD
donors on nine economic indicators measured in the single year 1989 (private
social expenditure, energy-per-GDP, electricity and patents per capita, real GDP
growth, CPI, trade openness, total tax revenue, GDP per capita). The benchmark
checks that the fitted synthetic West Germany reproduces the paper's printed
1989 balance table cell-by-cell, in all four columns: West Germany itself, the
multiple-outcomes synthetic control, the single-outcome synthetic control, and
the comparison-group simple average.

Provenance
----------
* Data: ``basedata/germany_augmented.csv`` (West Germany + 16 OECD donors; GDP
  per capita 1960-2003 plus the OECD predictor columns).
* Headline: Tian-Lee-Panchenko (2026, Econometrics Journal) Table 2 ("Balance on
  economic outcomes in 1989"), all 36 cells::

      Outcome (1989)          West Germany | Synth (multi) | Synth (single) | Sample mean
      Private social exp.            3.4   |      3.5      |      3.7       |     2.0
      Energy supply per GDP          0.2   |      0.1      |      0.1       |     0.1
      Electricity generation         9.0   |      8.7      |     10.1       |     7.6
      Triadic patent families        0.1   |      0.1      |      0.0       |     0.0
      Real GDP growth                3.9   |      4.1      |      3.5       |     3.5
      CPI                            2.8   |      3.1      |      4.0       |     5.5
      Trade openness                57.7   |     59.1      |     59.3       |    60.4
      Total tax revenue             36.2   |     34.1      |     32.9       |    33.7
      GDP per capita             18994.0   |  19029.8      |  19075.9       | 16493.8

  The reference side is a live captured run of the authors' own ``Germany.R``
  (their ``fn_W`` ``quadprog::solve.QP`` synthetic-control program), captured
  under ``benchmarks/reference/scmo_germany/`` with its provenance pinned -- not
  numbers transcribed from the printed ``Output/Ger_tab.txt``. The synthetic
  1989 balance on the mlsynth side is reconstructed from ``res.donor_weights``
  applied to the donors' 1989 indicator values (a pure read through the
  standardized weights accessor). The two data columns (West Germany's own
  values, the donor average) are cross-checked against the same run, which is
  what ties mlsynth's spec transforms -- the per-capita normalizations, the GDP
  and trade joins -- to the authors' read of ``all.xlsx``.
* The paper's reading of the table, "both synthetic controls are generally much
  closer to West Germany in the outcomes, compared with the simple average", is
  pinned as a count: the multiple-outcomes SC beats the donor average on 9 of
  the 9 outcomes and the single-outcome SC on 8 (it loses on total tax revenue),
  and the multiple-outcomes SC beats the single-outcome SC on all 9 -- matching
  on the nine 1989 outcomes balances every one of them better than matching on
  30 years of the GDP path.
* The deterministic ATTs / pre-fit RMSEs that mlsynth's SCMO reports for this
  panel are pinned as regression guards (the reference script reports the
  post-1990 effect only graphically -- Tian et al. Figure 1 -- so no ATT number
  is cross-validated).
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

_POP = "Population levels"
_TREATED = "West Germany"

# One row of Table 2: (reference-bundle slug, the ``_SPEC`` variable it names,
# tolerance on the two synthetic-control columns). The tolerances sit at the
# scale of each row -- they cover the gap between mlsynth's cvxpy simplex and
# the authors' quadprog solve.QP, which the captured run puts below 2e-4
# relative on every cell.
_ROWS = (
    ("social", "private_social_exp", 0.06),
    ("energy", "energy_gdp", 0.004),
    ("electricity", "electricity_pc", 0.15),
    ("patents", "patents_pc", 0.002),
    ("gdp_growth", "gdp_growth", 0.06),
    ("cpi", "cpi", 0.06),
    ("trade", "trade", 0.1),
    ("tax", "tax", 0.1),
    ("gdp_pc", "gdp_pc", 0.6),
)


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    """The 1989 indicator values by country, one column per ``_SPEC`` variable.

    Applies the same rules the matching matrix is built from (a bare column is
    read as a level; ``per_capita`` divides by the population column), without
    the cross-unit standardization -- Table 2 is printed in levels.
    """
    d89 = df[df["year"] == 1989].set_index("country")
    cols = {}
    for name, rule in _SPEC["vars"].items():
        col, op = (rule, "level") if isinstance(rule, str) else rule
        cols[name] = d89[col] / d89[_POP] if op == "per_capita" else d89[col]
    return pd.DataFrame(cols)


def run() -> dict:
    from mlsynth import SCMO

    df = pd.read_csv(_BASE / "germany_augmented.csv")
    df["Reunification"] = ((df["country"] == _TREATED) & (df["year"] >= 1990)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SCMO({"df": df, "outcome": "gdp", "treat": "Reunification",
                    "unitid": "country", "time": "year", "spec": _SPEC,
                    "schemes": ["separate", "concatenated", "averaged"],
                    "conformal_alpha": 0.1, "display_graphs": False}).fit()

    ind = _indicators(df)
    donors = ind.drop(index=_TREATED)
    con, sep = res.fits["concatenated"], res.fits["separate"]
    years = sorted(df["year"].unique())
    i89 = years.index(1989)

    def balance(fit, var: str) -> float:
        # Synthetic 1989 value = donor weights . donor 1989 indicator values.
        return float(sum(w * ind.loc[k, var] for k, w in fit.donor_weights.items()))

    def synth(fit, var: str) -> float:
        # The GDP cell uses the full synthetic path (all weights, no 4-decimal
        # rounding through the weights accessor), i.e. the pre-period
        # counterfactual at 1989.
        if var == "gdp_pc":
            return float(np.asarray(fit.counterfactual)[i89])
        return balance(fit, var)

    out: dict = {}
    closer = {"multi": 0, "single": 0, "multi_vs_single": 0}
    for slug, var, _tol in _ROWS:
        wg = float(ind.loc[_TREATED, var])
        mean = float(donors[var].mean())
        multi, single = synth(con, var), synth(sep, var)
        out[f"wg_{slug}_1989"] = wg
        out[f"multi_{slug}_1989"] = multi
        out[f"single_{slug}_1989"] = single
        out[f"mean_{slug}_1989"] = mean
        closer["multi"] += abs(multi - wg) < abs(mean - wg)
        closer["single"] += abs(single - wg) < abs(mean - wg)
        closer["multi_vs_single"] += abs(multi - wg) < abs(single - wg)

    # The paper's reading of Table 2, as counts over the nine outcomes.
    out["multi_closer_than_mean"] = float(closer["multi"])
    out["single_closer_than_mean"] = float(closer["single"])
    out["multi_closer_than_single"] = float(closer["multi_vs_single"])
    # Deterministic mlsynth regression guards (not paper numbers).
    out["concatenated_att"] = float(con.att)
    out["averaged_att"] = float(res.fits["averaged"].att)
    out["concatenated_pre_rmse"] = float(con.pre_rmse)
    out["separate_pre_rmse"] = float(sep.pre_rmse)
    return out


def comparison() -> dict:
    """mlsynth SCMO vs the authors' ``Germany.R``, 1989 balance quantity by
    quantity. Pairs the synthetic West Germany's 1989 indicator values (both the
    multiple-outcome and single-outcome synthetic controls) against the live
    captured ``fn_W`` run (``benchmarks/reference/scmo_germany/``). The two data
    columns of Table 2 are pinned against the same run in ``EXPECTED``.
    """
    m = run()
    rows = [{"quantity": f"{col}/{slug}", "mlsynth": round(float(m[f"{col}_{slug}_1989"]), 4),
             "reference": round(reference_value("scmo_germany", f"{col}_{slug}_1989"), 4)}
            for col in ("multi", "single") for slug, _var, _tol in _ROWS]
    cfg = {"outcome": "gdp", "treat": "Reunification", "unitid": "country",
           "time": "year", "spec": "9 indicators in 1989",
           "schemes": ["separate", "concatenated", "averaged"]}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SCMO", "config": cfg},
        "reference": {"impl": "Tian-Lee-Panchenko Germany.R (fn_W solve.QP, live run, captured)",
                      "version": "Tian, Lee & Panchenko (2026), Econometrics Journal"},
    }


# Deterministic (no resampling). All 36 cells of Table 2 are pinned from the
# live captured Germany.R run (benchmarks/reference/scmo_germany/) via
# reference_value. The two synthetic-control columns carry the per-row
# tolerances in _ROWS; the two data columns (West Germany, donor average) run
# through no solver, so they are held to floating-point agreement between the R
# read of all.xlsx and mlsynth's read of germany_augmented.csv. The three counts
# are exact, and the four ATT / pre-RMSE figures are mlsynth regression guards
# (the reference script has no ATT to cross-validate against).
_sg = lambda k: reference_value("scmo_germany", k)


def _data_tol(value: float) -> float:
    """Tolerance for a cell that is read, not estimated: 1e-6 relative, with an
    absolute floor for the near-zero rows (patents per capita)."""
    return max(1e-6, abs(value) * 1e-6)


EXPECTED = {}
for _slug, _var, _tol in _ROWS:
    for _col in ("multi", "single"):
        EXPECTED[f"{_col}_{_slug}_1989"] = (_sg(f"{_col}_{_slug}_1989"), _tol)
    for _col in ("wg", "mean"):
        EXPECTED[f"{_col}_{_slug}_1989"] = (
            _sg(f"{_col}_{_slug}_1989"), _data_tol(_sg(f"{_col}_{_slug}_1989")))
EXPECTED.update({
    "multi_closer_than_mean": (9.0, 0.0),
    "single_closer_than_mean": (8.0, 0.0),
    "multi_closer_than_single": (9.0, 0.0),
    "concatenated_att": (-1462.8, 5.0),
    "averaged_att": (-1720.4, 8.0),
    "concatenated_pre_rmse": (110.0, 3.0),
    "separate_pre_rmse": (74.3, 3.0),
})
