"""COMPSC Path A: Boussim (2026) Pennsylvania Alternative Energy Portfolio Standard.

Path A (scenario 1, paper only -- no replication package was published). Reproduces
Table 1, Table 2 and section 6.4 of Boussim, O. (2026), "Compositional Synthetic
Controls" (arXiv:2607.16991): the compositional effect of Pennsylvania's AEPS
(Act 213 of 2004) on its electricity generation mix, with the treatment date set
at T0 = 2003 (fourteen pre-treatment years, twenty post).

Run under the paper's geometry -- ``geometry="alr"`` with renewables as the
baseline category -- ``COMPSC.fit()`` matches every published number:

* Table 1: all nine donor weights to three decimals, Aitchison pre-treatment
  RMSPE 0.187, share RMSPE 1.34 percentage points;
* Table 2: all eleven reported years x six columns (per-category share effects,
  two log-ratio effects, Aitchison displacement);
* section 6.4: post/pre Aitchison RMSPE ratio 9.18, 36 retained units, exact
  p-value 4/36 = 0.111, with Michigan, Louisiana and Florida the three placebos
  exceeding Pennsylvania.

One difference from the printed table: mlsynth gives Arkansas a weight of
0.000924, and Table 1 lists nine donors. This was previously recorded here as a
solver difference, on the assumption that ``quadprog::solve.QP`` drives that
weight to zero. Running the author's replication script shows it does not --
``quadprog`` returns 0.000924 for Arkansas too, agreeing with mlsynth to 6e-10,
and reports ten active donors. The divergence is between the paper's table and
the author's own code; see ``compsc_pennsylvania_r``, which pins mlsynth against
that code. No reported statistic changes either way, and the gate below
tolerates weights under 1e-3 that the table omits.

The case also records the geometry correction that mlsynth ships as its default.
The paper's additive log-ratio map is not the isometry for the Aitchison metric
(the centered log-ratio is), so Algorithm 1's weights depend on which category is
nominated as the baseline. Switching to ``geometry="clr"`` moves the weights by
0.85 in L1, improves the Aitchison pre-treatment fit from 0.187 to 0.177, and
moves the 2022 natural-gas effect from +60.4 to +51.1 points. Both are reported
so the divergence is visible rather than buried.

Data: ``basedata/pa_aeps_generation.csv`` -- annual net generation in megawatt
hours by state and category, 1990-2023, built from the U.S. Energy Information
Administration state historical table ``annual_generation_state.xls`` (Total
Electric Power Industry). Categories follow section 6.1: natural gas (EIA
"Natural Gas" plus "Other Gases"); coal and oil ("Coal" plus "Petroleum"); and
renewables (conventional hydro, wind, solar, geothermal, wood and wood-derived
fuels, other biomass, other, and pumped storage). Nuclear is excluded and the
three categories are renormalised, which is what reproduces the paper's
pre-treatment balance. Donor pool follows section 6.1: states with a structural
zero in any category over the sample are dropped, as are Ohio and West Virginia
(which adopted comparable standards during the sample) and Vermont (whose gas and
fossil shares are each below 0.1 percent throughout), leaving 42 donors.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "pa_aeps_generation.csv")

_CATEGORIES = ["gas", "fossil", "renewables"]

# Boussim (2026) Table 1: donor weights and pre-treatment balance.
_PAPER_WEIGHTS = {
    "IL": 0.233, "WY": 0.208, "IA": 0.168, "NE": 0.149, "MA": 0.102,
    "CT": 0.048, "NJ": 0.043, "IN": 0.032, "MT": 0.016,
}
_PAPER_AITCHISON_RMSPE = 0.187
_PAPER_SHARE_RMSPE_PP = 1.34
_PAPER_PRE_SHARES_PCT = {"gas": 3.6, "fossil": 93.2, "renewables": 3.2}

# Boussim (2026) Table 2, even years plus 2023.
# year -> (ATT_gas, ATT_fos, ATT_ren in pp, LRTE_gas|fos, LRTE_gas|ren, delta_A)
_PAPER_TABLE2 = {
    2004: (4.6, -4.8, 0.2, 0.98, 0.87, 0.76),
    2006: (2.7, -1.9, -0.7, 0.34, 0.49, 0.36),
    2008: (7.1, -5.7, -1.4, 0.82, 1.04, 0.78),
    2010: (16.5, -10.5, -6.0, 1.46, 2.18, 1.57),
    2012: (25.3, -13.7, -11.6, 1.41, 2.43, 1.73),
    2014: (30.4, -18.8, -11.6, 1.94, 2.69, 1.96),
    2016: (37.3, -20.8, -16.5, 1.68, 2.54, 1.82),
    2018: (39.9, -22.4, -17.5, 1.66, 2.33, 1.70),
    2020: (52.1, -15.8, -36.3, 1.79, 3.08, 2.19),
    2022: (60.4, -24.5, -35.9, 2.37, 3.50, 2.53),
    2023: (59.1, -18.9, -40.2, 2.35, 3.35, 2.43),
}

# Boussim (2026) section 6.4.
_PAPER_PLACEBO = {"treated_ratio": 9.18, "n_retained": 36, "p_value": 4 / 36}
_PAPER_TOP_PLACEBOS = {"MI": 10.46, "LA": 10.22, "FL": 10.07}

_WEIGHT_FLOOR = 1e-3   # below this the paper reports the donor as unweighted


def _panel() -> pd.DataFrame:
    return pd.read_csv(_DATA)


def _fit(geometry: str = "alr", baseline: str = "renewables"):
    from mlsynth import COMPSC

    cfg = {
        "df": _panel(), "outcomes": _CATEGORIES, "treat": "treat",
        "unitid": "state", "time": "year", "target": "gas",
        "geometry": geometry, "display_graphs": False,
    }
    if geometry == "alr":
        cfg["baseline"] = baseline
    return COMPSC(cfg).fit()


def _years(res) -> list:
    return list(np.asarray(res.time_series.time_periods).ravel())


def run() -> dict:
    """Reproduce Tables 1-2 and section 6.4; return the max deviations."""
    res = _fit()
    years = _years(res)
    k = {c: i for i, c in enumerate(_CATEGORIES)}

    # --- Table 1: weights -------------------------------------------------
    fitted = {s: w for s, w in res.donor_weights.items() if w >= _WEIGHT_FLOOR}
    weight_diff = max(
        abs(fitted.get(s, 0.0) - w) for s, w in _PAPER_WEIGHTS.items()
    )
    extra = sorted(set(fitted) - set(_PAPER_WEIGHTS))

    pre = res.observed_composition[: years.index(2004)].mean(axis=0) * 100.0
    balance_diff = max(
        abs(pre[k[c]] - v) for c, v in _PAPER_PRE_SHARES_PCT.items()
    )

    # --- Table 2 ----------------------------------------------------------
    t2 = 0.0
    for year, paper in _PAPER_TABLE2.items():
        i = years.index(year)
        mine = (
            100 * res.share_effects[i, k["gas"]],
            100 * res.share_effects[i, k["fossil"]],
            100 * res.share_effects[i, k["renewables"]],
            res.log_ratio_effects[i, k["gas"], k["fossil"]],
            res.log_ratio_effects[i, k["gas"], k["renewables"]],
            res.aitchison_distance[i],
        )
        t2 = max(t2, max(abs(a - b) for a, b in zip(mine, paper)))

    # --- section 6.4 ------------------------------------------------------
    pb = res.placebo
    placebo_diff = max(
        abs(pb.treated_ratio - _PAPER_PLACEBO["treated_ratio"]),
        abs(pb.p_value - _PAPER_PLACEBO["p_value"]),
    )
    top_diff = max(
        abs(pb.ratios[s] - r) for s, r in _PAPER_TOP_PLACEBOS.items()
    )

    # --- the geometry correction -----------------------------------------
    clr = _fit(geometry="clr")
    l1 = sum(abs(clr.donor_weights[s] - res.donor_weights[s])
             for s in res.donor_weights)
    i22 = years.index(2022)

    # the paper prints three decimals, so the exact claim is that every weight
    # rounds to its published value -- more robust across solvers than a
    # knife-edge absolute tolerance
    rounded_mismatches = sum(
        round(fitted.get(s, 0.0), 3) != w for s, w in _PAPER_WEIGHTS.items()
    )

    return {
        "n_donors": len(res.donor_weights),
        "n_weighted_donors": len(fitted),
        "weights_rounded_mismatches": float(rounded_mismatches),
        "weights_max_diff_vs_paper": weight_diff,
        "n_extra_donors": float(len(extra)),
        "extra_donors": ", ".join(extra) or "none",
        "share_effects_abs_sum_max": float(
            np.abs(res.share_effects.sum(axis=1)).max()),
        "pre_balance_max_diff_pp": balance_diff,
        "aitchison_rmspe_pre": res.pre_treatment_aitchison_rmspe,
        "share_rmspe_pre_pp": 100 * res.pre_treatment_share_rmspe,
        "table2_max_diff": t2,
        "placebo_p_value": pb.p_value,
        "placebo_n_retained": pb.n_retained,
        "placebo_max_diff_vs_paper": max(placebo_diff, top_diff),
        # the correction mlsynth ships as its default
        "clr_l1_weight_shift": l1,
        "clr_aitchison_rmspe_pre": clr.pre_treatment_aitchison_rmspe,
        "alr_2022_att_gas_pp": 100 * res.share_effects[i22, k["gas"]],
        "clr_2022_att_gas_pp": 100 * clr.share_effects[i22, k["gas"]],
        "baseline_sensitivity_l1": res.baseline_sensitivity,
    }


# Gate. The paper's numbers carry their printed precision, so the tolerances
# below are "agrees at the precision Boussim (2026) reports": 3 decimals on the
# weights, 1 on percentage points, 2 on the log-ratio effects and delta_A. The
# clr_* rows are mlsynth's own corrected-geometry output, pinned as regression
# guards rather than checked against the paper.
EXPECTED = {
    # Table 1
    "n_donors": (42, 0.5),
    "n_weighted_donors": (9, 0.5),
    "weights_rounded_mismatches": (0.0, 0.5),
    "weights_max_diff_vs_paper": (0.0, 1e-3),
    "n_extra_donors": (0.0, 0.5),
    "pre_balance_max_diff_pp": (0.0, 0.05),
    "aitchison_rmspe_pre": (0.187, 5e-4),
    "share_rmspe_pre_pp": (1.34, 5e-3),
    # Table 2 -- all eleven years x six columns
    "table2_max_diff": (0.0, 0.05),
    # structural invariant of the geometry
    "share_effects_abs_sum_max": (0.0, 1e-12),
    # section 6.4
    "placebo_p_value": (4 / 36, 1e-6),
    "placebo_n_retained": (36, 0.5),
    "placebo_max_diff_vs_paper": (0.0, 0.01),
    # the correction mlsynth ships as its default (regression guards)
    "clr_l1_weight_shift": (0.8515, 5e-3),
    "clr_aitchison_rmspe_pre": (0.1773, 5e-4),
    "alr_2022_att_gas_pp": (60.4, 0.05),
    "clr_2022_att_gas_pp": (51.09, 0.05),
    "baseline_sensitivity_l1": (1.1605, 5e-3),
}


def comparison() -> dict:
    """Side-by-side mlsynth vs the paper's published tables."""
    res = _fit()
    years = _years(res)
    k = {c: i for i, c in enumerate(_CATEGORIES)}
    rows = []

    for state, w in _PAPER_WEIGHTS.items():
        rows.append({"quantity": f"weight[{state}]",
                     "mlsynth": float(res.donor_weights[state]),
                     "reference": w})
    rows.append({"quantity": "Aitchison RMSPE (pre)",
                 "mlsynth": float(res.pre_treatment_aitchison_rmspe),
                 "reference": _PAPER_AITCHISON_RMSPE})
    rows.append({"quantity": "share RMSPE (pre, pp)",
                 "mlsynth": float(100 * res.pre_treatment_share_rmspe),
                 "reference": _PAPER_SHARE_RMSPE_PP})
    for year, paper in _PAPER_TABLE2.items():
        i = years.index(year)
        rows.append({"quantity": f"ATT_gas[{year}] (pp)",
                     "mlsynth": float(100 * res.share_effects[i, k["gas"]]),
                     "reference": paper[0]})
        rows.append({"quantity": f"LRTE_gas|fos[{year}]",
                     "mlsynth": float(
                         res.log_ratio_effects[i, k["gas"], k["fossil"]]),
                     "reference": paper[3]})
    rows.append({"quantity": "placebo p-value",
                 "mlsynth": float(res.placebo.p_value),
                 "reference": _PAPER_PLACEBO["p_value"]})

    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "COMPSC",
            "config": {"geometry": "alr", "baseline": "renewables",
                       "outcomes": _CATEGORIES, "target": "gas"},
        },
        "reference": {
            "impl": "Boussim (2026) published tables (no replication package released)",
            "version": "arXiv:2607.16991v1",
        },
    }
