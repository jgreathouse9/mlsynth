"""Cross-validation benchmark: MicroSynth panel method vs the R ``microsynth``.

MicroSynth's ``weight_method="panel"`` is mlsynth's port of the panel-data
weighting in the R ``microsynth`` package (Robbins, Saunders & Kilmer 2017,
JASA; Robbins & Davenport 2021, JSS v97i02). This case reproduces the package's
canonical **Seattle Drug Market Intervention** example and cross-checks
mlsynth's per-period total treatment effects, weight totals, and balance against
the R package's output.

What the panel method actually does (read from ``microsynth/R/weights.r``)
------------------------------------------------------------------------
When ``match.out`` is supplied, ``microsynth::my.qp`` (a ``LowRankQP`` solve)
chooses control weights by a non-negative QP: **exactly** balance the
covariate totals (an intercept makes the weights sum to the treated count) and
**least-squares**-fit the lagged outcomes, with ``w >= 0``. This objective is
rank-deficient over a large control pool, so the counterfactual is *not*
identified by the constraints alone (the LP range of the period-13 effect on
this data is roughly ``[-392, +153]``); ``LowRankQP`` merely returns its
interior-point iterate. mlsynth adds a strictly-convex ridge that selects the
unique minimum-norm / maximum-ESS optimum -- which coincides with R's
interior-point solution to 3-4 significant figures, giving a genuine
cross-validation rather than a comparison of solver artifacts.

Provenance / scope
------------------
* Data: ``basedata/seattledmi.parquet`` -- the R ``microsynth`` package's
  ``seattledmi`` dataset (``data(seattledmi)``), trimmed to the columns this
  case uses (ID/time/Intervention + the four Table 2 outcomes + the 9 census
  covariates). Full
  panel: 9642 blocks x 16 periods, 39 treated (``Intervention`` turns on at
  ``time >= 13``).
* Reference: R ``microsynth`` 2.0.51 run via ``benchmarks/R/microsynth_seattle.R``
  with ``match.out=c("any_crime")``, ``match.covar`` = the 9 census vars,
  ``start.pre=1, end.pre=12, end.post=16`` -- a single-outcome configuration so
  the constraint set matches mlsynth's one-outcome MicroSynth exactly. R's
  per-period ``Plot.Stats$Difference`` (Treatment - synthetic Control) and
  weight totals are baked in below as the cross-validation target (R does not
  install in CI; the script documents regeneration).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "seattledmi.parquet"
_RSCRIPT_REF = Path(__file__).resolve().parents[1] / "R" / "microsynth_seattle.R"

# --- R microsynth 2.0.51 reference (single-outcome any_crime) ----------------
# Plot.Stats$Difference[any_crime, 13:16] from microsynth_seattle.R.
_R_EFFECTS = np.array([-33.059, -74.351, -45.450, -64.892])
_R_ATT = float(_R_EFFECTS.mean())                    # -54.438
_R_WEIGHT_SUM = 39.0                                 # weights sum to treated count

# --- JSS Table 2 top panel: multi-outcome joint match (summary(sea1)) ---------
# Cumulative Pct.Chng over post window 13:16, from microsynth_table2.R.
_TABLE2_PCT = {                                       # outcome -> Pct.Chng (%)
    "i_felony": -32.6, "i_misdemea": -37.3, "i_drugs": -15.8, "any_crime": -20.1,
}
_MATCH_OUT = ["i_felony", "i_misdemea", "i_drugs", "any_crime"]


def run() -> dict:
    from mlsynth import MicroSynth

    df = pd.read_parquet(_DATA)
    cov = ["TotalPop", "BLACK", "HISPANIC", "Males_1521", "HOUSEHOLDS",
           "FAMILYHOUS", "FEMALE_HOU", "RENTER_HOU", "VACANT_HOU"]

    cfg = {
        "df": df, "outcome": "any_crime", "treat": "Intervention",
        "unitid": "ID", "time": "time", "covariates": cov,
        "outcome_lag_periods": list(range(1, 13)),        # full pre-window 1..12
        "weight_method": "panel", "display_graphs": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MicroSynth({**cfg, "run_inference": False}).fit()
        # placebo-permutation inference (microsynth perm); modest n_perm keeps
        # the case fast -- the effect is ~7 SDs out, so it beats every placebo
        # and the p-value sits at the 1/(1+n_perm) floor.
        res_inf = MicroSynth({
            **cfg, "run_inference": True, "n_permutations": 24,
            "permutation_test": "lower", "seed": 99,
        }).fit()

    # --- JSS Table 2 top panel: 4-outcome joint match (one shared synthetic
    # control), cumulative Pct.Chng per outcome. ---
    pct_diffs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for o in _MATCH_OUT:
            r = MicroSynth({
                "df": df, "outcome": o, "treat": "Intervention", "unitid": "ID",
                "time": "time", "covariates": cov,
                "outcome_lag_periods": list(range(1, 13)),
                "match_outcomes": _MATCH_OUT, "weight_method": "panel",
                "run_inference": False, "display_graphs": False,
            }).fit()
            trt = float(np.sum(r.inputs.Y_T))
            con = float(np.sum(r.counterfactual))
            pct = 100.0 * (trt - con) / con
            pct_diffs.append(abs(pct - _TABLE2_PCT[o]))

    eff = np.asarray(res.gap_trajectory, dtype=float)     # per-period totals 13..16
    return {
        "table2_max_pct_diff_vs_R": float(max(pct_diffs)),
        # cross-validation vs R microsynth: per-period effects and ATT match
        "max_abs_effect_diff_vs_R": float(np.max(np.abs(eff - _R_EFFECTS))),
        "att_diff_vs_R": float(abs(res.att - _R_ATT)),
        # identified quantities both packages pin: exact balance + count
        "weight_sum_minus_treated_count": float(abs(sum(res.design.w) - _R_WEIGHT_SUM)),
        "max_abs_smd_after": float(np.max(np.abs(res.design.smd_after))),
        # mlsynth-pinned descriptors (deterministic on this panel)
        "att": float(res.att),
        "ess": float(res.design.ess),
        # placebo-permutation inference: significant crime reduction (one-sided)
        "perm_pvalue_lower": float(res_inf.inference.p_value),
    }


def _reference_live() -> dict:
    """Run R ``microsynth`` live; parse the single-outcome any_crime reference.

    Skips (``BenchmarkSkipped``) when ``Rscript`` or the ``microsynth`` package is
    unavailable, so a missing R toolchain never turns the suite red. Returns the
    per-period effects 13..16, the ATT (their mean), and the control weight sum.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (install R + the microsynth package)")
    probe = subprocess.run([rscript, "-e", "suppressMessages(library(microsynth))"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'microsynth' not installed")

    out = subprocess.run([rscript, str(_RSCRIPT_REF)], capture_output=True, text=True)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"microsynth reference failed: {out.stderr.strip()[-200:]}")

    m = re.search(r"post effects 13:16:\s*\n\s*\[1\]\s+([-\d.eE\s]+?)\n", out.stdout)
    att_m = re.search(r"ATT \(mean of 13:16\):\s*([-\d.eE]+)", out.stdout)
    wsum_m = re.search(r"control weight sum \(== treated count\):\s*([-\d.eE]+)", out.stdout)
    if not (m and att_m and wsum_m):
        raise BenchmarkSkipped("could not parse microsynth reference output")
    effects = np.array([float(x) for x in m.group(1).split()])
    return {"effects": effects, "att": float(att_m.group(1)),
            "weight_sum": float(wsum_m.group(1))}


def comparison() -> dict:
    """mlsynth MicroSynth vs R ``microsynth`` (run live), quantity by quantity.

    Per-period total effects 13..16, the ATT, and the control weight sum, paired
    for a side-by-side export. The reference side runs the R package live and
    ``BenchmarkSkipped``s when Rscript/microsynth is absent.
    """
    ref = _reference_live()                              # skips if R/microsynth absent

    from mlsynth import MicroSynth

    df = pd.read_parquet(_DATA)
    cov = ["TotalPop", "BLACK", "HISPANIC", "Males_1521", "HOUSEHOLDS",
           "FAMILYHOUS", "FEMALE_HOU", "RENTER_HOU", "VACANT_HOU"]
    cfg = {
        "outcome": "any_crime", "treat": "Intervention", "unitid": "ID",
        "time": "time", "covariates": cov,
        "outcome_lag_periods": list(range(1, 13)),
        "weight_method": "panel", "run_inference": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MicroSynth({**cfg, "df": df, "display_graphs": False}).fit()

    eff = np.asarray(res.gap_trajectory, dtype=float)     # per-period totals 13..16
    rows = [{"quantity": f"effect[period {13 + i}]",
             "mlsynth": round(float(eff[i]), 6),
             "reference": round(float(ref["effects"][i]), 6)}
            for i in range(len(eff))]
    rows.append({"quantity": "ATT", "mlsynth": round(float(res.att), 6),
                 "reference": round(ref["att"], 6)})
    rows.append({"quantity": "control_weight_sum",
                 "mlsynth": round(float(sum(res.design.w)), 6),
                 "reference": round(ref["weight_sum"], 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MicroSynth", "config": cfg},
        "reference": {"impl": "R package microsynth (via Rscript)",
                      "version": "microsynth (R, live)"},
    }


# The regularized panel QP reproduces R microsynth's LowRankQP solution on the
# Seattle DMI panel: per-period effects agree to ~0.1 crimes, the ATT to ~0.01,
# weights sum to the treated count (39), and covariate + lagged-outcome balance
# is exact (max |SMD| ~1e-10). ESS ~378 reflects the maximum-ESS (most diffuse)
# optimum the ridge selects.
EXPECTED = {
    "max_abs_effect_diff_vs_R": (0.10, 0.30),
    "att_diff_vs_R": (0.012, 0.10),
    "weight_sum_minus_treated_count": (0.0, 1e-3),
    "max_abs_smd_after": (0.0, 1e-6),
    "att": (-54.43, 0.5),
    "ess": (377.7, 25.0),
    "perm_pvalue_lower": (0.04, 0.001),       # 1/(1+24): beats every placebo
    # JSS Table 2 top panel: joint-match cumulative Pct.Chng matches to <0.5pp
    "table2_max_pct_diff_vs_R": (0.06, 0.5),
}
