"""Brazil pneumococcal vaccine: standard SC vs proximal, cross-validated vs live R.

Cross-validation against Qiu, Shi & Tchetgen Tchetgen, "Doubly robust proximal
synthetic control" -- their **Brazil 2010 PCV10** empirical analysis -- run to
contrast the *standard* synthetic control with the proximal (negative-control)
estimators on the identical age-group-9 panel.

The point of the panel is the contrast:

* Standard SC (Abadie), restricted to the three donor causes the authors use,
  tracks pre-period pneumonia but attributes essentially no effect to the
  vaccine (ATT ~ +0.4k hospitalisations, near null / slightly positive): a
  donor pool of other diseases cannot reconstruct the pneumonia counterfactual
  once the shared seasonal/reporting confounding is what the two sides have in
  common.
* The proximal estimators, using the same non-pneumonia causes as *negative
  controls*, recover a large reduction -- the outcome bridge ``h`` ~ -3646 and
  the doubly-robust ``DR`` ~ -2753 hospitalisations -- matching the authors'
  published Brazil numbers. That is the whole proximal thesis: the contamination
  that sinks standard SC is exactly what identifies the effect proximally.

The proximal cells are cross-validated cell-by-cell against live R in the
sibling case :mod:`benchmarks.cases.dr_proximal_brazil`; here they anchor the
standard-vs-proximal contrast and are re-checked against the *same* R script
(the outcome bridge ``h`` and the single-instrument ``DR[A]``).

Provenance / scope
------------------
* Data: ``basedata/pnas_brazil_age9.csv`` -- the ``InterventionEvaluatR``
  ``pnas_brazil`` dataset (Bruhn et al. PNAS 2017), age group 9, 2010-2011
  transition window dropped; vendored once (avoids the INLA dependency).
* mlsynth: standard SC is ``VanillaSC`` on the treated series plus the three
  donor causes; the proximal cells are ``PROXIMAL(methods=["DR-OID"])`` -- the
  over-identified doubly-robust config the authors use, whose outcome-bridge
  component reproduces their just-identified ``PI.h``.
* Reference: the authors' ``analysis.Rmd`` (commit ``3bcb5ec``), reproduced in
  ``benchmarks/R/dr_proximal_brazil.R``. R does not run in CI; ``comparison()``
  runs it live and skips when the toolchain is absent. R's ``Synth`` is not
  required here -- the standard-SC contrast is reported from mlsynth; only the
  proximal ``h``/``DR`` cells are cross-checked against R.

Numbers below are in hospitalisation units (phi * Y.scale), Y.scale = 18201.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from pathlib import Path

import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "pnas_brazil_age9.csv"
_RSCRIPT = Path(__file__).resolve().parents[1] / "R" / "dr_proximal_brazil.R"

_DONORS = ["cJ20_J22", "E00_99", "E40_46"]
_YSCALE = 18201.0

# Live-R converged reference (benchmarks/R/dr_proximal_brazil.R, reltol=1e-13),
# hospitalisation units. Both are well-conditioned and stable in R.
_R_BRIDGE_H = -3645.80
_R_DR_A = -2752.53


def _panel():
    """Long, max-scaled Brazil panel; treated = pneumonia (J12_18), t>84 post."""
    d = pd.read_csv(_DATA)
    causes = [c for c in d.columns if c not in ("date", "t")]
    for c in causes:
        d[c] = d[c] / d[c].max()
    long = d.melt(id_vars=["date", "t"], value_vars=causes,
                  var_name="cause", value_name="hosp")
    long["treat"] = ((long.cause == "J12_18") & (long.t > 84)).astype(int)
    pool = [c for c in causes if c not in ["J12_18"] + _DONORS]
    return long, pool


def _standard_sc(long):
    """Abadie standard SC on the three restricted donor causes."""
    from mlsynth import VanillaSC
    sub = long[long.cause.isin(["J12_18"] + _DONORS)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": sub, "outcome": "hosp", "treat": "treat", "unitid": "cause",
            "time": "t", "display_graphs": False,
        }).fit()


def _dr_oid(long, pool):
    """Over-identified DR proximal (single treatment instrument, subset A)."""
    from mlsynth import PROXIMAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PROXIMAL({
            "df": long, "outcome": "hosp", "treat": "treat", "unitid": "cause",
            "time": "t", "methods": ["DR-OID"], "donors": _DONORS,
            "outcome_instruments": pool,
            "treatment_instruments": ["A10_B99_nopneumo"],
            "display_graphs": False,
        }).fit().dr_oid


def run() -> dict:
    long, pool = _panel()
    sc = _standard_sc(long)
    dr = _dr_oid(long, pool)

    sc_att = sc.att * _YSCALE
    bridge_h = dr.metadata["outcome_bridge_att"] * _YSCALE   # h imputes counterfactual
    dr_att = dr.att * _YSCALE

    return {
        # standard SC ("for fun"): near-null, and its counterfactual DOES exist
        "standard_sc_att": sc_att,
        "standard_sc_pre_rmse": float(sc.pre_rmse),
        # proximal cells (hospitalisation units), mlsynth's robust optimum
        "bridge_h": bridge_h,
        "dr_A": dr_att,
        # cross-validation vs live-R converged values
        "h_diff_vs_R": abs(bridge_h - _R_BRIDGE_H),
        "drA_diff_vs_R": abs(dr_att - _R_DR_A),
        # the contrast: standard SC misses the effect proximal recovers.
        # |standard SC| is an order of magnitude below the proximal reduction.
        "proximal_over_standard": abs(dr_att) / abs(sc_att),
        # signs: standard SC ~ null/positive; the vaccine reduces pneumonia
        "standard_sc_is_near_null": 1.0 if abs(sc_att) < 1000.0 else 0.0,
        "proximal_is_negative": 1.0 if (bridge_h < 0 and dr_att < 0) else 0.0,
    }


def _reference_live() -> dict:
    """Run the authors' R analysis live; parse the converged DR/h values."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (install R + gmm, dplyr, tidyr)")
    probe = subprocess.run(
        [rscript, "-e", "suppressMessages({library(gmm);library(dplyr);library(tidyr)})"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R packages 'gmm'/'dplyr'/'tidyr' not installed")
    out = subprocess.run([rscript, str(_RSCRIPT), str(_DATA)],
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"R reference failed: {out.stderr.strip()[-200:]}")
    vals = {}
    for key in ["outcome_bridge_h", "DR_A"]:
        m = re.search(rf"^{re.escape(key)}:\s*([-\d.eE]+)", out.stdout, re.M)
        if m:
            vals[key] = float(m.group(1))
    if "outcome_bridge_h" not in vals or "DR_A" not in vals:
        raise BenchmarkSkipped("could not parse R reference output")
    return vals


def comparison() -> dict:
    """mlsynth standard-vs-proximal panorama; proximal cells vs the authors' live R."""
    ref = _reference_live()                       # skips if R/toolchain absent
    long, pool = _panel()
    sc = _standard_sc(long)
    dr = _dr_oid(long, pool)
    rows = [
        {"quantity": "standard_sc (VanillaSC)",
         "mlsynth": round(sc.att * _YSCALE, 3), "reference": None},
        {"quantity": "outcome_bridge_h",
         "mlsynth": round(dr.metadata["outcome_bridge_att"] * _YSCALE, 3),
         "reference": round(ref.get("outcome_bridge_h", float("nan")), 3)},
        {"quantity": "DR[A10_B99_nopneumo]",
         "mlsynth": round(dr.att * _YSCALE, 3),
         "reference": round(ref.get("DR_A", float("nan")), 3)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"standard": "VanillaSC", "proximal": "PROXIMAL(DR-OID)"},
        "reference": {"impl": "R gmm (authors' analysis.Rmd, commit 3bcb5ec, reltol=1e-13)",
                      "script": "benchmarks/R/dr_proximal_brazil.R",
                      "note": "standard SC reported from mlsynth (R's Synth not required)"},
    }


# On the Brazil PCV10 panel, mlsynth's standard SC (restricted donors) lands
# near null (+0.4k), while the proximal negative-control estimators recover the
# vaccine's pneumonia reduction: the outcome bridge h = -3645.8 matches live R
# to the digit and DR[A] = -2753.5 matches to <0.05%. The contrast is the
# benchmark: proximal effect magnitude is ~7x the standard-SC artefact.
EXPECTED = {
    "standard_sc_att": (409.122, 8.0),
    "standard_sc_pre_rmse": (0.169, 0.03),
    "bridge_h": (-3645.798, 3.0),
    "dr_A": (-2753.524, 4.0),
    "h_diff_vs_R": (0.0, 3.0),
    "drA_diff_vs_R": (1.0, 3.0),
    "proximal_over_standard": (6.73, 1.5),
    "standard_sc_is_near_null": (1.0, 0.0),
    "proximal_is_negative": (1.0, 0.0),
}
