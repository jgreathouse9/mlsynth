"""Cross-validation: over-identified DR proximal SC vs live R, Brazil vaccine.

Path: cross-validation against the authors' own R code (Qiu, Shi & Tchetgen
Tchetgen, "Doubly robust proximal synthetic control"), reproducing their
**Brazil 2010 pneumococcal-vaccine** empirical analysis cell-by-cell.

The estimator is mlsynth's ``PROXIMAL`` with ``methods=["DR-OID"]`` -- the
*over-identified* doubly-robust configuration the authors use empirically: the
outcome bridge ``h(W)`` is instrumented by the full pool of negative-control
disease causes, while the treatment bridge ``q(Z)`` uses only a small selected
subset (``#instruments != #donors``). The outcome is pneumonia hospitalisations
(``J12_18``); donors are three respiratory/metabolic causes; instruments are the
other ICD chapters -- a genuine proximal / negative-control design.

Why Brazil and not Kansas. The same paper's Kansas tax-cut analysis runs the
identical GMM, but there the treatment bridge ``q = exp(Z beta)`` separates on a
clean pre/post split: a flat valley with no stable optimum, where the *published*
table merely records where R's ``optim(BFGS)`` happened to stop (tightening R's
tolerance moves DR[Iowa] from -0.077 to -0.107). Brazil's negative-control causes
share seasonality/reporting confounders, so the ``q``-block is well-conditioned:
mlsynth's decoupled solver and live R agree to <0.05% on the well-identified
cells, with ``converged=True`` and a single basin. See the replication docs.

Provenance / scope
------------------
* Data: ``basedata/pnas_brazil_age9.csv`` -- the ``InterventionEvaluatR``
  ``pnas_brazil`` dataset (Bruhn et al. PNAS 2017), age group 9, with the
  2010-2011 transition window dropped, vendored once (avoids the INLA dependency).
* Reference: the authors' ``analysis.Rmd`` at commit ``3bcb5ec``, reproduced
  verbatim in ``benchmarks/R/dr_proximal_brazil.R`` with two documented edits
  (read the vendored CSV; tighten ``optim`` ``reltol`` to the genuine optimum).
  R does not run in CI; ``comparison()`` runs it live and skips when absent.

Numbers below are in hospitalisation units (phi * Y.scale), Y.scale = 18201.
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

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "pnas_brazil_age9.csv"
_RSCRIPT = Path(__file__).resolve().parents[1] / "R" / "dr_proximal_brazil.R"

_DONORS = ["cJ20_J22", "E00_99", "E40_46"]
_SUBSETS = {
    "A": ["A10_B99_nopneumo"],
    "AD": ["A10_B99_nopneumo", "D50_89"],
    "APD": ["A10_B99_nopneumo", "P05_07", "D50_89"],
}
# Live-R converged reference (benchmarks/R/dr_proximal_brazil.R, reltol=1e-13),
# hospitalisation units. h and DR[A] are well-conditioned and stable; the
# multi-instrument cells are mlsynth-robust but mildly path-sensitive in R.
_R_BRIDGE_H = -3645.80
_R_DR_A = -2752.43


def _panel():
    d = pd.read_csv(_DATA)
    causes = [c for c in d.columns if c not in ("date", "t")]
    yscale = float(d["J12_18"].max())
    for c in causes:
        d[c] = d[c] / d[c].max()
    long = d.melt(id_vars=["date", "t"], value_vars=causes,
                  var_name="cause", value_name="hosp")
    long["treat"] = ((long.cause == "J12_18") & (long.t > 84)).astype(int)
    pool = [c for c in causes if c not in ["J12_18"] + _DONORS]
    return long, pool, yscale


def _fit(long, pool, treatment_instruments):
    from mlsynth import PROXIMAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PROXIMAL({
            "df": long, "outcome": "hosp", "treat": "treat", "unitid": "cause",
            "time": "t", "methods": ["DR-OID"], "donors": _DONORS,
            "outcome_instruments": pool,
            "treatment_instruments": treatment_instruments,
            "display_graphs": False,
        }).fit().dr_oid


def run() -> dict:
    long, pool, ys = _panel()
    fits = {nm: _fit(long, pool, ti) for nm, ti in _SUBSETS.items()}

    att = {nm: f.att * ys for nm, f in fits.items()}
    bridge_h = fits["A"].metadata["outcome_bridge_att"] * ys   # h imputes alpha
    all_converged = all(f.metadata["converged"] and f.metadata["n_basins"] == 1
                        for f in fits.values())

    return {
        # converged ATTs (hospitalisation units); mlsynth's robust optimum
        "bridge_h": bridge_h,
        "dr_A": att["A"],
        "dr_AD": att["AD"],
        "dr_APD": att["APD"],
        # cross-validation vs live-R converged values on the well-identified cells
        "h_diff_vs_R": abs(bridge_h - _R_BRIDGE_H),
        "drA_diff_vs_R": abs(att["A"] - _R_DR_A),
        # every DR-OID cell pins a single well-conditioned basin
        "all_converged": 1.0 if all_converged else 0.0,
        # the vaccine reduces pneumonia hospitalisations (negative ATT)
        "dr_A_is_negative": 1.0 if att["A"] < 0 else 0.0,
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
    for key in ["outcome_bridge_h", "DR_A", "DR_AD", "DR_APD"]:
        m = re.search(rf"^{re.escape(key)}:\s*([-\d.eE]+)", out.stdout, re.M)
        if m:
            vals[key] = float(m.group(1))
    if "outcome_bridge_h" not in vals or "DR_A" not in vals:
        raise BenchmarkSkipped("could not parse R reference output")
    return vals


def comparison() -> dict:
    """mlsynth PROXIMAL(DR-OID) vs the authors' live R, cell by cell."""
    ref = _reference_live()                       # skips if R/toolchain absent
    long, pool, ys = _panel()
    fits = {nm: _fit(long, pool, ti) for nm, ti in _SUBSETS.items()}
    rows = [{"quantity": "outcome_bridge_h",
             "mlsynth": round(fits["A"].metadata["outcome_bridge_att"] * ys, 3),
             "reference": round(ref.get("outcome_bridge_h", float("nan")), 3)}]
    for nm in ("A", "AD", "APD"):
        rows.append({"quantity": f"DR[{'+'.join(_SUBSETS[nm])}]",
                     "mlsynth": round(fits[nm].att * ys, 3),
                     "reference": round(ref.get(f"DR_{nm}", float("nan")), 3)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PROXIMAL", "method": "DR-OID"},
        "reference": {"impl": "R gmm (authors' analysis.Rmd, commit 3bcb5ec, reltol=1e-13)",
                      "script": "benchmarks/R/dr_proximal_brazil.R"},
    }


# The over-identified DR proximal estimator reproduces the authors' Brazil
# vaccine analysis: the outcome bridge and single-instrument DR match live R to
# <0.05% (h exact), every cell converges to a single basin, and the vaccine's
# effect on pneumonia hospitalisations is a clear reduction (~2750-3650/Y.scale).
# Multi-instrument cells are pinned by mlsynth's robust solver (R is mildly
# path-sensitive there); tolerances absorb that and the HAC-kernel difference.
EXPECTED = {
    "bridge_h": (-3645.8, 3.0),
    "dr_A": (-2753.5, 4.0),
    "dr_AD": (-3613.5, 6.0),
    "dr_APD": (-3611.8, 12.0),
    "h_diff_vs_R": (0.0, 3.0),
    "drA_diff_vs_R": (1.1, 3.0),
    "all_converged": (1.0, 0.0),
    "dr_A_is_negative": (1.0, 0.0),
}
