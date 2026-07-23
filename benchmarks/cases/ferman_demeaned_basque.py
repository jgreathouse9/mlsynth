"""Golden cross-validation: Ferman & Pinto (2021)'s demeaned SC, run live in R.

Ferman and Pinto (2021), *"Synthetic controls with imperfect pretreatment fit"*
(Quantitative Economics 12:1197-1221), study synthetic control when the
pre-treatment fit is imperfect and propose a *demeaned* SC estimator -- donor
weights on the simplex (non-negative, summing to one) together with a free
intercept -- as a bias/variance improvement over difference-in-differences.

mlsynth does not ship a "demeaned SC" estimator by that name, but it does not
need to: the demeaned SC is exactly the ``MSCa`` variant of :class:`~mlsynth.TSSC`
(the Two-Step SC's simplex-plus-intercept model). This case validates the
*estimator*, not TSSC's variant-selection machinery -- it reaches into the
``MSCa`` fit and cross-checks it against the authors' own R code, **executed live
via ``Rscript``**: ``benchmarks/reference/ferman_demeaned_basque/reference.R``
reproduces ``_aux.R :: synth_control_est_demean`` verbatim (a ``quadprog`` QP) and
is run on the same shipped panel each time this case runs. It ``BenchmarkSkipped``s
when ``Rscript`` / ``quadprog`` is unavailable, so a missing R toolchain never
turns the suite red.

The panel is the Basque Country / ETA terrorism study (Abadie & Gardeazabal
2003), ``basedata/basque_data.csv``, treatment 1975. That year is the *identified*
regime -- twenty pre-treatment periods (1955-1974) against sixteen donors, so
``C < n`` and the demeaned-SC weights are unique. There mlsynth's ``MSCa`` QP and
the authors' ``quadprog`` QP agree value-for-value: donor weights to ~1e-4
(Cataluna 0.561, Rioja 0.279, Asturias 0.106, Madrid 0.054), the free intercept
to ~1e-5, and the ATT to ~4e-4 (-0.797 thousand-USD GDP per capita). (At a 1970
cutoff the panel is rank-deficient, ``C > n``; the demeaned-SC weights are then
non-unique and the two solvers legitimately land on different minimisers --
documented on the replication page, not pinned here.)

Provenance: Ferman & Pinto (2021), Quantitative Economics 12:1197-1221;
reference numbers produced by the authors' ``synth_control_est_demean`` run live
via ``Rscript`` (``quadprog``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DATA = os.path.abspath(os.path.join(_ROOT, "basedata", "basque_data.csv"))
_REF_R = os.path.abspath(os.path.join(
    _ROOT, "benchmarks", "reference", "ferman_demeaned_basque", "reference.R"))
_TREATED = "Basque Country (Pais Vasco)"
_DROP = ["Spain (Espana)", "Syntetic Basque Country"]
_TREAT_YEAR = 1975
_MLSYNTH_KW = {"outcome": "gdpcap", "treat": "treat", "unitid": "regionname",
               "time": "year", "display_graphs": False}
_DONORS = {"cataluna": "Cataluna", "rioja": "Rioja (La)",
           "asturias": "Principado De Asturias", "madrid": "Madrid (Comunidad De)"}


def _reference_live() -> dict:
    """Run the authors' demeaned-SC (``synth_control_est_demean``) live via R.

    Skips (``BenchmarkSkipped``) when ``Rscript`` or the ``quadprog`` package is
    unavailable. Returns ``{"att", "intercept", "weights": {region: w}}`` parsed
    from ``reference.R``'s JSON stdout.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (install R + the quadprog package)")
    probe = subprocess.run([rscript, "-e", "suppressMessages(library(quadprog))"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'quadprog' not installed")
    out = subprocess.run([rscript, _REF_R, _DATA], capture_output=True, text=True)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"demeaned-SC reference failed: {out.stderr.strip()[-200:]}")
    try:
        ref = json.loads(out.stdout.strip())
    except json.JSONDecodeError as exc:
        raise BenchmarkSkipped(f"could not parse reference.R output: {exc}") from exc
    return {"att": float(ref["values"]["att"]),
            "intercept": float(ref["values"]["intercept"]),
            "weights": {k: float(v) for k, v in ref["weights"].items()}}


def _msca_fit():
    """Fit TSSC on the Basque panel and return its MSCa (demeaned-SC) variant."""
    from mlsynth import TSSC

    d = pd.read_csv(_DATA)
    d = d[~d.regionname.isin(_DROP)].copy()
    d["treat"] = ((d.regionname == _TREATED) & (d.year >= _TREAT_YEAR)).astype(int)
    d = d[["regionname", "year", "gdpcap", "treat"]].dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TSSC({"df": d, **_MLSYNTH_KW}).fit().variants["MSCa"]


def _paired():
    """Run BOTH sides (live R + mlsynth MSCa) and align their donor weights."""
    ref = _reference_live()                              # skips if R/quadprog absent
    m = _msca_fit()
    ml = pd.Series(dict(m.donor_weights), dtype=float)
    rw = pd.Series(ref["weights"], dtype=float)
    keys = rw.index
    l1 = float((ml.reindex(keys).fillna(0.0) - rw).abs().sum())
    return m, ref, ml, rw, l1


def run() -> dict:
    m, ref, ml, rw, l1 = _paired()
    g = lambda name: float(ml.get(name, 0.0))
    return {
        # mlsynth MSCa reproduces the LIVE R demeaned-SC QP value-for-value
        "weight_l1_vs_r": l1,
        "att_absdiff_vs_r": float(abs(float(m.att) - ref["att"])),
        "intercept_absdiff_vs_r": float(abs(float(m.intercept) - ref["intercept"])),
        "agrees_with_r": float(l1 < 1e-3 and abs(float(m.att) - ref["att"]) < 1e-2),
        # mlsynth's own headline numbers (the identified 1975 solution)
        "cataluna_w": g("Cataluna"),
        "rioja_w": g("Rioja (La)"),
        "att": float(m.att),
        "intercept": float(m.intercept),
    }


def comparison() -> dict:
    """mlsynth ``TSSC`` MSCa (the demeaned-SC variant) vs Ferman & Pinto (2021)'s
    own R ``synth_control_est_demean`` (quadprog QP) run **live** via ``Rscript``,
    weight by weight. Both solve the identical simplex-plus-intercept least-squares
    program on the identified (``C < n``) Basque panel at 1975. Skips when the R
    toolchain is absent."""
    m, ref, ml, rw, l1 = _paired()
    g = lambda name: float(ml.get(name, 0.0))
    rows = [{"quantity": f"{lbl} weight", "mlsynth": round(g(reg), 4),
             "reference": round(ref["weights"][reg], 4)}
            for lbl, reg in [("Cataluna", "Cataluna"), ("Rioja", "Rioja (La)"),
                             ("Asturias", "Principado De Asturias"),
                             ("Madrid", "Madrid (Comunidad De)")]]
    rows.append({"quantity": "free intercept", "mlsynth": round(float(m.intercept), 4),
                 "reference": round(ref["intercept"], 4)})
    rows.append({"quantity": "ATT (GDP per capita)", "mlsynth": round(float(m.att), 4),
                 "reference": round(ref["att"], 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "TSSC", "variant": "MSCa (demeaned SC)",
                         "config": {"treat_year": _TREAT_YEAR,
                                    **{k: v for k, v in _MLSYNTH_KW.items()
                                       if k != "display_graphs"}}},
        "reference": {
            "impl": "authors' _aux.R :: synth_control_est_demean (quadprog QP, live via Rscript)",
            "version": "Ferman & Pinto (2021), Quantitative Economics 12:1197-1221",
        },
    }


# Cross-validation (scenario: full repo, live R). mlsynth's TSSC MSCa variant is
# Ferman & Pinto (2021)'s demeaned SC; on the identified Basque/ETA panel (1975,
# C<n) it reproduces their R quadprog QP -- run live via Rscript -- value-for-value:
# donor-weight L1 ~1e-4, ATT within ~4e-4, intercept within ~1e-5. Deterministic
# QP on both sides => solver-tight tolerances. Skips if Rscript/quadprog absent.
EXPECTED = {
    "weight_l1_vs_r": (0.0, 5e-3),
    "att_absdiff_vs_r": (0.0, 1e-2),
    "intercept_absdiff_vs_r": (0.0, 5e-3),
    "agrees_with_r": (1.0, 0.0),
    "cataluna_w": (0.56050, 5e-3),
    "rioja_w": (0.27918, 5e-3),
    "att": (-0.7966, 2e-2),
    "intercept": (0.59223, 5e-3),
}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2))
