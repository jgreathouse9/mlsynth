"""GEOLIFT cross-check vs augsynth: the GeoLift_Walkthrough fit.

Cross-validation of mlsynth's fixed-effect ridge ASCM against **augsynth**
(``ebenmichael/augsynth``, which GeoLift wraps) on the chicago+portland panel --
the gold-standard the replication contract asks for. ``geolift_walkthrough`` pins
the public ``GEOLIFT`` API against GeoLift's *published* walkthrough numbers; this
case goes further and matches augsynth's CV-selected ridge penalty, donor weights,
and post-period ATT. mlsynth reproduces augsynth to floating-point on lambda
(rel-diff ~1e-11), to ~7 decimals on every weight, and to 4 sig figs on the ATT.

The gate compares against a **captured** augsynth reference, not a fresh live run.
augsynth's ridge ASCM is a live R build: its CV-selected lambda is sensitive to
the exact CRAN dependency build (BLAS / osqp / matrix-routine versions), so a live
re-run's lambda can drift ~1% between environments even at a pinned augsynth SHA,
which would spuriously fail a stable mlsynth. mlsynth's own lambda is a
deterministic closed form (the ~1e-11 agreement is a formula match, not a knife-
edge CV argmin), so pinning the reference to the capture makes this a durable
regression test of *mlsynth* against a provenance-tracked augsynth run.

Reference: augsynth ``0.2.0 @ 7a90ea4`` on this panel, captured 2026-06-24 (see
``benchmarks/reference/geolift_augsynth_ref/comparison.csv``). To re-validate
against a fresh live augsynth run -- e.g. before re-pinning -- set
``GEOLIFT_AUGSYNTH_LIVE=1`` (needs ``benchmarks/R/install_augsynth.sh``); to
refresh the capture, re-run live and update ``_FROZEN_REFERENCE``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.geolift_helpers.marketselect.helpers.fit import fit_augsynth_once
from mlsynth.utils.geolift_helpers.marketselect.helpers.shaping import (
    aggregate_treated, donor_matrix,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "geolift_test_data.csv")
_RSCRIPT_REF = os.path.join(_ROOT, "benchmarks", "R", "augsynth_geolift.R")
_TREATED = frozenset({"chicago", "portland"})
_PRE = 90

# Captured augsynth reference: augsynth 0.2.0 @ 7a90ea4 on the chicago+portland
# GeoLift panel, run 2026-06-24. Source of record:
# benchmarks/reference/geolift_augsynth_ref/comparison.csv (the "reference"
# column). The gate compares mlsynth against this frozen capture so a live
# augsynth re-run's dependency-build-sensitive lambda cannot spuriously red the
# case; set GEOLIFT_AUGSYNTH_LIVE=1 to re-validate against live augsynth instead.
_FROZEN_REFERENCE = {
    "lambda": 1673101687.1,
    "att": 156.805406,
    "weights": {
        "cincinnati": 0.227306, "miami": 0.202933, "baton rouge": 0.133658,
        "minneapolis": 0.090122, "dallas": 0.074116, "nashville": 0.068726,
        "honolulu": 0.067404, "austin": 0.046682, "san diego": 0.045221,
        "reno": 0.030821, "san antonio": 0.0056, "houston": 0.004812,
        "new york": 0.004793,
    },
}


def _reference() -> dict:
    """The augsynth reference the gate compares against: the deterministic frozen
    capture by default, or a fresh live augsynth run when ``GEOLIFT_AUGSYNTH_LIVE=1``
    (which propagates ``BenchmarkSkipped`` if Rscript/augsynth is absent).
    """
    if os.environ.get("GEOLIFT_AUGSYNTH_LIVE") == "1":
        return _augsynth_reference()
    return _FROZEN_REFERENCE


def _augsynth_reference() -> dict:
    """Run augsynth via Rscript and parse its lambda / ATT / weights dump."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (run benchmarks/R/install_augsynth.sh)")
    # Confirm augsynth itself is installed before the (slower) fit.
    probe = subprocess.run(
        [rscript, "-e", "suppressMessages(library(augsynth))"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'augsynth' not installed")
    out = subprocess.run([rscript, _RSCRIPT_REF, _DATA],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"augsynth reference failed: {out.stderr.strip()[-200:]}")

    lam = att = pval = None
    weights: dict = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "LAMBDA":
            lam = float(parts[1])
        elif parts[0] == "ATT":
            att = float(parts[1])
        elif parts[0] == "PVAL":
            pval = float(parts[1])
        elif parts[0] == "W":                       # W <name (may contain spaces)> <weight>
            weights[" ".join(parts[1:-1])] = float(parts[-1])
    if lam is None or att is None:
        raise BenchmarkSkipped("could not parse augsynth reference output")
    return {"lambda": lam, "att": att, "pval": pval, "weights": weights}


def run() -> dict:
    ref = _reference()

    Ywide = geoex_dataprep(pd.read_csv(_DATA), "location", "date", "Y")["Ywide"]
    treated = aggregate_treated(Ywide, _TREATED, how="mean").to_numpy()
    donors = donor_matrix(Ywide, _TREATED)
    Y0 = donors.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_augsynth_once(
            treated[:_PRE], Y0[:_PRE], augment="ridge", fixed_effects=True,
            donor_names=[str(c) for c in donors.columns])
    gap = treated - fit.predict(Y0)
    att = float(np.mean(gap[_PRE:]))
    w = {str(c): float(wj) for c, wj in zip(donors.columns, fit.weights)}

    # Largest absolute weight gap over augsynth's non-zero donors.
    weight_max_abs_diff = max(abs(w.get(k, 0.0) - v) for k, v in ref["weights"].items())

    return {
        "lambda_rel_diff": abs(fit.lambda_ - ref["lambda"]) / ref["lambda"],
        "att_abs_diff": abs(att - ref["att"]),
        "weight_max_abs_diff": weight_max_abs_diff,
        "n_weights_match": float(len(w_nz := [v for v in w.values() if abs(v) > 1e-3])
                                 == len(ref["weights"])),
    }


def comparison() -> dict:
    """mlsynth's ridge ASCM vs live augsynth, quantity by quantity.

    Runs both sides live -- augsynth via Rscript and mlsynth's fixed-effect ridge
    ASCM on the identical chicago+portland panel -- and pairs the CV-selected ridge
    penalty (lambda), the post-period ATT, and each donor weight. Propagates the
    ``BenchmarkSkipped`` from ``_augsynth_reference`` when Rscript/augsynth is absent.
    """
    ref = _augsynth_reference()                     # skips if Rscript/augsynth absent

    Ywide = geoex_dataprep(pd.read_csv(_DATA), "location", "date", "Y")["Ywide"]
    treated = aggregate_treated(Ywide, _TREATED, how="mean").to_numpy()
    donors = donor_matrix(Ywide, _TREATED)
    Y0 = donors.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_augsynth_once(
            treated[:_PRE], Y0[:_PRE], augment="ridge", fixed_effects=True,
            donor_names=[str(c) for c in donors.columns])
    gap = treated - fit.predict(Y0)
    att = float(np.mean(gap[_PRE:]))
    w = {str(c): float(wj) for c, wj in zip(donors.columns, fit.weights)}

    rows = [{"quantity": "lambda", "mlsynth": round(float(fit.lambda_), 6),
             "reference": round(float(ref["lambda"]), 6)},
            {"quantity": "ATT", "mlsynth": round(att, 6),
             "reference": round(float(ref["att"]), 6)}]
    for k in sorted(ref["weights"], key=lambda s: -abs(ref["weights"][s])):
        rows.append({"quantity": f"weight[{k}]", "mlsynth": round(w.get(k, 0.0), 6),
                     "reference": round(float(ref["weights"][k]), 6)})

    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "GEOLIFT",
            "config": {"outcome": "Y", "treat": "treat", "unitid": "location",
                       "time": "date", "treated_units": sorted(_TREATED),
                       "augment": "ridge", "fixed_effects": True, "pre_periods": _PRE},
        },
        "reference": {"impl": "R package augsynth (via Rscript)",
                      "version": "augsynth (R, live)"},
    }


# mlsynth reproduces live augsynth to floating point. The tolerances pin a genuine
# match (not the looser "published walkthrough" gap): lambda within 1e-4 relative,
# every weight within 5e-4, the ATT within half a unit (~0.3% of 156.8).
EXPECTED = {
    "lambda_rel_diff": (0.0, 1e-4),
    "att_abs_diff": (0.0, 0.5),
    "weight_max_abs_diff": (0.0, 5e-4),
    "n_weights_match": (1.0, 0.5),
}
