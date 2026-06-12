"""GEOLIFT live cross-check vs augsynth: the GeoLift_Walkthrough fit.

Cross-validation against the *running* reference. ``geolift_walkthrough`` pins the
public ``GEOLIFT`` API against GeoLift's *published* walkthrough numbers; this case
goes further and checks mlsynth's fixed-effect ridge ASCM against **live augsynth**
(``ebenmichael/augsynth``, which GeoLift wraps) on the identical chicago+portland
panel -- the gold-standard cross-validation the replication contract asks for.

It shells out to ``benchmarks/R/augsynth_geolift.R`` (install the reference once
with ``benchmarks/R/install_augsynth.sh``) and compares the CV-selected ridge
penalty, the donor weights, and the post-period ATT. mlsynth matches augsynth to
floating-point on lambda (rel-diff ~1e-11), to ~7 decimals on every weight, and
to 4 sig figs on the ATT -- so the gaps are pinned near zero, not at the looser
"published walkthrough" tolerances. (The vignette's printed ATT 155.556 is an
older augsynth release; today's augsynth returns 156.8, which mlsynth reproduces.)

The reference is **commit-pinned** -- the install script freezes augsynth (and
every source-compiled dep) to a SHA -- so this is a stable timestamp, not a moving
tip. Numbers below are augsynth ``0.2.0 @ 7a90ea4`` (frozen 2026-06-12); refresh
by re-pinning ``install_augsynth.sh`` and updating ``EXPECTED``.

Skips itself (``BenchmarkSkipped``) when ``Rscript`` or ``augsynth`` is absent, so
it is a no-op in CI and runs only where the reference is installed.
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
    ref = _augsynth_reference()

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


# mlsynth reproduces live augsynth to floating point. The tolerances pin a genuine
# match (not the looser "published walkthrough" gap): lambda within 1e-4 relative,
# every weight within 5e-4, the ATT within half a unit (~0.3% of 156.8).
EXPECTED = {
    "lambda_rel_diff": (0.0, 1e-4),
    "att_abs_diff": (0.0, 0.5),
    "weight_max_abs_diff": (0.0, 5e-4),
    "n_weights_match": (1.0, 0.5),
}
