"""Cross-validation: VanillaSC outcome-only vs the original R ``Synth`` solver.

The original Abadie-Diamond-Hainmueller estimator ships as the CRAN R package
``Synth`` -- the same nested predictor/donor optimization that MSCMT (Becker &
Klossner 2018) and Malo et al. (2024) show can stop short of the global optimum.
This case runs that solver directly (no Python reimplementation) on the
California Proposition 99 tobacco panel under outcome-only matching: the pre-1989
``cigsale`` path serves as both the predictors ``X`` and the dependent fit
``Z``, and the result is checked against ``VanillaSC(backend="outcome-only")`` on
the identical panel.

Two things hold. First, the two agree on the canonical synthetic California: the
donor weights match to about 0.003 (Utah ~0.39, Montana ~0.23, Nevada ~0.20,
Connecticut ~0.11) and the ATT to a fraction of a pack. Second, and the point of
the comparison, ``VanillaSC``'s exact outcome QP reaches a strictly lower
pre-period loss than the original solver (pre-period SSR ~52.130 against
~52.136): the genuine R reference lands just short of the optimum mlsynth
attains. This is the MSCMT/Malo thesis -- the data-driven nested SCM solvers can
fail to reach the global optimum -- demonstrated against the original ``Synth``
solver itself rather than a port, and it is why the paper benchmarks
``VanillaSC`` against the reference rather than re-deriving it.

Skips (``BenchmarkSkipped``) when ``Rscript`` or the ``Synth`` package is
unavailable, so a missing R toolchain never turns the suite red.

Provenance: Abadie, Diamond & Hainmueller (2010), JASA 105(490) and the R
package ``Synth`` (Abadie, Diamond & Hainmueller 2011, JSS 42(13)); California
Prop 99 panel ``basedata/california_panel.csv``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import reference_value

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "california_panel.csv")
_RSCRIPT_REF = os.path.join(os.path.dirname(__file__), "..", "R",
                            "synth_outcome_ref.R")


def _panel():
    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    years = np.array(sorted(d.year.unique()))
    T0 = int((years < 1989).sum())                 # 1970-1988 = 19 pre periods
    wide = d.pivot(index="state", columns="year", values="cigsale")
    donors = [s for s in wide.index if s != "California"]
    Y0 = wide.loc[donors].to_numpy().T             # periods x donors
    y = wide.loc["California"].to_numpy()          # periods
    return d, donors, Y0, y, T0


def _reference_weights(donors, Y0, y, T0) -> np.ndarray:
    """Run the original R ``Synth`` solver outcome-only; return donor weights."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (install R + the Synth package)")
    probe = subprocess.run([rscript, "-e", "suppressMessages(library(Synth))"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'Synth' not installed")

    with tempfile.TemporaryDirectory() as tmp:
        # units x periods, treated (California) row LAST -- the script's contract.
        Y = np.vstack([Y0.T, y[None, :]])
        np.savetxt(os.path.join(tmp, "Y.csv"), Y, delimiter=",")
        pd.DataFrame({"T0": [T0], "T": [Y.shape[1]], "trt_row": [Y.shape[0]]}).to_csv(
            os.path.join(tmp, "meta.csv"), index=False)
        out = subprocess.run([rscript, os.path.abspath(_RSCRIPT_REF), tmp],
                             capture_output=True, text=True)
        wp = os.path.join(tmp, "w_ref.csv")
        if out.returncode != 0 or not os.path.exists(wp):
            raise BenchmarkSkipped(f"Synth reference failed: {out.stderr.strip()[-200:]}")
        return pd.read_csv(wp)["w"].to_numpy()


def run() -> dict:
    d, donors, Y0, y, T0 = _panel()
    w_ref = _reference_weights(donors, Y0, y, T0)   # skips if R/Synth absent

    from mlsynth import VanillaSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treat", "unitid": "state",
            "time": "year", "backend": "outcome-only", "seed": 0,
            "display_graphs": False,
        }).fit()
    w_ml = np.array([float(res.weights.donor_weights.get(s, 0.0)) for s in donors])

    # Pre-period loss each solver actually attains, on the same panel.
    ssr_ml = float(np.sum((y[:T0] - Y0[:T0] @ w_ml) ** 2))
    ssr_ref = float(np.sum((y[:T0] - Y0[:T0] @ w_ref) ** 2))
    # ATT (post-period mean gap) under each weight vector.
    att_ml = float(np.mean((y - Y0 @ w_ml)[T0:]))
    att_ref = float(np.mean((y - Y0 @ w_ref)[T0:]))

    wmap_ml = dict(zip(donors, w_ml))
    return {
        "weight_max_abs_dev": float(np.max(np.abs(w_ml - w_ref))),
        "att_max_abs_diff": abs(att_ml - att_ref),
        "att_outcome_only": att_ml,
        "weight_utah": float(wmap_ml.get("Utah", 0.0)),
        "vanillasc_pre_ssr": ssr_ml,
        "synth_pre_ssr": ssr_ref,
        # Non-negative: VanillaSC's exact QP is never worse than the R solver.
        "ssr_synth_minus_vanillasc": ssr_ref - ssr_ml,
    }


# Deterministic (exact outcome QP vs Synth's default nested optim, fixed seed).
# VanillaSC reproduces the published ADH synthetic California (Utah 0.3939, ATT
# -19.51363) and matches R's Synth donor-by-donor to ~0.003, while reaching a
# pre-period SSR ~0.007 lower than the original solver -- the optimum the nested
# solver stops just short of.
EXPECTED = {
    "weight_max_abs_dev": (0.0, 0.02),         # vs R Synth, donor by donor
    "att_max_abs_diff": (0.0, 0.2),            # ATT agreement (packs)
    "att_outcome_only": (-19.51363, 0.02),     # canonical ADH ATT
    "weight_utah": (0.3939, 0.01),             # canonical synthetic California
    "vanillasc_pre_ssr": (52.1296, 0.05),      # the optimum mlsynth attains
    # pinned from the captured reference bundle (benchmarks/reference/synth_prop99/)
    # so the constant and the inspectable R run cannot drift apart.
    "synth_pre_ssr": (reference_value("synth_prop99", "synth_pre_ssr"), 0.1),
    "ssr_synth_minus_vanillasc": (0.0067, 0.0067),  # >= 0: never worse than R
}
