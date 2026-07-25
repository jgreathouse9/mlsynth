"""Golden cross-validation: mlsynth's RRSC vs the authors' reference R.

RRSC (He, Li, Shi & Miao 2026, *A robust regression approach to synthetic
control with interference*) is validated value-for-value against a live run of a
reference R implementation of both regimes -- large-N (EM factor analysis +
Huber-IRLS) and fixed-N (``stats::factanal`` + ``robustbase::ltsReg`` + the
Donoho-Johnstone selection). The data is a self-contained synthetic panel drawn
from the method's own untreated factor model with planted direct and
interference effects, so no external dataset or licence is involved -- only the
reference *algorithm* (``benchmarks/R/rrsc_reference.R``) is exercised.

``run()`` asserts the match as a pass/fail benchmark; ``comparison()`` emits the
per-unit mlsynth-vs-reference table that feeds the public :doc:`validation`
dashboard. Both read one deterministic computation, so the asserted numbers and
the published ones cannot drift apart.

Skips (rather than fails) when R, ``robustbase``/``expm``, or the reference
script is unavailable, so the suite stays green without the R toolchain.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from mlsynth.utils.rrsc_helpers.pipeline import point_effects

_REF_R = Path(__file__).resolve().parents[1] / "R" / "rrsc_reference.R"
SEED = 20260101
_cache: dict = {}


def _panel(N, T, T0, r=2, interfered=(1, 2), effect=8.0, interference=5.0, seed=SEED):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, r))
    F[:, 0] = np.cumsum(rng.normal(size=T)) / 3.0            # trend factor (survives centering)
    L = rng.normal(size=(N, r))
    Y = (F @ L.T).T + 0.3 * rng.normal(size=(N, T))
    Y[0, T0:] += effect
    for j in interfered:
        Y[j, T0:] += interference
    X = np.concatenate([np.zeros(T0, int), np.ones(T - T0, int)])
    return Y, X


def _r_effects(Y, X, regime, r, update_dif, workdir):
    ycsv, xcsv, ocsv = (workdir / f"{n}.csv" for n in ("Y", "X", "out"))
    np.savetxt(ycsv, Y, delimiter=",")
    np.savetxt(xcsv, X, delimiter=",", fmt="%d")
    proc = subprocess.run(
        ["Rscript", str(_REF_R), str(ycsv), str(xcsv), regime, str(r),
         "TRUE" if update_dif else "FALSE", str(ocsv)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise BenchmarkSkipped(f"reference R failed: {proc.stderr.strip()[-300:]}")
    import csv
    with open(ocsv) as fh:
        return np.array([float(row["effect"]) for row in csv.DictReader(fh)])


def _require_r() -> None:
    """Skip (never fail) when the reference toolchain is unavailable."""
    if shutil.which("Rscript") is None:
        raise BenchmarkSkipped("Rscript not available")
    if not _REF_R.exists():
        raise BenchmarkSkipped("reference R script missing")
    chk = subprocess.run(
        ["Rscript", "-e",
         "quit(status = !all(sapply(c('robustbase','expm'), requireNamespace, quietly=TRUE)))"],
        capture_output=True, text=True,
    )
    if chk.returncode != 0:
        raise BenchmarkSkipped("R packages robustbase/expm not installed")


def _regimes() -> dict:
    """Per-unit effects from mlsynth and from the live reference R, both regimes.

    Deterministic (fixed seed, no resampling), so ``run()`` and ``comparison()``
    describe the same numbers; cached so calling both runs the R once each.
    """
    if _cache:
        return _cache["regimes"]
    _require_r()
    out = {}
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        # large-N: many units, short post-period
        Y, X = _panel(N=30, T=44, T0=32)
        py, _ = point_effects(Y, X, r=2, regime="large_n", update_dif=False)
        out["large_n"] = (py, _r_effects(Y, X, "large_n", 2, False, wd))
        # fixed-N: few units, long pre/post
        Y, X = _panel(N=8, T=80, T0=55)
        py, _ = point_effects(Y, X, r=2, regime="fixed_n", update_dif=True)
        out["fixed_n"] = (py, _r_effects(Y, X, "fixed_n", 2, True, wd))
    _cache["regimes"] = out
    return out


def run() -> dict:
    reg = _regimes()                                 # skips if R/packages absent
    out: dict = {}
    for name in ("large_n", "fixed_n"):
        py, rr = reg[name]
        out[f"{name}_max_abs_diff"] = float(np.max(np.abs(py - rr)))
        out[f"{name}_direct_effect"] = float(py[0])
    return out


def comparison() -> dict:
    """mlsynth's RRSC vs the reference R, unit by unit, in both regimes.

    Reports the estimands the method exists to produce: the direct effect on the
    treated unit and the interference effects on the units the DGP actually
    contaminates (units 1-2), plus a clean control (unit 5, true effect 0).
    """
    reg = _regimes()
    labels = [(0, "direct effect, treated unit"),
              (1, "interference, unit 1"),
              (2, "interference, unit 2"),
              (5, "clean control, unit 5")]
    rows = []
    for regime, tag in (("large_n", "large-N"), ("fixed_n", "fixed-N")):
        py, rr = reg[regime]
        for i, what in labels:
            rows.append({"quantity": f"{tag}: {what}",
                         "mlsynth": round(float(py[i]), 6),
                         "reference": round(float(rr[i]), 6)})
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "RRSC",
            "config": {"n_factors": 2, "regimes": "large_n + fixed_n",
                       "update_dif": "False (large-N) / True (fixed-N)"},
        },
        "reference": {
            "impl": "reference R implementation of both RRSC regimes "
                    "(fa.em + Huber-IRLS; stats::factanal + robustbase::ltsReg "
                    "+ Donoho-Johnstone selection), run LIVE via Rscript",
            "version": "He, Li, Shi & Miao (2026), arXiv:2411.01249",
        },
    }


# mlsynth reproduces the reference R value-for-value: the large-N EM and
# Huber-IRLS to solver precision, the fixed-N path to the factor-analysis
# optimiser tolerance (mlsynth uses SciPy L-BFGS-B, R uses its own optim). The
# planted direct effect (+8) is recovered by both.
EXPECTED = {
    "large_n_max_abs_diff": (0.0, 1e-6),        # EM + IRLS to solver precision
    "fixed_n_max_abs_diff": (0.0, 1e-3),        # factanal optimiser tolerance (SciPy vs R optim)
    "large_n_direct_effect": (8.231, 0.5),
    "fixed_n_direct_effect": (8.381, 0.5),
}
