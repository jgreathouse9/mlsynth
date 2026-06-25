"""On-demand clone + live runner for the authors' SPSC R package.

Single Proxy Synthetic Control ships a reference R implementation at
https://github.com/qkrcks0218/SPSC (Park & Tchetgen Tchetgen 2025, JCI
20230079). The package is a single self-contained source file
(``R/SPSC.R``) depending only on ``MASS`` and ``splines``, so we clone it at a
pinned commit (codeload tarball when the git proxy 403s) and ``source()`` it
from a throwaway ``Rscript`` -- no package install needed. ``run_reference``
runs the reference ``SPSC()`` on a supplied panel and parses its effect path
and per-period standard errors back into NumPy.

Everything raises :class:`BenchmarkSkipped` when ``Rscript``, the clone, or the
reference run is unavailable, so the SPSC cross-checks skip gracefully off-line.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/qkrcks0218/SPSC.git"
_COMMIT = "054f1fbb7536352b498270c57016b667689f9e69"
_CACHE = Path(__file__).resolve().parent / ".cache" / "qkrcks0218-SPSC"


def _ensure_clone() -> Path:
    marker = _CACHE / "R" / "SPSC.R"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("SPSC clone missing R/SPSC.R")
    return _CACHE


def _have_rscript() -> bool:
    try:
        subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def run_reference(y: np.ndarray, W: np.ndarray, T0: int, *,
                  detrend: bool = True, att_degree: int = 0,
                  detrend_linear: bool = False,
                  ridge_lambda: float | None = None,
                  conformal_periods=None,
                  conformal_alpha: float = 0.05) -> dict:
    """Run the reference ``SPSC()`` on ``(y, W, T0)`` and parse its output.

    ``att_degree`` and ``detrend_linear`` mirror the package's ``att.ft`` and
    ``detrend.ft`` (``att_degree=1`` is a linear effect path; ``detrend_linear``
    swaps the spline trend for the linear ``(1, t)`` trend). Returns
    ``{"effect_path", "path_se"}`` as NumPy arrays.

    When ``conformal_periods`` (1-based post-period indices) is given, the
    reference's pointwise conformal prediction intervals are computed at level
    ``conformal_alpha`` and returned as ``{"conformal_lb", "conformal_ub"}`` too
    (the achievable discrete level needs ``conformal_alpha >= 2/(T0+1)``).

    Raises
    ------
    BenchmarkSkipped
        If ``Rscript``, the clone, or the reference run is unavailable.
    """
    if not _have_rscript():
        raise BenchmarkSkipped("Rscript not available for the SPSC reference")
    clone = _ensure_clone()
    y = np.asarray(y, float).ravel()
    W = np.asarray(W, float)
    T1 = len(y) - int(T0)
    lam_line = ('lambda.type="cv", lambda.value=NULL' if ridge_lambda is None
                else f'lambda.type="fix", lambda.value={float(ridge_lambda)}')
    att_ft = ("function(t){matrix(c(1,t),1,2)}" if att_degree == 1
              else "function(t){matrix(c(1),1,1)}")
    detrend_ft = ("function(t){matrix(c(1,t),1,2)}" if detrend_linear
                  else "function(t){Spline.Trend(t,T0,df=5)}")
    want_conformal = conformal_periods is not None
    if want_conformal:
        periods_r = "c(" + ",".join(str(int(p)) for p in conformal_periods) + ")"
        conf_line = (f"conformal.period={periods_r}, conformal.cover=FALSE, "
                     f"true.effect=NULL, conformal.interval=TRUE, "
                     f"conformal.pvalue={float(conformal_alpha)}")
        conf_dump = ('cat("CILB", paste(s$conformal.interval[,1], collapse=" "), "\\n")\n'
                     'cat("CIUB", paste(s$conformal.interval[,2], collapse=" "), "\\n")')
    else:
        conf_line = ("conformal.period=NULL, conformal.cover=FALSE, "
                     "true.effect=NULL, conformal.interval=FALSE, "
                     "conformal.pvalue=0.05")
        conf_dump = ""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savetxt(tmp / "y.csv", y, delimiter=",")
        np.savetxt(tmp / "W.csv", W, delimiter=",")
        script = f"""
suppressMessages({{library(MASS); library(splines)}})
source("{clone / 'R' / 'SPSC.R'}")
y <- scan("{tmp / 'y.csv'}", quiet=TRUE)
W <- as.matrix(read.csv("{tmp / 'W.csv'}", header=FALSE))
T0 <- {int(T0)}; T1 <- {T1}
s <- SPSC(Y.Pre=y[1:T0], Y.Post=y[T0+1:T1], W.Pre=W[1:T0,], W.Post=W[T0+1:T1,],
          detrend={'TRUE' if detrend else 'FALSE'}, detrend.ft={detrend_ft},
          Y.basis=function(y){{matrix(c(y),1,1)}}, att.ft={att_ft},
          {lam_line}, lambda.grid=seq(-6,2,by=0.5), bootstrap.num=0,
          {conf_line})
cat("PATH", paste(as.numeric(s$ATT), collapse=" "), "\\n")
cat("SE", paste(as.numeric(s$ASE.ATT), collapse=" "), "\\n")
{conf_dump}
"""
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True,
                              text=True, cwd=str(tmp))
        if proc.returncode != 0:
            raise BenchmarkSkipped(f"SPSC reference run failed: {proc.stderr[-300:]}")
    path, se, lb, ub = None, None, None, None
    for line in proc.stdout.splitlines():
        if line.startswith("PATH"):
            path = np.array([float(x) for x in line.split()[1:]])
        elif line.startswith("SE"):
            se = np.array([float(x) for x in line.split()[1:]])
        elif line.startswith("CILB"):
            lb = np.array([float(x) for x in line.split()[1:]])
        elif line.startswith("CIUB"):
            ub = np.array([float(x) for x in line.split()[1:]])
    if path is None or se is None:  # pragma: no cover - defensive
        raise BenchmarkSkipped("could not parse SPSC reference output")
    out = {"effect_path": path, "path_se": se}
    if want_conformal:
        if lb is None or ub is None:  # pragma: no cover - defensive
            raise BenchmarkSkipped("could not parse SPSC conformal interval")
        out["conformal_lb"], out["conformal_ub"] = lb, ub
    return out
