"""On-demand clone + live runner for Hollingsworth & Wing's SCUL R package.

Synthetic Control Using Lasso ships a reference R implementation at
https://github.com/hollina/scul (Hollingsworth & Wing 2022). We clone it at a
pinned commit (codeload tarball when the git proxy 403s) and ``source()`` its
core functions from a throwaway ``Rscript`` -- no package install needed.
``run_reference`` runs the reference ``SCUL()`` on the shipped California
(Proposition 99) cigarette panel and parses its selected penalty, donor weights,
synthetic series and ATT back into NumPy, alongside a long-format CSV of the
exact panel so the mlsynth side fits the *identical* data.

Everything raises :class:`BenchmarkSkipped` when ``Rscript``, ``glmnet``, the
clone, or the reference run is unavailable, so the cross-check skips gracefully.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/hollina/scul.git"
_COMMIT = "121b588cf8e4eebcea68518b57a7c754040153bb"
_CACHE = Path(__file__).resolve().parent / ".cache" / "hollina-scul"
_TREATMENT_BEGINS_AT = 19          # tutorial: row 19 (the post-1988 period)
_NUMBER_INITIAL = 5
_TRAINING_POST = 7


def _ensure_clone() -> Path:
    marker = _CACHE / "R" / "SCUL.R"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("SCUL clone missing R/SCUL.R")
    return _CACHE


def _have_rscript() -> bool:
    try:
        subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def run_reference(out_dir: Path) -> dict:
    """Run the reference ``SCUL()`` on the shipped cigarette panel.

    Writes ``scul_panel_long.csv`` (state, year, cigsale, retprice, treat) into
    ``out_dir`` for the mlsynth side and returns
    ``{"ridge_lambda", "att", "cohens_d", "series" (T,), "year" (T,)}``.

    Raises
    ------
    BenchmarkSkipped
        If ``Rscript``, the clone, or the reference run is unavailable.
    """
    if not _have_rscript():
        raise BenchmarkSkipped("Rscript not available for the SCUL reference")
    clone = _ensure_clone()
    long_csv = out_dir / "scul_panel_long.csv"
    script = f"""
ok <- requireNamespace("glmnet", quietly=TRUE)
if(!ok){{ cat("NO_GLMNET\\n"); quit(status=0) }}
suppressMessages(library(glmnet))
source("{clone / 'R' / 'mysd.R'}"); source("{clone / 'R' / 'getOptcv.scul.R'}")
source("{clone / 'R' / 'SCUL.R'}")
load("{clone / 'data' / 'cigarette_sales.RData'}")
cs <- cigarette_sales
TBA <- {_TREATMENT_BEGINS_AT}L
donorcols <- setdiff(names(cs), c("year","cigsale_6","retprice_6"))
Xall <- as.matrix(cs[, donorcols])
y.actual <- matrix(cs$cigsale_6, ncol=1)
res <- SCUL(
  PostPeriodLength = nrow(y.actual)-TBA+1, PrePeriodLength = TBA-1,
  NumberInitialTimePeriods = {_NUMBER_INITIAL}L, OutputFilePath = ".",
  x.DonorPool.PreTreatment = Xall[1:(TBA-1),,drop=FALSE],
  y.PreTreatment = matrix(cs$cigsale_6[1:(TBA-1)], ncol=1),
  x.DonorPool = Xall, time = data.frame(year=cs$year), y.actual = y.actual,
  TreatmentBeginsAt = TBA, TrainingPostPeriodLength = {_TRAINING_POST}L,
  cvOption = "lambda.median", plotCV = FALSE)
att <- mean(res$y.actual[TBA:length(res$y.actual)] - res$y.scul[TBA:length(res$y.scul)])
cat(sprintf("LAMBDA %.10f\\n", res$CrossValidatedLambda))
cat(sprintf("ATT %.8f\\n", att))
cat(sprintf("COHENSD %.8f\\n", res$CohensD))
cat("SERIES", paste(sprintf("%.8f", as.numeric(res$y.scul)), collapse=" "), "\\n")
cat("YEAR", paste(cs$year, collapse=" "), "\\n")
# long panel for mlsynth (state = FIPS suffix; treat = California post-treatment)
fips <- unique(sub("^[a-z]+_", "", donorcols))
rows <- do.call(rbind, lapply(c(6, as.integer(fips)), function(f){{
  data.frame(state=f, year=cs$year,
             cigsale=cs[[paste0("cigsale_", f)]],
             retprice=cs[[paste0("retprice_", f)]])
}}))
rows$treat <- as.integer(rows$state==6 & rows$year>=cs$year[TBA])
write.csv(rows, "{long_csv}", row.names=FALSE)
"""
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise BenchmarkSkipped(f"SCUL reference run failed: {proc.stderr[-300:]}")
    if "NO_GLMNET" in proc.stdout:
        raise BenchmarkSkipped("glmnet not available for the SCUL reference")
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("LAMBDA"):
            out["ridge_lambda"] = float(line.split()[1])
        elif line.startswith("ATT"):
            out["att"] = float(line.split()[1])
        elif line.startswith("COHENSD"):
            out["cohens_d"] = float(line.split()[1])
        elif line.startswith("SERIES"):
            out["series"] = np.array([float(x) for x in line.split()[1:]])
        elif line.startswith("YEAR"):
            out["year"] = np.array([int(x) for x in line.split()[1:]])
    if "ridge_lambda" not in out or not long_csv.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("could not parse SCUL reference output")
    return out
