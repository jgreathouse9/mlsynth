"""Live runner for the appendix Stan program behind BFSC (Pinkney 2021).

mlsynth's :class:`mlsynth.BFSC` ports the paper's appendix Stan model to NumPyro.
This module compiles and runs that reference Stan program directly via ``rstan``
-- i.e. it installs (compiles) Stan's code at runtime -- on a single-treated
panel, and parses the treated unit's posterior counterfactual back into NumPy so
a case can compare mlsynth against the author's own reference on identical data.

The Stan model lives at ``benchmarks/reference/bfsc.stan`` (a verbatim
transcription of the paper's appendix). Everything raises
:class:`BenchmarkSkipped` when ``Rscript`` or ``rstan`` is unavailable or the
reference run fails, so the cross-check skips gracefully off the daily reference
runner.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_STAN = Path(__file__).resolve().parent / "bfsc.stan"


def _have_rstan() -> bool:
    """True iff Rscript runs and the rstan package is importable."""
    try:
        proc = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("rstan", quietly=TRUE))'],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return proc.stdout.strip().endswith("TRUE")


def run_reference(
    out_dir: Path,
    *,
    data_csv: str,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: str,
    first_treated_period: int,
    n_factors: int = 8,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 4,
    seed: int = 1,
) -> dict:
    """Run the appendix Stan BFSC on a long panel and parse the treated posterior.

    Reads ``data_csv`` (long format: ``unit``/``time``/``outcome`` columns),
    puts ``treated_unit`` in row 0, treats ``first_treated_period`` as the first
    post period, compiles ``bfsc.stan`` into ``out_dir`` and samples it.

    Returns ``{"years" (T,), "cf_mean" (T,), "cf_lo" (T,), "cf_hi" (T,),
    "att", "sigma", "pre_rmse", "rhat_sigma"}`` -- the treated unit's posterior
    counterfactual (unscaled), the mean post-period ATT, the idiosyncratic scale,
    and the pre-period RMSE, all from Stan.

    Raises
    ------
    BenchmarkSkipped
        If ``Rscript``/``rstan`` is unavailable or the reference run fails.
    """
    if not _have_rstan():
        raise BenchmarkSkipped("Rscript/rstan not available for the BFSC reference")

    # Compile in a scratch copy so rstan's auto-written .rds never lands in-repo.
    stan_copy = out_dir / "bfsc.stan"
    shutil.copyfile(_STAN, stan_copy)
    data_abs = str(Path(data_csv).resolve())

    script = f"""
suppressMessages({{library(rstan); library(data.table)}})
options(mc.cores={n_chains}); rstan_options(auto_write=TRUE)
d <- fread("{data_abs}")
W <- dcast(d, {time} ~ {unit}, value.var="{outcome}"); setorder(W, {time})
years <- W${time}; treated <- "{treated_unit}"
donors <- setdiff(colnames(W)[-1], treated); units <- c(treated, donors)
Ymat <- t(as.matrix(W[, ..units]))          # J x T, treated is row 1
J <- nrow(Ymat); T <- ncol(Ymat)
n_post <- sum(years >= {first_treated_period}); T0 <- T - n_post
sdat <- list(T=T, J=J, L={n_factors}L, P=0, X=matrix(0,0,J), Y=Ymat, trt_times=n_post)
fit <- stan("{stan_copy}", data=sdat, chains={n_chains}L,
            warmup={n_warmup}L, iter={n_warmup + n_samples}L, seed={seed}L, refresh=0,
            control=list(adapt_delta=0.95, max_treedepth=14))
tr <- extract(fit, "synth_out")$synth_out[,1,]   # draws x T (treated counterfactual)
cf_mean <- colMeans(tr); cf_lo <- apply(tr,2,quantile,.05); cf_hi <- apply(tr,2,quantile,.95)
sig <- extract(fit,"sigma")$sigma
obs <- Ymat[1,]; gap <- obs - cf_mean
rhat_sigma <- summary(fit, pars="sigma")$summary[,"Rhat"]
cat("SIGMA", sprintf("%.8f", mean(sig)), "\\n")
cat("RHATSIGMA", sprintf("%.6f", rhat_sigma), "\\n")
cat("PRERMSE", sprintf("%.8f", sqrt(mean(gap[1:T0]^2))), "\\n")
cat("ATT", sprintf("%.8f", mean(gap[(T0+1):T])), "\\n")
cat("YEARS", paste(years, collapse=" "), "\\n")
cat("CFMEAN", paste(sprintf("%.8f", cf_mean), collapse=" "), "\\n")
cat("CFLO", paste(sprintf("%.8f", cf_lo), collapse=" "), "\\n")
cat("CFHI", paste(sprintf("%.8f", cf_hi), collapse=" "), "\\n")
"""
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:  # compile/sample failure -> skip, not fail
        raise BenchmarkSkipped(f"BFSC Stan reference run failed: {proc.stderr[-400:]}")

    out: dict = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key in ("SIGMA", "RHATSIGMA", "PRERMSE", "ATT"):
            out[{"SIGMA": "sigma", "RHATSIGMA": "rhat_sigma",
                 "PRERMSE": "pre_rmse", "ATT": "att"}[key]] = float(parts[1])
        elif key == "YEARS":
            out["years"] = np.array([int(x) for x in parts[1:]])
        elif key in ("CFMEAN", "CFLO", "CFHI"):
            out[{"CFMEAN": "cf_mean", "CFLO": "cf_lo", "CFHI": "cf_hi"}[key]] = \
                np.array([float(x) for x in parts[1:]])
    if "att" not in out or "cf_mean" not in out:  # pragma: no cover - defensive
        raise BenchmarkSkipped("could not parse BFSC Stan reference output")
    return out


# Convenience wrapper for the shipped Prop 99 cigarette panel.
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "smoking_data.csv"


def run_prop99(out_dir: Path, **kw) -> dict:
    """Run the Stan reference on the shipped Prop 99 panel (California, 1989)."""
    return run_reference(
        out_dir, data_csv=str(_DATA), outcome="cigsale", unit="state",
        time="year", treated_unit="California", first_treated_period=1989, **kw)
