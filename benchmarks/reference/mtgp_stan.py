"""Live runner for the reference Stan program behind MTGP (Ben-Michael et al. 2023).

mlsynth's :class:`mlsynth.MTGP` ports the paper's Gaussian model (the replication
package's ``code/stan/normal.stan``) to NumPyro. This module compiles and runs
that reference Stan program directly via ``rstan`` -- i.e. it installs (compiles)
Stan's code at runtime -- on a single-treated panel, and parses the treated
unit's posterior counterfactual back into NumPy so a case can compare mlsynth
against the authors' own reference on identical data.

The Stan model lives at ``benchmarks/reference/mtgp.stan`` (a verbatim
transcription of the package's ``normal.stan``). To make the cross-check a clean
cell-for-cell comparison, the reference is fed the identical inverse-population
noise scaling mlsynth uses (``pop.mean() / pop``), so both engines see the same
standardized data. Everything raises :class:`BenchmarkSkipped` when ``Rscript``
or ``rstan`` is unavailable or the reference run fails, so the cross-check skips
gracefully off the daily reference runner.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_STAN = Path(__file__).resolve().parent / "mtgp.stan"


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
    population: str,
    treated_unit: str,
    first_treated_period: int,
    n_factors: int = 5,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    max_tree_depth: int = 13,
    seed: int = 1,
) -> dict:
    """Run the reference Stan MTGP on a long panel and parse the treated posterior.

    Reads ``data_csv`` (long: ``unit``/``time``/``outcome``/``population``
    columns), builds the ``(N periods, D units)`` outcome matrix with
    ``treated_unit`` in column 1, masks that unit's cells from
    ``first_treated_period`` on, scales the noise by ``pop.mean() / pop``
    (matching :func:`mlsynth.utils.mtgp_helpers.setup.prepare_mtgp_inputs`),
    compiles ``mtgp.stan`` into ``out_dir`` and samples it.

    Returns ``{"years" (T,), "cf_mean" (T,), "cf_lo" (T,), "cf_hi" (T,),
    "att", "pre_rmse", "lengthscale_f", "lengthscale_global", "rhat_lengthscale_f"}``
    -- the treated unit's posterior counterfactual (the latent no-intervention
    surface's treated column), the mean post-period ATT, the pre-period RMSE, and
    the GP length-scales, all from Stan.

    Raises
    ------
    BenchmarkSkipped
        If ``Rscript``/``rstan`` is unavailable or the reference run fails.
    """
    if not _have_rstan():
        raise BenchmarkSkipped("Rscript/rstan not available for the MTGP reference")

    stan_copy = out_dir / "mtgp.stan"
    shutil.copyfile(_STAN, stan_copy)
    data_abs = str(Path(data_csv).resolve())

    script = f"""
suppressMessages({{library(rstan); library(data.table)}})
options(mc.cores={n_chains}); rstan_options(auto_write=TRUE)
d <- fread("{data_abs}")
W  <- dcast(d, {time} ~ {unit}, value.var="{outcome}");    setorder(W, {time})
Wp <- dcast(d, {time} ~ {unit}, value.var="{population}");  setorder(Wp, {time})
years <- W${time}; treated <- "{treated_unit}"
donors <- setdiff(colnames(W)[-1], treated); units <- c(treated, donors)
Y   <- as.matrix(W[,  ..units])       # N x D, treated is column 1
pop <- as.matrix(Wp[, ..units])       # N x D
N <- nrow(Y); D <- ncol(Y)
inv_pop <- mean(pop) / pop            # match mlsynth's pop.mean()/pop scaling
n_post <- sum(years >= {first_treated_period}); T0 <- N - n_post
# control cells: everything except the treated column's post-period rows,
# in column-major (to_vector) order to match Stan's to_vector(y)
mask <- matrix(TRUE, N, D)            # TRUE = control (in likelihood)
mask[(T0 + 1):N, 1] <- FALSE         # treated column, post-period, masked out
control_idx <- which(as.vector(mask))
sdat <- list(N=N, D=D, n_k_f={n_factors}L, x=as.numeric(years), y=Y,
             inv_population=inv_pop, num_treated=n_post,
             control_idx=as.integer(control_idx))
fit <- stan("{stan_copy}", data=sdat, chains={n_chains}L,
            warmup={n_warmup}L, iter={n_warmup + n_samples}L, seed={seed}L, refresh=0,
            control=list(adapt_delta={target_accept}, max_treedepth={max_tree_depth}L))
f  <- extract(fit, "f")$f             # draws x N x D
cf <- f[, , 1]                        # draws x N (treated counterfactual surface)
cf_mean <- colMeans(cf); cf_lo <- apply(cf,2,quantile,.025); cf_hi <- apply(cf,2,quantile,.975)
obs <- Y[, 1]; gap <- obs - cf_mean
hp  <- summary(fit, pars=c("lengthscale_f","lengthscale_global"))$summary
cat("LSF", sprintf("%.8f", hp["lengthscale_f","mean"]), "\\n")
cat("LSG", sprintf("%.8f", hp["lengthscale_global","mean"]), "\\n")
cat("RHATLSF", sprintf("%.6f", hp["lengthscale_f","Rhat"]), "\\n")
cat("PRERMSE", sprintf("%.8f", sqrt(mean(gap[1:T0]^2))), "\\n")
cat("ATT", sprintf("%.8f", mean(gap[(T0+1):N])), "\\n")
cat("YEARS", paste(years, collapse=" "), "\\n")
cat("CFMEAN", paste(sprintf("%.8f", cf_mean), collapse=" "), "\\n")
cat("CFLO", paste(sprintf("%.8f", cf_lo), collapse=" "), "\\n")
cat("CFHI", paste(sprintf("%.8f", cf_hi), collapse=" "), "\\n")
"""
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:  # compile/sample failure -> skip, not fail
        raise BenchmarkSkipped(f"MTGP Stan reference run failed: {proc.stderr[-400:]}")

    out: dict = {}
    _scalars = {"LSF": "lengthscale_f", "LSG": "lengthscale_global",
                "RHATLSF": "rhat_lengthscale_f", "PRERMSE": "pre_rmse", "ATT": "att"}
    _arrays = {"CFMEAN": "cf_mean", "CFLO": "cf_lo", "CFHI": "cf_hi"}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key in _scalars:
            out[_scalars[key]] = float(parts[1])
        elif key == "YEARS":
            out["years"] = np.array([int(float(x)) for x in parts[1:]])
        elif key in _arrays:
            out[_arrays[key]] = np.array([float(x) for x in parts[1:]])
    if "att" not in out or "cf_mean" not in out:  # pragma: no cover - defensive
        raise BenchmarkSkipped("could not parse MTGP Stan reference output")
    return out


# Convenience wrapper for the shipped California (APPS) homicide panel.
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "mtgp_california.csv"


def run_california(out_dir: Path, **kw) -> dict:
    """Run the Stan reference on the shipped California panel (APPS, treated 2007)."""
    return run_reference(
        out_dir, data_csv=str(_DATA), outcome="homicide_rate", unit="state",
        time="year", population="population", treated_unit="CA",
        first_treated_period=2007, **kw)
