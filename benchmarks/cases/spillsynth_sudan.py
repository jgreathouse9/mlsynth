"""SPILLSYNTH SAR vs the authors' Rcpp sampler (Sakaguchi-Tagawa 2026, Sudan).

Cross-validation. The 2011 Sudan secession application of Sakaguchi & Tagawa
(2026, *The Econometrics Journal*, doi:10.1093/ectj/utag006): a spatial-
autoregressive (SAR) Bayesian synthetic control with a trade-weighted spatial
weight matrix, treated unit "the Sudans" (North + South recombined), 2000-2015,
treatment in 2011 (the partition year is a transition and is dropped from the
effect series). mlsynth's ``SPILLSYNTH(method="sar")`` is laid against the
authors' own RcppArmadillo sampler run on the committed nonproprietary panel.

The targets are pinned from a live run of the authors' code captured in
``benchmarks/reference/spillsynth_sudan/`` (their ``sc_spillover`` at 1e5 draws,
seed 20251022). Compared quantities, chosen to be convention-robust:

* ``rho`` -- the spatial-autocorrelation parameter. It is *weakly identified*
  here (the authors' reported effective sample size is ~390 even at 1e6 draws),
  so this is a converged-posterior match, not a smoke-run match: at a matched 1e5
  draws the authors' C++ gives ~0.43 and mlsynth ~0.42, both on the authors'
  reported 0.427. The tolerance reflects that weak identification.
* ``effect_2012_pct`` / ``effect_2015_pct`` -- the GDP-per-capita effect on the
  Sudans, ``(observed - counterfactual)/counterfactual``, in 2012 and 2015.
* ``egypt_share`` / ``kenya_share`` -- each country's share of the total
  (top-8) spillover magnitude. Shares are unit-free, so they sidestep a global
  scale convention between the two codebases' per-donor spillover reporting; the
  spillover *ranking* (Egypt, then Kenya) is what the paper highlights.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import reference_value

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")
_ITER, _BURN, _SEED = 20000, 10000, 20251022


def _need(name: str) -> str:
    p = os.path.abspath(os.path.join(_BASE, name))
    if not os.path.exists(p):
        raise BenchmarkSkipped(f"{name} not available")
    return p


def _fit():
    from mlsynth import SPILLSYNTH

    panel = pd.read_csv(_need("sudan_panel.csv"))
    W = pd.read_csv(_need("sudan_W_matrix.csv"), index_col=0)
    w = pd.read_csv(_need("sudan_w_vector.csv"), index_col=0).squeeze()
    gdp, covs = panel.columns[3], list(panel.columns[4:10])
    panel["treated"] = ((panel["country"] == "Sudan") & (panel["year"] >= 2011)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SPILLSYNTH({
            "df": panel, "outcome": gdp, "treat": "treated", "unitid": "country",
            "time": "year", "method": "sar", "spatial_W": W, "spatial_w": w,
            "covariates": covs, "p_factors": 1, "mcmc_iter": _ITER, "mcmc_burn": _BURN,
            "step_rho": 0.02, "mcmc_seed": _SEED, "display_graphs": False}).fit()
    return res, panel, gdp


def _metrics(res, panel, gdp) -> dict:
    sud = panel[panel["country"] == "Sudan"].sort_values("year")
    yr, obs = sud["year"].to_numpy(), sud[gdp].to_numpy(float)
    cf = np.asarray(res.time_series.counterfactual_outcome, float).ravel()
    pct = 100.0 * (obs - cf) / cf
    # per-donor mean spillover, most-negative first; shares of the top-8 magnitude
    spill = {k.split(",")[0]: float(np.mean(v)) for k, v in res.spillover_effects.items()}
    ranked = sorted(spill.items(), key=lambda kv: kv[1])
    top8 = ranked[:8]
    denom = sum(abs(v) for _, v in top8) or 1.0
    return {
        "rho": float(res.sar.rho_hat),
        "effect_2012_pct": float(pct[yr == 2012][0]),
        "effect_2015_pct": float(pct[yr == 2015][0]),
        "egypt_share": abs(top8[0][1]) / denom,
        "kenya_share": abs(top8[1][1]) / denom,
    }


def run() -> dict:
    res, panel, gdp = _fit()
    return _metrics(res, panel, gdp)


def comparison() -> dict:
    res, panel, gdp = _fit()
    m = _metrics(res, panel, gdp)
    rows = [
        {"quantity": q, "mlsynth": round(m[k], 6),
         "reference": round(reference_value("spillsynth_sudan", k), 6)}
        for q, k in [("rho", "rho"),
                     ("effect_2012_pct", "effect_2012_pct"),
                     ("effect_2015_pct", "effect_2015_pct"),
                     ("egypt_spillover_share", "egypt_share"),
                     ("kenya_spillover_share", "kenya_share")]
    ]
    cfg = {"outcome": "GDP per capita", "treat": "treated", "unitid": "country",
           "time": "year", "method": "sar", "p_factors": 1,
           "mcmc_iter": _ITER, "mcmc_burn": _BURN, "mcmc_seed": _SEED}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SPILLSYNTH", "config": cfg},
        "reference": {"impl": "Sakaguchi-Tagawa RcppArmadillo sc_spillover "
                              "(method=sar, live run on the nonproprietary panel, captured)",
                      "version": "Zenodo doi:10.5281/zenodo.19066186"},
    }


# Targets pinned from the captured authors'-R run (benchmarks/reference/
# spillsynth_sudan/). rho's tolerance is wide on purpose -- it is weakly
# identified (authors' ESS ~390 at 1e6 draws); the effect and spillover shares
# are tight.
_r = lambda k: reference_value("spillsynth_sudan", k)
EXPECTED = {
    "rho": (_r("rho"), 0.05),
    "effect_2012_pct": (_r("effect_2012_pct"), 1.5),
    "effect_2015_pct": (_r("effect_2015_pct"), 2.0),
    "egypt_share": (_r("egypt_share"), 0.04),
    "kenya_share": (_r("kenya_share"), 0.04),
}
