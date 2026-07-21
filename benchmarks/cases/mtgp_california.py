r"""MTGP cross-validation: California (APPS) vs the reference Stan program.

Cross-validation against the *running* reference. mlsynth's :class:`mlsynth.MTGP`
(Multitask Gaussian Process synthetic control, Ben-Michael et al. 2023) ports the
paper's Gaussian model (the replication package's ``code/stan/normal.stan``) to
NumPyro. This case compiles and runs that Stan program live via ``rstan``
(:func:`benchmarks.reference.mtgp_stan.run_california`) on the study's shipped
California panel (gun-homicide rates per 100,000 across the 50 states, 1997--2018,
California treated in 2007 when the Armed and Prohibited Persons System began
enforcement), feeds the *identical* panel to mlsynth's estimator, and checks the
two posteriors agree.

Both engines use the same spec (rank-5 coregionalization, squared-exponential
time kernels, population-scaled noise via ``pop.mean() / pop``, seeded) and
budget. MTGP is a Bayesian GP factor model, so the cells carry MCMC tolerances;
the residual is Monte-Carlo error between two independent NUTS runs of one model,
not a specification gap. The identified quantities -- the counterfactual path, the
ATT, and the GP length-scales -- mix and match; the raw coregionalization loadings
are rotation/sign non-identified and are not compared. On this panel the two tell
the same story: California's homicide rate falls about one per 100,000 below its
counterfactual after 2007, with a credible band that widens through the post-period.

Requires the ``[bayes]`` extra (NumPyro) for mlsynth and ``rstan`` for the
reference; skips gracefully when either is absent.

Provenance: Ben-Michael, Arbour, Feller, Franks & Raphael (2023), AOAS 17(2),
985--1016, replication-package Stan (``normal.stan``); FBI UCR homicide rates +
Census population for the panel.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

# Shared sampling budget for BOTH engines (kept modest so the live Stan compile +
# sample stays tractable on the daily reference runner).
_NF, _WARMUP, _DRAWS, _CHAINS = 5, 1000, 1000, 4


def _fit_both():
    """Run mlsynth MTGP and the live Stan reference on the identical panel."""
    import numpy as np
    import pandas as pd

    from benchmarks.reference.mtgp_stan import run_california, _DATA
    from mlsynth import MTGP

    try:
        import numpyro  # noqa: F401
    except ImportError as exc:
        from benchmarks.compare import BenchmarkSkipped
        raise BenchmarkSkipped("MTGP needs NumPyro (pip install 'mlsynth[bayes]').") \
            from exc

    with tempfile.TemporaryDirectory() as tmp:
        ref = run_california(Path(tmp), n_factors=_NF, n_warmup=_WARMUP,
                             n_samples=_DRAWS, n_chains=_CHAINS, seed=1)  # skips w/o rstan

    d = pd.read_csv(_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MTGP({
            "df": d, "outcome": "homicide_rate", "treat": "apps",
            "unitid": "state", "time": "year", "population": "population",
            "n_factors": _NF, "n_warmup": _WARMUP, "n_samples": _DRAWS,
            "n_chains": _CHAINS, "seed": 0, "display_graphs": False,
        }).fit()

    yr = np.asarray(res.time_series.time_periods)
    order = np.argsort(yr)
    ml = {
        "years": yr[order],
        "cf_mean": np.asarray(res.time_series.counterfactual_outcome)[order],
        "att": float(res.att),
        "lengthscale_f": float(res.weights.summary_stats["lengthscale_f"]),
        "lengthscale_global": float(res.weights.summary_stats["lengthscale_global"]),
        "pre_rmse": float(res.fit_diagnostics.rmse_pre),
    }
    return ml, ref


def run() -> dict:
    import numpy as np

    ml, ref = _fit_both()
    cf_corr = float(np.corrcoef(ml["cf_mean"], ref["cf_mean"])[0, 1])
    cf_max = float(np.max(np.abs(ml["cf_mean"] - ref["cf_mean"])))
    return {
        "mean_att": ml["att"],
        "att_negative": 1.0 if ml["att"] < 0.0 else 0.0,
        "cf_corr": cf_corr,
        "cf_max_vs_ref": cf_max,
        "att_vs_ref": float(abs(ml["att"] - ref["att"])),
        "lengthscale_f_vs_ref": float(abs(ml["lengthscale_f"] - ref["lengthscale_f"])),
        "pre_rmse_vs_ref": float(abs(ml["pre_rmse"] - ref["pre_rmse"])),
    }


def comparison() -> dict:
    """Side-by-side mlsynth MTGP vs the reference Stan on the California panel.

    Pairs the identified headline quantities -- the mean post-2007 ATT, the GP
    time-factor length-scale, and the pre-period fit -- live from both
    implementations, for the committed ``comparison.csv`` / ``comparisons.xlsx``
    workbook. Propagates ``BenchmarkSkipped`` when NumPyro/rstan is absent.
    """
    ml, ref = _fit_both()
    rows = [
        {"quantity": "ATT (post-2007 mean, per 100k)",
         "mlsynth": round(ml["att"], 3), "reference": round(ref["att"], 3)},
        {"quantity": "lengthscale_f (posterior mean)",
         "mlsynth": round(ml["lengthscale_f"], 3),
         "reference": round(ref["lengthscale_f"], 3)},
        {"quantity": "pre-period RMSE",
         "mlsynth": round(ml["pre_rmse"], 3), "reference": round(ref["pre_rmse"], 3)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "MTGP",
                         "config": {"n_factors": _NF, "n_warmup": _WARMUP,
                                    "n_samples": _DRAWS, "n_chains": _CHAINS,
                                    "seed": 0}},
        "reference": {"impl": "replication-package Stan (via Rscript + rstan)",
                      "version": "Ben-Michael et al. 2023 AOAS normal.stan (live)"},
    }


# Cross-validated live against the authors' replication-package Stan on the shipped
# California APPS panel (CA treated 2007, n_factors=5, 1000 warmup / 1000 draws, 4
# chains, adapt_delta 0.9). Both engines see the identical pop.mean()/pop noise
# scaling, so the counterfactual matches cell-for-cell; tolerances bracket the
# Monte-Carlo error between two independent NUTS runs (the /replicate port matched
# corr 0.99993, ATT -1.029 vs -1.031, length-scales within ~2%).
EXPECTED = {
    "mean_att": (-1.03, 0.6),          # ~1 per 100k decline post-APPS
    "att_negative": (1.0, 0.0),        # negative effect (paper's finding)
    "cf_corr": (1.0, 0.005),           # counterfactual paths track (>= 0.995)
    "cf_max_vs_ref": (0.0, 0.3),       # worst-cell gap (MC error on a ~6 rate level)
    "att_vs_ref": (0.0, 0.3),          # ATT agreement, per 100k
    "lengthscale_f_vs_ref": (0.0, 0.15),  # GP time length-scale agreement
    "pre_rmse_vs_ref": (0.0, 0.2),     # pre-period fit agreement, per 100k
}
