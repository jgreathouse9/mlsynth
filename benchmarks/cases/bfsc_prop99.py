"""BFSC cross-validation: California (Proposition 99) vs the appendix Stan.

Cross-validation against the *running* reference. mlsynth's :class:`mlsynth.BFSC`
(Bayesian Factor Synthetic Control, Pinkney 2021) ports the paper's appendix Stan
program to NumPyro. This case compiles and runs that Stan program live via
``rstan`` (:func:`benchmarks.reference.bfsc_stan.run_prop99`) on the shipped Prop
99 cigarette panel (California treated in 1989, 38 donor states, 1970--2000),
feeds the *identical* panel to mlsynth's estimator, and checks the two posteriors
agree -- the second independent Stan cross-check of BFSC after German
reunification (``bfsc_germany``).

Both samplers use the same spec (8 factors, adapt_delta 0.95, seeded) and budget.
BFSC is a Bayesian factor model, so the cells carry MCMC tolerances; the residual
is Monte-Carlo error between two independent NUTS runs of one model, not a
specification gap. The identified quantities -- the counterfactual path, the ATT,
and :math:`\\sigma` -- mix and match; the raw factor loadings are rotation/sign
non-identified and are not compared (Stan flags this too). On this panel the two
tell the same story: a widening decline in cigarette sales (~-16 packs) with a
credible band wide enough to reach zero.

Requires the ``[bayes]`` extra (NumPyro) for mlsynth and ``rstan`` for the
reference; skips gracefully when either is absent.

Provenance: Pinkney (2021), arXiv:2103.16244, appendix Stan; Abadie, Diamond &
Hainmueller (2010) for the Prop 99 panel.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

# Shared sampling budget for BOTH engines (kept modest so the live Stan compile +
# sample stays tractable on the daily reference runner).
_NF, _WARMUP, _DRAWS, _CHAINS = 8, 1000, 1000, 4


def _fit_both():
    """Run mlsynth BFSC and the live Stan reference on the identical panel."""
    import numpy as np
    import pandas as pd

    from benchmarks.reference.bfsc_stan import run_prop99, _DATA
    from mlsynth import BFSC

    try:
        import numpyro  # noqa: F401
    except ImportError as exc:
        from benchmarks.compare import BenchmarkSkipped
        raise BenchmarkSkipped("BFSC needs NumPyro (pip install 'mlsynth[bayes]').") \
            from exc

    with tempfile.TemporaryDirectory() as tmp:
        ref = run_prop99(Path(tmp), n_factors=_NF, n_warmup=_WARMUP,
                         n_samples=_DRAWS, n_chains=_CHAINS, seed=1)  # skips w/o rstan

    d = pd.read_csv(_DATA)
    d["treat"] = d["Proposition 99"].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = BFSC({
            "df": d, "outcome": "cigsale", "treat": "treat",
            "unitid": "state", "time": "year",
            "n_factors": _NF, "n_warmup": _WARMUP, "n_samples": _DRAWS,
            "n_chains": _CHAINS, "target_accept": 0.95, "seed": 0,
            "display_graphs": False,
        }).fit()

    # align mlsynth quantities to the reference's year grid
    yr = np.asarray(res.time_series.time_periods)
    order = np.argsort(yr)
    ml = {
        "years": yr[order],
        "cf_mean": np.asarray(res.time_series.counterfactual_outcome)[order],
        "att": float(res.att),
        "sigma": float(res.weights.summary_stats["sigma_post_mean"]),
        "pre_rmse": float(res.fit_diagnostics.rmse_pre),
    }
    return ml, ref


def run() -> dict:
    import numpy as np

    ml, ref = _fit_both()
    # both grids are sorted ascending by construction
    cf_corr = float(np.corrcoef(ml["cf_mean"], ref["cf_mean"])[0, 1])
    cf_max = float(np.max(np.abs(ml["cf_mean"] - ref["cf_mean"])))
    return {
        "mean_att": ml["att"],
        "att_negative": 1.0 if ml["att"] < 0.0 else 0.0,
        "cf_corr": cf_corr,
        "cf_max_vs_ref": cf_max,
        "att_vs_ref": float(abs(ml["att"] - ref["att"])),
        "sigma_vs_ref": float(abs(ml["sigma"] - ref["sigma"])),
        "pre_rmse_vs_ref": float(abs(ml["pre_rmse"] - ref["pre_rmse"])),
    }


def comparison() -> dict:
    """Side-by-side mlsynth BFSC vs the appendix Stan on the Prop 99 panel.

    Pairs the identified headline quantities -- the mean post-1989 ATT, the
    idiosyncratic scale, and the pre-period fit -- live from both implementations,
    for the committed ``comparison.csv`` / ``comparisons.xlsx`` workbook.
    Propagates ``BenchmarkSkipped`` when NumPyro/rstan is absent.
    """
    ml, ref = _fit_both()
    rows = [
        {"quantity": "ATT (post-1989 mean)",
         "mlsynth": round(ml["att"], 3), "reference": round(ref["att"], 3)},
        {"quantity": "sigma (posterior mean)",
         "mlsynth": round(ml["sigma"], 4), "reference": round(ref["sigma"], 4)},
        {"quantity": "pre-period RMSE",
         "mlsynth": round(ml["pre_rmse"], 3), "reference": round(ref["pre_rmse"], 3)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "BFSC",
                         "config": {"n_factors": _NF, "n_warmup": _WARMUP,
                                    "n_samples": _DRAWS, "n_chains": _CHAINS,
                                    "seed": 0}},
        "reference": {"impl": "author appendix Stan (via Rscript + rstan)",
                      "version": "Pinkney 2021 arXiv:2103.16244 appendix (live)"},
    }


# Cross-validated live against the author's appendix Stan on the shipped Prop 99
# panel (California treated 1989, L=8, 1000 warmup / 1000 draws, 4 chains,
# adapt_delta 0.95). The two independent NUTS runs agree on the identified
# quantities; tolerances bracket the Monte-Carlo error between them (the long
# 2000/2000 run tightens ATT agreement to ~0.6 packs and cf-path corr to 0.9999).
EXPECTED = {
    "mean_att": (-16.0, 7.0),          # widening decline in cigarette sales
    "att_negative": (1.0, 0.0),        # negative effect (classical finding)
    "cf_corr": (1.0, 0.01),            # counterfactual paths track (>= 0.99)
    "cf_max_vs_ref": (0.0, 5.0),       # worst-cell gap, packs (MC error, wide band)
    "att_vs_ref": (0.0, 4.0),          # ATT agreement, packs
    "sigma_vs_ref": (0.0, 0.03),       # idiosyncratic scale agreement
    "pre_rmse_vs_ref": (0.0, 0.6),     # pre-period fit agreement, packs
}
