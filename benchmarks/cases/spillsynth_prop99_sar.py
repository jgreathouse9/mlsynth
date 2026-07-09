"""SPILLSYNTH (Bayesian Spatial SAR) cross-validation: California Proposition 99.

Cross-validates mlsynth's ``SPILLSYNTH(method="sar")`` -- the spatial-autoregressive
Bayesian SCM of Sakaguchi & Tagawa (2026), *"Identification and Bayesian Inference
for Synthetic Control Methods with Spillover Effects"* (The Econometrics Journal,
doi:10.1093/ectj/utag006) -- against the California Proposition 99 tutorial of
Carlos Mendez (*"Bayesian Spatial Synthetic Control"*,
https://carlos-mendez.org/post/r_sc_bayes_spatial/), whose Stage 3 runs the
authors' SAR via the ``sc_spillover`` Rcpp helpers. The live R reference is
captured in ``benchmarks/reference/spillsynth_prop99_sar/`` (R 4.3.3, seed
20251022).

California is treated from 1988; 38 contiguous-US donor states; the 38x38 donor
adjacency (``california_W_matrix.csv``) and California's contiguity vector
(``california_w_vector.csv``) are vendored from the tutorial's
``california_smoking.rda``. Outcome ``cigsale``; covariate ``retprice``;
``p_factors=1``; ``step_rho=0.01`` -- the tutorial's Stage-3 configuration.

What is cross-validated, and why not rho in the full model
----------------------------------------------------------
The spatial core is cross-implementation exact: with ``p_factors=0`` and no
covariate, mlsynth's ``rho`` matches the tutorial's Rcpp to four decimals
(mlsynth 0.849 vs R 0.8492). The estimand reproduces: the SAR ATT is ~-17 packs
per capita (R -16.44), the spatial correction is more negative than the
SUTVA-imposed SCM (spillover to donors inflates the classical estimate), and
Nevada -- California's border state with the textbook cross-border cigarette
flow -- dominates the spillover ranking by an order of magnitude in both codes.

The full-model ``rho`` is deliberately NOT pinned tightly. With an AR(1) latent
factor and the spatial autoregression both explaining cross-sectional co-movement,
``rho`` sits on a near-flat ridge (weakly identified); every full conditional in
the R kernel is algebraically identical to mlsynth's NumPy sampler, yet the two
settle in different regions of the ridge (this tutorial helper near ~0.24, mlsynth
near ~0.41) while the ATT agrees within ~5%. Against the authors' canonical Rcpp
(see ``spillsynth_sudan``) mlsynth's ``rho`` matches to 0.004; this tutorial helper
is an "inspired-by" variant. So the case cross-checks the identified quantities and
only asserts ``rho`` lies in the weak-identification band.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")
_REF = load_reference("spillsynth_prop99_sar")["values"]
TREAT_YEAR = 1988
SEED = 20251022


def _panel():
    panel = pd.read_csv(os.path.join(_BASE, "california_panel.csv"))
    panel["treatment"] = ((panel.state == "California") &
                          (panel.year >= TREAT_YEAR)).astype(int)
    W = pd.read_csv(os.path.join(_BASE, "california_W_matrix.csv"))
    w = pd.read_csv(os.path.join(_BASE, "california_w_vector.csv"))
    return panel, W, w


def _fit(covariates, p_factors, M, burn):
    from mlsynth import SPILLSYNTH

    panel, W, w = _panel()
    cfg = {"df": panel, "outcome": "cigsale", "treat": "treatment",
           "unitid": "state", "time": "year", "method": "sar",
           "spatial_W": W, "spatial_w": w, "p_factors": p_factors,
           "mcmc_iter": M, "mcmc_burn": burn, "step_rho": 0.01,
           "mcmc_seed": SEED, "ci_level": 0.95, "display_graphs": False}
    if covariates:
        cfg["covariates"] = covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH(cfg).fit()


def run() -> dict:
    bare = _fit(None, 0, 6000, 3000)                    # identified spatial core
    full = _fit(["retprice"], 1, 6000, 3000)            # tutorial headline config

    spill = {lab: float(np.mean(v)) for lab, v in full.sar.spillover_panel.items()}
    top_state, top_val = min(spill.items(), key=lambda kv: kv[1])  # most negative
    others_max = max(abs(v) for s, v in spill.items() if s != top_state)

    return {
        # spatial core matches the tutorial's Rcpp to ~4 dp
        "bare_rho_abs_diff_vs_R": float(abs(bare.sar.rho_hat - _REF["bare_rho"])),
        "bare_att": float(bare.att),
        # estimand: SAR ATT reproduces the tutorial (R -16.44)
        "full_att_abs_diff_vs_R": float(abs(full.att - _REF["full_att"])),
        # spatial correction is more negative than SUTVA-imposed SCM
        "spatial_more_negative_than_scm": float(abs(full.att) > abs(full.att_scm)),
        # Nevada dominates the spillover ranking, by an order of magnitude
        "nevada_is_top_spillover": float(top_state == "Nevada"),
        "nevada_dominates": float(abs(top_val) > 3.0 * others_max),
        # full-model rho is weakly identified: assert only the ridge band
        "full_rho_in_weakid_band": float(0.15 <= full.sar.rho_hat <= 0.55),
    }


# Deterministic (seeded MCMC). Cross-validates mlsynth's SAR against the Mendez /
# Sakaguchi-Tagawa California Prop 99 tutorial Rcpp reference on the IDENTIFIED
# quantities: the spatial core matches to ~4 dp, the ATT reproduces within ~5%,
# the spatial correction exceeds the classical SCM, and Nevada dominates the
# spillover ranking. Full-model rho is weakly identified (AR(1) factor vs spatial
# ridge) and only checked to lie in the band spanning R (~0.24) and mlsynth (~0.41).
EXPECTED = {
    "bare_rho_abs_diff_vs_R": (0.0, 0.01),        # mlsynth 0.849 vs R 0.8492
    "bare_att": (-22.0, 1.0),                     # identified spatial-core ATT
    "full_att_abs_diff_vs_R": (0.0, 1.5),         # mlsynth ~-17 vs R -16.44
    "spatial_more_negative_than_scm": (1.0, 0.0),
    "nevada_is_top_spillover": (1.0, 0.0),
    "nevada_dominates": (1.0, 0.0),
    "full_rho_in_weakid_band": (1.0, 0.0),
}


def comparison() -> dict:
    """mlsynth SAR vs the tutorial's Rcpp reference on the identified quantities."""
    bare = _fit(None, 0, 6000, 3000)
    full = _fit(["retprice"], 1, 6000, 3000)
    spill = {lab: float(np.mean(v)) for lab, v in full.sar.spillover_panel.items()}
    nevada = spill.get("Nevada", float("nan"))
    rows = [
        {"quantity": "bare_rho", "mlsynth": round(float(bare.sar.rho_hat), 4),
         "reference": _REF["bare_rho"]},
        {"quantity": "full_att", "mlsynth": round(float(full.att), 4),
         "reference": _REF["full_att"]},
        {"quantity": "nevada_spillover", "mlsynth": round(nevada, 4),
         "reference": _REF["nevada_spillover"]},
        {"quantity": "full_rho (weakly identified)",
         "mlsynth": round(float(full.sar.rho_hat), 4), "reference": _REF["full_rho"]},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SPILLSYNTH",
                         "config": {"method": "sar", "outcome": "cigsale",
                                    "covariates": ["retprice"], "p_factors": 1,
                                    "treat": "treatment", "unitid": "state",
                                    "time": "year"}},
        "reference": {"impl": "Mendez tutorial Rcpp sc_spillover (cmg777)",
                      "version": "carlos-mendez.org/post/r_sc_bayes_spatial (fetched 2026-07-09)"},
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
