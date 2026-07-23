"""Cross-validation: VanillaSC vs Jens Hainmueller's R ``Synth`` (j-hai/Synth) on
Prop 99, point estimates *and* the split-conformal band.

Path A / cross-validation, input scenario 3 (the authors' runnable R package).
Two things are checked against ``Synth`` 1.2.0 -- the maintained package with
Hainmueller's new ``synth_inference()`` -- on the California Proposition 99 panel
under the canonical ADH (2010) predictor spec:

1. The synthetic control itself. ``VanillaSC(backend="mscmt")`` reproduces the
   package's ``synth()`` donor weights to about 0.02 and the ATT to a fraction of
   a pack. mlsynth reaches a *lower* pre-period RMSPE (1.754 vs 1.791) -- the
   MSCMT/Malo thesis that the nested V-search can stop short of the global
   optimum, here against the original ``Synth`` solver itself.

2. The split-conformal band. ``inference="conformal_split"`` is mlsynth's port of
   ``synth_inference(method="conformal")``: a constant half-width ``q``, the
   ``ceil((n+1)(1-alpha))``-th order statistic of the absolute pre-period gaps.
   On a *shared* set of gaps the two constructions agree value-for-value --
   feeding the package's own pre-period gaps to ``split_conformal_quantile``
   returns its ``conformal_q`` (6.113436) exactly. On its own (better-fitting)
   synthetic mlsynth's band is slightly tighter (``q`` = 5.90 vs 6.11), a direct
   consequence of the lower pre-period RMSPE.

Provenance
----------
* Data: ``basedata/augmented_cali_long.csv`` -- the ADH Prop 99 panel (39 states,
  1970-2000), the same file ``vanillasc_prop99`` uses. California is ``stateno``
  5; predictors ``loginc`` / ``p_cig`` / ``pct15-24`` averaged over 1980-1988,
  ``pc_beer`` over 1984-1988, plus ``cigsale`` at 1975 / 1980 / 1988.
* Reference: R ``Synth`` 1.2.0 (Hainmueller, https://github.com/j-hai/Synth) via
  ``benchmarks/R/synth_jhai_prop99.R`` (``synth()`` + ``synth_inference()``),
  captured in ``benchmarks/reference/synth_jhai_prop99/`` with provenance pinned.
  R does not install from CRAN in CI; the reference is baked and read here.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "augmented_cali_long.csv")
_REF = load_reference("synth_jhai_prop99")
_DONORS = ("Utah", "Nevada", "Montana", "Colorado", "Connecticut")
_LAGS = (1975, 1980, 1988)


def _fit():
    from mlsynth import VanillaSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    for L in _LAGS:
        d[f"cig{L}"] = d["state"].map(d[d.year == L].set_index("state")["cigsale"])
    covs = ["loginc", "p_cig", "pct15-24", "pc_beer",
            "cig1975", "cig1980", "cig1988"]
    windows = {"loginc": (1980, 1988), "p_cig": (1980, 1988),
               "pct15-24": (1980, 1988), "pc_beer": (1984, 1988),
               "cig1975": (1975, 1975), "cig1980": (1980, 1980),
               "cig1988": (1988, 1988)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treated", "unitid": "state",
            "time": "year", "covariates": covs, "covariate_windows": windows,
            "backend": "mscmt", "canonical_v": "min.loss.w", "seed": 0,
            "inference": "conformal_split", "display_graphs": False,
        }).fit()


def run() -> dict:
    from mlsynth.utils.inferutils import split_conformal_quantile

    res = _fit()
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    v = _REF["values"]
    weight_dev = max(abs(w.get(s, 0.0) - v[f"w_{s}"]) for s in _DONORS)

    # Value-for-value: mlsynth's split-conformal quantile on the PACKAGE's own
    # pre-period gaps must return the package's conformal_q exactly (same formula).
    q_construct = split_conformal_quantile(_REF["gap_pre"], 0.05)

    return {
        # --- cross-validation of the synthetic control (solver-difference band) ---
        "weight_max_abs_dev_vs_jhai": weight_dev,
        "att_abs_diff_vs_jhai": abs(float(res.effects.att) - v["att"]),
        # mlsynth reaches a strictly lower pre-period RMSPE (Malo/MSCMT thesis)
        "prefit_improvement_over_jhai": v["pre_rmspe"] - float(res.fit_diagnostics.rmse_pre),
        # --- split-conformal construction: value-for-value with j-hai/Synth ---
        "split_conformal_q_value_match": abs(q_construct - v["conformal_q"]),
        # --- deterministic mlsynth descriptors (regression guards) ---
        "mlsynth_att": float(res.effects.att),
        "mlsynth_pre_rmspe": float(res.fit_diagnostics.rmse_pre),
        "mlsynth_conformal_q": float(res.inference.details["conformal_q"]),
        "n_donors_used": float(sum(1 for x in w.values() if x > 1e-3)),
    }


def comparison() -> dict:
    """mlsynth VanillaSC vs j-hai/Synth, quantity by quantity."""
    from mlsynth.utils.inferutils import split_conformal_quantile

    res = _fit()
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    v = _REF["values"]
    rows = [{"quantity": f"w[{s}]", "mlsynth": round(w.get(s, 0.0), 4),
             "reference": round(v[f"w_{s}"], 4)} for s in _DONORS]
    rows += [
        {"quantity": "ATT", "mlsynth": round(float(res.effects.att), 3),
         "reference": round(v["att"], 3)},
        {"quantity": "pre_RMSPE", "mlsynth": round(float(res.fit_diagnostics.rmse_pre), 3),
         "reference": round(v["pre_rmspe"], 3)},
        {"quantity": "conformal_q (own synthetic)",
         "mlsynth": round(float(res.inference.details["conformal_q"]), 3),
         "reference": round(v["conformal_q"], 3)},
        {"quantity": "conformal_q (value-for-value, shared gaps)",
         "mlsynth": round(split_conformal_quantile(_REF["gap_pre"], 0.05), 6),
         "reference": round(v["conformal_q"], 6)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC",
                         "config": {"backend": "mscmt",
                                    "inference": "conformal_split"}},
        "reference": {"impl": "R Synth (j-hai/Synth, synth + synth_inference)",
                      "version": "Synth 1.2.0"},
    }


# Tolerances. The weight/ATT gap is the two nested V-solvers' spread on Prop 99
# (deterministic at seed=0), centred on the measured value; the pre-fit
# improvement is asserted positive (mlsynth reaches the lower loss). The
# split-conformal value-match is exact -- the two implement the same order
# statistic -- so its tolerance covers only float noise. The mlsynth-side
# descriptors are deterministic regression guards.
EXPECTED = {
    "weight_max_abs_dev_vs_jhai": (0.020, 0.010),   # Montana/Colorado split
    "att_abs_diff_vs_jhai": (0.26, 0.25),
    "prefit_improvement_over_jhai": (0.037, 0.037),  # > 0: mlsynth fits better
    "split_conformal_q_value_match": (0.0, 1e-6),    # value-for-value
    "mlsynth_att": (-18.98, 0.5),
    "mlsynth_pre_rmspe": (1.754, 0.1),
    "mlsynth_conformal_q": (5.90, 0.15),
    "n_donors_used": (5.0, 1.0),
}
