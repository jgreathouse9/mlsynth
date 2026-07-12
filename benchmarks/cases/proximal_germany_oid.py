"""Cross-validation: mlsynth PROXIMAL (PIOID) vs the authors' own manuscript code.

Cross-validation against the canonical implementation. mlsynth's over-identified
proximal-inference method (:class:`mlsynth.PROXIMAL`, ``methods=["PIOID"]``) is
checked, value for value, against the authors' own replication of Shi, Li, Yu,
Miao, Kuchibhotla, Hu & Tchetgen Tchetgen (2026), *"Theory for Identification and
Inference with Synthetic Controls: A Proximal Causal Inference Framework"* (JASA)
-- `KenLi93/proximal_sc_manuscript <https://github.com/KenLi93/proximal_sc_manuscript>`_
(``NC_nocov`` for the point estimate, ``NC_nocov_gmm`` for the GMM/Newey-West
interval) -- on the paper's 1990 German-reunification application.

The estimand is the over-identified proximal outcome bridge: the 6 donor
countries ``W`` (Austria, Italy, Japan, Netherlands, Switzerland, USA -- the
units Cattaneo et al. give nonzero SC weight) are instrumented by the remaining
10 OECD countries ``Z``, a single GDP-per-capita outcome throughout,

    omega = (W'Z Z'W)^{-1} W'Z Z'Y   (pre-period 1960-1990, no covariates),
    ATT   = mean over 1991-2003 of (Y - W omega).

This is distinct from mlsynth's just-identified variable-proxy ``PI`` (one proxy
variable per donor, validated on Panic 1907 against ``freshtaste/proximal``);
here the proxies are a *distinct set of donor units*, which is why the method is
``PIOID`` (``donors`` = W, ``outcome_instruments`` = Z).

Value for value
---------------
The one-step-GMM (identity-weight) optimum is unique, so mlsynth reproduces the
paper's PI headline exactly: ATT -1709 USD. The ATT standard error follows the
authors' ``NC_nocov_gmm`` -- a joint GMM over ``theta = (tau, omega)`` with the
over-identified sandwich and a Bartlett/Newey-West HAC at the manuscript's lag
``q = 10`` (``pioid_hac_lag=10``, mlsynth's default) -- reproducing the paper's
GMM PI 90% interval (-2806, -616) USD. The ``scpi_germany`` GDP is in thousands
of USD, so the pinned values are in thousands (-1.7091, etc.).

Reference (live captured run)
-----------------------------
``benchmarks/reference/proximal_germany_oid/reference.R`` carries the authors'
method and runs it on the in-repo ``scpi_germany``; its ATT, ATT SE, 90% CI and
donor weights are captured under ``benchmarks/reference/proximal_germany_oid/``
and pinned here via :func:`reference_value` / :func:`load_reference`. The upstream
repo has no licence, so the method is reproduced on the public in-repo data
rather than vendored (as with ``sbc_germany``). Regenerate with
``python benchmarks/reference/generate.py proximal_germany_oid``.

Provenance
----------
* Data: ``basedata/scpi_germany.csv`` -- the Abadie, Diamond & Hainmueller (2015)
  / Cattaneo et al. (2025) West German reunification panel (17 OECD countries,
  1960-2003; West Germany treated from 1991). Outcome ``gdp`` (per-capita GDP,
  thousands of USD).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_TREATED = "West Germany"
_W = ["Austria", "Italy", "Japan", "Netherlands", "Switzerland", "USA"]

_REF = load_reference("proximal_germany_oid")
_REF_WEIGHTS = _REF["weights"]
REF_ATT = reference_value("proximal_germany_oid", "att")
REF_SE = reference_value("proximal_germany_oid", "att_se")
REF_CI_LB = reference_value("proximal_germany_oid", "ci90_lb")
REF_CI_UB = reference_value("proximal_germany_oid", "ci90_ub")
REF_ATT_CPI = reference_value("proximal_germany_oid", "att_cpi")

_Z90 = 1.6448536269514722


def _mlsynth_pioid(simplex: bool = False):
    from mlsynth import PROXIMAL

    df = pd.read_csv(_BASE / "scpi_germany.csv")[["country", "year", "gdp"]].dropna()
    df["treat"] = ((df["country"] == _TREATED) & (df["year"] > 1990)).astype(int)
    Z = [c for c in df["country"].unique() if c not in _W + [_TREATED]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = PROXIMAL({
            "df": df, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "donors": _W, "outcome_instruments": Z,
            "methods": ["PIOID"], "pioid_simplex": simplex,
            "display_graphs": False,
        }).fit().methods["PIOID"]
    weights = {str(k): float(v) for k, v in fit.donor_weights.items()}
    se = float(fit.att_se) if fit.att_se is not None else float("nan")
    return weights, float(fit.att), se


def run() -> dict:
    w_ml, att_ml, se_ml = _mlsynth_pioid()
    lo, hi = att_ml - _Z90 * se_ml, att_ml + _Z90 * se_ml
    w_diff = max(abs(w_ml.get(d, 0.0) - wr) for d, wr in _REF_WEIGHTS.items())

    # Constrained proximal inference (cPI): simplex-constrained bridge.
    wc_ml, att_cpi_ml, _ = _mlsynth_pioid(simplex=True)
    wc = np.array(list(wc_ml.values()))

    return {
        "mls_att": att_ml,
        "mls_att_se": se_ml,
        # mlsynth PIOID vs the authors' manuscript code, thousands of USD.
        "att_abs_diff_vs_manuscript": float(abs(att_ml - REF_ATT)),
        "se_abs_diff_vs_manuscript": float(abs(se_ml - REF_SE)),
        "ci90_lb_abs_diff": float(abs(lo - REF_CI_LB)),
        "ci90_ub_abs_diff": float(abs(hi - REF_CI_UB)),
        "weight_max_abs_diff_vs_manuscript": float(w_diff),
        "att_negative": 1.0 if att_ml < 0 else 0.0,
        # cPI (simplex): matches the paper's -1719 USD; weights on the simplex.
        "mls_att_cpi": att_cpi_ml,
        "att_cpi_abs_diff_vs_manuscript": float(abs(att_cpi_ml - REF_ATT_CPI)),
        "cpi_weights_on_simplex": 1.0 if (wc >= -1e-8).all() and abs(wc.sum() - 1.0) < 1e-6 else 0.0,
    }


def comparison() -> dict:
    """mlsynth PIOID vs ``KenLi93/proximal_sc_manuscript``, quantity by quantity.

    Lays the mlsynth over-identified PI fit against the authors' own code on the
    German-reunification panel: the ATT, its GMM/Newey-West standard error, the
    90% CI bounds, and the donor weights. The reference side is a live captured
    ``NC_nocov`` + ``NC_nocov_gmm`` run in
    ``benchmarks/reference/proximal_germany_oid/`` (values in thousands of USD).
    """
    w_ml, att_ml, se_ml = _mlsynth_pioid()
    lo, hi = att_ml - _Z90 * se_ml, att_ml + _Z90 * se_ml

    _wc, att_cpi_ml, _ = _mlsynth_pioid(simplex=True)
    rows = [
        {"quantity": "ATT (PI)", "mlsynth": round(att_ml, 6), "reference": round(REF_ATT, 6)},
        {"quantity": "ATT SE", "mlsynth": round(se_ml, 6), "reference": round(REF_SE, 6)},
        {"quantity": "90% CI lower", "mlsynth": round(lo, 6), "reference": round(REF_CI_LB, 6)},
        {"quantity": "90% CI upper", "mlsynth": round(hi, 6), "reference": round(REF_CI_UB, 6)},
        {"quantity": "ATT (cPI, simplex)", "mlsynth": round(att_cpi_ml, 6),
         "reference": round(REF_ATT_CPI, 6)},
    ]
    for donor, w_ref in sorted(_REF_WEIGHTS.items(), key=lambda kv: -abs(kv[1])):
        rows.append({"quantity": f"weight[{donor}]",
                     "mlsynth": round(float(w_ml.get(donor, 0.0)), 6),
                     "reference": round(float(w_ref), 6)})

    cfg = {"outcome": "gdp", "treat": "treat", "unitid": "country", "time": "year",
           "estimator": "PROXIMAL", "methods": ["PIOID"], "pioid_hac_lag": 10}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PROXIMAL/PIOID (over-identified outcome bridge)",
                         "config": cfg},
        "reference": {"impl": "KenLi93/proximal_sc_manuscript NC_nocov + NC_nocov_gmm "
                              "(over-identified, Newey-West q=10), live run, captured",
                      "version": "Shi et al. (2026) JASA, on in-repo scpi_germany.csv"},
    }


# The one-step-GMM identity-weight optimum is unique, so mlsynth PIOID reproduces
# the paper's PI ATT (-1709 USD, i.e. -1.7091 thousand) exactly; the ATT SE
# follows the authors' NC_nocov_gmm (joint over-identified GMM sandwich, Bartlett
# HAC at the manuscript's Newey-West lag q=10), reproducing the paper's GMM PI 90%
# CI (-2806, -616) USD. Targets are pinned from the live captured R run
# (benchmarks/reference/proximal_germany_oid/) via reference_value/load_reference;
# the *_abs_diff tolerances are the actual mlsynth-vs-R gap (numerical).
EXPECTED = {
    "mls_att": (REF_ATT, 1e-4),                        # -1.7091 (thousand USD)
    "mls_att_se": (REF_SE, 1e-4),                      # 0.6655
    "att_abs_diff_vs_manuscript": (0.0, 1e-4),
    "se_abs_diff_vs_manuscript": (0.0, 1e-4),
    "ci90_lb_abs_diff": (0.0, 1e-4),                   # 90% CI reproduces the paper
    "ci90_ub_abs_diff": (0.0, 1e-4),
    "weight_max_abs_diff_vs_manuscript": (0.0, 1e-3),
    "att_negative": (1.0, 0.0),
    # cPI: the simplex-constrained bridge reproduces the paper's -1719 USD.
    "mls_att_cpi": (REF_ATT_CPI, 1e-3),                # -1.7189 (thousand USD)
    "att_cpi_abs_diff_vs_manuscript": (0.0, 1e-3),
    "cpi_weights_on_simplex": (1.0, 0.0),
}
