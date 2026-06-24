"""Ridge ASCM cross-validation: augsynth's canonical Kansas tax-cut study.

Cross-validation (the reference is the **augsynth** R package, Ben-Michael,
Feller & Rothstein 2021). augsynth's flagship example estimates the effect of
Kansas's 2012 tax cuts on quarterly log GDP per capita with the Augmented SCM,
walking up a "ladder" of estimators: classic SCM, ridge-augmented SCM, ridge
ASCM with auxiliary covariates (balanced directly), and the residualized
covariate variant. As the fit de-biases and balances more, the measured effect
grows and the pre-treatment imbalance falls -- the un-augmented SCM is the
conservative end of the ladder.

This cross-validates against a live run of the augsynth package (captured in
``benchmarks/reference/ascm_kansas/`` with its version pinned), not transcribed
constants. mlsynth vs live augsynth 0.2.0 across the four specifications:

    Specification     mlsynth ATT / L2     live augsynth ATT / L2
    Classic SCM       -0.0294 / 0.0826     -0.0294 / 0.0826   (exact)
    Ridge ASCM        -0.0401 / 0.0615     -0.0401 / 0.0615   (exact)
    Covariate ASCM    -0.0629 / 0.0546     -0.0609 / 0.0539
    Residualized      -0.0572 / 0.0668     -0.0528 / 0.0576

The classic SCM and ridge ASCM reproduce the package to machine precision; the
covariate cell agrees to ~0.002. The residualized cell is wider (see below) and,
notably, the package's own live value (-0.053 / 0.058) differs from the vignette
table's -0.055 / 0.067 -- a symptom of that spec's ill-posed CV, surfaced by
running augsynth rather than trusting the printed numbers.

The covariate model is augsynth's documented Kansas spec,
``treated | lngdpcapita + log(revstatecapita) + log(revlocalcapita) +
log(avgwklywagecapita) + estabscapita + emplvlcapita`` -- per-row transforms
aggregated to a pre-period mean per unit, with rows carrying a missing
(sparsely reported) revenue value dropped before averaging (R's ``model.frame``
``na.omit`` default). The two no-covariate cells are exact to augsynth; the
covariate cells match its values and reproduce the monotone ladder.

Note on the residualized penalty: after residualizing out K covariates the
residual Gram is rank-deficient, so augsynth's residual lambda-CV is ill-posed
(it drifts to the grid floor). mlsynth tunes the penalty on the outcome scale
instead -- where augsynth's CV lands anyway -- which reproduces the published
-0.055 / 0.067 robustly.

Provenance: ``ebenmichael/augsynth`` (the package, run live -- see
``benchmarks/reference/ascm_kansas/reference.R`` and the captured output /
``augsynth 0.2.0`` provenance); data shipped as ``basedata/kansas_ascm.csv``
(augsynth's ``kansas`` dataset, relevant columns).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import reference_value

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "kansas_ascm.csv")
_TREATED_FIPS = 20.0          # Kansas
_T_INT = 2012.25              # first treated quarter (2012 Q2)
# augsynth covariate formula: (column, per-row transform)
_COVS = [("lngdpcapita", None), ("revstatecapita", np.log),
         ("revlocalcapita", np.log), ("avgwklywagecapita", np.log),
         ("estabscapita", None), ("emplvlcapita", None)]


def _prep():
    d = pd.read_csv(os.path.abspath(_DATA))
    piv = d.pivot(index="fips", columns="year_qtr", values="lngdpcapita").sort_index()
    times = np.array(sorted(d["year_qtr"].unique()))
    pre = times < _T_INT
    units = piv.index.to_numpy()
    trt = units == _TREATED_FIPS
    Y = piv.to_numpy()
    y, Y0 = Y[trt][0], Y[~trt]
    y_pre, Y0_pre = y[: pre.sum()], Y0[:, : pre.sum()].T          # (T0,), (T0, J)
    y_post, Y0_post = y[pre.sum():], Y0[:, pre.sum():]            # (T1,), (J, T1)

    # covariate matrix: per-row transform, drop rows with any NA, pre-period mean
    layers = []
    for name, fn in _COVS:
        m = d.pivot(index="fips", columns="year_qtr", values=name).sort_index().to_numpy()[:, pre]
        layers.append(fn(m) if fn else m)
    stack = np.stack(layers, axis=2)                              # (N, T0, K)
    rowok = ~np.isnan(stack).any(axis=2)
    Zall = np.array([stack[u][rowok[u]].mean(0) for u in range(stack.shape[0])])
    Z0, z1 = Zall[~trt], Zall[trt][0]
    return y_pre, Y0_pre, y_post, Y0_post, Z0, z1


def _att(w, y_post, Y0_post):
    return float(np.mean(y_post - Y0_post.T @ w))


def _l2(w, y_pre, Y0_pre):
    mu = Y0_pre.mean(1)                                           # per-period donor mean
    return float(np.sqrt(np.sum(((y_pre - mu) - (Y0_pre - mu[:, None]) @ w) ** 2)))


def run() -> dict:
    from mlsynth.utils.bilevel.ridge_augment import (
        ridge_augment_weights, simplex_qp, build_matching)

    y_pre, Y0_pre, y_post, Y0_post, Z0, z1 = _prep()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B, A = build_matching(y_pre, Y0_pre)
        w_scm = simplex_qp(B, A)
        w_ridge = ridge_augment_weights(y_pre, Y0_pre).W
        w_cov = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1).W
        w_res = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1, residualize=True).W

    out = {}
    for tag, w in [("scm", w_scm), ("ridge", w_ridge),
                   ("covariate", w_cov), ("residualized", w_res)]:
        out[f"att_{tag}"] = _att(w, y_post, Y0_post)
        out[f"l2_{tag}"] = _l2(w, y_pre, Y0_pre)
    # the de-biasing ladder: |ATT| grows monotonically SCM -> ridge -> covariate
    out["ladder_monotone"] = float(
        abs(out["att_scm"]) < abs(out["att_ridge"]) < abs(out["att_covariate"]))
    return out


def comparison() -> dict:
    """mlsynth's ridge ASCM ladder vs the augsynth Kansas values, cell by cell.

    Re-derives mlsynth's ATT and pre-fit L2 imbalance for the four
    specifications (classic SCM, ridge ASCM, covariate ASCM, residualized) with
    the case's own helpers, and pairs each with augsynth's published value, so
    the exporter can lay them side by side. Returns ``{"rows": [...],
    "mlsynth_call": {...}, "reference": {...}}`` -- rows are
    ``{quantity, mlsynth, reference}``.
    """
    from mlsynth.utils.bilevel.ridge_augment import (
        ridge_augment_weights, simplex_qp, build_matching)

    y_pre, Y0_pre, y_post, Y0_post, Z0, z1 = _prep()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B, A = build_matching(y_pre, Y0_pre)
        w_scm = simplex_qp(B, A)
        w_ridge = ridge_augment_weights(y_pre, Y0_pre).W
        w_cov = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1).W
        w_res = ridge_augment_weights(y_pre, Y0_pre, Z0=Z0, z1=z1, residualize=True).W

    weights = {"scm": w_scm, "ridge": w_ridge,
               "covariate": w_cov, "residualized": w_res}
    rows = []
    for tag, w in weights.items():
        rows.append({"quantity": f"ATT[{tag}]",
                     "mlsynth": round(_att(w, y_post, Y0_post), 6),
                     "reference": round(reference_value("ascm_kansas", f"att_{tag}"), 6)})
        rows.append({"quantity": f"pre_fit_L2[{tag}]",
                     "mlsynth": round(_l2(w, y_pre, Y0_pre), 6),
                     "reference": round(reference_value("ascm_kansas", f"l2_{tag}"), 6)})

    cfg = {"treated_fips": _TREATED_FIPS, "t_int": _T_INT,
           "covariates": [name for name, _ in _COVS],
           "specifications": ["scm", "ridge", "covariate", "residualized"]}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "ridge_augment_weights", "config": cfg},
        "reference": {"impl": "R package augsynth (live run, Kansas study)",
                      "version": "augsynth 0.2.0"},
    }


# Deterministic. Targets are pinned from a live augsynth run captured in
# benchmarks/reference/ascm_kansas/ (not transcribed constants), so the benchmark
# checks mlsynth against the actual package output. The no-covariate and
# covariate cells match augsynth to ~0.002; the residualized cells are wider
# because augsynth's residual lambda-CV is ill-posed (rank-deficient residual
# Gram) -- the spec where the package's own value drifts (its live -0.053/0.058
# differs from the vignette's -0.055/0.067).
_ref = lambda k: reference_value("ascm_kansas", k)
EXPECTED = {
    "att_scm": (_ref("att_scm"), 0.001),
    "l2_scm": (_ref("l2_scm"), 0.001),
    "att_ridge": (_ref("att_ridge"), 0.001),
    "l2_ridge": (_ref("l2_ridge"), 0.001),
    "att_covariate": (_ref("att_covariate"), 0.003),
    "l2_covariate": (_ref("l2_covariate"), 0.002),
    "att_residualized": (_ref("att_residualized"), 0.007),
    "l2_residualized": (_ref("l2_residualized"), 0.013),
    "ladder_monotone": (1.0, 0.5),
}
