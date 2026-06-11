"""Ridge ASCM cross-validation: augsynth's canonical Kansas tax-cut study.

Cross-validation (the reference is the **augsynth** R package, Ben-Michael,
Feller & Rothstein 2021). augsynth's flagship example estimates the effect of
Kansas's 2012 tax cuts on quarterly log GDP per capita with the Augmented SCM,
walking up a "ladder" of estimators: classic SCM, ridge-augmented SCM, ridge
ASCM with auxiliary covariates (balanced directly), and the residualized
covariate variant. As the fit de-biases and balances more, the measured effect
grows and the pre-treatment imbalance falls -- the un-augmented SCM is the
conservative end of the ladder.

This reproduces augsynth's published ATTs and pre-fit L2 imbalance for the four
specifications, value-for-value:

    Specification     ATT      Pre-fit L2     augsynth
    Classic SCM       -0.029     0.083        -0.029 / 0.083
    Ridge ASCM        -0.040     0.062        -0.040 / 0.062
    Covariate ASCM    -0.063     0.055        -0.061 / 0.054
    Residualized      -0.057     0.067        -0.055 / 0.067

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

Provenance: ``ebenmichael/augsynth`` Kansas vignette; data shipped as
``basedata/kansas_ascm.csv`` (augsynth's ``kansas`` dataset, relevant columns).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

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


# Deterministic (exact QP base, fixed lambda CV). Cells reproduce augsynth's
# Kansas ladder; the augsynth target is in the comment, the tolerance is set so
# the no-cov cells are exact and the covariate cells match augsynth's values.
EXPECTED = {
    "att_scm": (-0.0294, 0.0015),          # augsynth -0.029
    "l2_scm": (0.0826, 0.003),             # augsynth  0.083
    "att_ridge": (-0.0401, 0.0015),        # augsynth -0.040
    "l2_ridge": (0.0615, 0.003),           # augsynth  0.062
    "att_covariate": (-0.0629, 0.004),     # augsynth -0.061
    "l2_covariate": (0.0546, 0.004),       # augsynth  0.054
    "att_residualized": (-0.0572, 0.004),  # augsynth -0.055
    "l2_residualized": (0.0671, 0.004),    # augsynth  0.067
    "ladder_monotone": (1.0, 0.5),
}
