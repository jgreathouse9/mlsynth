"""Cross-validation: PCR-SC confidence intervals (Shen et al.) and their coverage.

Validates mlsynth's frequentist PCR-SC confidence intervals against the authors'
reference implementation at https://github.com/deshen24/panel-data-regressions
(the *Same Root Different Leaves* paper, Shen et al.). Two checks:

1. **Variance cross-validation** -- mlsynth's ported variance estimators
   (:func:`mlsynth.utils.clustersc_helpers.pcr.inference._var_homo` / ``_var_jack``)
   must equal the reference ``var.py`` cell-for-cell on identical inputs.
2. **Coverage validity** -- reproducing the repo's ``simulation.py`` Monte Carlo
   (calibrated to the Proposition 99 panel), the **doubly-robust (DR)** variance
   attains ~95% coverage for *all three* estimands (μ_hz, μ_vt, μ_dr), whereas a
   single-source variance under-covers the estimand it is not designed for --
   the paper's central message (e.g. the vertical variance covers the horizontal
   estimand only ~63% of the time).

Provenance
----------
* Reference: deshen24/panel-data-regressions @ 51e2170 (pinned), cloned on
  demand by :mod:`benchmarks.reference.clone_panel_regressions`; no licence, so
  imported (``var`` / ``regr`` / ``rank``) and its bundled Prop 99 data are used
  from the clone, never vendored.
* DGP: the repo's own ``simulation.py`` calibration -- low-rank approximation of
  the donor pre-outcomes, OLS-fit response models, Gaussian resampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference.clone_panel_regressions import ensure_clone, import_reference
from mlsynth.utils.clustersc_helpers.pcr.inference import _var_homo, _var_jack

DATASET = "prop99"
TREATED = "California"
SPECTRAL_ENERGY = 0.999
N_MATCH = 30        # samples for the var cell-by-cell cross-validation
MATCH_SEED = 123
N_ITERS = 500       # coverage Monte Carlo iterations
COVERAGE_SEED = 0
Z = 1.96


def _calibrate(repo, rank_mod):
    """Reproduce simulation.py's calibration on the Prop 99 panel."""
    from sklearn.linear_model import LinearRegression

    base = repo / "data" / DATASET
    df_pre = pd.read_csv(base / "pre_outcomes.csv")
    df_post = pd.read_csv(base / "post_outcomes.csv")
    units = df_post.unit.unique()
    donors = units[units != TREATED]
    post_cols = list(df_post.drop(columns=["unit"]).columns)

    Y0_obs = df_pre.loc[df_pre["unit"].isin(donors)].drop(columns=["unit"]).values
    y_n_obs = df_pre.loc[df_pre["unit"] == TREATED].drop(columns=["unit"]).values.flatten()
    y_t_obs = df_post.loc[df_post["unit"].isin(donors), post_cols[0]].values.flatten()

    s = np.linalg.svd(Y0_obs, compute_uv=False)
    k = rank_mod.spectral_rank(s, t=SPECTRAL_ENERGY)
    Y0, Hu, Hv, Hu_perp, Hv_perp = rank_mod.svt(Y0_obs, max_rank=k)

    regr = LinearRegression(fit_intercept=False)
    alpha = regr.fit(Y0_obs, y_t_obs).coef_
    beta = regr.fit(Y0_obs.T, y_n_obs).coef_
    y_n = Y0.T @ beta
    y_t = Y0 @ alpha
    N0, T0 = Y0.shape
    sig_t = np.diag((np.linalg.norm(Hu_perp @ y_t_obs) ** 2 / (N0 - k)) * np.ones(N0))
    sig_n = np.diag((np.linalg.norm(Hv_perp @ y_n_obs) ** 2 / (T0 - k)) * np.ones(T0))
    return dict(Y0=Y0, Hu=Hu, Hv=Hv, Hu_perp=Hu_perp, Hv_perp=Hv_perp,
                alpha=alpha, beta=beta, y_n=y_n, y_t=y_t, sig_n=sig_n, sig_t=sig_t,
                regr=regr)


def run() -> dict:
    repo = ensure_clone()
    ref_var, _ref_regr, ref_rank = import_reference()
    c = _calibrate(repo, ref_rank)
    Y0, alpha, beta = c["Y0"], c["alpha"], c["beta"]
    Hu, Hv, Hu_perp, Hv_perp = c["Hu"], c["Hv"], c["Hu_perp"], c["Hv_perp"]
    y_n, y_t, sig_n, sig_t, regr = c["y_n"], c["y_t"], c["sig_n"], c["sig_t"], c["regr"]

    # --- (1) variance cross-validation: mlsynth vs reference, cell-by-cell ---
    d_homo = d_jack = 0.0
    np.random.seed(MATCH_SEED)
    for _ in range(N_MATCH):
        yn = np.random.multivariate_normal(y_n, sig_n)
        yt = np.random.multivariate_normal(y_t, sig_t)
        ah = regr.fit(Y0, yt).coef_
        bh = regr.fit(Y0.T, yn).coef_
        rh = ref_var.var_est(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp, v_alg="homoskedastic")
        mh = _var_homo(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp)
        rj = ref_var.var_est(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp, v_alg="jackknife")
        mj = _var_jack(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp)
        d_homo = max(d_homo, max(abs(a - b) for a, b in zip(rh, mh)))
        d_jack = max(d_jack, max(abs(a - b) for a, b in zip(rj, mj)))

    # --- (2) coverage Monte Carlo using mlsynth's variance estimators ---
    np.random.seed(COVERAGE_SEED)
    cp = {v: {e: 0 for e in ("hz", "vt", "dr")} for v in ("hz", "vt", "dr")}
    for _ in range(N_ITERS):
        yn = np.random.multivariate_normal(y_n, sig_n)
        yt = np.random.multivariate_normal(y_t, sig_t)
        mu = {"hz": float(np.dot(yn, Hv @ alpha)),
              "vt": float(np.dot(yt, Hu @ beta)),
              "dr": float(np.dot(alpha, Y0.T @ beta))}
        ah = regr.fit(Y0, yt).coef_
        bh = regr.fit(Y0.T, yn).coef_
        pred = float(np.dot(yn, ah))
        vh, vv, trA = _var_homo(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp)
        var = {"hz": vh, "vt": vv, "dr": max(0.0, vh + vv - trA)}
        for vn, vval in var.items():
            se = Z * np.sqrt(max(vval, 0.0))
            for en, mval in mu.items():
                if pred - se <= mval <= pred + se:
                    cp[vn][en] += 1

    cov = {vn: {en: cp[vn][en] / N_ITERS for en in cp[vn]} for vn in cp}
    return {
        "var_homo_match_max_abs_diff": float(d_homo),
        "var_jack_match_max_abs_diff": float(d_jack),
        "dr_coverage_mu_hz": cov["dr"]["hz"],
        "dr_coverage_mu_dr": cov["dr"]["dr"],
        "dr_min_coverage": float(min(cov["dr"].values())),
        "vt_coverage_mu_hz": cov["vt"]["hz"],
    }


def comparison() -> dict:
    """mlsynth's ported variance estimators vs the authors' ``var.var_est``.

    The exact (cell-by-cell) cross-validation side of the benchmark: on one
    calibrated draw from the Prop 99 DGP, lay mlsynth's :func:`_var_homo` /
    :func:`_var_jack` next to the reference ``var.var_est`` component by
    component (each returns the horizontal variance, vertical variance, and the
    cross trace ``tr(A)``). Skips (via ``ensure_clone`` /
    ``import_reference``) when git or the network is unavailable.
    """
    repo = ensure_clone()                       # skips if the clone is blocked
    ref_var, _ref_regr, ref_rank = import_reference()
    c = _calibrate(repo, ref_rank)
    Y0, regr = c["Y0"], c["regr"]
    Hu_perp, Hv_perp = c["Hu_perp"], c["Hv_perp"]
    y_n, y_t, sig_n, sig_t = c["y_n"], c["y_t"], c["sig_n"], c["sig_t"]

    np.random.seed(MATCH_SEED)
    yn = np.random.multivariate_normal(y_n, sig_n)
    yt = np.random.multivariate_normal(y_t, sig_t)
    ah = regr.fit(Y0, yt).coef_
    bh = regr.fit(Y0.T, yn).coef_

    rh = ref_var.var_est(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp, v_alg="homoskedastic")
    mh = _var_homo(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp)
    rj = ref_var.var_est(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp, v_alg="jackknife")
    mj = _var_jack(yn, yt, Y0, ah, bh, Hu_perp, Hv_perp)

    comps = ("var_hz", "var_vt", "tr_A")
    rows = []
    for kind, mvals, rvals in (("homo", mh, rh), ("jack", mj, rj)):
        for name, mv, rv in zip(comps, mvals, rvals):
            rows.append({"quantity": f"{kind}/{name}",
                         "mlsynth": round(float(mv), 6),
                         "reference": round(float(rv), 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "ClusterSC (pcr.inference _var_homo/_var_jack)",
                         "config": {"dataset": DATASET, "treated": TREATED,
                                    "spectral_energy": SPECTRAL_ENERGY,
                                    "match_seed": MATCH_SEED}},
        "reference": {"impl": "deshen24/panel-data-regressions var.var_est "
                              "(homoskedastic + jackknife)",
                      "version": "git 51e2170d33463bbf403f23fe8a72cbf66bcc34ef"},
    }


# The variance cross-validation is exact (a line-by-line port), so it is pinned
# at ~0 with a tight 1e-9 band. Coverage probabilities are Monte-Carlo (seed 0,
# 500 iters) and depend on the numpy Gaussian sampler, so they carry wider
# bands; the binding facts are that the doubly-robust variance is ~valid for
# every estimand (dr_min_coverage near 0.95) while the single-source vertical
# variance under-covers the horizontal estimand (well below nominal).
EXPECTED = {
    "var_homo_match_max_abs_diff": (0.0, 1e-9),
    "var_jack_match_max_abs_diff": (0.0, 1e-9),
    "dr_coverage_mu_hz": (0.952, 0.06),
    "dr_coverage_mu_dr": (0.916, 0.06),
    "dr_min_coverage": (0.916, 0.07),     # DR stays approximately valid (>~0.85)
    "vt_coverage_mu_hz": (0.632, 0.15),   # single-source VT under-covers (<~0.78)
}
