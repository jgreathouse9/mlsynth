"""German reunification -- Hsiao & Zhou (2024), LP and FB against their Figures 1-2.

Hsiao, C. and Zhou, Q. (2024), *Panel treatment effects measurement: Factor or
linear projection modelling?*, Journal of Applied Econometrics 39(7):1332-1358,
`10.1002/jae.3081 <https://doi.org/10.1002/jae.3081>`_. Replication package:
`10.15456/jae.2024145.0725725591 <https://doi.org/10.15456/jae.2024145.0725725591>`_.

Section 7 fits the paper's two predictors -- a linear projection of the treated
series on the contemporaneous controls (LP) and a principal-component factor
model (FB) -- to the Abadie, Diamond & Hainmueller (2015) panel, 17 OECD
countries over 1960-2003 with West Germany treated from 1991. It reports the
result as two figures and prints no number, and the replication package ships
data with no code, so the referent here is the figures themselves: both are
vector graphics, and ``benchmarks/reference/hz_germany/digitise_figures.py``
recovers the plotted series from the PDF. That bundle's README documents the two
calibrations and their agreement to 4e-4.

What is compared
----------------
The LP half is a cross-validation and it passes: mlsynth's PDA family already
produces the paper's LP counterfactual, with ``method="fs"`` landing 0.009 log
points from the plotted path and a faithful port of the paper's own eq. (16)
landing 0.030.

The FB half is a negative result, pinned here so it is not rediscovered. The
plotted FB path misses the observed series in sample by RMSE 0.623 log points,
which no principal-component fit of this panel does -- the specified top-five
fit misses by 0.011, and a search over all 6188 five-subsets of the panel's
components gets no closer than 0.563 to the plotted path. Two independent
correct implementations, a faithful port of eqs. (30)-(31) and ``FMA``
(Li & Sonnier 2023), agree with each other to 0.018 and both sit 0.41 from it.

That gap is what drives the paper's headline. Its note 15 opens SE^2_{t,FB}
with the mean squared in-sample residual, so the plotted 95% half-width of 1.300
is close to the 1.96 * 0.623 = 1.221 that the figure's own fit implies. The
claim that the FB intervals cover zero while the LP intervals do not is a
property of that fit, not of the factor approach: with a correct FB the
half-width is 0.047.

The case therefore asserts three things -- that PDA reproduces the LP path, that
the two correct FB implementations agree with each other, and that both differ
from the published FB path by the margin recorded above.

The spike that produced this, and the decision not to build from the paper, are
recorded in ``agents/future_integrations.md`` section 22.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from mlsynth import FMA, PDA
from mlsynth.config_models import FMAConfig, PDAConfig
from mlsynth.utils.datautils import dataprep

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REF = _ROOT / "benchmarks" / "reference" / "hz_germany"

FIRST_POST_YEAR = 1991
N_FACTORS = 5          # note 13: Bai-Ng IC with a maximum of 5 returns 5

EXPECTED = {
    # --- the referent itself, so a bad re-digitisation cannot pass silently ---
    # Figure 2's effects equal the observed series minus Figure 1's paths
    "figures_internally_consistent_maxabs": (0.0, 1e-3),
    # note 13's factor count, recomputed from the panel
    "bai_ng_factor_count": (5.0, 0.0),

    # --- LP: mlsynth already produces the paper's linear projection ---
    "pda_fs_rmse_vs_paper_lp": (0.0086, 0.004),
    "pda_best_rmse_vs_paper_lp": (0.0086, 0.004),
    "hz_lp_port_rmse_vs_paper_lp": (0.0299, 0.010),

    # --- FB: two correct implementations agree, and neither is the figure ---
    "fma_vs_hz_fb_port_rmse": (0.0182, 0.010),
    "hz_fb_port_rmse_vs_paper_fb": (0.4182, 0.060),
    "fma_rmse_vs_paper_fb": (0.4105, 0.060),
    # the published FB path's in-sample miss, and a correct one's
    "paper_fb_pre_rmse": (0.6230, 0.010),
    "correct_fb_pre_rmse_max": (0.0115, 0.005),
    # note 15's interval is consistent with that miss: 1.96 * 0.623 = 1.221
    "paper_fb_halfwidth_over_sigma_implied": (1.0644, 0.060),
    # a correct FB's half-width, against the 1.300 the paper plots
    "correct_fb_halfwidth": (0.0465, 0.020),
}


def _panel() -> pd.DataFrame:
    d = pd.read_stata(_ROOT / "basedata" / "repgermany.dta")
    d["loggdp"] = np.log(d["gdp"].astype(float))
    d["treated"] = ((d.country == "West Germany")
                    & (d.year >= FIRST_POST_YEAR)).astype(int)
    return d


def _pc_loadings(Y_pre: np.ndarray, r: int) -> np.ndarray:
    """Eq. (17) under the paper's ``Sigma_lambda = I_r`` branch, so
    ``Lambda = sqrt(N) V``.

    The counterfactual is invariant to ``Lambda -> Lambda A`` and so would not
    care, but the note-15 interval is not: its third term
    ``(1/T) sigma_1^2 f_t' f_t`` is stated in an ``f`` whose scale is fixed by
    this normalisation, and reading it with an orthonormal ``Lambda`` inflates
    that term by a factor of N.
    """
    _, _, Vt = np.linalg.svd(Y_pre, full_matrices=False)
    return np.sqrt(Y_pre.shape[1]) * Vt[:r].T


def _fb(Y_pre: np.ndarray, Y_post: np.ndarray, r: int):
    """Eqs. (30)-(31) plus the note-15 interval."""
    Lam = _pc_loadings(Y_pre, r)
    lam1, Lc = Lam[0], Lam[1:]
    T0 = Y_pre.shape[0]
    F_pre = Y_pre @ Lam / Y_pre.shape[1]          # eq. (18)
    cf_pre = F_pre @ lam1
    sigma2 = float(np.mean((Y_pre[:, 0] - cf_pre) ** 2))
    gram_inv = np.linalg.inv(Lc.T @ Lc)
    F_post = Y_post[:, 1:] @ Lc @ gram_inv.T
    cf_post = F_post @ lam1
    U_c = Y_pre[:, 1:] - F_pre @ Lc.T
    sandwich = gram_inv @ Lc.T @ (U_c.T @ U_c / T0) @ Lc @ gram_inv
    se = np.sqrt(sigma2 + float(lam1 @ sandwich @ lam1)
                 + sigma2 * np.einsum("ij,ij->i", F_post, F_post) / T0)
    return cf_pre, cf_post, se


def _lp(Y_pre: np.ndarray, Y_post: np.ndarray):
    """Eq. (16) with the Remark 2 intercept, applied out of sample by eq. (32)."""
    X = np.column_stack([np.ones(len(Y_pre)), Y_pre[:, 1:]])
    Xp = np.column_stack([np.ones(len(Y_post)), Y_post[:, 1:]])
    w = np.linalg.solve(X.T @ X, X.T @ Y_pre[:, 0])
    return X @ w, Xp @ w


def _bai_ng_icp2(Y_pre: np.ndarray, r_max: int) -> int:
    T, N = Y_pre.shape
    _, _, Vt = np.linalg.svd(Y_pre, full_matrices=False)
    scores = []
    for r in range(1, r_max + 1):
        V = Vt[:r].T                              # orthonormal, so V V' projects
        resid = Y_pre - Y_pre @ V @ V.T
        scores.append(np.log(float(np.mean(resid ** 2)))
                      + r * ((N + T) / (N * T)) * np.log(min(N, T)))
    return int(np.argmin(scores)) + 1


def run() -> dict:
    warnings.simplefilter("ignore")
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    gold1 = pd.read_csv(_REF / "gold_figure1_paths.csv")
    gold2 = pd.read_csv(_REF / "gold_figure2_effects.csv")

    d = _panel()
    prep = dataprep(d, "country", "year", "loggdp", "treated")
    T0 = prep["pre_periods"]
    Y = np.column_stack([prep["y"], prep["donor_matrix"]])
    Y_pre, Y_post = Y[:T0], Y[T0:]
    observed = gold1["actual_observed"].to_numpy()

    out: dict = {}

    # the referent checks itself: Figure 2 must equal observed minus Figure 1
    consistency = max(
        float(np.abs((observed[T0:] - gold1[f"{k}_counterfactual"].to_numpy()[T0:])
                     - gold2[f"{k}_effect"].to_numpy()).max())
        for k in ("lp", "fb"))
    out["figures_internally_consistent_maxabs"] = consistency
    out["bai_ng_factor_count"] = float(_bai_ng_icp2(Y_pre, 5))

    paper_lp = gold1["lp_counterfactual"].to_numpy()[T0:]
    paper_fb = gold1["fb_counterfactual"].to_numpy()[T0:]

    # --- LP ---
    _, lp_post = _lp(Y_pre, Y_post)
    out["hz_lp_port_rmse_vs_paper_lp"] = rmse(lp_post, paper_lp)

    pda_rmse = {}
    for meth in ("hcw", "fs", "LASSO", "l2"):
        res = PDA(PDAConfig(df=d, unitid="country", time="year", outcome="loggdp",
                            treat="treated", display_graphs=False, method=meth)).fit()
        cf = np.asarray(res.time_series.counterfactual_outcome).ravel()[T0:]
        pda_rmse[meth] = rmse(cf, paper_lp)
    out["pda_fs_rmse_vs_paper_lp"] = pda_rmse["fs"]
    out["pda_best_rmse_vs_paper_lp"] = min(pda_rmse.values())

    # --- FB ---
    fb_pre, fb_post, fb_se = _fb(Y_pre, Y_post, N_FACTORS)
    res = FMA(FMAConfig(df=d, unitid="country", time="year", outcome="loggdp",
                        treat="treated", display_graphs=False)).fit()
    fma_cf = np.asarray(res.time_series.counterfactual_outcome).ravel()

    out["fma_vs_hz_fb_port_rmse"] = rmse(fb_post, fma_cf[T0:])
    out["hz_fb_port_rmse_vs_paper_fb"] = rmse(fb_post, paper_fb)
    out["fma_rmse_vs_paper_fb"] = rmse(fma_cf[T0:], paper_fb)

    paper_fb_pre_rmse = rmse(gold1["fb_counterfactual"].to_numpy()[:T0], observed[:T0])
    out["paper_fb_pre_rmse"] = paper_fb_pre_rmse
    out["correct_fb_pre_rmse_max"] = max(rmse(fb_pre, Y_pre[:, 0]),
                                         rmse(fma_cf[:T0], Y_pre[:, 0]))

    paper_hw = float(((gold2["fb_upper"] - gold2["fb_lower"]) / 2).mean())
    out["paper_fb_halfwidth_over_sigma_implied"] = paper_hw / (1.96 * paper_fb_pre_rmse)
    out["correct_fb_halfwidth"] = float(np.mean(1.96 * fb_se))
    return out
