"""TWSF -- Shen's section 7.1 design: exact recovery, calibrated variance, coverage.

Shen, D. (2026), *Causal Forecasting in Panel Data: A Two-Way Synthetic
Forecasting Approach*, `arXiv:2606.18512
<https://arxiv.org/abs/2606.18512>`_. No code release.

Path B. The paper's simulation draws donor factors, forms treated and control
time factors from a shared harmonic basis, scales the signal so
``max |<u_i, v_t(d)>| <= 0.8`` and adds ``N(0, 0.1^2)`` noise. Everything in
that description is reported in v2 except the numeric entries of the two 4x8
loading matrices, which are given only structurally -- fixed across
replications, with the lowest-frequency harmonic absent under control and
present under treatment. This case draws a non-degenerate pair respecting that
structure, which is what the v1-to-v2 gate established is sufficient:
`benchmarks/reference/twsf_spike/` on the review branch carries the full run
and `agents/future_integrations.md` section 17 the history.

What is checked
---------------
Three properties, in the order that separates a broken port from a broken
design, which is the sequence section 17 arrived at the hard way.

Algebra. With ``sigma = 0`` the forecast is exact, because the treated time
factor is a sum of harmonics and so satisfies a linear recursion of order at
most the lag length. Any error in the Page-block layout, the companion
recursion or the bilinear combination shows up here as a finite error, at
machine precision, not as a shift in a coverage rate.

Variance. Empirical standard deviation over mean plug-in standard error. The
paper's plug-in formula was already exact under v1, when its coverage was not,
so this is the diagnostic that separates "the variance is wrong" from "the
design is wrong".

Coverage. The gate. Nominal 90%, measured at the largest panel where the
theory's asymptotics have room to work. Coverage is below nominal at the
smallest panels for a spectral reason: the lag length equals ``n`` in this
design, so a short window cannot resolve the longest harmonic and the Page
matrix is near-degenerate at the oracle rank. That is consistent with an
asymptotic theory, and the case asserts the large-panel value, not a
uniform one.

The replication budget here is smaller than the paper's 100 x 10, since the
suite runs on every pull request; the tolerances are set for this budget. The
full-budget run gave coverage 0.908 / 0.885 / 0.892 at ``n = 150`` for horizons
1 / 5 / 10.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlsynth import TWSF
from mlsynth.config_models import TWSFConfig

SIGMA = 0.10
RANK = 4
N_LATENT, N_NOISE = 25, 4          # 100 replications per cell
Z90 = 1.6448536269514722

EXPECTED = {
    # algebra: sigma = 0 must recover the estimand exactly
    "noiseless_max_abs_error": (0.0, 1e-8),
    # Variance: empirical SD over mean plug-in SE, 1.0 being a calibrated
    # formula. The band is set by the replication budget here, not by how much
    # miscalibration would be tolerable -- 100 replications put roughly 7% of
    # Monte Carlo error on an SD estimate. The full-budget run gave 0.894 to
    # 1.165 across the grid. The direction that would matter is a ratio well
    # *above* one, which is an interval narrower than the sampling spread it
    # claims to cover; below one is conservative, and shows up as the coverage
    # rows landing above nominal, not below.
    "sd_over_se_h1": (1.00, 0.30),
    "sd_over_se_h5": (1.00, 0.30),
    # coverage at the large panel, nominal 0.90
    "coverage_h1": (0.90, 0.07),
    "coverage_h5": (0.90, 0.07),
    # the spectral story behind the small-panel shortfall
    "page_signal_spread_large_panel_below": (1.0, 0.0),
}


def _loadings(seed: int = 0):
    rng = np.random.default_rng(seed)
    A1 = rng.standard_normal((RANK, 8))
    A0 = A1.copy()
    A0[:, :2] = 0.0                 # lowest-frequency harmonic absent under control
    return A0, A1


def _basis(t: np.ndarray, T_star: int) -> np.ndarray:
    w = (2 * np.pi / (8 * T_star), 2 * np.pi / 12, 2 * np.pi / 37, 2 * np.pi / 91)
    return np.stack([
        np.sin(w[0] * t - np.pi / 3), np.cos(w[0] * t - np.pi / 3),
        np.sin(w[1] * t), np.cos(w[1] * t),
        np.sin(w[2] * t), np.cos(w[2] * t),
        np.sin(w[3] * t), np.cos(w[3] * t),
    ])


def _panel(n_donors: int, T0: int, T1: int, horizon: int, sigma: float,
           seed: int):
    """Return the long frame and the noiseless treated path past the panel."""
    rng = np.random.default_rng(seed)
    A0, A1 = _loadings()
    xi = rng.standard_normal((n_donors, RANK - 1))
    xi = (xi - xi.mean(0)) / xi.std(0)
    U = np.column_stack([np.ones(n_donors), xi])
    lam = rng.dirichlet(np.ones(n_donors))
    u_target = lam @ U

    T = T0 + T1
    T_star = T + horizon
    b = _basis(np.arange(1, T_star + 1), T_star)
    V0, V1 = A0 @ b, A1 @ b
    peak = max(np.abs(np.vstack([U, u_target]) @ V0).max(),
               np.abs(np.vstack([U, u_target]) @ V1).max())
    V0, V1 = V0 * (0.8 / peak), V1 * (0.8 / peak)

    rows = []
    noise = rng.normal(0, sigma, (n_donors + 1, T)) if sigma else \
        np.zeros((n_donors + 1, T))
    for i in range(n_donors):
        for k in range(T):
            v = V1[:, k] if k >= T0 else V0[:, k]
            rows.append(dict(unit=f"d{i}", time=k + 1,
                             y=U[i] @ v + noise[i, k], treat=int(k >= T0)))
    for k in range(T):
        rows.append(dict(unit="target", time=k + 1,
                         y=u_target @ V0[:, k] + noise[n_donors, k], treat=0))
    truth = np.array([u_target @ V1[:, T + h] for h in range(horizon)])
    return pd.DataFrame(rows), truth


def _fit(df, horizon, L, k_z):
    return TWSF(TWSFConfig(
        df=df, outcome="y", unitid="unit", time="time", treat="treat",
        target="target", L=L, k_y=RANK, k_z=k_z, horizon=horizon,
        multistep="recursive", alpha=0.10, display_graphs=False)).fit()


def run() -> dict:
    warnings.simplefilter("ignore")
    out: dict = {}

    # ---- algebra ----------------------------------------------------------
    worst = 0.0
    for horizon in (1, 5):
        df, truth = _panel(30, 60, 240, horizon, sigma=0.0, seed=11)
        cf = np.asarray(_fit(df, horizon, L=20, k_z=8)
                        .time_series.counterfactual_outcome, dtype=float)
        worst = max(worst, float(np.max(np.abs(cf - truth))))
    out["noiseless_max_abs_error"] = worst

    # ---- variance and coverage on the large panel -------------------------
    n = 60
    for horizon in (1, 5):
        errs, ses, cov = [], [], []
        for s in range(N_LATENT):
            for r in range(N_NOISE):
                df, truth = _panel(n, n, 8 * (n + 10), horizon, sigma=SIGMA,
                                   seed=5000 + s * 50 + r)
                res = _fit(df, horizon, L=n // 2, k_z=8)
                cf = np.asarray(res.time_series.counterfactual_outcome,
                                dtype=float)
                se = np.asarray(res.inference.details["std_error_path"],
                                dtype=float)
                e = float(np.mean(cf) - np.mean(truth))
                sm = float(np.mean(se))
                errs.append(e); ses.append(sm)
                cov.append(abs(e) <= Z90 * sm)
        errs = np.asarray(errs)
        out[f"sd_over_se_h{horizon}"] = float(errs.std(ddof=1) / np.mean(ses))
        out[f"coverage_h{horizon}"] = float(np.mean(cov))

    # ---- the spectral story behind the small-panel shortfall --------------
    from mlsynth.utils.twsf_helpers.pipeline import page_blocks
    from mlsynth.utils.twsf_helpers.setup import prepare_twsf_inputs
    spreads = {}
    for n in (12, 60):
        df, _ = _panel(n, n, 8 * (n + 10), 1, sigma=0.0, seed=77)
        inp = prepare_twsf_inputs(df, "y", "unit", "time", "treat", "target",
                                  horizon=1)
        Z, _z, _W = page_blocks(inp.Y_donors_post, L=max(3, n // 2))
        sv = np.linalg.svd(Z, compute_uv=False)
        k = min(8, sv.size)          # the oracle rank, or all of them if L < 8
        spreads[n] = float(sv[0] / sv[k - 1])
    # the large panel's retained signal must be better conditioned
    out["page_signal_spread_large_panel_below"] = float(
        spreads[60] < spreads[12])
    out["_page_spread_small"] = spreads[12]
    out["_page_spread_large"] = spreads[60]
    return out
