"""SPILLSYNTH (SAR Bayesian) Path-B: cross-validation vs the paper's Table 5.2.

Cross-validates mlsynth's ``SPILLSYNTH(method="sar")`` -- the spatial-autoregressive
Bayesian SCM of Sakaguchi & Tagawa (2026), *"Identification and Bayesian Inference
for Synthetic Control Methods with Spillover Effects"* (arXiv 2408.00291) -- against
the authors' own published simulation table (their Section-5.2 Monte Carlo, the
committed ``mc_study_N=16T0=20`` cells of their replication package).

The reference DGP is reproduced exactly
(:func:`mlsynth.utils.spillsynth_helpers.sar.simulation.simulate_sar_panel`: rook
lattice, true synthetic weights ``alpha = (0.5, -0.2, 0.4, 0.4, ...)``, treated
weight on the first four units, innovation variance ``sigma2 = 1``, effect
``tau ~ N(1, 1)``). Over Monte Carlo draws at ``N = 16, T0 = 20, T1 = 10`` the
paper reports -- and mlsynth reproduces, **bias defined as truth − estimate as in
the reference** -- the proposed method (SCSPILL) is essentially unbiased at every
``rho`` while ordinary SCM's bias grows with the spillover:

  =====  =================  =================  =================
  rho    SCSPILL bias       SCM bias (paper)   SCM bias (mlsynth)
  =====  =================  =================  =================
  0.0    ~0.00 (paper -0.002)  -0.003            ~0.00
  0.3    ~0.00 (paper -0.003)  +0.092            ~+0.11
  0.8    ~0.00 (paper +0.001)  +0.504            ~+0.60
  =====  =================  =================  =================

and the proposed method's 95% credible interval covers at ~0.95 (paper
0.94-0.96). Path B (cross-validation): mlsynth runs the *authors' DGP* and
reproduces the *authors' published* bias / coverage pattern -- SCSPILL unbiased
and well-covered where SCM is badly biased. The MCMC is seeded; tolerances absorb
the 30-draw Monte Carlo noise (the paper averages many more replications).
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.spillsynth_helpers.sar.simulation import simulate_sar_panel

M = 30                 # Monte Carlo draws per cell (each SAR fit ~1 s)
_RHOS = (0.0, 0.3, 0.8)
_SIGMA2 = 1.0          # the paper's innovation variance


def _fit(sample):
    from mlsynth import SPILLSYNTH

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH({
            "df": sample.df, "outcome": "y", "treat": "d", "unitid": "unit",
            "time": "time", "method": "sar",
            "spatial_W": sample.spatial_W, "spatial_w": sample.spatial_w,
            "p_factors": 0, "mcmc_iter": 3000, "mcmc_burn": 1000,
            "step_rho": 0.02, "mcmc_seed": 1, "display_graphs": False,
        }).fit()


def _cell(rho: float):
    """Return (SCSPILL bias, SCSPILL coverage, SCM bias) in the paper's convention."""
    sp_err = np.empty(M)
    scm_err = np.empty(M)
    cov = np.empty(M)
    for s in range(M):
        sample = simulate_sar_panel(rho=rho, n_rows=4, n_cols=4, T=30, T0=20,
                                    sigma2=_SIGMA2, seed=s)
        res = _fit(sample)
        # reference defines bias = truth - estimate
        sp_err[s] = sample.true_att - res.att
        scm_err[s] = sample.true_att - res.att_scm
        lo, hi = res.sar.ate_ci
        cov[s] = 1.0 if lo <= sample.true_att <= hi else 0.0
    return float(sp_err.mean()), float(cov.mean()), float(scm_err.mean())


def run() -> dict:
    cells = {rho: _cell(rho) for rho in _RHOS}
    sp_bias = {rho: c[0] for rho, c in cells.items()}
    cover = {rho: c[1] for rho, c in cells.items()}
    scm_bias = {rho: c[2] for rho, c in cells.items()}
    return {
        "scspill_abs_bias_max": float(max(abs(b) for b in sp_bias.values())),
        "scm_bias_rho03": scm_bias[0.3],
        "scm_bias_rho08": scm_bias[0.8],
        "scspill_coverage_mean": float(np.mean(list(cover.values()))),
        # the de-biasing: at rho=0.8 SCSPILL bias is far smaller than SCM's
        "scspill_debiases_scm_rho08": float(abs(sp_bias[0.8]) < 0.25 * abs(scm_bias[0.8])),
        "scm_bias_grows_with_rho": float(
            abs(scm_bias[0.8]) > abs(scm_bias[0.3]) > 0.01),
    }


# Deterministic (seeded MCMC). Tolerances absorb the 30-draw Monte Carlo noise.
# Reproduces Sakaguchi-Tagawa's published Section-5.2 cells (N=16, T0=20): the
# proposed SCSPILL method is unbiased and ~95%-covered at every rho, while
# ordinary SCM's bias grows with the spillover (paper: +0.092 at rho=0.3, +0.504
# at rho=0.8). mlsynth's SCM bias runs a touch higher (fewer reps, 3000 vs 5000
# MCMC iters) but reproduces the sign and growth pattern.
EXPECTED = {
    "scspill_abs_bias_max": (0.006, 0.012),   # unbiased (paper |bias| <= 0.003)
    "scm_bias_rho03": (0.10, 0.06),           # paper +0.092
    "scm_bias_rho08": (0.58, 0.18),           # paper +0.504
    "scspill_coverage_mean": (0.94, 0.08),    # paper 0.94-0.96
    "scspill_debiases_scm_rho08": (1.0, 0.0),
    "scm_bias_grows_with_rho": (1.0, 0.0),
}
