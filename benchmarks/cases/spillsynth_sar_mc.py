"""SPILLSYNTH (SAR Bayesian) Path-B: spatial-spillover recovery Monte Carlo.

Validates mlsynth's ``SPILLSYNTH(method="sar")`` -- the spatial-autoregressive
Bayesian SCM of Sakaguchi & Tagawa (2026), *"Identification and Bayesian
Inference for Synthetic Control Methods with Spillover Effects"* (arXiv
2408.00291) -- against the paper's own simulation DGP
(:func:`mlsynth.utils.spillsynth_helpers.sar.simulation.simulate_sar_panel`).

The control outcomes follow a spatial autoregression on a rook lattice, so a
treatment on the treated unit propagates to the controls and biases a naive
synthetic control. The SAR estimator, which models that propagation, recovers
both the spatial coefficient and the treatment effect. The case pins the paper's
two headline claims:

* **Recovery** -- over Monte Carlo draws at the true :math:`\\rho = 0.6`, the
  posterior :math:`\\widehat\\rho` recovers the truth and the SAR ATT lands
  closer to the realised effect than the spillover-biased naive SCM ATT
  (SAR beats SCM in the large majority of draws);
* **Nesting** -- at :math:`\\rho = 0` (no spillover) the SAR estimator collapses
  to ordinary SCM: its ATT and the naive SCM ATT coincide, and the posterior
  :math:`\\widehat\\rho` is near zero.

Path B (the authors' simulation DGP): the MCMC is seeded, so the cells are
deterministic; tolerances absorb the residual posterior-sampling noise.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.spillsynth_helpers.sar.simulation import simulate_sar_panel

M = 8            # Monte Carlo draws (each SAR fit ~1 s)
_TRUE_RHO = 0.6


def _fit(sample):
    from mlsynth import SPILLSYNTH

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH({
            "df": sample.df, "outcome": "y", "treat": "d", "unitid": "unit",
            "time": "time", "method": "sar",
            "spatial_W": sample.spatial_W, "spatial_w": sample.spatial_w,
            "p_factors": 0, "mcmc_iter": 3000, "mcmc_burn": 1000,
            "step_rho": 0.05, "mcmc_seed": 1, "display_graphs": False,
        }).fit()


def run() -> dict:
    rhos = np.empty(M)
    beats = 0
    for s in range(M):
        sample = simulate_sar_panel(rho=_TRUE_RHO, seed=s)
        res = _fit(sample)
        rhos[s] = float(res.sar.rho_hat)
        beats += int(abs(res.att - sample.true_att) < abs(res.att_scm - sample.true_att))

    # rho = 0 nesting: SAR collapses to ordinary SCM.
    nest = simulate_sar_panel(rho=0.0, seed=0)
    res0 = _fit(nest)

    return {
        "rho_hat_mean": float(rhos.mean()),
        "sar_beats_scm_frac": beats / M,
        "rho0_att_gap": float(abs(res0.att - res0.att_scm)),
        "rho0_rho_hat": float(res0.sar.rho_hat),
    }


# Deterministic (seeded MCMC). Tolerances absorb the residual posterior-sampling
# noise at M=8. Reproduced facts (Sakaguchi-Tagawa): the SAR estimator recovers
# rho=0.6 and beats the spillover-biased naive SCM, and nests ordinary SCM at
# rho=0 (matching ATTs, near-zero posterior rho).
EXPECTED = {
    "rho_hat_mean": (0.58, 0.10),          # recovers true rho = 0.6
    "sar_beats_scm_frac": (1.0, 0.30),     # SAR closer to truth than naive SCM
    "rho0_att_gap": (0.03, 0.06),          # rho=0: SAR ATT == SCM ATT
    "rho0_rho_hat": (0.09, 0.12),          # rho=0: posterior rho near zero
}
