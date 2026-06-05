"""Two-step driver for the SAR spillover SCM (``method='sar'``).

Runs Step 1 (horseshoe ``alpha``) and Step 2 (SAR ``rho`` + nuisances), then
reads off the treatment effect on the treated unit and the spillover effects on
the controls via the identification plug-ins. Effects follow the authors'
convention: the point estimate plugs in the posterior means ``(alpha_hat,
rho_hat)``; the credible interval is swept over the ``rho`` posterior with
``alpha`` fixed at ``alpha_hat``.
"""
from __future__ import annotations

import numpy as np

from .sampler import (
    hs_alpha_gibbs,
    sar_full_sampler,
    spillover_effects,
    treated_counterfactual,
)
from .structures import SARFit, SARInputs


def run_sar(
    inputs: SARInputs,
    *,
    p_factors: int = 1,
    M: int = 6000,
    burn: int = 2000,
    step_rho: float = 0.02,
    seed: int = 0,
    ci_level: float = 0.95,
) -> SARFit:
    """Fit the SAR spillover SCM and assemble a :class:`SARFit`."""
    rng = np.random.default_rng(seed)
    T0 = inputs.T0
    Y0, Yc = inputs.Y0, inputs.Yc
    Wn, wn = inputs.Wn, inputs.wn
    Y0_pre, Yc_pre = Y0[:T0], Yc[:T0]
    Y0_post, Yc_post = Y0[T0:], Yc[T0:]

    # Step 1: synthetic weights.
    a_draws = hs_alpha_gibbs(rng, Y0_pre, Yc_pre, M, burn)
    alpha_hat = a_draws.mean(axis=0)

    # Step 2: spatial parameter (+ nuisances).
    Xpre = inputs.X[:T0] if inputs.X is not None else None
    sar = sar_full_sampler(rng, Yc_pre, alpha_hat, wn, Wn, M, burn,
                           X=Xpre, p=int(p_factors), step_rho=step_rho)
    rho_draws = sar["rho"]
    rho_hat = float(rho_draws.mean())
    beta_hat = None if sar["beta"] is None else sar["beta"].mean(axis=0)

    lo_q, hi_q = (1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0

    # --- point estimate at (alpha_hat, rho_hat) ---
    ycf_sp = treated_counterfactual(Y0_post, Yc_post, Wn, wn, alpha_hat, rho_hat)
    gap_sp = Y0_post - ycf_sp
    # SCM special case: rho = 0.
    ycf_scm = Yc_post @ alpha_hat
    gap_scm = Y0_post - ycf_scm

    # --- spillover panel + ATE band, swept over rho (alpha fixed) ---
    n_post = Y0_post.shape[0]
    spill_sum = np.zeros((n_post, inputs.N))
    ate_draws = np.empty(rho_draws.shape[0])
    for i, r in enumerate(rho_draws):
        cf = treated_counterfactual(Y0_post, Yc_post, Wn, wn, alpha_hat, r)
        ate_draws[i] = float(np.mean(Y0_post - cf))
        spill_sum += spillover_effects(Y0_post, Yc_post, Wn, wn, alpha_hat, r)
    spill_mean = spill_sum / rho_draws.shape[0]
    spillover_panel = {lab: spill_mean[:, j]
                       for j, lab in enumerate(inputs.control_labels)}
    ate_ci = (float(np.quantile(ate_draws, lo_q)),
              float(np.quantile(ate_draws, hi_q)))

    return SARFit(
        att_sp=float(gap_sp.mean()),
        att_scm=float(gap_scm.mean()),
        gap_sp=gap_sp,
        gap_scm=gap_scm,
        counterfactual_sp=ycf_sp,
        counterfactual_scm=ycf_scm,
        spillover_panel=spillover_panel,
        ate_ci=ate_ci,
        rho_hat=rho_hat,
        rho_ci=(float(np.quantile(rho_draws, lo_q)), float(np.quantile(rho_draws, hi_q))),
        rho_draws=rho_draws,
        sigma2_hat=float(sar["s2"].mean()),
        alpha_hat=alpha_hat,
        alpha_labels=inputs.control_labels,
        acc_rho=float(sar["acc"]),
        p_factors=int(p_factors),
        ci_level=ci_level,
        beta_hat=beta_hat,
        metadata={"M": M, "burn": burn, "step_rho": step_rho, "seed": seed},
    )
