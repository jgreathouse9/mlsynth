"""Orchestration pipeline for Spatial Synthetic Difference-in-Differences.

Implements Algorithm 1 of Serenini & Masek (2024):

1. Compute SDID unit / time weights using *only* the pure controls as
   donors (Arkhangelsky et al. 2021 QPs duplicated in :mod:`.weights`).
2. Fix the per-unit weight as
   :math:`\\omega_i = 1 / N_{\\mathrm{tr}}` for directly-treated units,
   :math:`\\omega_i = 1 / N_{\\mathrm{sp}}` for indirectly-treated units,
   and SDID-fit :math:`\\omega_i` for pure controls.
3. Run the weighted two-way FE regression

   .. math::

      (\\widehat \\tau, \\widehat \\tau_s, \\widehat \\mu, \\widehat \\alpha,
       \\widehat \\beta)
      = \\arg \\min \\sum_{i, t}
        \\bigl[ Y_{it}
                - \\mu - \\alpha_i - \\beta_t
                - \\tau D_{it}
                - \\tau_s (WD)_{it}
        \\bigr]^2
        \\widehat \\omega_i\\, \\widehat \\lambda_t.

   The augmented design recovers the direct effect :math:`\\widehat \\tau`
   and the spillover coefficient :math:`\\widehat \\tau_s` jointly.
4. The implied population ATE is
   :math:`\\widehat \\tau (1 + \\overline{WD})`
   with :math:`\\overline{WD}` the average exposure across directly +
   indirectly treated units (paper eq. 14).
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..results_helpers import make_weights_results
from .structures import SpSyDiDInputs, SpSyDiDResults
from .weights import (
    compute_regularization,
    fit_time_weights,
    fit_unit_weights,
)


def run_spsydid(inputs: SpSyDiDInputs) -> SpSyDiDResults:
    """Run Algorithm 1 of Serenini & Masek (2024)."""

    Y = inputs.outcome_matrix
    D = inputs.treatment_matrix
    WD = inputs.exposure_matrix
    N, T = Y.shape
    T0 = inputs.T0
    T_post = T - T0
    direct = inputs.direct_indices
    spillover = inputs.spillover_indices
    pure = inputs.pure_control_indices
    N_tr = direct.size
    N_sp = spillover.size

    # ------------------------------------------------------------------
    # Step 1: SDID unit & time weights using PURE CONTROLS as donors and
    # the mean of directly-treated units as the target trajectory.
    # ------------------------------------------------------------------
    pure_pre = Y[pure][:, :T0].T            # shape (T0, N_pure)
    pure_post = Y[pure][:, T0:].T           # shape (T_post, N_pure)
    treated_pre_mean = Y[direct][:, :T0].mean(axis=0)  # shape (T0,)

    zeta = compute_regularization(pure_pre, num_post_periods=T_post)

    intercept_omega, omega_pure = fit_unit_weights(
        donor_outcomes_pre=pure_pre,
        mean_treated_outcome_pre=treated_pre_mean,
        zeta=zeta,
    )
    if omega_pure is None:
        raise MlsynthEstimationError("SpSyDiD: SDID unit-weight QP failed.")

    intercept_lambda, lambda_pre = fit_time_weights(
        donor_outcomes_pre=pure_pre,
        mean_donor_outcomes_post=pure_post.mean(axis=0),
    )
    if lambda_pre is None:
        raise MlsynthEstimationError("SpSyDiD: SDID time-weight QP failed.")

    # ------------------------------------------------------------------
    # Step 2: assemble the full omega / lambda vectors for the WLS step.
    # ------------------------------------------------------------------
    omega = np.zeros(N, dtype=float)
    omega[pure] = omega_pure
    omega[direct] = 1.0 / N_tr
    if N_sp > 0:
        omega[spillover] = 1.0 / N_sp

    # Time weights: SDID-fit for the pre-period, uniform 1/T_post for the post.
    lam = np.empty(T, dtype=float)
    lam[:T0] = lambda_pre
    lam[T0:] = 1.0 / T_post

    # ------------------------------------------------------------------
    # Step 3: weighted two-way FE regression with the spatial term.
    # Build the design matrix directly:
    #   columns = [intercept, alpha_2 ... alpha_N, beta_2 ... beta_T, D, WD]
    # row count = N * T.
    # ------------------------------------------------------------------
    flat_Y = Y.flatten()                  # row-major (unit, time)
    flat_D = D.flatten()
    flat_WD = WD.flatten()

    weight_vec = np.outer(omega, lam).flatten()
    sqrt_w = np.sqrt(weight_vec)

    # Build design matrix. Use unit_0 and time_0 as the reference categories
    # to absorb the intercept (mu); fixed effects enter as dummies for
    # i = 2..N and t = 2..T.
    n_obs = N * T
    n_cols = 1 + (N - 1) + (T - 1) + 2
    X = np.zeros((n_obs, n_cols), dtype=float)
    X[:, 0] = 1.0                                   # intercept
    # Unit dummies: rows i*T : (i+1)*T have a 1 in column (1 + i - 1) for i>=1.
    for i in range(1, N):
        X[i * T : (i + 1) * T, i] = 1.0
    # Time dummies: rows i*T + t for i in 0..N-1 have 1 in col (1 + N-1 + t-1)
    # for t >= 1.
    base = 1 + (N - 1)
    for t in range(1, T):
        idx = np.arange(N) * T + t
        X[idx, base + t - 1] = 1.0
    # D and WD columns:
    X[:, -2] = flat_D
    X[:, -1] = flat_WD

    # Weighted least squares: rescale rows by sqrt(w) and solve.
    Xw = X * sqrt_w[:, None]
    Yw = flat_Y * sqrt_w
    try:
        beta_hat, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise MlsynthEstimationError(
            f"SpSyDiD final WLS failed: {exc}"
        ) from exc

    tau = float(beta_hat[-2])
    tau_s = float(beta_hat[-1])

    # ------------------------------------------------------------------
    # ATE = tau * (1 + bar(WD)) where bar(WD) is the average exposure
    # across the directly + indirectly treated units in the post-period.
    # ------------------------------------------------------------------
    treated_union = np.concatenate([direct, spillover]) if N_sp > 0 else direct
    bar_WD = float(WD[treated_union, T0:].mean()) if treated_union.size > 0 else 0.0
    aite = tau_s
    ate = tau * (1.0 + bar_WD) if bar_WD != 0.0 else tau

    unit_weights = {
        inputs.unit_names[i]: float(omega[i]) for i in range(N)
    }

    metadata = {
        "N_direct": int(N_tr),
        "N_spillover": int(N_sp),
        "N_pure": int(pure.size),
        "T0": int(T0),
        "T_post": int(T_post),
        "mean_exposure_post_treated": bar_WD,
        "sdid_zeta": float(zeta),
        "sdid_omega_intercept": float(intercept_omega) if intercept_omega is not None else None,
        "sdid_lambda_intercept": float(intercept_lambda) if intercept_lambda is not None else None,
    }

    # Donor weights = the pure-control SDID unit weights (omega); the
    # directly/indirectly treated rows carry the 1/N aggregation weights.
    pure_names = [inputs.unit_names[i] for i in inputs.pure_control_indices]
    donor_weights = {n: float(unit_weights[n]) for n in pure_names}
    weights_res = make_weights_results(
        donor_weights,
        constraint="SDID unit weights over pure controls (>= 0, sum to 1)",
        extra={
            "time_weights": [float(x) for x in lambda_pre],
            "omega_intercept": metadata.get("sdid_omega_intercept"),
            "lambda_intercept": metadata.get("sdid_lambda_intercept"),
        },
    )

    return SpSyDiDResults(
        inputs=inputs,
        att=tau,
        aite=aite,
        ate=ate,
        unit_weights=unit_weights,
        time_weights=lambda_pre,
        zeta=float(zeta),
        weights=weights_res,
        metadata=metadata,
    )
