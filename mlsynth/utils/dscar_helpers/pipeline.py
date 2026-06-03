"""End-to-end Dynamic Synthetic Control orchestrator.

Wraps the per-period weight solver in :mod:`.weights` with the
algorithm's recursive structure (Section 2.2 of Zheng & Chen 2024):

* For the pre-period and the first post-period, match on the observed
  lagged outcome :math:`Y_{i, t-1}`.
* For each subsequent post-period :math:`t > T_0 + 1`, replace the
  treated unit's lagged-outcome target by the previously-estimated
  counterfactual :math:`\\widehat \\mu_{t-1}(0)`. This is the dynamic
  matching that makes the bias term in eq. (2.11) stochastically
  small.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .inference import (
    fdr_unconfoundedness_test, normalised_placebo_test,
)
from .structures import DSCARFit, DSCARInputs
from .weights import solve_dsc_weights, variable_importance


def _build_period_targets(
    inputs: DSCARInputs,
    t: int,
    Y0_hat_so_far: np.ndarray,
):
    """Build (Z1, Z0, active_donor_mask) for time period ``t``.

    The mask marks the donor units that have **no missing values** in
    the matching variables (lagged outcome, covariates, current
    outcome) at this period. This mirrors the per-period complete-case
    donor selection used by the R reference (Section "select records
    with complete variables").

    Returns
    -------
    Z1 : np.ndarray
        Length-``k`` target vector (treated covariates + lagged
        outcome). Treated-side missingness is averaged through (NaN is
        ignored via ``nanmean``).
    Z0 : np.ndarray
        Shape ``(k, n_active)`` donor targets for the active donors only.
    active : np.ndarray
        Length-``n_donors`` boolean mask of active donors at time ``t``.
    """
    n_treated = inputs.n_treated
    n_donors = inputs.N - n_treated
    p = inputs.X.shape[2]

    # Treated-mean covariates at time t (nanmean tolerates a couple of
    # missing treated rows at a given hour).
    x1_exog = (
        np.nanmean(inputs.X[:n_treated, t, :], axis=0) if p > 0 else np.zeros(0)
    )
    # Treated lagged-outcome target.
    if t == 0 or t <= inputs.T0:
        # Pre-period and the first post-period use the observed lag.
        if t >= 1:
            y1_lag = float(np.nanmean(inputs.Y_lag1[:n_treated, t]))
        else:
            y1_lag = float(np.nanmean(inputs.Y_lag1[:n_treated, t]))
    else:
        # Post-period (t > T0): use the previous estimated counterfactual.
        y1_lag = float(Y0_hat_so_far[t - 1])

    use_lag = not np.isnan(y1_lag)
    z1_parts = ([y1_lag] if use_lag else []) + list(x1_exog)
    Z1 = np.asarray(z1_parts, dtype=float)

    # Per-period complete-case donor mask.
    donor_y = inputs.Y[n_treated:, t]
    donor_lag = inputs.Y_lag1[n_treated:, t]
    donor_x = inputs.X[n_treated:, t, :] if p > 0 else np.zeros((n_donors, 0))
    complete = np.isfinite(donor_y)
    if use_lag:
        complete &= np.isfinite(donor_lag)
    if p > 0:
        complete &= np.all(np.isfinite(donor_x), axis=1)
    if not np.any(complete):
        return Z1, np.zeros((Z1.size, 0), dtype=float), complete

    # Donor side: stack [Y_lag1[active, t], X[active, t, :]].
    z0_rows = []
    if use_lag:
        z0_rows.append(donor_lag[complete])
    for k in range(p):
        z0_rows.append(donor_x[complete, k])
    Z0 = np.vstack(z0_rows) if z0_rows else np.zeros((0, int(complete.sum())))
    return Z1, Z0, complete


def run_dsc(
    inputs: DSCARInputs,
    *,
    el_tolerance: float = 1e-2,
    placebo_reps: int = 0,
    do_fdr_test: bool = True,
    fdr_alpha: float = 0.05,
    seed: int = 0,
) -> DSCARFit:
    """Walk the panel period by period and assemble the DSCARFit.

    Parameters
    ----------
    inputs : DSCARInputs
        Output of :func:`prepare_dsc_inputs`.
    el_tolerance : float
        Mean-absolute-mismatch threshold for triggering EL refinement.
    placebo_reps : int
        If ``> 0``, run a normalised placebo test (Section 3.2) with
        this many random control-only "treated" draws and populate
        ``DSCARFit.se`` / ``DSCARFit.placebo_atts``.
    do_fdr_test : bool
        If ``True``, run the FDR-controlled per-pre-period
        unconfoundedness test (Section 3.1) and populate
        ``DSCARFit.pre_period_pvalues`` /
        ``DSCARFit.pre_period_min_pvalue_adj``.
    fdr_alpha : float
        Significance level for the FDR test.
    seed : int
        RNG seed for placebo draws.
    """
    N, T = inputs.Y.shape
    n_treated = inputs.n_treated
    n_donors = N - n_treated
    Y_treated_mean = inputs.Y[:n_treated].mean(axis=0)        # (T,)

    # Per-period OLS variable-importance V (eq. ``paramt`` in the R
    # reference): pre-period uses the FULL panel, post-period uses
    # donors only (since the treated post-period outcomes contain the
    # treatment effect we are trying to estimate).
    V_pre = variable_importance(
        inputs.Y, inputs.X, inputs.Y_lag1, T0=inputs.T0,
    )
    V_post = variable_importance(
        inputs.Y[n_treated:], inputs.X[n_treated:],
        inputs.Y_lag1[n_treated:], T0=inputs.T0,
    )
    V = V_pre.copy()
    V[inputs.T0:] = V_post[inputs.T0:]
    weights = np.zeros((T, n_donors))
    Y0_hat = np.zeros(T)
    n_exact = 0

    for t in range(T):
        Z1, Z0, active = _build_period_targets(inputs, t, Y0_hat)
        # Number of donors that survived per-period complete-case filter.
        n_active = int(active.sum())
        if Z1.size == 0 or n_active == 0:
            # No matching constraint or no usable donors -> uniform weights
            # over whatever donors *do* have a valid outcome at t.
            outcome_ok = np.isfinite(inputs.Y[n_treated:, t])
            if outcome_ok.any():
                w_full = np.zeros(n_donors)
                w_full[outcome_ok] = 1.0 / outcome_ok.sum()
                used_el = False
            else:
                w_full = np.zeros(n_donors)
                used_el = False
        else:
            # V diagonal must align with the rows of Z that we actually built.
            # Z has either (1 + p) rows when the lag was included, or p rows
            # when it wasn't. V[t, 0] is the lag coefficient |rho_t|; V[t, 1..p]
            # are the |beta_t, k| covariate coefficients.
            p = inputs.X.shape[2]
            use_lag = (Z0.shape[0] == 1 + p)
            if use_lag:
                V_diag = V[t, : 1 + p]
            else:
                V_diag = V[t, 1 : 1 + p]
            V_diag = np.where(V_diag > 1e-12, V_diag, 1.0)
            w_active, used_el = solve_dsc_weights(
                Z1, Z0, V_diag, el_tolerance=el_tolerance,
            )
            w_full = np.zeros(n_donors)
            w_full[active] = w_active
        weights[t, :] = w_full
        # Synthetic outcome: only the active (well-defined) donors contribute.
        Y_t_donors = inputs.Y[n_treated:, t]
        Y_t_donors = np.where(np.isfinite(Y_t_donors), Y_t_donors, 0.0)
        Y0_hat[t] = float(Y_t_donors @ w_full)
        if used_el:
            n_exact += 1

    gap = Y_treated_mean - Y0_hat
    post = slice(inputs.T0, T)
    mu1 = float(Y_treated_mean[post].mean())
    mu0 = float(Y0_hat[post].mean())
    att = mu1 - mu0
    att_relative = 1.0 - (mu1 / mu0) if mu0 != 0 else float("nan")

    # Optional inference.
    se: Optional[float] = None
    placebo_atts: Optional[np.ndarray] = None
    if placebo_reps > 0:
        placebo_atts, se = normalised_placebo_test(
            inputs=inputs,
            weights=weights,
            att_observed=att,
            placebo_reps=placebo_reps,
            el_tolerance=el_tolerance,
            seed=seed,
        )

    pre_pvals: Optional[np.ndarray] = None
    min_p_adj: Optional[float] = None
    if do_fdr_test:
        pre_pvals, min_p_adj = fdr_unconfoundedness_test(
            inputs=inputs,
            weights=weights,
            gap=gap,
        )

    return DSCARFit(
        weights=weights,
        Y0_hat=Y0_hat,
        Y_treated_mean=Y_treated_mean,
        gap=gap,
        att=att,
        att_relative=att_relative,
        se=se,
        placebo_atts=placebo_atts,
        pre_period_pvalues=pre_pvals,
        pre_period_min_pvalue_adj=min_p_adj,
        n_exact_matched_periods=n_exact,
        v_diagonal=V,
    )
