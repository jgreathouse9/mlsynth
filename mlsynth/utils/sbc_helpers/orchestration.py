"""Top-level SBC procedure.

Steps 1-4 of Shi, Xi, Xie (2025), Section 3.1:

    1. Hamilton-filter every unit's pre-treatment series into
       (trend, cycle).
    2. Use the treated unit's pre-treatment lags + its own AR
       coefficients to extrapolate the post-treatment trend.
    3. Fit a simplex SCM on the donor cycles to impute the treated
       cycle over the post-treatment window.
    4. Combine: y_hat_{1, t}(0) = tau_hat_{1, t} + c_hat_{1, t}.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .hamilton import fit_hamilton_filter
from .structures import SBCDesign, SBCInputs
from .synthetic import solve_sbc_weights
from .trend_forecast import forecast_treated_trend


def solve_sbc(
    inputs: SBCInputs,
    *,
    h: int,
    p: int,
    weights_mode: str = "simplex",
) -> SBCDesign:
    """Run the four SBC steps end-to-end.

    Parameters
    ----------
    inputs : SBCInputs
        From ``prepare_sbc_inputs``.
    h, p : int
        Hamilton-filter horizon and lag count.
    weights_mode : {"simplex", "unrestricted"}
        See ``synthetic.solve_sbc_weights``.

    Returns
    -------
    SBCDesign
    """

    Y_full = inputs.Y_full
    T = inputs.T
    T0 = inputs.T0
    N = Y_full.shape[1]
    # The Hamilton projection is an h-step forecast, so the counterfactual
    # is only well-defined for the first h post-treatment periods (paper
    # Step 4; the authors' code uses Fh = h). Cap the horizon accordingly.
    horizon = min(h, T - T0)

    # --- Step 1: trend / cycle decomposition -----------------------------
    # Treated unit: detrend on the PRE window only (its post outcomes are
    # contaminated by treatment). Donors: detrend on the FULL series, since
    # the donor cycles are needed in the post window and donors are
    # untreated. This mirrors the authors' replication code.
    treated_fit = fit_hamilton_filter(Y_full[:T0, 0], h=h, p=p)
    donor_fits = [
        fit_hamilton_filter(Y_full[:, j], h=h, p=p) for j in range(1, N)
    ]
    # Donor cycles over the full sample, (T, N-1); NaN in the first h+p-1 rows.
    donor_cycles_full = np.column_stack([f.cycle_pre for f in donor_fits])

    # Pre-treatment rows where the treated Hamilton filter produced a cycle.
    valid_pre = ~np.isnan(treated_fit.cycle_pre)   # length T0
    pre_idx = np.where(valid_pre)[0]
    if pre_idx.size < 2:
        from ...exceptions import MlsynthEstimationError
        raise MlsynthEstimationError(
            "SBC needs at least 2 effective pre-period observations after "
            f"applying the Hamilton filter; got {pre_idx.size}."
        )

    cycles_treated = treated_fit.cycle_pre[pre_idx]          # (n_pre,)
    cycles_donors = donor_cycles_full[pre_idx]               # (n_pre, N-1)

    # --- Step 3: SCM weights on the pre-treatment cycles -----------------
    weights, intercept = solve_sbc_weights(
        cycles_treated=cycles_treated,
        cycles_donors=cycles_donors,
        weights_mode=weights_mode,
    )

    # Pre-period in-sample cycle fit RMSE (diagnostic).
    pred_pre = cycles_donors @ weights
    if intercept is not None:
        pred_pre = pred_pre + intercept
    pre_cycle_rmse = float(np.sqrt(np.mean((cycles_treated - pred_pre) ** 2)))

    # --- Step 2: extrapolate treated trend over the (capped) horizon -----
    if horizon > 0:
        trend_forecast = forecast_treated_trend(
            y_target=inputs.y_target,
            treated_fit=treated_fit,
            T0=T0,
            horizon=horizon,
        )

        # --- Step 3 (cont'd): synthetic cycle over the horizon ----------
        # Post synthetic cycle = full-sample donor cycles at the post rows,
        # combined with the pre-fitted weights.
        post_idx = np.arange(T0, T0 + horizon)
        cycle_forecast = donor_cycles_full[post_idx] @ weights
        if intercept is not None:
            cycle_forecast = cycle_forecast + intercept

        counterfactual_post = trend_forecast + cycle_forecast
    else:
        trend_forecast = np.empty(0, dtype=float)
        cycle_forecast = np.empty(0, dtype=float)
        counterfactual_post = np.empty(0, dtype=float)

    return SBCDesign(
        weights=weights,
        weights_mode=weights_mode,
        intercept=intercept,
        treated_hamilton=treated_fit,
        donor_hamiltons=donor_fits,
        trend_forecast=trend_forecast,
        cycle_forecast=cycle_forecast,
        counterfactual_post=counterfactual_post,
        pre_cycle_rmse=pre_cycle_rmse,
    )


def summarize_effects(
    inputs: SBCInputs, design: SBCDesign
):
    """Build the full-window counterfactual and the ATT.

    Returns
    -------
    att : float
        ``mean(y_target - counterfactual)`` over post-treatment periods,
        or ``np.nan`` when no post window exists.
    counterfactual_full : np.ndarray
        Length-``T`` series. Pre-treatment is the observed treated
        series (no counterfactual estimated there); post-treatment is
        ``design.counterfactual_post``.
    treatment_effect : np.ndarray
        Length-``T`` ``y_target - counterfactual_full``. Zero pre,
        non-trivial post.
    """

    T0 = inputs.T0
    hz = int(design.counterfactual_post.shape[0])   # capped at h
    counterfactual_full = inputs.y_target.astype(float).copy()
    if hz > 0:
        counterfactual_full[T0:T0 + hz] = design.counterfactual_post
        # Beyond the h-step horizon the counterfactual is not estimated.
        counterfactual_full[T0 + hz:] = np.nan
        att = float(np.mean(
            inputs.y_target[T0:T0 + hz] - design.counterfactual_post
        ))
    else:
        att = float("nan")

    treatment_effect = inputs.y_target - counterfactual_full
    return att, counterfactual_full, treatment_effect
