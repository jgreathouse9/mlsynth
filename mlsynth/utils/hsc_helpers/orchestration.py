"""Top-level HSC solve and effect summary.

``solve_hsc`` runs the full pipeline -- select ``rho`` by rolling-origin CV,
fit the profiled donor QP on the whole pre-period, forecast the smooth
residual into the post-period, and assemble the counterfactual. The
post-treatment counterfactual is

    Y_hat_post = X_post @ omega + forecast(E),

the donor-matched component plus the extrapolated smooth component.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .forecast import forecast_smooth
from .formulation import sdid_ridge_coefficient
from .optimization import fit_at_rho, select_rho_by_cv
from .structures import HSCDesign, HSCInputs


def solve_hsc(
    inputs: HSCInputs,
    q: int,
    rho_grid: Sequence[float],
    n_splits: int = 3,
    ridge: object = 1e-6,
    forecaster: str = "arima110",
    solver: Optional[object] = None,
) -> HSCDesign:
    """Run HSC end-to-end and return the fitted design.

    ``ridge`` is either a float (relative ridge ``ridge * trace(X'WX)/N``) or
    the string ``"sdid"``, which uses the data-driven SDID-style coefficient
    ``zeta^2 T0`` (computed once from the full pre-period donors and the
    post-treatment horizon) for diversified donor weights.
    """

    X_pre, Y_pre, X_post = inputs.X_pre, inputs.Y_pre, inputs.X_post

    if isinstance(ridge, str):
        if ridge != "sdid":
            raise ValueError(f"Unknown HSC ridge mode '{ridge}'. Use a float or 'sdid'.")
        ridge_abs = sdid_ridge_coefficient(X_pre, inputs.n_post)
        ridge_float = 0.0
    else:
        ridge_abs = None
        ridge_float = float(ridge)

    selected_rho, cv_curve = select_rho_by_cv(
        X_pre, Y_pre, q, rho_grid, n_splits, ridge_float, forecaster, solver,
        ridge_abs=ridge_abs,
    )
    omega, E_pre = fit_at_rho(
        X_pre, Y_pre, selected_rho, q, ridge_float, solver, ridge_abs=ridge_abs
    )

    n_post = inputs.n_post
    smooth_forecast = forecast_smooth(E_pre, n_post, forecaster)
    donor_match_post = X_post @ omega
    counterfactual_post = donor_match_post + smooth_forecast

    return HSCDesign(
        selected_rho=selected_rho,
        q=q,
        omega=omega,
        smooth_pre=E_pre,
        donor_match_pre=X_pre @ omega,
        counterfactual_post=counterfactual_post,
        smooth_forecast=smooth_forecast,
        donor_match_post=donor_match_post,
        cv_curve=cv_curve,
        forecaster=forecaster,
    )


def summarize_effects(
    inputs: HSCInputs, design: HSCDesign
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Assemble ATT, full-timeline counterfactual, and per-period effects.

    Returns
    -------
    att : float
        Mean post-period treatment effect.
    counterfactual_full : np.ndarray
        Pre-period in-sample fit (``X_pre @ omega + E``) stacked with the
        post-period counterfactual, shape ``(T,)``.
    treatment_effect : np.ndarray
        ``Y_post - counterfactual_post``, shape ``(n_post,)``.
    """

    cf_pre = design.donor_match_pre + design.smooth_pre
    counterfactual_full = np.concatenate([cf_pre, design.counterfactual_post])
    treatment_effect = inputs.Y_post - design.counterfactual_post
    att = float(np.mean(treatment_effect)) if treatment_effect.size else 0.0
    return att, counterfactual_full, treatment_effect
