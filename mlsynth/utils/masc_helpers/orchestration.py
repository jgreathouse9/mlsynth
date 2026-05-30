"""End-to-end MASC pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .crossval import _aggregate_covariates, cross_validate
from .estimation import (
    masc_combine,
    nearest_neighbor_weights,
    sc_simplex_weights,
)
from .structures import MASCFit, MASCInputs


def run_masc(
    inputs: MASCInputs,
    *,
    m_grid: Optional[Sequence[int]] = None,
    min_preperiods: Optional[int] = None,
    set_f: Optional[Sequence[int]] = None,
    fold_weights: Optional[np.ndarray] = None,
    forecast_minlength: int = 1,
    forecast_maxlength: int = 1,
    solver: Optional[str] = None,
) -> MASCFit:
    """Run MASC end-to-end on ``inputs``.

    1. Cross-validate ``(m, phi)`` via rolling-origin CV.
    2. Refit SC and matching at ``m_hat`` on the *full* pre-period.
    3. Combine with the analytic ``phi_hat``.
    4. Form the ATT as ``mean(Y_treated_post - YJ_post @ weights)``.
    """
    Y_treated = inputs.Y_treated
    Y_donors = inputs.Y_donors
    T0 = inputs.T0

    m_hat, phi_hat, cv_error, cv_grid, by_fold = cross_validate(
        Y_treated, Y_donors, inputs.treatment_period,
        m_grid=m_grid, min_preperiods=min_preperiods, set_f=set_f,
        fold_weights=fold_weights,
        forecast_minlength=forecast_minlength,
        forecast_maxlength=forecast_maxlength,
        solver=solver,
        cov_treated_panel=inputs.cov_treated_panel,
        cov_donors_panel=inputs.cov_donors_panel,
        covariate_names=inputs.covariate_names,
        time_index=inputs.time_index,
        covariate_windows=inputs.covariate_windows,
    )

    Y0_pre = Y_treated[:T0]
    YJ_pre = Y_donors[:T0]
    w_match = nearest_neighbor_weights(Y0_pre, YJ_pre, m_hat)
    # For the final refit, aggregate covariates over the entire pre-period
    # (matches the R reference's behaviour outside the CV folds).
    X_treated, X_donors = _aggregate_covariates(
        inputs.cov_treated_panel, inputs.cov_donors_panel,
        inputs.covariate_names, inputs.time_index,
        pre_end_period=T0,
        covariate_windows=inputs.covariate_windows,
    )
    w_sc = sc_simplex_weights(
        Y0_pre, YJ_pre,
        X_treated=X_treated, X_donors=X_donors,
        solver=solver,
    )
    w_masc = masc_combine(w_match, w_sc, phi_hat)

    counterfactual = Y_donors @ w_masc
    gap = Y_treated - counterfactual
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    att = float(gap[T0:].mean())
    donor_weights = dict(zip(inputs.donor_labels, w_masc.tolist()))

    return MASCFit(
        att=att,
        weights=w_masc,
        weights_match=w_match,
        weights_sc=w_sc,
        phi_hat=phi_hat,
        m_hat=m_hat,
        counterfactual=counterfactual,
        gap=gap,
        pre_rmse=pre_rmse,
        cv_error=cv_error,
        cv_error_by_fold=by_fold,
        cv_grid=cv_grid,
        donor_weights=donor_weights,
    )
