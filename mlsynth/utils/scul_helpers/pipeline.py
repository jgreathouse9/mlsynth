"""SCUL orchestration: inputs -> lasso fit -> placebo inference -> results."""

from __future__ import annotations

import numpy as np

from .estimate import fit_scul
from .inference import placebo_pvalue
from .structures import SCULFit, SCULInputs, SCULResults


def run_scul(inputs: SCULInputs, config) -> SCULResults:
    """Fit SCUL on prepared ``inputs`` and assemble :class:`SCULResults`."""
    T0 = inputs.T0
    fit = fit_scul(
        inputs.y, inputs.donor_matrix, T0,
        number_initial_periods=config.number_initial_periods,
        training_post_length=config.training_post_length,
        cv_option=config.cv_option,
    )
    counterfactual = fit["counterfactual"]
    gap = inputs.y - counterfactual
    att = float(np.mean(gap[T0:])) if T0 < inputs.T else float("nan")

    weights = fit["weights"]
    donor_weights = {
        f"{inputs.col_unit[k]}:{inputs.col_variable[k]}": float(weights[k])
        for k in np.where(np.abs(weights) > 1e-10)[0]
    }

    p_value, n_placebo = None, None
    if config.inference and T0 < inputs.T:
        p_value, n_placebo = placebo_pvalue(
            inputs, gap,
            number_initial_periods=config.number_initial_periods,
            training_post_length=config.training_post_length,
            cv_option=config.cv_option,
            cohensd_threshold=config.cohensd_threshold,
        )

    scul_fit = SCULFit(
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        weights=weights,
        intercept=fit["intercept"],
        ridge_lambda=fit["ridge_lambda"],
        cohens_d=fit["cohens_d"],
        donor_weights=donor_weights,
        p_value=p_value,
        n_placebo=n_placebo,
        metadata={"cv_option": config.cv_option},
    )
    return SCULResults(inputs=inputs, fit=scul_fit)
