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

    # scpi lasso prediction intervals on the fitted signed coefficients plus the
    # intercept (Chernozhukov et al. 2021 / scpi Table 3). The compatible set is
    # the L1 ball at the realised budget ||w||_1; the intercept is the (free)
    # constant/KM block.
    scpi_obj = None
    if getattr(config, "compute_scpi_pi", False) and T0 < inputs.T:
        from ..clustersc_helpers.scpi_pi import scpi_pi_inference
        from ...exceptions import MlsynthEstimationError
        l1 = float(np.sum(np.abs(weights)))
        try:
            scpi_obj = scpi_pi_inference(
                inputs.y, inputs.donor_matrix, T0,
                np.concatenate([weights, [float(fit["intercept"])]]),
                constraint={"name": "lasso", "Q": l1}, constant=True,
                sims=config.scpi_sims, alpha=config.scpi_alpha,
                e_method=config.scpi_e_method, seed=0,
                periods=list(inputs.time_labels[T0:]),
            )
        except (MlsynthEstimationError, ValueError, ImportError):
            scpi_obj = None

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
        scpi=scpi_obj,
        metadata={"cv_option": config.cv_option},
    )
    return SCULResults(inputs=inputs, fit=scul_fit)
