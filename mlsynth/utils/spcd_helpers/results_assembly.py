"""Assemble the standardized BaseEstimatorResults summary for SPCD.

Packages an :class:`SPCDDesign` together with its preprocessed
:class:`SPCDInputs` into the project's standardized result pydantic
models defined in :mod:`mlsynth.config_models`:

    EffectsResults          : ATT (mean post-period synthetic gap)
    FitDiagnosticsResults   : pre/post RMSE of the synthetic gap
    TimeSeriesResults       : synthetic treated/control/gap trajectories
    WeightsResults          : per-unit signed contrast weights
    MethodDetailsResults    : variant, weights mode, alpha/lam/beta, iters

The result is wrapped in a :class:`BaseEstimatorResults` so SPCD's
public ``results.summary`` matches the shape used by the rest of the
mlsynth estimator suite.
"""

from __future__ import annotations

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from .structures import SPCDDesign, SPCDInputs
from .treatment_effect import compute_att_and_fit


def build_summary(
    design: SPCDDesign, inputs: SPCDInputs
) -> BaseEstimatorResults:
    """Build the standardized result bundle for an SPCD design.

    Parameters
    ----------
    design : SPCDDesign
        Output of :func:`solve_spcd`.
    inputs : SPCDInputs
        Preprocessed panel data used in the fit.

    Returns
    -------
    BaseEstimatorResults
        Bundle with ``effects``, ``fit_diagnostics``, ``time_series``,
        ``weights``, and ``method_details`` populated.
    """

    att, rmse_pre, rmse_post = compute_att_and_fit(
        Y_pre=inputs.Y_pre,
        Y_post=inputs.Y_post,
        treated_weights=design.treated_weights,
        control_weights=design.control_weights,
    )

    unit_labels = inputs.unit_index.labels
    donor_weights = {
        str(unit_labels[i]): float(design.contrast_weights[i])
        for i in range(len(unit_labels))
    }

    pre_periods = inputs.pre_time_index.labels
    if inputs.post_time_index is not None:
        post_periods = inputs.post_time_index.labels
        time_periods = np.concatenate([pre_periods, post_periods])
    else:
        time_periods = pre_periods

    return BaseEstimatorResults(
        effects=EffectsResults(att=att),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=rmse_pre,
            rmse_post=rmse_post,
        ),
        time_series=TimeSeriesResults(
            observed_outcome=design.synthetic_treated,
            counterfactual_outcome=design.synthetic_control,
            estimated_gap=design.synthetic_gap,
            time_periods=time_periods,
        ),
        weights=WeightsResults(
            donor_weights=donor_weights,
            summary_stats={
                "n_treated": int(design.n_treated),
                "n_control": int(len(unit_labels) - design.n_treated),
                "treated_unit_labels": [
                    str(x) for x in design.selected_unit_labels.tolist()
                ],
            },
        ),
        method_details=MethodDetailsResults(
            method_name=f"SPCD ({design.variant}, weights={design.weights_mode})",
            parameters_used={
                "variant": design.variant,
                "weights": design.weights_mode,
                "alpha_ridge": design.alpha_ridge,
                "lam_balance": design.lam_balance,
                "beta": design.beta,
                "n_iterations": design.n_iterations,
                "converged": design.converged,
                "max_iter_reached": (not design.converged),
            },
        ),
    )
