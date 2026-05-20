"""Assemble the standardized BaseEstimatorResults summary for SPCD.

Packages an :class:`SPCDDesign` together with its preprocessed
:class:`SPCDInputs` and optional inference / power outputs into the
project's standardized result pydantic models defined in
:mod:`mlsynth.config_models`:

    EffectsResults          : ATT (mean post-period synthetic gap)
    FitDiagnosticsResults   : pre/post RMSE of the synthetic gap +
                              MDE / detectability in additional_metrics
    TimeSeriesResults       : synthetic treated/control/gap trajectories
    WeightsResults          : per-unit signed contrast weights
    InferenceResults        : conformal p-value, CI, pointwise bands
    MethodDetailsResults    : variant, weights mode, alpha/lam/beta, iters

The result is wrapped in a :class:`BaseEstimatorResults` so SPCD's
public ``results.summary`` matches the shape used by the rest of the
mlsynth estimator suite.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from .inference import SPCDConformalResult
from .power import SPCDPowerAnalysis
from .structures import SPCDDesign, SPCDInputs
from .treatment_effect import compute_att_and_fit


def build_summary(
    design: SPCDDesign,
    inputs: SPCDInputs,
    conformal: Optional[SPCDConformalResult] = None,
    power: Optional[SPCDPowerAnalysis] = None,
) -> BaseEstimatorResults:
    """Build the standardized result bundle for an SPCD design.

    Parameters
    ----------
    design : SPCDDesign
        Output of :func:`solve_spcd_with_holdout` (or :func:`solve_spcd`).
    inputs : SPCDInputs
        Preprocessed panel data used in the fit.
    conformal : SPCDConformalResult, optional
        Moving-block conformal CI for the post-period ATT. ``None``
        when no post period is supplied or inference was disabled.
    power : SPCDPowerAnalysis, optional
        MDE / detectability output. ``None`` when inference was
        disabled or the holdout window was too small.

    Returns
    -------
    BaseEstimatorResults
        Bundle with ``effects``, ``fit_diagnostics``, ``time_series``,
        ``weights``, ``inference``, and ``method_details`` populated.
        Fields are honest about absence: ATT is ``None`` (not 0.0) when
        no post period is supplied.
    """

    # ------------------------------------------------------------------
    # Effects: ATT is None when no post period, else mean(post_gap).
    # Note: rmse_pre / rmse_post here are computed using treated_weights
    # and control_weights independently (the value reported in the
    # design's synthetic_gap already uses both, so we recompute against
    # the original matrices to keep the diagnostic explicit).
    # ------------------------------------------------------------------
    att_raw, rmse_pre, rmse_post = compute_att_and_fit(
        Y_pre=inputs.Y_pre,
        Y_post=inputs.Y_post,
        treated_weights=design.treated_weights,
        control_weights=design.control_weights,
    )
    att = att_raw if inputs.Y_post is not None else None

    unit_labels = inputs.unit_index.labels
    treated_weights_by_unit = {
        str(unit_labels[i]): float(design.treated_weights[i])
        for i in range(len(unit_labels))
        if design.treated_weights[i] != 0.0
    }
    control_weights_by_unit = {
        str(unit_labels[i]): float(design.control_weights[i])
        for i in range(len(unit_labels))
        if design.control_weights[i] != 0.0
    }

    pre_periods = inputs.pre_time_index.labels
    if inputs.post_time_index is not None:
        post_periods = inputs.post_time_index.labels
        time_periods = np.concatenate([pre_periods, post_periods])
    else:
        time_periods = pre_periods

    # ------------------------------------------------------------------
    # Inference block.
    # ------------------------------------------------------------------
    inference_result: Optional[InferenceResults] = None
    if conformal is not None:
        inference_result = InferenceResults(
            p_value=conformal.p_value,
            ci_lower=conformal.ci_lower,
            ci_upper=conformal.ci_upper,
            standard_error=None,
            confidence_level=1.0 - conformal.alpha,
            method=conformal.method,
            details={
                "block_size": conformal.block_size,
                "alpha": conformal.alpha,
                "pointwise_lower": conformal.pointwise_lower,
                "pointwise_upper": conformal.pointwise_upper,
            },
        )

    # ------------------------------------------------------------------
    # Fit diagnostics: include MDE and detectability in additional_metrics.
    # ------------------------------------------------------------------
    additional_metrics: dict = {}
    if power is not None:
        additional_metrics.update(
            {
                "mde_tau": power.mde_tau,
                "mde_pct": power.mde_pct,
                "mde_baseline": power.baseline,
                "mde_critical_stat": power.critical_stat,
                "mde_feasible": power.feasible,
                "mde_n_post": power.n_post,
                "mde_alpha": power.alpha,
                "mde_power_target": power.power_target,
            }
        )
        if power.detectability is not None:
            additional_metrics["detectability_curve"] = power.detectability

    fit_diagnostics = FitDiagnosticsResults(
        rmse_pre=rmse_pre,
        rmse_post=rmse_post,
        additional_metrics=additional_metrics or None,
    )

    return BaseEstimatorResults(
        effects=EffectsResults(att=att),
        fit_diagnostics=fit_diagnostics,
        time_series=TimeSeriesResults(
            observed_outcome=design.synthetic_treated,
            counterfactual_outcome=design.synthetic_control,
            estimated_gap=design.synthetic_gap,
            time_periods=time_periods,
        ),
        weights=WeightsResults(
            treated_weights_by_unit=treated_weights_by_unit,
            control_weights_by_unit=control_weights_by_unit,
            summary_stats={
                "n_treated": int(design.n_treated),
                "n_control": int(len(unit_labels) - design.n_treated),
                "treated_unit_labels": [
                    str(x) for x in design.selected_unit_labels.tolist()
                ],
            },
        ),
        inference=inference_result,
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
