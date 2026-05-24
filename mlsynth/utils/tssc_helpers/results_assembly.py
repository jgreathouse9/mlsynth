"""Assemble the standardized BaseEstimatorResults for TSSC.

Packages the recommended variant's fit plus the Step-1 selection record
into the project's standardized result models, so TSSC's public
``summary`` matches the rest of the mlsynth suite.
"""

from __future__ import annotations

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
from .structures import TSSCInputs, TSSCSelection, TSSCVariantFit


def build_summary(
    inputs: TSSCInputs,
    variant: TSSCVariantFit,
    selection: TSSCSelection,
) -> BaseEstimatorResults:
    """Standardized bundle for the Step-1-recommended TSSC variant."""

    ci_lower, ci_upper = variant.att_ci
    half_width = (ci_upper - ci_lower) / 2.0
    # Approximate SE from the symmetric 95% bootstrap CI (z = 1.96).
    standard_error = half_width / 1.959963985 if np.isfinite(half_width) else None

    test_summary = {
        name: {
            "statistic": t.statistic,
            "ci_lower": t.ci_lower,
            "ci_upper": t.ci_upper,
            "rejected": t.rejected,
        }
        for name, t in selection.tests.items()
    }

    return BaseEstimatorResults(
        effects=EffectsResults(att=variant.att),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=variant.rmse_pre,
            rmse_post=variant.rmse_post,
            additional_metrics={"r2_pre": variant.r2_pre},
        ),
        time_series=TimeSeriesResults(
            observed_outcome=inputs.y,
            counterfactual_outcome=variant.counterfactual,
            estimated_gap=variant.gap,
            time_periods=inputs.time_labels,
        ),
        weights=WeightsResults(
            donor_weights=variant.donor_weights,
            summary_stats={
                "n_donors": int(inputs.n_donors),
                "intercept": variant.intercept,
                "treated_unit": str(inputs.treated_unit_name),
            },
        ),
        inference=InferenceResults(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            standard_error=standard_error,
            method="bootstrap",
            details={"selection_alpha": selection.alpha},
        ),
        method_details=MethodDetailsResults(
            method_name=f"TSSC ({selection.recommended})",
            parameters_used={
                "recommended_method": selection.recommended,
                "selection_alpha": selection.alpha,
                "subsample_size": selection.subsample_size,
                "n_subsamples": selection.n_subsamples,
                "decision_path": list(selection.decision_path),
                "restriction_tests": test_summary,
            },
        ),
    )
