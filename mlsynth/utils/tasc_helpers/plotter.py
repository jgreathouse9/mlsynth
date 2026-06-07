"""Plotting helper for TASC.

Wraps ``mlsynth.utils.resultutils.plot_estimates`` so we get the standard
observed-vs-counterfactual chart with the posterior-based CI band shaded
behind the counterfactual curve.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from ..resultutils import plot_estimates
from .structures import TASCResults


def plot_tasc(
    results: TASCResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, dict] = False,
    time_axis_label: str = "Time",
    outcome_label: str | None = None,
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
) -> None:
    """Render the TASC counterfactual against the observed series."""

    inputs = results.inputs
    inference = results.inference_detail

    counterfactuals = [inference.counterfactual]
    if isinstance(counterfactual_color, str):
        cf_colors = [counterfactual_color]
    else:
        cf_colors = list(counterfactual_color)

    ci_array = np.column_stack([inference.ci_lower, inference.ci_upper])

    processed = {
        "Ywide": inputs.Ywide,
        "pre_periods": inputs.pre_periods,
    }

    plot_estimates(
        processed_data_dict=processed,
        time_axis_label=time_axis_label,
        unit_identifier_column_name=unit_label,
        outcome_variable_label=outcome_label or "Outcome",
        treatment_name_label=treatment_label,
        treated_unit_name=str(inputs.treated_unit_name),
        observed_outcome_series=inputs.y_target,
        counterfactual_series_list=counterfactuals,
        estimation_method_name="TASC",
        counterfactual_names=[f"TASC {inputs.treated_unit_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
        uncertainty_intervals_array=ci_array,
    )
