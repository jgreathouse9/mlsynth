"""Plotting helper for mlSC.

Wraps ``mlsynth.utils.resultutils.plot_estimates`` so we get the standard
observed-vs-counterfactual chart, where "observed" is the aggregate treated
series and "counterfactual" is ``X_disagg @ omega``.
"""

from __future__ import annotations

from typing import List, Union

from ..resultutils import plot_estimates
from .structures import MLSCResults


def plot_mlsc(
    results: MLSCResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, dict, str] = False,
    time_axis_label: str = "Time",
    outcome_label: str = "Outcome",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
) -> None:
    """Render the mlSC aggregate counterfactual against the observed series."""

    inputs = results.inputs
    inference = results.inference

    if isinstance(counterfactual_color, str):
        cf_colors = [counterfactual_color]
    else:
        cf_colors = list(counterfactual_color)

    processed = {
        "Ywide": inputs.Ywide_agg,
        "pre_periods": inputs.T0,
    }

    plot_estimates(
        processed_data_dict=processed,
        time_axis_label=time_axis_label,
        unit_identifier_column_name=unit_label,
        outcome_variable_label=outcome_label,
        treatment_name_label=treatment_label,
        treated_unit_name=str(inputs.treated_unit_name),
        observed_outcome_series=inputs.Y_agg_treated,
        counterfactual_series_list=[inference.counterfactual],
        estimation_method_name="mlSC",
        counterfactual_names=[f"mlSC {inputs.treated_unit_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
