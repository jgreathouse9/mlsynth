"""Plot helper for SparseSC.

Wraps :func:`mlsynth.utils.resultutils.plot_estimates` so the
observed-vs-counterfactual chart works with the typed results object.
"""

from __future__ import annotations

from typing import List, Union

from ..resultutils import plot_estimates
from .structures import SparseSCResults


def plot_sparse_sc(
    results: SparseSCResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
) -> None:
    """Render observed vs SparseSC counterfactual on the treated unit."""
    inputs = results.inputs

    if isinstance(counterfactual_color, str):
        cf_colors = [counterfactual_color]
    else:
        cf_colors = list(counterfactual_color)

    processed = {"Ywide": inputs.Ywide, "pre_periods": inputs.T0_total}

    plot_estimates(
        processed_data_dict=processed,
        time_axis_label=time_axis_label,
        unit_identifier_column_name=unit_label,
        outcome_variable_label=inputs.outcome,
        treatment_name_label=treatment_label,
        treated_unit_name=str(inputs.treated_unit_name),
        observed_outcome_series=inputs.Y1,
        counterfactual_series_list=[results.counterfactual],
        estimation_method_name="SparseSC",
        counterfactual_names=[f"SparseSC {inputs.treated_unit_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
