"""Plot helper for RMSI: observed vs. imputed counterfactual."""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates
from .structures import RMSIResults


def plot_rmsi(
    results: RMSIResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Render observed vs. RMSI-imputed counterfactual for the treated units."""
    inputs = results.inputs
    T0 = inputs.T0
    time_labels = inputs.time_labels
    cf_colors = [counterfactual_color] if isinstance(counterfactual_color, str) \
        else list(counterfactual_color)

    if inputs.treated_idx.size == 1:
        treated_name = str(inputs.unit_names[int(inputs.treated_idx[0])])
    else:
        treated_name = f"Treated mean (n={inputs.treated_idx.size})"

    observed = np.asarray(results.treated_mean, dtype=float)
    counterfactual = np.asarray(results.synthetic_mean, dtype=float)
    ywide = pd.DataFrame(
        {treated_name: observed},
        index=pd.Index(time_labels, name=time_axis_label),
    )
    plot_estimates(
        processed_data_dict={"Ywide": ywide, "pre_periods": T0},
        time_axis_label=time_axis_label,
        unit_identifier_column_name=unit_label,
        outcome_variable_label=outcome_label,
        treatment_name_label=treatment_label,
        treated_unit_name=treated_name,
        observed_outcome_series=observed,
        counterfactual_series_list=[counterfactual],
        estimation_method_name="RMSI",
        counterfactual_names=[f"RMSI {treated_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
