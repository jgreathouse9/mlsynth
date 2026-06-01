"""Plot helper for the MSQRT estimator.

MSQRT produces a synthetic counterfactual for every treated unit, so the
canonical observed-vs-synthetic synthetic-control chart applies. With one
treated unit its trajectory is drawn against its synthetic; with multiple
treated units (the paper's high-dimensional regime) the cross-unit means are
plotted -- the same series the ATT averages over.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates
from .structures import MSQRTResults


def plot_msqrt(
    results: MSQRTResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Render observed vs. synthetic for the MSQRT result."""
    inputs = results.inputs
    T0 = inputs.T0
    time_labels = inputs.time_labels

    cf_colors = [counterfactual_color] if isinstance(counterfactual_color, str) \
        else list(counterfactual_color)

    if inputs.m == 1:
        treated_name = str(inputs.treated_names[0])
    else:
        treated_name = f"Treated mean (n={inputs.m})"

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
        estimation_method_name="MSQRT",
        counterfactual_names=[f"MSQRT {treated_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
