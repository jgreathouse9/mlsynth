"""Plot helper for SNN.

Wraps :func:`mlsynth.utils.resultutils.plot_estimates` so the canonical
observed-vs-counterfactual synthetic-control chart works with the typed
:class:`SNNResults` object. For a single treated unit (e.g. California
under Proposition 99) the treated trajectory is plotted against its SNN
counterfactual; with multiple treated units the cross-unit averages are
plotted.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates
from .structures import SNNResults


def plot_snn(
    results: SNNResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Render observed vs SNN counterfactual for the treated unit(s)."""
    inputs = results.inputs
    treated = inputs.treated_idx
    T0 = inputs.T0
    time_labels = inputs.time_labels

    if isinstance(counterfactual_color, str):
        cf_colors = [counterfactual_color]
    else:
        cf_colors = list(counterfactual_color)

    if treated.size == 1:
        i = int(treated[0])
        observed = inputs.Y[i]
        counterfactual = results.counterfactual_matrix[i]
        treated_name = str(inputs.unit_names[i])
    else:
        # Average observed and counterfactual across treated units.
        observed = inputs.Y[treated].mean(axis=0)
        counterfactual = results.counterfactual_matrix[treated].mean(axis=0)
        treated_name = f"Treated mean (n={treated.size})"

    ywide = pd.DataFrame(
        {treated_name: observed}, index=pd.Index(time_labels, name=time_axis_label)
    )
    processed = {"Ywide": ywide, "pre_periods": T0}

    plot_estimates(
        processed_data_dict=processed,
        time_axis_label=time_axis_label,
        unit_identifier_column_name=unit_label,
        outcome_variable_label=outcome_label,
        treatment_name_label=treatment_label,
        treated_unit_name=treated_name,
        observed_outcome_series=np.asarray(observed, dtype=float),
        counterfactual_series_list=[np.asarray(counterfactual, dtype=float)],
        estimation_method_name="SNN",
        counterfactual_names=[f"SNN {treated_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
