"""Plot helper for SpSyDiD.

Renders the directly-treated group's observed mean trajectory against its
SDID synthetic control (the pure-control units reweighted by the fitted
unit weights plus the SDID level intercept), with a vertical line at the
treatment date -- the spatial-SDID analogue of the canonical
observed-vs-counterfactual chart. The reported ``att`` is the
exposure-adjusted *direct* effect from the weighted regression, so the
visual post-period gap is illustrative of (not numerically identical to)
the estimate.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates
from .structures import SpSyDiDResults


def plot_spsydid(
    results: SpSyDiDResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Observed directly-treated mean vs SDID synthetic from pure controls."""
    inputs = results.inputs
    Y = inputs.outcome_matrix
    direct = inputs.direct_indices
    pure = inputs.pure_control_indices
    T0 = inputs.T0
    time_labels = inputs.time_labels

    if isinstance(counterfactual_color, str):
        cf_colors = [counterfactual_color]
    else:
        cf_colors = list(counterfactual_color)

    observed = Y[direct].mean(axis=0)               # directly-treated mean (T,)

    # Reconstruct the SDID synthetic: intercept + sum_j omega_pure_j Y_jt.
    omega_pure = np.array(
        [results.unit_weights[inputs.unit_names[j]] for j in pure], dtype=float
    )
    intercept = results.metadata.get("sdid_omega_intercept") or 0.0
    counterfactual = intercept + omega_pure @ Y[pure]   # (T,)

    treated_name = "Directly treated (mean)"
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
        estimation_method_name="SpSyDiD",
        counterfactual_names=["SpSyDiD synthetic"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
