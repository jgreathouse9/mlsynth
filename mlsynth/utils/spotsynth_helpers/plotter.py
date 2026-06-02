"""Plot helper for SPOTSYNTH: treated vs. screened synthetic control."""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates
from .structures import SpotSynthResults


def plot_spotsynth(
    results: SpotSynthResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Render the treated series against the spillover-screened synthetic control."""
    inputs = results.inputs
    treated_name = str(inputs.treated_name)
    cf_colors = [counterfactual_color] if isinstance(counterfactual_color, str) \
        else list(counterfactual_color)

    observed = np.asarray(inputs.y, dtype=float)
    counterfactual = np.asarray(results.counterfactual, dtype=float)
    ywide = pd.DataFrame(
        {treated_name: observed},
        index=pd.Index(inputs.time_labels, name=time_axis_label),
    )
    n_excl = results.screen.excluded_idx.size
    cf_name = f"SPOTSYNTH {treated_name} ({n_excl} donor(s) screened out)"

    # Pass the Bayesian posterior-predictive credible band when available.
    extra = {}
    if results.counterfactual_lower is not None:
        extra["uncertainty_intervals_array"] = np.column_stack([
            np.asarray(results.counterfactual_lower, dtype=float),
            np.asarray(results.counterfactual_upper, dtype=float),
        ])

    try:
        plot_estimates(
            processed_data_dict={"Ywide": ywide, "pre_periods": inputs.T0},
            time_axis_label=time_axis_label,
            unit_identifier_column_name=unit_label,
            outcome_variable_label=outcome_label,
            treatment_name_label=treatment_label,
            treated_unit_name=treated_name,
            observed_outcome_series=observed,
            counterfactual_series_list=[counterfactual],
            estimation_method_name="SPOTSYNTH",
            counterfactual_names=[cf_name],
            treated_series_color=treated_color,
            save_plot_config=save,
            counterfactual_series_colors=cf_colors,
            **extra,
        )
    except TypeError:
        # plot_estimates signature without uncertainty support -- draw plainly.
        plot_estimates(
            processed_data_dict={"Ywide": ywide, "pre_periods": inputs.T0},
            time_axis_label=time_axis_label,
            unit_identifier_column_name=unit_label,
            outcome_variable_label=outcome_label,
            treatment_name_label=treatment_label,
            treated_unit_name=treated_name,
            observed_outcome_series=observed,
            counterfactual_series_list=[counterfactual],
            estimation_method_name="SPOTSYNTH",
            counterfactual_names=[cf_name],
            treated_series_color=treated_color,
            save_plot_config=save,
            counterfactual_series_colors=cf_colors,
        )
