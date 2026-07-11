"""Plotting for DPSC: observed treated path vs the private synthetic control."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates


def plot_dpsc(config, results) -> None:
    """Observed treated path vs the differentially private synthetic counterfactual."""
    ts = results.time_series
    obs = np.asarray(ts.observed_outcome, float)
    cf = np.asarray(ts.counterfactual_outcome, float)
    time_labels = np.asarray(ts.time_periods)
    pre = int(results.additional_outputs["pre_periods"])

    processed = {"Ywide": pd.DataFrame(index=pd.Index(time_labels)), "pre_periods": pre}
    plot_estimates(
        processed_data_dict=processed,
        time_axis_label=config.time,
        unit_identifier_column_name=config.unitid,
        outcome_variable_label=config.outcome,
        treatment_name_label=config.treat,
        treated_unit_name=str(results.additional_outputs["treated_name"]),
        observed_outcome_series=obs,
        counterfactual_series_list=[cf],
        estimation_method_name="DPSC",
        treated_series_color=config.treated_color,
        counterfactual_series_colors=[config.counterfactual_color[0]],
        counterfactual_names=["DPSC private synthetic"],
        save_plot_config=config.save,
    )
