"""Plotting for BEAST: observed vs immunized synthetic, with the effect band."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates


def plot_beast(config, results) -> None:
    """Observed treated path vs the BEAST synthetic counterfactual + 95% band."""
    ts = results.time_series
    obs = np.asarray(ts.observed_outcome, float)
    cf = np.asarray(ts.counterfactual_outcome, float)
    time_labels = np.asarray(ts.time_periods)
    pre = int(results.additional_outputs["pre_periods"])

    # Counterfactual uncertainty band (post only): cf = obs - effect, so the
    # effect interval [pi_lower, pi_upper] maps to [obs - pi_upper, obs - pi_lower].
    det = results.inference.details
    band = None
    if det.get("pi_lower") is not None:
        lo = np.full_like(obs, np.nan)
        hi = np.full_like(obs, np.nan)
        lo[pre:] = obs[pre:] - np.asarray(det["pi_upper"], float)
        hi[pre:] = obs[pre:] - np.asarray(det["pi_lower"], float)
        band = np.column_stack([lo, hi])

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
        estimation_method_name="BEAST",
        treated_series_color=config.treated_color,
        counterfactual_series_colors=[config.counterfactual_color[0]],
        counterfactual_names=["BEAST synthetic"],
        save_plot_config=config.save,
        uncertainty_intervals_array=band,
    )
