"""Plot helper for MC-NNM.

MC-NNM produces a genuine fitted counterfactual for every cell, so the
canonical observed-vs-counterfactual synthetic-control chart applies
directly. For a single treated unit (e.g. California under Prop 99) the
treated trajectory is plotted against its MC-NNM counterfactual; with
multiple treated units the cross-unit averages are plotted.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from ..resultutils import plot_estimates
from .structures import MCNNMResults


def _is_staggered(results: MCNNMResults) -> bool:
    """More than one distinct adoption time among the treated units."""
    D = results.inputs.D
    adopt = {int(np.argmax(D[i] == 1)) for i in results.inputs.treated_idx}
    return len(adopt) > 1


def _plot_event_study(results, treated_color, counterfactual_color, save,
                      outcome_label):
    """Event-study plot: average effect by time relative to adoption."""
    import os
    import matplotlib.pyplot as plt

    es = results.event_study
    if not es:
        return
    es_items = sorted(es.items())
    e = np.array([k for k, _ in es_items])
    v = np.array([val for _, val in es_items])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="grey", lw=1, ls=":")
    ax.axvline(-0.5, color="grey", lw=1.5, ls="--", label="adoption")
    ax.plot(e, v, "-o", color=counterfactual_color, ms=4, lw=1.8)
    ax.set_xlabel("Time relative to adoption")
    ax.set_ylabel(f"Effect on {outcome_label}")
    ax.set_title(f"MC-NNM event study (overall ATT = {results.att:+.3f})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    if save:
        fname = save if isinstance(save, str) else "MC-NNM_event_study.png"
        if not os.path.splitext(fname)[1]:
            fname += ".png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(fname)}")
        plt.close(fig)
    else:
        plt.show()


def plot_mcnnm(
    results: MCNNMResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Render the MC-NNM result.

    For a single adoption time this is the canonical observed-vs-
    counterfactual trajectory; under staggered adoption (multiple adoption
    times) it is an event-study plot (average effect by time relative to
    each unit's adoption), which avoids blending cohorts at different event
    times on a single calendar axis.
    """
    inputs = results.inputs
    treated = inputs.treated_idx
    T0 = inputs.T0
    time_labels = inputs.time_labels

    cf_color_single = (counterfactual_color if isinstance(counterfactual_color, str)
                       else (counterfactual_color[0] if counterfactual_color else "red"))
    if _is_staggered(results):
        _plot_event_study(results, treated_color, cf_color_single, save,
                          outcome_label)
        return

    cf_colors = [counterfactual_color] if isinstance(counterfactual_color, str) \
        else list(counterfactual_color)

    if treated.size == 1:
        i = int(treated[0])
        observed = inputs.Y[i]
        counterfactual = results.counterfactual_matrix[i]
        treated_name = str(inputs.unit_names[i])
    else:
        observed = inputs.Y[treated].mean(axis=0)
        counterfactual = results.counterfactual_matrix[treated].mean(axis=0)
        treated_name = f"Treated mean (n={treated.size})"

    ywide = pd.DataFrame(
        {treated_name: observed}, index=pd.Index(time_labels, name=time_axis_label)
    )
    plot_estimates(
        processed_data_dict={"Ywide": ywide, "pre_periods": T0},
        time_axis_label=time_axis_label,
        unit_identifier_column_name=unit_label,
        outcome_variable_label=outcome_label,
        treatment_name_label=treatment_label,
        treated_unit_name=treated_name,
        observed_outcome_series=np.asarray(observed, dtype=float),
        counterfactual_series_list=[np.asarray(counterfactual, dtype=float)],
        estimation_method_name="MC-NNM",
        counterfactual_names=[f"MC-NNM {treated_name}"],
        treated_series_color=treated_color,
        save_plot_config=save,
        counterfactual_series_colors=cf_colors,
    )
