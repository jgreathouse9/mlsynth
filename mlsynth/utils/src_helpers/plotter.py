"""Plotting for SRC: the treated trajectory vs the SRC counterfactual.

Delegates to the shared :class:`~mlsynth.utils.plotting.Plotter` so SRC matches
the library's visual archetype.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from ..plotting import Plotter, mlsynth_style
from .structures import SRCResults


def plot_src(
    results: SRCResults,
    *,
    outcome: str,
    time: str,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str] = False,
) -> None:
    """Overlay the treated unit and its SRC counterfactual (``bias + Y0 (theta*w)``)."""
    import matplotlib.pyplot as plt

    inputs = results.inputs
    times = np.asarray(inputs.time_index)

    with mlsynth_style():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plotter = Plotter(
            treated_color=treated_color,
            counterfactual_colors=counterfactual_color,
        )
        plotter.observed_vs_counterfactual(
            times,
            inputs.Y_treated,
            results.counterfactual,
            labels=["SRC"],
            treated_label=inputs.treated_label,
            intervention=inputs.intervention_time,
            outcome=outcome,
            time=time,
            title="Treated vs. SRC counterfactual",
            ax=ax,
        )
        fig.tight_layout()
        if save:
            fname = save if isinstance(save, str) else "src_fit.png"
            fig.savefig(fname, dpi=130, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
