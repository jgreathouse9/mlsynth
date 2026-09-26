"""Presentation for SL. Returns its figure; never shows, saves or prints it.

Both panels come from the shared :class:`~mlsynth.utils.plotting.Plotter`, so SL
draws the library's archetype in the library's style and a change to either
reaches SL with every other estimator. SL adds one thing the archetype has no
reason to carry: a second vertical marker at the expert-training split, since
Algorithm 1 divides the pre-treatment window in two and a reader cannot
otherwise see which stretch scored the experts.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from ..plotting import Plotter, mlsynth_style


def plot_sl(results: Any, *, outcome: str = "Outcome", time: str = "Time",
            treated_color: str = "black",
            counterfactual_color: Union[str, List[str]] = "#3b78e7",
            title: Optional[str] = None):
    """Observed against the ensemble counterfactual, with the gap beneath.

    Parameters
    ----------
    results : SLResults
        A fitted result.
    outcome, time : str
        Axis labels.
    treated_color, counterfactual_color : str or list of str
        Line colours. A list is accepted because ``BaseEstimatorConfig`` types
        the counterfactual colour that way.
    title : str, optional
        Overrides the default, which reports the library size, the weights'
        perplexity and the test's p-value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    f, ts = results.fit, results.time_series
    times = np.asarray(ts.time_periods)
    observed = np.asarray(ts.observed_outcome, dtype=float)
    counterfactual = np.asarray(ts.counterfactual_outcome, dtype=float)
    gap = np.asarray(ts.estimated_gap, dtype=float)
    # Algorithm 1's first split: the periods before it trained the experts, the
    # ones between it and the treatment scored them.
    split = times[results.inputs.T0 - f.weight_periods]

    with mlsynth_style():
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
        plotter = Plotter(treated_color=treated_color,
                          counterfactual_colors=counterfactual_color)
        plotter.observed_vs_counterfactual(
            times, observed, counterfactual,
            labels=["Synthetic learner"],
            treated_label=results.inputs.treated_label,
            intervention=ts.intervention_time,
            outcome=outcome, time=time,
            title=title or (
                f"SL: {len(f.experts)} experts, effective K "
                f"{f.effective_k:.2f}, p = {f.p_value:.3f}"),
            ax=ax)
        plotter.gap(times, gap, intervention=ts.intervention_time,
                    outcome=outcome, time=time,
                    title="Estimated gap", ax=bx)
        for axis, label in ((ax, "Expert-training split"), (bx, None)):
            axis.axvline(split, color=plotter.intervention_color,
                         linestyle=":", linewidth=1.2, label=label)
        ax.legend()
        ax.set_xlabel("")          # the axes are shared; one label, on the bottom
        fig.tight_layout()
    return fig
