"""Presentation for SL. Returns its figure; never shows, saves or prints it."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np


def plot_sl(results: Any, *, treated_color: str = "black",
            counterfactual_color: Union[str, List[str]] = "#3b78e7",
            title: Optional[str] = None):
    """Observed against the ensemble counterfactual, with the gap beneath.

    Parameters
    ----------
    results : SLResults
        A fitted result.
    treated_color, counterfactual_color : str
        Line colours.
    title : str, optional
        Overrides the default.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # BaseEstimatorConfig types this as List[str]; matplotlib wants a scalar.
    colour = (counterfactual_color[0] if isinstance(counterfactual_color, list)
              else counterfactual_color)
    f = results.fit
    ts = results.time_series
    x = np.arange(len(ts.observed_outcome))
    T0 = results.inputs.T0
    split = T0 - f.weight_periods

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(x, ts.observed_outcome, color=treated_color, label="observed")
    ax.plot(x, ts.counterfactual_outcome, color=colour,
            linestyle="--", label="synthetic learner")
    ax.axvline(split, color="grey", linestyle=":", linewidth=1)
    ax.axvline(T0, color="grey", linestyle="-", linewidth=1)
    ax.set_ylabel("outcome")
    ax.legend(frameon=False)
    ax.set_title(title or (
        f"SL: {len(f.experts)} experts, effective K {f.effective_k:.2f}, "
        f"p = {f.p_value:.3f}"))

    bx.axhline(0.0, color="grey", linewidth=1)
    bx.plot(x, ts.estimated_gap, color=colour)
    bx.axvline(split, color="grey", linestyle=":", linewidth=1)
    bx.axvline(T0, color="grey", linestyle="-", linewidth=1)
    bx.set_ylabel("gap")
    bx.set_xlabel("period")
    fig.tight_layout()
    return fig
