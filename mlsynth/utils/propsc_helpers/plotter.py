"""Plotting for PROPSC: one observed-vs-counterfactual panel per proportion."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import PROPSCResults


def plot_propsc(
    results: PROPSCResults,
    time_labels: np.ndarray,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    intervention_index: Optional[int] = None,
    save: Union[bool, str] = False,
) -> None:
    """Draw a grid of observed vs synthetic paths, one panel per proportion.

    Parameters
    ----------
    results : PROPSCResults
        Fitted result whose ``proportions`` supply the paths.
    time_labels : np.ndarray
        Time axis labels, shape ``(T,)``.
    treated_color : str
        Colour of the treated-average path.
    counterfactual_color : str or list of str
        Colour of the synthetic path (first entry used if a list).
    intervention_index : int, optional
        Column index of the first treated period; a vertical marker is drawn
        there when supplied.
    save : bool or str
        If truthy, save the figure to this path (str) or ``propsc.png`` (True).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dependency
        raise MlsynthPlottingError(f"matplotlib unavailable: {exc}") from exc

    cf_color = (counterfactual_color[0]
                if isinstance(counterfactual_color, list) and counterfactual_color
                else counterfactual_color)
    props = results.proportions
    K = len(props)
    ncol = min(3, K)
    nrow = int(np.ceil(K / ncol))
    x = np.asarray(time_labels)

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, fit in enumerate(props):
        ax = axes[k // ncol][k % ncol]
        ax.plot(x, fit.observed, color=treated_color, label="Treated (avg)")
        ax.plot(x, fit.counterfactual, color=cf_color, linestyle="--",
                label="Synthetic")
        if intervention_index is not None and 0 <= intervention_index < len(x):
            ax.axvline(x[intervention_index], color="grey", alpha=0.4)
        ax.set_title(f"{fit.name}  (ATT={fit.att:+.3f})")
        ax.set_xlabel("Time")
        if k == 0:
            ax.legend(frameon=False, fontsize=8)
    for j in range(K, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        f"PROPSC ({results.method.upper()}) — sum of ATTs = "
        f"{results.sum_constraint:+.2e}")
    fig.tight_layout()

    if save:
        path = save if isinstance(save, str) else "propsc.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
