"""Plotting for COMPSC: the composition against its counterfactual, plus the
Aitchison displacement that summarizes the whole compositional shift."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import COMPSCResults


def plot_compsc(
    results: COMPSCResults,
    time_labels: np.ndarray,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    intervention_index: Optional[int] = None,
    save: Union[bool, str] = False,
) -> None:
    """Draw observed vs counterfactual shares, one panel per category, and a
    final panel with the Aitchison distance between the two compositions.

    Parameters
    ----------
    results : COMPSCResults
        Fitted result whose ``categories`` supply the paths.
    time_labels : np.ndarray
        Time axis labels, shape ``(T,)``.
    treated_color : str
        Colour of the observed path.
    counterfactual_color : str or list of str
        Colour of the counterfactual path (first entry used if a list).
    intervention_index : int, optional
        Period index of the first treated period; a vertical marker is drawn
        there when supplied.
    save : bool or str
        If truthy, save the figure to this path (str) or ``compsc.png`` (True).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dependency
        raise MlsynthPlottingError(f"matplotlib unavailable: {exc}") from exc

    cf_color = (counterfactual_color[0]
                if isinstance(counterfactual_color, list) and counterfactual_color
                else counterfactual_color)
    cats = results.categories
    n_panels = len(cats) + 1
    ncol = min(3, n_panels)
    nrow = int(np.ceil(n_panels / ncol))
    x = np.asarray(time_labels)

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, fit in enumerate(cats):
        ax = axes[k // ncol][k % ncol]
        ax.plot(x, fit.observed, color=treated_color, label="Observed")
        ax.plot(x, fit.counterfactual, color=cf_color, linestyle="--",
                label="Counterfactual")
        if intervention_index is not None and 0 <= intervention_index < len(x):
            ax.axvline(x[intervention_index], color="grey", alpha=0.4)
        ax.set_title(f"{fit.name}  (ATT={fit.att:+.3f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Share")
        if k == 0:
            ax.legend(frameon=False, fontsize=8)

    ax = axes[len(cats) // ncol][len(cats) % ncol]
    ax.plot(x, results.aitchison_distance, color=treated_color)
    if intervention_index is not None and 0 <= intervention_index < len(x):
        ax.axvline(x[intervention_index], color="grey", alpha=0.4)
    ax.set_title("Compositional displacement")
    ax.set_xlabel("Time")
    ax.set_ylabel("Aitchison distance")

    for j in range(n_panels, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    subtitle = f"COMPSC ({results.geometry})"
    if results.baseline is not None:
        subtitle += f", baseline = {results.baseline}"
    subtitle += f" — sum of share effects = {results.sum_constraint:+.2e}"
    fig.suptitle(subtitle)
    fig.tight_layout()

    if save:
        path = save if isinstance(save, str) else "compsc.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
