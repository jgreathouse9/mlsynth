"""Plot helper for LPCA.

The observed-versus-counterfactual chart comes from the standardized
:meth:`mlsynth.config_models.BaseEstimatorResults.plot`, which LPCA populates
like every other effect estimator. What is specific to this method is the local
spectrum: the singular values of the treated unit's neighbourhood, and the rank
the ratio rule kept from them. That is the diagnostic the paper leaves to the
practitioner, so it gets its own view.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .structures import LPCAResults


def plot_local_spectrum(
    results: LPCAResults, *, ax: Optional[Any] = None, save: Any = False
) -> Any:
    """Draw the treated unit's local singular values and the rank kept.

    Parameters
    ----------
    results : LPCAResults
    ax : matplotlib Axes, optional
    save : bool or str
        Path to save to, or False.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    spectrum = np.asarray(results.singular_values, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    order = np.arange(1, spectrum.size + 1)
    ax.plot(order, spectrum, "-o", ms=5, lw=1.6, color="black")
    kept = results.local_rank
    ax.axvline(kept + 0.5, ls="--", lw=1.4, color="red",
               label=f"rank kept = {kept}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")
    ax.set_xticks(order)
    active = results.metadata.get("rank_rule_active", True)
    suffix = "" if active else "  (ratio rule inert at K <= 15)"
    ax.set_title(f"LPCA local spectrum, K = {results.n_neighbours}{suffix}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    if save:
        ax.figure.savefig(save if isinstance(save, str) else "lpca_spectrum.png",
                          dpi=150, bbox_inches="tight")
    return ax
