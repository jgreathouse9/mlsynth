"""Multivariate observed-vs-counterfactual plot for CMBSTS.

The standardized ``result.plot()`` draws the treated series; this helper lays
out all ``d`` group series with their pointwise credible bands.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import CMBSTSResults


def plot_cmbsts(results: CMBSTSResults, ax: Optional[Any] = None, save: Any = False) -> Any:
    """Plot each group series' observed path against its CMBSTS counterfactual.

    Parameters
    ----------
    results : CMBSTSResults
        A fitted CMBSTS result.
    ax : matplotlib Axes, optional
        Unused for the multi-panel layout; accepted for API symmetry.
    save : bool or str
        If a path string, save the figure there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib is a hard dep
        raise MlsynthPlottingError(f"CMBSTS plotting needs matplotlib: {exc}") from exc

    try:
        inp = results.inputs
        det = results.inference_detail
        d = inp.d
        T0 = inp.T0
        labels = np.asarray(inp.time_labels)
        fig, axes = plt.subplots(d, 1, figsize=(8, 3 * d), squeeze=False, sharex=True)
        for s in range(d):
            a = axes[s, 0]
            a.plot(labels, inp.Y[:, s], color="black", label="Observed")
            a.plot(labels, det.counterfactual_full[:, s], color="red",
                   linestyle="--", label="Counterfactual")
            post_idx = np.arange(T0, inp.T)
            a.fill_between(labels[post_idx],
                           inp.Y[post_idx, s] - det.effect_upper[:, s],
                           inp.Y[post_idx, s] - det.effect_lower[:, s],
                           color="red", alpha=0.15)
            a.axvline(labels[T0] if T0 < inp.T else labels[-1], color="grey", linestyle=":")
            a.set_title(str(inp.series_names[s]))
            a.legend(loc="best", fontsize=8)
        fig.tight_layout()
        if isinstance(save, str):
            fig.savefig(save, bbox_inches="tight")
        return fig
    except MlsynthPlottingError:  # pragma: no cover - defensive error translation
        raise
    except Exception as exc:  # pragma: no cover - defensive error translation
        raise MlsynthPlottingError(f"CMBSTS plotting failed: {exc}") from exc
