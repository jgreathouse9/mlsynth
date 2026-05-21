"""Event-study plot helper for SDID.

Wraps :func:`mlsynth.utils.resultutils.SDID_plot`, which already knows
how to render an event-study chart with placebo CI bands, so the new
``MLSC``-style typed-results object can be plotted without duplicating
visualization code.
"""

from __future__ import annotations

from typing import Any, Dict

from ..resultutils import SDID_plot
from .structures import SDIDResults


def plot_sdid(results: SDIDResults, **plot_kwargs: Any) -> None:
    """Render the SDID event-study chart from a typed results object."""

    # ``SDID_plot`` consumes the raw dict shape; the raw payload is kept on
    # ``results.raw`` precisely so we don't have to re-marshal anything.
    SDID_plot(sdid_results_dict=results.raw, **plot_kwargs)
