"""Display plot for SDID.

A single treated cohort is drawn as an observed-versus-counterfactual chart
through the shared in-house ``Plotter`` (``mlsynth.utils.plotting``), so SDID
looks like every other single-treated-unit estimator. A staggered design (more
than one adoption cohort) keeps the pooled event-study chart rendered by
:func:`mlsynth.utils.resultutils.SDID_plot`, the only sensible aggregate view
when cohorts adopt at different times.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...exceptions import MlsynthPlottingError
from ..resultutils import SDID_plot
from .structures import SDIDResults


def plot_sdid(results: SDIDResults, **plot_kwargs: Any) -> None:
    """Render the SDID display plot, choosing the view by treated structure."""

    if len(results.cohorts) == 1:
        _plot_single_cohort(results, **plot_kwargs)
    else:
        # ``SDID_plot`` consumes the raw dict shape; the raw payload is kept on
        # ``results.raw`` precisely so we don't have to re-marshal anything.
        SDID_plot(sdid_results_dict=results.raw, **plot_kwargs)


def _plot_single_cohort(results: SDIDResults, *, title: str | None = None,
                        **_ignored: Any) -> None:
    """Observed treated series vs the SDID counterfactual, one panel."""
    import matplotlib.pyplot as plt

    from ..plotting import Plotter, mlsynth_style

    inputs = results.inputs
    ts = results.time_series
    t = np.asarray(ts.time_periods).ravel()
    observed = np.asarray(ts.observed_outcome, dtype=float).ravel()
    counterfactual = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    treated = str(inputs.treated_unit_name)
    time_name = getattr(getattr(inputs.Ywide, "index", None), "name", None) or "Time"

    with mlsynth_style():
        plotter = Plotter.from_config(getattr(results, "plot_config", None))
        ax = plotter.observed_vs_counterfactual(
            times=t,
            observed=observed,
            counterfactuals=[counterfactual],
            labels=["SDID counterfactual"],
            treated_label=treated,
            intervention=ts.intervention_time,
            outcome=inputs.outcome,
            time=time_name,
            title=title or f"Synthetic Difference-in-Differences: {treated}",
        )
        fig = ax.figure
        try:
            plt.show()
        except Exception as exc:
            raise MlsynthPlottingError(f"SDID plotting failed: {exc}") from exc
        finally:
            plt.close(fig)
