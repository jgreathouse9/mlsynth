"""Display plot for CLUSTERSC results.

A single-panel observed-vs-counterfactual chart drawn through the shared in-house
``Plotter`` (``mlsynth.utils.plotting``), so CLUSTERSC looks like every other
estimator. When both the PCR and RPCA estimators are run, their counterfactuals
are overlaid on the one panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from ..plotting import Plotter, mlsynth_style, select_pi_bands
from .structures import CLUSTERSCResults


def _scpi_full_band(post_lower, post_upper, T: int, T0: int):
    """Align post-period SCPI bounds to the full T-length axis (NaN pre)."""
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    post_lower = np.asarray(post_lower, dtype=float).ravel()
    post_upper = np.asarray(post_upper, dtype=float).ravel()
    n = min(len(post_lower), T - T0)
    lo[T0:T0 + n] = post_lower[:n]
    hi[T0:T0 + n] = post_upper[:n]
    return lo, hi


def plot_clustersc(results: CLUSTERSCResults, *, plot_bands: str = "pointwise"):
    """Observed treated series vs the PCR / RPCA counterfactual(s).

    When SCPI prediction intervals were computed (``compute_scpi_pi=True``),
    ``plot_bands`` selects which band(s) to shade around the primary
    counterfactual: ``"pointwise"`` (default), ``"simultaneous"``, or ``"both"``.
    Returns the Matplotlib axis drawn on.
    """

    inputs = results.inputs
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:
        t = np.arange(inputs.T)

    counterfactuals: list[np.ndarray] = []
    labels: list[str] = []
    if results.pcr is not None:
        counterfactuals.append(np.asarray(results.pcr.counterfactual, dtype=float))
        labels.append("PCR counterfactual")
    if results.rpca is not None:
        counterfactuals.append(np.asarray(results.rpca.counterfactual, dtype=float))
        labels.append("RPCA counterfactual")

    intervention = t[inputs.T0] if 0 <= inputs.T0 < t.size else None

    # SCPI prediction-interval band(s), aligned to the full axis, shaded around
    # the primary counterfactual (the first series).
    pointwise = simultaneous = None
    sc = getattr(results.cluster_inference, "scpi", None) if results.cluster_inference else None
    if sc is not None:
        pointwise = _scpi_full_band(sc.cf_lower, sc.cf_upper, inputs.T, inputs.T0)
        simultaneous = _scpi_full_band(sc.cf_lower_simul, sc.cf_upper_simul,
                                       inputs.T, inputs.T0)
    interval, interval2, interval_label = select_pi_bands(
        pointwise, simultaneous, plot_bands)

    with mlsynth_style():
        plotter = Plotter.from_config(None)
        ax = plotter.observed_vs_counterfactual(
            times=t,
            observed=np.asarray(inputs.treated_outcome, dtype=float),
            counterfactuals=counterfactuals,
            labels=labels,
            treated_label=str(inputs.treated_unit_name),
            intervention=intervention,
            interval=interval, interval_label=interval_label, interval2=interval2,
            outcome=inputs.outcome_name or "outcome",
            time=inputs.time_name or "time",
            title=f"CLUSTERSC: {inputs.treated_unit_name}",
        )
        fig = ax.figure
        try:
            plt.show()
        except Exception as exc:
            raise MlsynthPlottingError(f"CLUSTERSC plotting failed: {exc}") from exc
        finally:
            plt.close(fig)
    return ax
