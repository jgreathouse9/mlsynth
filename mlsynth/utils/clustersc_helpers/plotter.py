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
from ..plotting import Plotter, mlsynth_style
from .structures import CLUSTERSCResults


def plot_clustersc(results: CLUSTERSCResults) -> None:
    """Observed treated series vs the PCR / RPCA counterfactual(s)."""

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

    with mlsynth_style():
        plotter = Plotter.from_config(None)
        ax = plotter.observed_vs_counterfactual(
            times=t,
            observed=np.asarray(inputs.treated_outcome, dtype=float),
            counterfactuals=counterfactuals,
            labels=labels,
            treated_label=str(inputs.treated_unit_name),
            intervention=intervention,
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
