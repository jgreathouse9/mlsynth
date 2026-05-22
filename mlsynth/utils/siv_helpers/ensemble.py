"""Event-study plot for SIV results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import SIVResults


def plot_event_study(results: SIVResults) -> None:
    """Plot per-period reduced-form coefficients with the T0 reference line."""

    inference = results.inference
    if inference.event_study_coefs.size == 0:
        raise MlsynthPlottingError(
            "No event-study coefficients available -- enable "
            "inference_method='conformal' to populate them."
        )

    coefs = inference.event_study_coefs
    t = np.asarray(results.inputs.time_index.labels)
    if t.shape[0] != coefs.shape[0]:
        t = np.arange(coefs.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="grey", lw=1, ls=":")
    ax.axvline(t[results.inputs.T0 - 1], color="red", lw=1, alpha=0.6,
               label="Treatment start")
    ax.plot(t, coefs, "o-", lw=1.5, label="Reduced-form event coef")
    ax.set_title("SIV event-study coefficients")
    ax.set_xlabel("Period")
    ax.set_ylabel("theta_t")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
