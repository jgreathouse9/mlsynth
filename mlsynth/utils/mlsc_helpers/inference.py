"""Counterfactual path and effect summaries for mlSC.

The aggregate-level counterfactual is just ``X_disagg @ omega`` evaluated
across all ``T`` periods. The pre-period RMSE and post-period mean gap come
from comparing this trajectory against the observed aggregate treated
outcome.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .structures import MLSCInference, MLSCInputs


def counterfactual_path(inputs: MLSCInputs, omega: np.ndarray) -> MLSCInference:
    """Build the counterfactual and gap series."""
    cf = inputs.X_disagg @ omega
    gap = inputs.Y_agg_treated - cf
    return MLSCInference(counterfactual=cf, gap=gap)


def summarize_effects(inputs: MLSCInputs, inference: MLSCInference) -> Tuple[float, float]:
    """Return ``(att, pre_rmse)`` for the aggregate treated unit."""
    T0 = inputs.T0
    T = inputs.T
    gap = inference.gap

    if T > T0:
        att = float(np.mean(gap[T0:]))
    else:
        att = float("nan")

    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    return att, pre_rmse
