"""Validate native trajectories before including an estimate in a study."""

from typing import Any

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..vanillasc_helpers.structures import VanillaSCResults
from .structures import FittedPaths


def extract_paths(
    fitted: VanillaSCResults, name: str, periods: list[Any]
) -> FittedPaths:
    """Extract common-period trajectories and donor weights without rounding."""
    series = fitted.time_series
    times = list(series.time_periods)
    if times != periods:
        raise MlsynthDataError("Estimator changed the common trajectory periods.")
    observed = np.asarray(series.observed_outcome, dtype=float).reshape(-1)
    synthetic = np.asarray(series.counterfactual_outcome, dtype=float).reshape(-1)
    donor_weights = fitted.weights.donor_weights
    if (
        observed.shape != (len(times),)
        or synthetic.shape != observed.shape
        or not np.isfinite(
            np.r_[observed, synthetic, list(donor_weights.values())]
        ).all()
    ):
        raise MlsynthDataError("Estimator returned missing or nonfinite paths/weights.")
    gap = observed - synthetic
    if not np.isfinite(gap).all():
        raise MlsynthDataError("Estimator returned a nonfinite trajectory.")
    return FittedPaths(
        paths=pd.DataFrame(
            {
                "spec_id": name,
                "time": times,
                "observed": observed,
                "synthetic": synthetic,
                "gap": gap,
            }
        ),
        weights=pd.DataFrame(
            [
                {"spec_id": name, "donor": donor, "weight": weight}
                for donor, weight in donor_weights.items()
            ]
        ),
    )
