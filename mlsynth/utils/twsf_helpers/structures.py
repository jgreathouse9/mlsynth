"""Frozen containers for the TWSF estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class TWSFInputs:
    """The two sides of the panel TWSF consumes.

    Attributes
    ----------
    y_target_pre : np.ndarray
        The focal unit's outcomes over the unit-side window, shape ``(T0,)``.
    Y_donors_pre : np.ndarray
        The donors' outcomes over the same window, shape ``(N1, T0)``. Every
        unit is under control here, which is what makes the cross-sectional
        relationship estimable.
    Y_donors_post : np.ndarray
        The donors' outcomes over the treated window, shape ``(N1, T1)``. Every
        donor is treated throughout, which is what makes the treated regime's
        dynamics estimable.
    donor_names : list of str
    target_name : str
    unit_side_end : object
        Time label at which the unit-side window closes (the first treatment
        date among the donors).
    time_side_start : object
        Time label at which the treated window opens (the last treatment date
        among the donors, so every donor is treated across it).
    forecast_labels : list
        Labels for the forecast dates, extrapolated past the panel.
    staggered : bool
        Whether the donors adopted on different dates.
    """

    y_target_pre: np.ndarray
    Y_donors_pre: np.ndarray
    Y_donors_post: np.ndarray
    donor_names: List[str]
    target_name: str
    unit_side_end: object
    time_side_start: object
    forecast_labels: list = field(default_factory=list)
    staggered: bool = False


@dataclass(frozen=True)
class TWSFFit:
    """What the two regressions and their combination produced."""

    forecast: np.ndarray          # (h,) treated potential outcome past the panel
    std_error: np.ndarray         # (h,) plug-in standard deviation
    lower: np.ndarray
    upper: np.ndarray
    beta: np.ndarray              # (N1,) unit-side weights
    alpha: np.ndarray             # (L,) one-step temporal rule
    sigma2: float
    n_blocks: int
    multistep: str
