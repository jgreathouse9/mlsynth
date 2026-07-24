"""Frozen input container for DROSC (the boundary between ``dataprep`` and the
estimation/inference engine)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass(frozen=True)
class DROSCInputs:
    """Canonical DROSC inputs derived from :func:`mlsynth.utils.datautils.dataprep`.

    Attributes
    ----------
    y : np.ndarray
        Treated-unit outcome over all periods, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes over all periods, shape ``(T, N)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    time_labels : np.ndarray
        Period labels, length ``T``.
    donor_names : List[Any]
        Donor unit labels, length ``N``.
    treated_name : Any
        Treated-unit label.
    """

    y: np.ndarray
    donor_matrix: np.ndarray
    pre_periods: int
    time_labels: np.ndarray
    donor_names: List[Any]
    treated_name: Any

    @property
    def Y0(self) -> np.ndarray:
        return self.y[: self.pre_periods]

    @property
    def Y1(self) -> np.ndarray:
        return self.y[self.pre_periods:]

    @property
    def X0(self) -> np.ndarray:
        return self.donor_matrix[: self.pre_periods]

    @property
    def X1(self) -> np.ndarray:
        return self.donor_matrix[self.pre_periods:]
