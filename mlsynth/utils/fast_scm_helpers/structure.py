from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from .fast_scm_bb_helpers import Solution


@dataclass
class WeightVectors:
    treated: np.ndarray
    control: np.ndarray
    control_sparse: np.ndarray = field(init=False)

    def __post_init__(self):
        self.control_sparse = np.where(np.abs(self.control) < 1e-8, 0.0, self.control)


@dataclass
class PredictionVectors:
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    effects: np.ndarray
    residuals_E: np.ndarray
    residuals_B: np.ndarray


@dataclass
class Losses:
    loss_E: float
    nmse_E: float
    nmse_B: float


@dataclass
class Identification:
    solution: Solution
    treated_idx: np.ndarray
    tuple_id: str = field(init=False)

    def __post_init__(self):
        self.tuple_id = getattr(self.solution, "label", f"Tuple_{list(self.treated_idx)}")


@dataclass
class SEDCandidate:
    identification: Identification
    weights: WeightVectors
    predictions: PredictionVectors
    losses: Losses
    mde_results: Optional[dict] = None

    def __post_init__(self):
        if self.mde_results is None:
            self.mde_results = {}