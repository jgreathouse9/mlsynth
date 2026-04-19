from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from .fast_scm_bb_helpers import Solution


@dataclass
class WeightVectors:
    """
    Container for weight vectors defining the synthetic treated and control units.

    Attributes
    ----------
    treated : np.ndarray, shape (k,)
        Weights over the selected treated units (subset of indices).
        These come from the branch-and-bound solution.
    control : np.ndarray, shape (N,)
        Synthetic control weights over all units, obtained from the control QP.
        Entries at treated indices are zero.
    control_sparse : np.ndarray, shape (N,)
        Sparsified version of `control` with small-magnitude values set to zero.

    Notes
    -----
    - `treated` defines the synthetic treated unit:
        X[:, treated_idx] @ treated
    - `control` defines the synthetic control unit:
        X @ control
    - `control_sparse` is intended for interpretability (reporting / inspection),
      not for downstream computation.
    """
    treated: np.ndarray
    control: np.ndarray
    control_sparse: np.ndarray = field(init=False)

    def __post_init__(self):
        self.control_sparse = np.where(np.abs(self.control) < 1e-8, 0.0, self.control)


@dataclass
class PredictionVectors:
    """
    Container for all constructed time series in the synthetic experiment.

    Attributes
    ----------
    synthetic_treated : np.ndarray, shape (T,)
        Time series of the synthetic treated unit constructed from selected units.
    synthetic_control : np.ndarray, shape (T,)
        Time series of the synthetic control unit.
    effects : np.ndarray, shape (T,)
        Estimated treatment effects at each time point:
            synthetic_treated - synthetic_control
    residuals_E : np.ndarray
        Effects restricted to the estimation period.
    residuals_B : np.ndarray
        Effects restricted to the baseline/backcast (validation) period.

    Notes
    -----
    - `effects` is the primary object used for inference and power analysis.
    - Residual splits (E, B) are used for:
        * model fit (E)
        * validation / placebo diagnostics (B)
    """
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    effects: np.ndarray
    residuals_E: np.ndarray
    residuals_B: np.ndarray


@dataclass
class Losses:
    """
    Container for fit metrics used to evaluate candidate solutions.

    Attributes
    ----------
    loss_E : float
        Objective value from the branch-and-bound stage (estimation period).
    nmse_E : float
        Normalized mean squared error over the estimation period.
    nmse_B : float
        Normalized mean squared error over the baseline/backcast period.

    Notes
    -----
    - `loss_E` is the optimization objective (used during search).
    - `nmse_E` measures in-sample fit.
    - `nmse_B` is the primary diagnostic for out-of-sample fit and robustness.
    """
    loss_E: float
    nmse_E: float
    nmse_B: float


@dataclass
class Identification:
    """
    Identification metadata for a candidate synthetic experiment.

    Attributes
    ----------
    solution : Solution
        Original branch-and-bound solution object.
    treated_idx : np.ndarray
        Indices of units selected as treated.
    tuple_id : str
        Human-readable identifier for the candidate.

    Notes
    -----
    - `tuple_id` is derived from `solution.label` if available,
      otherwise constructed from `treated_idx`.
    - This object links downstream evaluation results back to the
      original combinatorial selection.
    """
    solution: Solution
    treated_idx: np.ndarray
    tuple_id: str = field(init=False)

    def __post_init__(self):
        self.tuple_id = getattr(self.solution, "label", f"Tuple_{list(self.treated_idx)}")


@dataclass
class SEDCandidate:
    """
    Complete representation of a candidate synthetic experiment design.

    Attributes
    ----------
    identification : Identification
        Metadata describing the selected treated units.
    weights : WeightVectors
        Treated and control weight vectors.
    predictions : PredictionVectors
        Constructed time series (treated, control, effects, residuals).
    losses : Losses
        Fit metrics for evaluation and ranking.
    mde_results : dict, optional
        Results from minimum detectable effect (MDE) analysis.

    Notes
    -----
    This object encapsulates the full pipeline output for a single candidate:

    1. Selection (Branch-and-Bound)
        - `identification.solution`
        - `identification.treated_idx`

    2. Construction (Control QP)
        - `weights.treated`
        - `weights.control`

    3. Time Series
        - `predictions.synthetic_treated`
        - `predictions.synthetic_control`
        - `predictions.effects`

    4. Evaluation
        - `losses` (fit metrics)

    5. Power Analysis (optional)
        - `mde_results`

    - Designed to be the core unit passed through ranking, reporting,
      and downstream inference.
    - Mutated in-place by later stages (e.g., power analysis).
    """
    identification: Identification
    weights: WeightVectors
    predictions: PredictionVectors
    losses: Losses
    mde_results: Optional[dict] = None

    def __post_init__(self):
        if self.mde_results is None:
            self.mde_results = {}
