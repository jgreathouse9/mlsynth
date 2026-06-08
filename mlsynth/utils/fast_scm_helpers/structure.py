from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Iterable

from pydantic import ConfigDict, Field as PydField

from ...config_models import DesignResult
from .fast_scm_bb_helpers import Solution
#from .fast_scm_setup import IndexSet



@dataclass(frozen=True)
class IndexSet:
    """
    Immutable bidirectional index mapping between arbitrary labels and integer indices.

    This class provides a lightweight utility for converting between human-readable
    unit/time labels and integer indices used in NumPy arrays.

    Attributes
    ----------
    labels : np.ndarray
        Ordered array of labels (e.g., unit IDs or time periods).
    label_to_idx : Dict[Any, int]
        Mapping from label → integer index.

    Methods
    -------
    from_labels(labels)
        Construct IndexSet from an iterable of labels.
    get_labels(indices)
        Convert integer indices to labels.
    get_index(labels)
        Convert labels to integer indices.

    Notes
    -----
    - This structure is immutable (`frozen=True`).
    - Intended for consistent indexing across panel datasets.
    """

    labels: np.ndarray
    label_to_idx: Dict[Any, int]

    @classmethod
    def from_labels(cls, labels: Iterable[Any]) -> "IndexSet":
        """
        Construct an IndexSet from an iterable of labels.

        Parameters
        ----------
        labels : Iterable[Any]
            Ordered sequence of unique identifiers.

        Returns
        -------
        IndexSet
            Mapping object for label-index conversions.
        """
        labels = np.asarray(list(labels))
        return cls(
            labels=labels,
            label_to_idx={label: i for i, label in enumerate(labels)}
        )

    def get_labels(self, indices):
        """
        Convert integer indices into corresponding labels.

        Parameters
        ----------
        indices : array-like
            Integer indices.

        Returns
        -------
        np.ndarray
            Corresponding labels.
        """
        return self.labels[np.asarray(indices)]

    def get_index(self, labels):
        """
        Convert labels into integer indices.

        Parameters
        ----------
        labels : array-like
            Input labels.

        Returns
        -------
        np.ndarray
            Integer indices corresponding to input labels.
        """
        return np.array([self.label_to_idx[l] for l in labels])

    def __len__(self):
        """Return number of labels."""
        return len(self.labels)

    def __iter__(self):
        """Iterate over labels."""
        return iter(self.labels)

    def __array__(self):
        """Return labels as NumPy array."""
        return self.labels

    def __repr__(self):
        return f"IndexSet(n={len(self.labels)})"





@dataclass
class Inference:
    ate: Optional[float] = None
    p_value: Optional[float] = None

    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    treated_col_idx: Optional[list] = field(default_factory=list)

# =========================================================
# WEIGHTS
# =========================================================
@dataclass
class WeightVectors:
    treated: np.ndarray
    control: np.ndarray
    control_sparse: np.ndarray = field(init=False)

    def __post_init__(self):
        self.control_sparse = np.where(np.abs(self.control) < 1e-8, 0.0, self.control)


# =========================================================
# PREDICTIONS
# =========================================================
@dataclass
class PredictionVectors:
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    effects: np.ndarray
    residuals_E: np.ndarray
    residuals_B: np.ndarray


# =========================================================
# LOSSES
# =========================================================
@dataclass
class Losses:
    loss_E: float
    nmse_E: float
    nmse_B: float
    # synthetic treated vs synthetic control
    rmse_sc_E: float
    rmse_sc_B: float

    # synthetic treated vs population target
    rmse_pop_E: float
    rmse_pop_B: float


# =========================================================
# IDENTIFICATION
# =========================================================
@dataclass
class Identification:
    solution: Solution
    treated_idx: np.ndarray
    tuple_id: str = field(init=False)

    def __post_init__(self):
        self.tuple_id = getattr(
            self.solution,
            "label",
            f"Tuple_{list(self.treated_idx)}"
        )


# =========================================================
# CANDIDATE OBJECT
# =========================================================
@dataclass
class SEDCandidate:
    identification: Identification
    weights: WeightVectors
    predictions: PredictionVectors
    losses: Losses

    inference: Inference = field(default_factory=Inference)

    treated_weight_dict: Dict[str, float] = field(default_factory=dict)
    control_weight_dict: Dict[str, float] = field(default_factory=dict)

    # -------- Useful derived helpers --------
    @property
    def treated_size(self) -> int:
        return len(self.identification.treated_idx)

    @property
    def control_idx(self) -> np.ndarray:
        return np.where(self.weights.control > 1e-6)[0]

    @property
    def control_size(self) -> int:
        return len(self.control_idx)


# =========================================================
# TIME METADATA
# =========================================================
@dataclass
class TimeInfo:
    n_total: int
    n_pre: int
    n_fit_time: int    # Time periods used for estimation (excluding covariates)
    n_blank_time: int  # Time periods used for the blank/validation window
    n_fit: int         # Total features in estimation (Time + Covariates)
    n_blank: int       # Typically same as n_blank_time
    n_post: int
    index: IndexSet


# =========================================================
# UNIT METADATA
# =========================================================
@dataclass
class UnitInfo:
    n_units_total: int
    treated_labels: List[str]
    control_labels: List[str]

    @property
    def treated_size(self) -> int:
        return len(self.treated_labels)

    @property
    def control_size(self) -> int:
        return len(self.control_labels)


# =========================================================
# FINAL RESULTS OBJECT
# =========================================================
class LEXSCMResults(DesignResult):
    """Top-level container returned by :meth:`mlsynth.LEXSCM.fit`.

    A :class:`~mlsynth.config_models.DesignResult` (the experimental-design
    family): LEXSCM *chooses which units to treat* before any intervention, so
    its output is a design that resolves to an effect report. The standardized
    design fields (``report`` -- an :class:`~mlsynth.config_models.EffectResult`
    built from the realized post-fit; ``selected_units``; ``assignment``;
    ``design_weights``; ``power``; ``metadata``) are populated by the
    orchestrator. The LEXSCM-specific search/diagnostic structure is kept on the
    fields below.

    Parameters
    ----------
    summary : pd.DataFrame
        Ranked shortlist of candidate designs.
    best_candidate : SEDCandidate
        The selected design (winner of the lexicographic search).
    all_candidates : list of SEDCandidate
        Every evaluated candidate.
    bnb_metadata : dict
        Branch-and-bound search diagnostics + the recommendation record.
    time : TimeInfo
        Pre/blank/post period structure.
    units : UnitInfo
        Treated / control unit labels.
    outcome : str
        Outcome column name.
    y_pop_mean_t : np.ndarray
        Population-mean outcome series over time.
    post_fit : SyntheticControlPostFit or None
        Standardized post-fit diagnostics (ATE / total / RMSE / SMD / power)
        from the chosen design; ``None`` if post-fit assembly failed. The
        ``report`` field is the contract-standard view of the same realization.
    """

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="allow")

    summary: Any
    best_candidate: SEDCandidate
    all_candidates: List[SEDCandidate]
    bnb_metadata: Dict[str, Any]
    time: TimeInfo
    units: UnitInfo
    outcome: str
    y_pop_mean_t: np.ndarray = PydField(default_factory=lambda: np.array([]))
    post_fit: Optional[Any] = None
