"""Frozen, NumPy-first containers for Forward-Selected SCM (FSCM).

FSCM (Cerulli 2024) treats the *number of donors* as a complexity parameter
governing a bias--variance trade-off: a richer donor pool fits the
pre-treatment window better in sample but injects variance from poorly
correlated donors out of sample. A forward stepwise selection grows a nested
donor sequence (greedy on the training-period fit), and a two-interval-time
out-of-sample validation on the held-out tail of the pre-period picks the
donor count that minimizes test RMSPE.

Everything below is pure NumPy; units/time are addressed through
:class:`IndexSet`. The only DataFrame touchpoint is ``setup``.

References
----------
Cerulli, G. (2024). Optimal initial donor selection for the synthetic control
method. Economics Letters, 244, 111976.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..fast_scm_helpers.structure import IndexSet


@dataclass(frozen=True)
class FSCMInputs:
    """Preprocessed, NumPy-only panel for the FSCM engine.

    Parameters
    ----------
    unit_index : IndexSet
        All ``N`` donor units (column order of ``Y``).
    time_index : IndexSet
        All ``T`` periods (row order of ``y`` and ``Y``).
    y : np.ndarray
        Treated-unit outcome over all periods, shape ``(T,)``.
    Y : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    T0 : int
        Number of pre-treatment periods; post is ``T2 = T - T0``.
    treated_label : Any
        Identifier of the treated unit.
    cov_treated : np.ndarray, optional
        Treated covariate predictor values, shape ``(P,)`` -- each covariate
        averaged over its (Abadie) aggregation window. Empty if no covariates.
    cov_donors : np.ndarray, optional
        Donor covariate predictor values, shape ``(N, P)``. Empty if none.
    covariate_names : list of str
        Names of the ``P`` covariate columns.
    match_idx : np.ndarray, optional
        Time-row indices of "special predictor" periods -- specific
        pre-treatment periods whose outcome value is matched directly
        (e.g. the 1975/1980/1988 cigarette-sales values in Abadie's spec).
    match_periods : list
        Labels of those periods, for provenance.
    metadata : dict
        Free-form provenance.
    """

    unit_index: IndexSet
    time_index: IndexSet
    y: np.ndarray
    Y: np.ndarray
    T0: int
    treated_label: Any
    cov_treated: np.ndarray = field(default_factory=lambda: np.empty(0))
    cov_donors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    covariate_names: List[str] = field(default_factory=list)
    match_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    match_periods: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def T2(self) -> int:
        return self.T - self.T0

    @property
    def n_donors(self) -> int:
        return int(self.Y.shape[1])

    @property
    def donor_labels(self) -> np.ndarray:
        return np.asarray(self.unit_index.labels)

    @property
    def has_covariates(self) -> bool:
        return len(self.covariate_names) > 0

    @property
    def has_match_periods(self) -> bool:
        return self.match_idx.size > 0

    @property
    def has_predictors(self) -> bool:
        return self.has_covariates or self.has_match_periods


@dataclass(frozen=True)
class FSCMSelectionPath:
    """The forward-selection / cross-validation trace (Cerulli Fig. 2--3).

    Each entry ``k`` describes the nested model with ``k`` donors: the donor
    added at that step, the in-sample pre-period RMSPE, and the out-of-sample
    rolling-origin CV RMSPE used to choose the optimal size.
    """

    sizes: np.ndarray                 # (K,) number of donors: 1..K
    order: List[Any]                  # donor labels in the order selected
    train_rmspe: np.ndarray           # (K,) in-sample pre-period RMSPE
    test_rmspe: np.ndarray            # (K,) rolling-origin CV RMSPE
    optimal_size: int                 # argmin test_rmspe (number of donors)


@dataclass(frozen=True)
class FSCMResults:
    """Top-level container returned by :meth:`mlsynth.FSCM.fit`."""

    inputs: FSCMInputs
    selected_donors: List[Any]        # labels of the donor set with weight
    weights: np.ndarray               # (n_selected,) simplex weights
    donor_weights: Dict[Any, float]   # full pool, zeros off the selected set
    counterfactual: np.ndarray        # (T,)
    gap: np.ndarray                   # (T,) = y - counterfactual
    att: float                        # mean post-period gap
    selection_path: Optional[FSCMSelectionPath] = None  # None when no forward selection
    fit_diagnostics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.selected_donors)

    @property
    def pre_rmse(self) -> float:
        return float(self.fit_diagnostics.get("pre_rmse", np.nan))
