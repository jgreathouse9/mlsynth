"""Frozen, NumPy-first containers for Synthetic Control with Multiple Outcomes.

SCMO (Tian-Lee-Panchenko 2024; Sun-Ben-Michael-Feller 2025) builds the
synthetic control by matching the treated unit to donors on a **matching
matrix** ``Z`` assembled from one or more related outcomes/predictors --
optionally across several pre-treatment periods -- rather than a single
outcome's long trajectory. Everything below is pure NumPy; the only
DataFrame touchpoint is :func:`mlsynth.utils.scmo_helpers.setup.prepare_scmo_inputs`.

Units and time are addressed through :class:`IndexSet` (immutable
label<->integer maps) so downstream code never reaches back into pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# IndexSet currently lives in fast_scm_helpers.structure on this branch;
# on main it is re-homed to helperutils -- a one-line import swap at sync.
from ..fast_scm_helpers.structure import IndexSet


# Weighting schemes.
CONCATENATED = "concatenated"   # Tian-Lee-Panchenko: stack all standardized columns
AVERAGED = "averaged"           # Sun-Ben-Michael-Feller: match the average of outcomes
SEPARATE = "separate"           # conventional per-outcome baseline
MA = "MA"                       # model-average of concatenated & averaged


@dataclass(frozen=True)
class SCMOInputs:
    """Preprocessed, NumPy-only panel for the SCMO engine.

    Parameters
    ----------
    unit_index : IndexSet
        All ``N`` units; row order of ``Y`` and ``Z``.
    time_index : IndexSet
        All ``T`` periods; column order of ``Y``.
    treated_idx : int
        Row index (into ``unit_index``) of the treated unit.
    donor_idx : np.ndarray
        Row indices of the donor pool.
    Y : np.ndarray
        Primary-outcome panel, shape ``(N, T)`` (rows = units).
    T0 : int
        Number of pre-treatment periods.
    Z : np.ndarray
        Standardized matching matrix, shape ``(N, P)`` (rows = units).
    predictor_labels : Sequence
        Length-``P`` labels for the columns of ``Z``.
    metadata : dict
        Free-form provenance (spec, demean flag, dropped columns, ...).
    """

    unit_index: IndexSet
    time_index: IndexSet
    treated_idx: int
    donor_idx: np.ndarray
    Y: np.ndarray
    T0: int
    Z: np.ndarray
    predictor_labels: Sequence
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.Y.shape[1])

    @property
    def n_donors(self) -> int:
        return int(self.donor_idx.shape[0])

    @property
    def donor_labels(self) -> np.ndarray:
        return self.unit_index.get_labels(self.donor_idx)

    @property
    def treated_label(self) -> Any:
        return self.unit_index.get_labels([self.treated_idx])[0]

    @property
    def y_treated(self) -> np.ndarray:
        return self.Y[self.treated_idx]

    @property
    def Y_donors(self) -> np.ndarray:
        return self.Y[self.donor_idx]

    @property
    def Z_treated(self) -> np.ndarray:
        return self.Z[self.treated_idx]

    @property
    def Z_donors(self) -> np.ndarray:
        return self.Z[self.donor_idx]


@dataclass(frozen=True)
class SCMOMethodFit:
    """A single weighting-scheme fit (concatenated / averaged / separate / MA)."""

    name: str
    weights: np.ndarray              # (n_donors,) donor weights
    counterfactual: np.ndarray       # (T,)
    gap: np.ndarray                  # (T,)
    att: float
    pre_rmse: float
    donor_weights: Dict[Any, float]
    att_se: Optional[float] = None
    ci: Tuple[float, float] = (float("nan"), float("nan"))
    p_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SCMOResults:
    """Top-level container returned by :meth:`mlsynth.SCMO.fit`."""

    inputs: SCMOInputs
    fits: Dict[str, SCMOMethodFit]
    selected_variant: str = CONCATENATED
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def _primary(self) -> SCMOMethodFit:
        return self.fits.get(self.selected_variant, next(iter(self.fits.values())))

    @property
    def att(self) -> float:
        return self._primary.att

    @property
    def counterfactual(self) -> np.ndarray:
        return self._primary.counterfactual

    @property
    def gap(self) -> np.ndarray:
        return self._primary.gap

    @property
    def donor_weights(self) -> Dict[Any, float]:
        return self._primary.donor_weights

    @property
    def pre_rmse(self) -> float:
        return self._primary.pre_rmse

    def att_by_method(self) -> Dict[str, float]:
        return {name: fit.att for name, fit in self.fits.items()}
