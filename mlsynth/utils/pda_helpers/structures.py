"""Frozen, NumPy-first containers for the Panel Data Approach (PDA).

PDA (Hsiao, Ching & Wan 2012) predicts a treated unit's untreated
counterfactual by a linear regression on the control units fit over the
pre-treatment window, then extrapolates out-of-sample. ``mlsynth`` exposes
three high-dimensional PDA variants, each in its own subpackage with the
inference theory from its own paper:

* ``l2``    -- L2-relaxation (Shi & Wang 2024).
* ``lasso`` -- L1/LASSO (Li & Bell 2017).
* ``fs``    -- forward-selected PDA (Shi & Huang 2023).

Everything below is pure NumPy; units/time are addressed through
:class:`IndexSet`. The only DataFrame touchpoint is ``setup``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..fast_scm_helpers.structure import IndexSet

# Method names.
L2 = "l2"
LASSO = "lasso"
FS = "fs"


@dataclass(frozen=True)
class PDAInputs:
    """Preprocessed, NumPy-only panel for the PDA engine.

    Parameters
    ----------
    unit_index : IndexSet
        All ``N`` donor units (column order of ``X``).
    time_index : IndexSet
        All ``T`` periods (row order of ``y`` and ``X``).
    y : np.ndarray
        Treated-unit outcome over all periods, shape ``(T,)``.
    X : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    T0 : int
        Number of pre-treatment periods (``T1``); post is ``T2 = T - T0``.
    treated_label : Any
        Identifier of the treated unit.
    metadata : dict
        Free-form provenance.
    """

    unit_index: IndexSet
    time_index: IndexSet
    y: np.ndarray
    X: np.ndarray
    T0: int
    treated_label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def T2(self) -> int:
        return self.T - self.T0

    @property
    def n_donors(self) -> int:
        return int(self.X.shape[1])

    @property
    def donor_labels(self) -> np.ndarray:
        return np.asarray(self.unit_index.labels)


@dataclass(frozen=True)
class PDAMethodFit:
    """A single PDA-variant fit (l2 / lasso / fs)."""

    name: str
    beta: np.ndarray                 # donor coefficients
    intercept: float
    counterfactual: np.ndarray       # (T,)
    gap: np.ndarray                  # (T,)  = y - counterfactual
    att: float                       # mean post-period gap
    att_se: float
    ci: Tuple[float, float]
    p_value: float
    donor_weights: Dict[Any, float]
    selected_donors: Optional[List[Any]] = None      # fs / lasso support
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PDAResults:
    """Top-level container returned by :meth:`mlsynth.PDA.fit`."""

    inputs: PDAInputs
    fits: Dict[str, PDAMethodFit]
    selected_variant: str = FS
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def _primary(self) -> PDAMethodFit:
        return self.fits.get(self.selected_variant, next(iter(self.fits.values())))

    @property
    def att(self) -> float:
        return self._primary.att

    @property
    def att_se(self) -> float:
        return self._primary.att_se

    @property
    def counterfactual(self) -> np.ndarray:
        return self._primary.counterfactual

    @property
    def gap(self) -> np.ndarray:
        return self._primary.gap

    @property
    def donor_weights(self) -> Dict[Any, float]:
        return self._primary.donor_weights

    def att_by_method(self) -> Dict[str, float]:
        return {name: fit.att for name, fit in self.fits.items()}

    def se_by_method(self) -> Dict[str, float]:
        return {name: fit.att_se for name, fit in self.fits.items()}
