"""Typed results of the exact backend, carried by ``BilevelSolution.metadata["exact"]``.

Arrays are indexed like the predictors (``ExactDetails.predictor_names``) or the donors
of the problem that was solved; the ``*_by_name`` helpers key them by predictor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DiagonalRationalizers:
    """The diagonal predictor weights that select the donor weights (paper, sec. 4.5).

    ``logdet`` is the log-det maximiser and ``center`` the analytic centre of the set,
    each summing to one; ``support`` lists the predictors some rationaliser weights,
    ``forced_zero`` those every rationaliser leaves at zero, and ``n_implicit`` counts the
    inactive donors' inequalities that hold with equality for every rationaliser.
    """

    logdet: np.ndarray
    center: np.ndarray
    support: Tuple[int, ...]
    forced_zero: Tuple[int, ...]
    n_implicit: int


@dataclass(frozen=True)
class PDRationalizers:
    """The positive definite predictor weights that select the donor weights, on tr V = 1.

    ``logdet`` and ``center`` are K x K matrices, the log-det maximiser and the analytic
    centre; ``logdet_min_eig`` is the log-det maximiser's smallest eigenvalue. With ranges
    requested, ``range_lo`` and ``range_hi`` bound each diagonal entry V_kk over the set
    (Table 4's identified ranges); otherwise they are None.
    """

    logdet: np.ndarray
    center: np.ndarray
    logdet_min_eig: float
    range_lo: Optional[np.ndarray] = None
    range_hi: Optional[np.ndarray] = None

    @property
    def logdet_shares(self) -> np.ndarray:
        """V_kk / tr V of the log-det maximiser, the shares Table 4 reports."""
        return np.diag(self.logdet) / np.trace(self.logdet)

    @property
    def center_shares(self) -> np.ndarray:
        """V_kk / tr V of the analytic centre."""
        return np.diag(self.center) / np.trace(self.center)


@dataclass(frozen=True)
class FaceCounts:
    """Faces of the donor hull by dimension 0 to K - 1 (Table 3): all of them, those of the
    facets visible from the treated unit, and the visible faces that hold an attainable point."""

    total: Tuple[int, ...]
    visible: Tuple[int, ...]
    attainable: Tuple[int, ...]


@dataclass(frozen=True)
class ExactDetails:
    """What the exact backend found, beyond the donor and predictor weights.

    ``v_class`` is the predictor-weight class searched (``"diagonal"`` or ``"pd"``) and
    ``regime`` the case the treated unit falls in: ``"exterior"`` (outside the donor
    hull, matched on a face of its visible facets) or ``"interior"`` (inside it, an exact
    match chosen by the vanishing penalty). ``diagonal`` holds the diagonal rationalisers
    of the donor weights (None in the positive definite class, whose optimum no diagonal
    V need select) and ``pd`` the positive definite ones. ``n_visible_facets`` and
    ``facets`` (the visible facets whose simplex fit is the positive definite optimum,
    several when it lies on a face they share) apply to the exterior case;
    ``face_counts`` is filled on request.
    """

    v_class: str
    regime: str
    predictor_names: Tuple[str, ...]
    validation_loss: float
    pd: PDRationalizers
    diagonal: Optional[DiagonalRationalizers] = None
    n_visible_facets: Optional[int] = None
    facets: Optional[Tuple[Tuple[int, ...], ...]] = None
    face_counts: Optional[FaceCounts] = None
    search: Mapping[str, Any] = field(default_factory=dict)

    def by_name(self, values) -> Dict[str, float]:
        """``{predictor: value}`` for a length-K vector such as ``diagonal.center``."""
        return {n: float(v) for n, v in zip(self.predictor_names, np.asarray(values, float))}
