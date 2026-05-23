"""Frozen dataclasses for the MC-NNM estimator.

Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *Matrix Completion
Methods for Causal Panel Data Models* (JASA). MC-NNM imputes the missing
(treated) entries of the outcome matrix by nuclear-norm-regularised
low-rank matrix completion with unregularised two-way fixed effects, then
forms treatment effects as observed minus imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class MCNNMInputs:
    """Preprocessed panel for MC-NNM.

    Attributes
    ----------
    Y : np.ndarray
        Observed outcomes, shape ``(N, T)``.
    mask : np.ndarray
        Observation indicator, shape ``(N, T)``; ``1`` observed (control
        / pre-treatment), ``0`` missing (treated post-treatment).
    D : np.ndarray
        Treatment indicators, shape ``(N, T)`` (``1`` where treated).
    treated_idx : np.ndarray
        Indices of ever-treated units.
    T0 : int
        First treated period.
    unit_names : list
    time_labels : np.ndarray
    """

    Y: np.ndarray
    mask: np.ndarray
    D: np.ndarray
    treated_idx: np.ndarray
    T0: int
    unit_names: List[Any]
    time_labels: np.ndarray

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]


@dataclass(frozen=True)
class MCNNMResults:
    """Top-level container returned by :meth:`mlsynth.MCNNM.fit`.

    Attributes
    ----------
    inputs : MCNNMInputs
    att : float
        Average treatment effect on the treated (observed minus imputed
        over the treated cells).
    counterfactual : np.ndarray
        Full fitted matrix ``L + Gamma + Delta``, shape ``(N, T)``; on
        treated cells this is the imputed untreated potential outcome.
    effects : np.ndarray
        Per-cell effects (observed minus imputed) on treated cells; ``NaN``
        elsewhere, shape ``(N, T)``.
    att_by_period : dict
        ``{period_label: mean effect across treated units}`` post-treatment.
    L : np.ndarray
        Estimated low-rank matrix, shape ``(N, T)``.
    gamma : np.ndarray
        Unit fixed effects, shape ``(N,)``.
    delta : np.ndarray
        Time fixed effects, shape ``(T,)``.
    best_lambda : float
        Cross-validation-selected singular-value threshold.
    rank : int
        Numerical rank of ``L`` (singular values > 1e-6 of the max).
    inference : object, optional
        :class:`MCNNMInference` when ``inference=True``; else ``None``.
    metadata : dict
    """

    inputs: MCNNMInputs
    att: float
    counterfactual: np.ndarray
    effects: np.ndarray
    att_by_period: Dict[Any, float]
    L: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    best_lambda: float
    rank: int
    inference: Optional["MCNNMInference"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCNNMInference:
    """Leave-one-control jackknife inference for the MC-NNM ATT.

    Attributes
    ----------
    method : str
        ``"jackknife"``.
    se : float
    ci : tuple of float
    alpha_level : float
    n_jackknife : int
    """

    method: str
    se: float
    ci: tuple
    alpha_level: float
    n_jackknife: int
