"""Frozen dataclasses for the Synthetic Nearest Neighbors (SNN) estimator.

Agarwal, A., Dahleh, M., Shah, D. & Shen, D. (2021). *"Causal Matrix
Completion."* arXiv:2109.15154.

In the causal/panel setting, SNN treats the treated units' post-treatment
potential outcomes :math:`Y(0)` as the missing entries of the outcome
matrix and imputes them by matrix completion (synthetic nearest
neighbors), then forms treatment effects as observed minus imputed. It
generalises the Synthetic Interventions / synthetic-control estimator to
arbitrary missingness patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SNNInputs:
    """Preprocessed panel for SNN.

    Attributes
    ----------
    Y : np.ndarray
        Observed outcomes, shape ``(N, T)``.
    D : np.ndarray
        Treatment indicators, shape ``(N, T)``; ``1`` where treated.
    treated_idx : np.ndarray
        Indices of ever-treated units.
    T0 : int
        First treated period (post-treatment is ``t >= T0``).
    unit_names : list
        Length-``N`` unit identifiers.
    time_labels : np.ndarray
        Length-``T`` period labels.
    """

    Y: np.ndarray
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
class SNNResults:
    """Top-level container returned by :meth:`mlsynth.SNN.fit`.

    Attributes
    ----------
    inputs : SNNInputs
        Preprocessed panel.
    att : float
        Average treatment effect on the treated, over imputed
        treated post-treatment cells.
    counterfactual : np.ndarray
        Outcome matrix with treated post-treatment :math:`Y(0)` imputed,
        shape ``(N, T)``.
    effects : np.ndarray
        Per-cell treatment effects (observed minus imputed) for treated
        post cells; ``NaN`` elsewhere, shape ``(N, T)``.
    att_by_period : dict
        ``{period_label: mean effect across treated units}`` for
        post-treatment periods.
    feasible : np.ndarray
        Boolean mask of cells SNN could impute, shape ``(N, T)``.
    inference : object, optional
        :class:`SNNInference` when ``inference=True``; ``None`` otherwise.
    metadata : dict
        Free-form diagnostics.
    """

    inputs: SNNInputs
    att: float
    counterfactual: np.ndarray
    effects: np.ndarray
    att_by_period: Dict[Any, float]
    feasible: np.ndarray
    inference: Optional["SNNInference"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SNNInference:
    """Jackknife inference for the SNN ATT.

    Leaves out one control (anchor) unit at a time, re-imputes, and uses
    the spread of the resulting ATTs to form a standard error and
    confidence interval.

    Attributes
    ----------
    method : str
        ``"jackknife"``.
    se : float
        Jackknife standard error of the ATT.
    ci : tuple of float
        Two-sided confidence interval for the ATT.
    alpha_level : float
        Level used for ``ci``.
    n_jackknife : int
        Number of leave-one-control re-fits used.
    """

    method: str
    se: float
    ci: tuple
    alpha_level: float
    n_jackknife: int
