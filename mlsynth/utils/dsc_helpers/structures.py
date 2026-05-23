"""Frozen dataclasses for the Distributional Synthetic Control (DSC) estimator.

DSC (Gunsilius 2023; asymptotic theory in Zhang, Zhang & Zhang 2026)
operates on micro-level panels: for each ``(unit, time)`` cell the user
supplies multiple individual observations and the estimator works with
the *empirical quantile function* of that cell rather than its mean.

The output therefore lives at the quantile level: per post-period we
return a counterfactual quantile function alongside the observed one,
and per quantile we expose the *quantile treatment effect* (QTE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DSCInputs:
    """Preprocessed micro-panel data for DSC.

    Attributes
    ----------
    cell_samples : dict
        Mapping ``(unit_id, time_id) -> 1-D np.ndarray`` of individual
        outcomes for that cell. Each array's length is :math:`n_{jt}`
        (the within-cell sample size).
    unit_names : list
        Length-``(J + 1)`` ordering of unit ids. ``unit_names[0]`` is
        the treated unit, the remaining ``J`` are donors.
    time_labels : np.ndarray
        Length-``T`` ordering of time-period labels.
    T : int
        Total number of panel periods.
    T0 : int
        Number of pre-treatment periods.
    treated_unit_name : Any
        Label of the treated unit.
    """

    cell_samples: Dict[Tuple[Any, Any], np.ndarray]
    unit_names: List[Any]
    time_labels: np.ndarray
    T: int
    T0: int
    treated_unit_name: Any

    @property
    def J(self) -> int:
        """Number of donor units (excluding the treated unit)."""
        return len(self.unit_names) - 1

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class QTECurve:
    """Quantile treatment effect at a single post-period.

    Attributes
    ----------
    time_label : Any
        Time-period identifier.
    quantiles : np.ndarray
        Quantile grid in ``(0, 1)``, shape ``(Q,)``.
    observed : np.ndarray
        Empirical treated quantile function evaluated at ``quantiles``,
        shape ``(Q,)``. Under treatment this is :math:`F^{-1}_{Y_{1t, I}}`.
    counterfactual : np.ndarray
        DSC counterfactual quantile function, shape ``(Q,)``. Estimate
        of :math:`F^{-1}_{Y_{1t, N}}`.
    qte : np.ndarray
        ``observed - counterfactual``, shape ``(Q,)``. Quantile
        treatment effect.
    """

    time_label: Any
    quantiles: np.ndarray
    observed: np.ndarray
    counterfactual: np.ndarray
    qte: np.ndarray


@dataclass(frozen=True)
class DSCResults:
    """Top-level container returned by :meth:`mlsynth.DSC.fit`.

    Attributes
    ----------
    inputs : DSCInputs
        Preprocessed micro-panel.
    donor_weights : dict
        Mapping ``{donor_unit_name: weight}`` for the aggregated
        weight vector :math:`\\widehat w = \\sum_t \\lambda_t \\widehat w_t`.
        All weights are non-negative and sum to one (the simplex
        constraint of Gunsilius 2023).
    period_weights : dict
        Mapping ``{time_label: weight_vector}`` for the per-pre-period
        weights :math:`\\widehat w_t` (each a J-vector). Useful for
        diagnostic inspection of pre-period heterogeneity.
    lambda_weights : np.ndarray
        Length-``T0`` pre-period aggregation weights.
    qte_curves : list of QTECurve
        Per-post-period QTE objects. Index 0 is post-period 1.
    average_qte : np.ndarray
        QTE averaged over post-periods (uniform mean), shape ``(Q,)``.
    att : float
        Mean-aggregated treatment effect on the treated, obtained by
        averaging the QTE over both quantiles and post-periods. Useful
        as a single scalar summary; the QTE itself remains the
        primary object.
    pre_period_wasserstein : np.ndarray
        Pre-period 2-Wasserstein loss per ``t \\in T_0``, shape
        ``(T0,)``. Lower is tighter pre-period fit.
    metadata : dict
        Free-form pipeline diagnostics (M, quantile grid method,
        random seed, etc.).
    """

    inputs: DSCInputs
    donor_weights: Dict[Any, float]
    period_weights: Dict[Any, np.ndarray]
    lambda_weights: np.ndarray
    qte_curves: List[QTECurve]
    average_qte: np.ndarray
    att: float
    pre_period_wasserstein: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
