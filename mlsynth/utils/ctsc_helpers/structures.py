"""Frozen dataclasses for the Continuous-Treatment Synthetic Control (CTSC).

Powell, D. (2022). *"Synthetic Control Estimation Beyond Comparative Case
Studies: Does the Minimum Wage Reduce Employment?"* Journal of Business &
Economic Statistics 40(3):1302-1314.

CTSC generalises synthetic control to **continuous and/or multi-valued
treatments** in panels where there is no clean treated / never-treated
split -- every unit has a time-varying treatment (e.g. every U.S. state
has a minimum wage that changes over time). It jointly estimates a
unit-specific treatment-slope vector :math:`\\alpha_i` and a synthetic
control over the other units' untreated outcomes, and reports a
population-weighted average marginal effect.

The paper names the estimator "GSC" (generalized synthetic control);
mlsynth uses **CTSC** to avoid confusion with Xu (2017)'s differently
constructed Generalized Synthetic Control (``gsynth``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class CTSCInputs:
    """Preprocessed panel for CTSC.

    Attributes
    ----------
    Y : np.ndarray
        Outcomes, shape ``(n, T)``.
    D : np.ndarray
        Treatment / explanatory variables, shape ``(n, T, K)``.
    unit_names : list
        Length-``n`` unit identifiers.
    time_labels : np.ndarray
        Length-``T`` period labels.
    treatment_names : list of str
        Length-``K`` names of the treatment / explanatory variables.
    population_weights : np.ndarray
        Per-unit weights :math:`\\pi_i` for the average effect; sum to one.
    """

    Y: np.ndarray
    D: np.ndarray
    unit_names: List[Any]
    time_labels: np.ndarray
    treatment_names: List[str]
    population_weights: np.ndarray

    @property
    def n(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    @property
    def K(self) -> int:
        return self.D.shape[2]


@dataclass(frozen=True)
class CTSCResults:
    """Top-level container returned by :meth:`mlsynth.CTSC.fit`.

    Attributes
    ----------
    inputs : CTSCInputs
        Preprocessed panel.
    average_effect : np.ndarray
        Population-weighted average marginal effect, shape ``(K,)``
        (paper eq. 7).
    unit_effects : np.ndarray
        Unit-specific treatment slopes :math:`\\alpha_i`, shape ``(n, K)``.
    unit_weight_matrix : np.ndarray
        All-units synthetic-control weight matrix, shape ``(n, n)``;
        ``[i, i] = 0``, each row non-negative and summing to one. CTSC
        builds a synthetic control for *every* unit, so there is no single
        treated-unit donor vector -- row ``i`` is unit ``i``'s weights.
    fit_metric : np.ndarray
        Per-unit fit weights :math:`\\Omega_i`, shape ``(n,)`` (smaller =
        better synthetic control; paper eq. 6).
    objective : float
        Minimised objective value.
    weights : WeightsResults, optional
        Standardized weights. Since CTSC has no single treated unit,
        ``donor_weights`` holds the cross-unit average weight each donor
        receives; the full per-unit matrix is ``unit_weight_matrix``.
    inference : object, optional
        :class:`CTSCInference` when inference is run; ``None`` otherwise.
    metadata : dict
        Free-form diagnostics.
    """

    inputs: CTSCInputs
    average_effect: np.ndarray
    unit_effects: np.ndarray
    unit_weight_matrix: np.ndarray
    fit_metric: np.ndarray
    objective: float
    weights: Optional[Any] = None
    inference: Optional["CTSCInference"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CTSCInference:
    """Wald / sign-flip inference for CTSC (paper Section 4).

    Tests :math:`H_0: \\alpha^{AE,(k)} = a_0` per treatment variable via a
    Wald statistic on the orthogonal moment scores, calibrated by a
    Rademacher (sign-flip) randomization distribution that permits arbitrary
    within-unit and cross-unit dependence.

    Attributes
    ----------
    method : str
        ``"sign_flip_wald"``.
    null_value : np.ndarray
        Tested null :math:`a_0` per variable, shape ``(K,)``.
    wald_stat : np.ndarray
        Wald statistics per variable, shape ``(K,)``.
    p_value : np.ndarray
        Randomization p-values per variable, shape ``(K,)``.
    se : np.ndarray
        Standard errors of the average effect per variable, shape ``(K,)``.
    n_draws : int
        Number of Rademacher draws.
    """

    method: str
    null_value: np.ndarray
    wald_stat: np.ndarray
    p_value: np.ndarray
    se: np.ndarray
    n_draws: int
