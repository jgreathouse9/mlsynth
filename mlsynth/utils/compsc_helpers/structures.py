"""Typed containers for the COMPSC pipeline.

``CompscInputs`` is the frozen ``(J, T, K)`` view assembled by ``setup.py``;
``CompscCategoryFit`` is one category's paths and share effect;
``CompscPlacebo`` carries the permutation test; ``COMPSCResults`` is the public
``COMPSC.fit()`` return -- an
:class:`~mlsynth.config_models.EffectResult` whose flat accessors delegate to
the target category, with the whole composition kept as typed fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class CompscInputs:
    """Assembled compositional panel for the COMPSC pipeline.

    Parameters
    ----------
    P : np.ndarray
        ``(J, T, K)`` share array, treated unit in row 0, donors after;
        pre-periods first. Every ``P[j, t, :]`` is strictly positive and sums
        to one.
    T0 : int
        Number of pre-treatment periods.
    outcomes : Sequence[str]
        The ``K`` share column names, in slice order.
    treated_label : Any
        Label of the treated unit.
    donor_labels : Sequence
        Labels of the ``J-1`` donors, in row order.
    time_labels : np.ndarray
        Time labels in period order.
    target_index : int
        Index into ``outcomes`` whose effect drives the flat accessors.
    baseline_index : int
        Index into ``outcomes`` used as the reference category by
        ``geometry="alr"``.
    """

    P: np.ndarray
    T0: int
    outcomes: Sequence[str]
    treated_label: Any
    donor_labels: Sequence
    time_labels: np.ndarray
    target_index: int
    baseline_index: int


@dataclass(frozen=True)
class CompscCategoryFit:
    """Per-category paths and share effect.

    Parameters
    ----------
    name : str
        Share (outcome) column name.
    att : float
        Mean post-treatment share effect, in share units.
    observed : np.ndarray
        Observed share path, shape ``(T,)``.
    counterfactual : np.ndarray
        Counterfactual share path, shape ``(T,)``.
    gap : np.ndarray
        ``observed - counterfactual``, shape ``(T,)``; the per-period ATT of
        paper eq. (2).
    """

    name: str
    att: float
    observed: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray


@dataclass(frozen=True)
class CompscPlacebo:
    """Abadie permutation test in the Aitchison metric (Algorithm 1, Step 6).

    Parameters
    ----------
    p_value : float
        Share of retained units whose post/pre Aitchison RMSPE ratio is at
        least the treated unit's.
    treated_ratio : float
        The treated unit's post/pre ratio.
    ratios : Dict[Any, float]
        Ratio for every unit, keyed by label.
    pre_rmspe : Dict[Any, float]
        Pre-treatment Aitchison RMSPE for every unit, keyed by label.
    retained : Tuple[Any, ...]
        Labels surviving the pre-fit screen (the treated unit always survives).
    n_retained : int
        Size of the comparison set, including the treated unit.
    screen : float
        The pre-fit screen multiple that was applied.
    """

    p_value: float
    treated_ratio: float
    ratios: Dict[Any, float]
    pre_rmspe: Dict[Any, float]
    retained: Tuple[Any, ...]
    n_retained: int
    screen: float


class COMPSCResults(BaseEstimatorResults):
    """Public ``COMPSC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult`: the standardized
    sub-models (``effects`` / ``time_series`` / ``weights`` / ``inference`` /
    ``fit_diagnostics`` / ``method_details``) are populated for the *target*
    category, so the flat accessors (``att`` / ``counterfactual`` / ``gap`` /
    ``donor_weights`` / ``pre_rmse``) resolve through the base contract. The
    whole composition stays in the typed fields below.

    Parameters
    ----------
    categories : Tuple[CompscCategoryFit, ...]
        Per-category paths and effects, in outcome order.
    observed_composition : np.ndarray
        ``(T, K)`` observed shares for the treated unit.
    counterfactual_composition : np.ndarray
        ``(T, K)`` counterfactual shares; the Frechet barycenter of the donor
        compositions under the Aitchison metric. Strictly positive, rows sum
        to one.
    share_effects : np.ndarray
        ``(T, K)`` per-period share effects (paper eq. 2). Rows sum to zero.
    log_ratio_effects : np.ndarray
        ``(T, K, K)`` pairwise log-ratio treatment effects (paper eq. 3).
    aitchison_distance : np.ndarray
        ``(T,)`` Aitchison distance between observed and counterfactual
        composition (paper eq. 4): a scalar summary of total compositional
        displacement.
    att_vector : np.ndarray
        The ``K`` mean post-treatment share effects, aligned with ``categories``.
    sum_constraint : float
        ``sum(att_vector)``; zero to machine precision by construction.
    pre_treatment_aitchison_rmspe : float
        Pre-treatment root mean squared Aitchison distance -- the quantity the
        ``clr`` objective minimizes.
    pre_treatment_share_rmspe : float
        Pre-treatment root mean squared Euclidean share error, for comparison
        with level-scale synthetic controls.
    placebo : CompscPlacebo, optional
        The permutation test, when ``inference="placebo"``.
    geometry : str
        ``"clr"`` or ``"alr"``.
    baseline : str, optional
        Reference category, when ``geometry="alr"``.
    target : str
        Name of the category driving the flat accessors.
    baseline_sensitivity : float, optional
        Largest L1 shift in the fitted weights across the ``K`` possible
        ``alr`` baselines. Under ``clr`` the objective has no baseline, so a
        large value is a warning that the paper's Algorithm 1 is sensitive to
        an arbitrary labelling choice on this panel. ``None`` when ``K < 3``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    categories: Tuple[CompscCategoryFit, ...]
    observed_composition: np.ndarray
    counterfactual_composition: np.ndarray
    share_effects: np.ndarray
    log_ratio_effects: np.ndarray
    aitchison_distance: np.ndarray
    att_vector: np.ndarray
    sum_constraint: float
    pre_treatment_aitchison_rmspe: float
    pre_treatment_share_rmspe: float
    placebo: Optional[CompscPlacebo]
    geometry: str
    baseline: Optional[str]
    target: str
    baseline_sensitivity: Optional[float]


COMPSCResults.model_rebuild()
