"""Typed containers for the PROPSC pipeline.

``PropscInputs`` is the frozen ``(N, T, K)`` view assembled by ``setup.py``;
``PropscProportionFit`` is one proportion's per-period paths and effect;
``PROPSCResults`` is the public ``PROPSC.fit()`` return -- an
:class:`~mlsynth.config_models.EffectResult` whose flat accessors delegate to
the target proportion, with the full compositional vector kept as typed fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class PropscInputs:
    """Assembled compositional panel for the PROPSC pipeline.

    Parameters
    ----------
    Y : np.ndarray
        ``(N, T, K)`` outcome array; controls first, treated last; pre-periods
        first. Each ``Y[i, t, :]`` sums to the composition total.
    N0, T0 : int
        Control-unit count and pre-treatment period count.
    outcomes : Sequence[str]
        The ``K`` proportion column names, in slice order.
    donor_labels : Sequence
        Labels of the ``N0`` control units (row order of the control block).
    treated_labels : Sequence
        Labels of the ``N - N0`` treated units.
    time_labels : np.ndarray
        Time labels in column order.
    target_index : int
        Index into ``outcomes`` whose effect drives the flat accessors.
    """

    Y: np.ndarray
    N0: int
    T0: int
    outcomes: Sequence[str]
    donor_labels: Sequence
    treated_labels: Sequence
    time_labels: np.ndarray
    target_index: int


@dataclass(frozen=True)
class PropscProportionFit:
    """Per-proportion effect and trajectories.

    Parameters
    ----------
    name : str
        Proportion (outcome) column name.
    att : float
        Treatment effect on this proportion (SDID/SC/DID double difference).
    se : float
        Jackknife standard error.
    ci : Tuple[float, float]
        95 percent confidence interval ``att +/- 1.96 * se``.
    p_value : float
        Two-sided normal p-value ``2 * (1 - Phi(|att / se|))``.
    observed : np.ndarray
        Treated-average outcome path, shape ``(T,)``.
    counterfactual : np.ndarray
        Synthetic control path (with SDID intercept shift), shape ``(T,)``.
    gap : np.ndarray
        ``observed - counterfactual``, shape ``(T,)``.
    donor_weights : Dict[Any, float]
        Common unit weights, shared across all proportions.
    """

    name: str
    att: float
    se: float
    ci: Tuple[float, float]
    p_value: float
    observed: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    donor_weights: Dict[Any, float]


class PROPSCResults(BaseEstimatorResults):
    """Public ``PROPSC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult`: the standardized
    sub-models (``effects`` / ``time_series`` / ``weights`` / ``inference`` /
    ``fit_diagnostics`` / ``method_details``) are populated for the *target*
    proportion, so the flat accessors (``att`` / ``att_ci`` / ``counterfactual``
    / ``gap`` / ``donor_weights`` / ``pre_rmse``) resolve through the base
    contract. The full compositional output stays in the typed fields below.

    Parameters
    ----------
    proportions : Tuple[PropscProportionFit, ...]
        Per-proportion effects and trajectories, in outcome order.
    att_vector : np.ndarray
        The ``K`` ATTs, aligned with ``proportions``.
    se_vector : np.ndarray
        The ``K`` jackknife standard errors.
    sum_constraint : float
        ``sum(att_vector)``; zero to machine precision by construction.
    method : str
        ``"sdid"``, ``"sc"``, or ``"did"``.
    target : str
        Name of the proportion driving the flat accessors.
    time_weights : np.ndarray
        Common time weights (empty for ``sc``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    proportions: Tuple[PropscProportionFit, ...]
    att_vector: np.ndarray
    se_vector: np.ndarray
    sum_constraint: float
    method: str
    target: str
    time_weights: np.ndarray


PROPSCResults.model_rebuild()
