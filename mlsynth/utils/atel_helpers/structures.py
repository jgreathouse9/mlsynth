"""Typed containers for the ATEL estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class ATELInputs:
    """The arrays Lee (2026) Steps 1-5 run on.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome panel ``(N + 1, T)`` with the treated unit in row 0.
    covariates : np.ndarray
        Covariate cube ``(N + 1, T, P)`` in the same unit and period order.
    n_pre : int
        Number of pre-treatment periods.
    unit_labels : np.ndarray
        Unit labels, treated first, length ``N + 1``.
    time_labels : np.ndarray
        Period labels, length ``T``.
    covariate_names : tuple of str
        Covariate column names, matching the cube's last axis.
    """

    outcomes: np.ndarray
    covariates: np.ndarray
    n_pre: int
    unit_labels: np.ndarray
    time_labels: np.ndarray
    covariate_names: Tuple[str, ...]

    @property
    def n_donors(self) -> int:
        """Number of donor units."""
        return int(self.outcomes.shape[0] - 1)

    @property
    def n_periods(self) -> int:
        """Total number of periods."""
        return int(self.outcomes.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.n_periods - int(self.n_pre)

    @property
    def treated_unit_name(self) -> Any:
        """Label of the treated unit."""
        return self.unit_labels[0]


class ATELResults(BaseEstimatorResults):
    """Container returned by :meth:`mlsynth.ATEL.fit`.

    An :class:`~mlsynth.config_models.EffectResult`. ATEL and the ATT are
    different estimands, so both are reported and neither stands in for the
    other: ``effects.att`` is the unweighted post-period mean gap that the
    standardized surface computes from the two outcome paths, and ``atel`` is
    the kernel-weighted localized estimate the paper is about. ``inference``
    carries the interval, standard error and p-value for ``atel``.

    Parameters
    ----------
    atel : float
        The localized estimate.
    bandwidth : float
        Bandwidth used, whether supplied or cross-validated.
    n_factors : int
        Number of factors, and the sieve width.
    factors : np.ndarray
        Estimated factors ``(T, n_factors)``.
    loadings : np.ndarray
        Time-varying loadings over the post-period ``(n_factors, T1)``.
    kernel_weights : np.ndarray
        Localization weights on each post-period, length ``T1``. They do not
        sum to one, so a constant effect ``d`` gives ``atel`` of ``d`` times
        their mass and not ``d`` itself.
    implied_donor_weights : np.ndarray
        The weight each donor carries at each post-period, ``(N, T1)``. The
        counterfactual is ``sum_i Y_it * implied_donor_weights[i, t]`` exactly.
        These come from a projection, so they are unconstrained in sign and do
        not sum to one. The standardized ``weights`` slot carries each donor's
        path averaged against the localization kernel, which is the weight it
        carries in the reported estimate.
    pointwise_standard_errors : np.ndarray
        Per-period standard error of the counterfactual, length ``T1``. These
        need the treated unit's loading path to be independent and normal across
        periods, a stronger assumption than the interval on ``atel`` uses.
    inputs : ATELInputs or None
        The ingested panel.
    diagnostics : dict
        Free-form diagnostics: the cross-validation grid, whether the selected
        bandwidth is an endpoint of it, and the effective windows.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    atel: float
    bandwidth: float
    n_factors: int
    factors: np.ndarray
    loadings: np.ndarray
    kernel_weights: np.ndarray
    implied_donor_weights: np.ndarray
    pointwise_standard_errors: np.ndarray
    inputs: Optional[ATELInputs] = None
    diagnostics: Dict[str, Any] = PydField(default_factory=dict)

    @property
    def kernel_mass(self) -> float:
        """Sum of the localization weights divided by the post-period window."""
        return float(self.kernel_weights.sum() / self.diagnostics["post_window"])
