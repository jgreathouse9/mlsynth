"""Typed result containers for the SDID pipeline.

All matrices follow ``mlsynth``'s ``(T, N)`` orientation (rows = time,
columns = unit), matching :func:`mlsynth.utils.datautils.dataprep`. The
Ciccia (2024) quantities are surfaced as first-class fields rather than
buried in a nested metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SDIDInputs:
    """Pre-processed two-DataFrame view of the SDID panel.

    Parameters
    ----------
    cohorts_dict : Dict[int, Dict[str, Any]]
        The cohort-keyed payload consumed by the math helpers. Keys are
        cohort adoption periods (integers); values follow the schema of
        ``estimate_cohort_sdid_effects``.
    treated_unit_name : Any
        Label of the (canonical) treated aggregate. For staggered designs
        this is the label of an arbitrary representative treated unit.
    donor_names : Sequence
        Labels of the donor units in the order matching the donor matrices.
    time_labels : np.ndarray
        Time labels in original order.
    n_pre : int
        Pre-treatment period count (relative to the earliest cohort).
    n_post : int
        Post-treatment period count.
    Ywide : Any
        The wide outcome frame produced by ``dataprep``; kept for plotting.
    outcome : str
        Outcome variable name.
    """

    cohorts_dict: Dict[int, Dict[str, Any]]
    treated_unit_name: Any
    donor_names: Sequence
    time_labels: np.ndarray
    n_pre: int
    n_post: int
    Ywide: Any
    outcome: str


@dataclass(frozen=True)
class SDIDEventEffect:
    """Single event-time effect with placebo-based SE and CI."""

    ell: int
    tau: float
    se: float
    ci: Tuple[float, float]


@dataclass(frozen=True)
class SDIDCohort:
    """Per-cohort SDID estimator output (Ciccia 2024 Eqs. 2 and 3).

    Parameters
    ----------
    adoption_period : int
        Treatment-onset period for this cohort.
    n_treated : int
        Number of treated units in this cohort (``N_tr^a``).
    n_post : int
        Number of post-treatment periods for this cohort (``T_tr^a``).
    att : float
        Cohort ATT ``tau_a^sdid`` (Equation 2 of Ciccia 2024).
    att_se : float
        Placebo-based standard error for ``att``.
    att_ci : Tuple[float, float]
        95 percent confidence interval for ``att``.
    event_effects : Dict[int, SDIDEventEffect]
        Cohort-specific event-time effects ``tau_{a, ell}^sdid``
        (Equation 3), keyed by event time ``ell`` (negative for pre,
        non-negative for post).
    actual : np.ndarray
        Mean treated-unit outcome trajectory, shape ``(T,)``.
    counterfactual : np.ndarray
        Bias-corrected synthetic control trajectory, shape ``(T,)``.
    """

    adoption_period: int
    n_treated: int
    n_post: int
    att: float
    att_se: float
    att_ci: Tuple[float, float]
    event_effects: Dict[int, SDIDEventEffect]
    actual: np.ndarray
    counterfactual: np.ndarray


@dataclass(frozen=True)
class SDIDEventStudy:
    """Pooled event-study estimator (Ciccia 2024 Equation 6).

    Parameters
    ----------
    event_times : np.ndarray
        Event-time offsets ``ell`` covered by the pooled estimator.
    tau : np.ndarray
        Pooled effects ``tau_ell^sdid``, aligned with ``event_times``.
    se : np.ndarray
        Placebo-based standard errors aligned with ``tau``.
    ci : np.ndarray
        Length-2 CI tuples aligned with ``tau``, shape ``(L, 2)``.
    """

    event_times: np.ndarray
    tau: np.ndarray
    se: np.ndarray
    ci: np.ndarray


@dataclass(frozen=True)
class SDIDInference:
    """Overall ATT and placebo inference (Ciccia 2024 Equation 7).

    Parameters
    ----------
    att : float
        Treated-unit-weighted aggregate ATT across cohorts.
    se : float
        Placebo-based standard error.
    ci : Tuple[float, float]
        95 percent confidence interval.
    p_value : float
        Two-sided p-value from the placebo distribution
        ``(|placebo| >= |att|) + 1) / (B + 1)``.
    placebo_att : np.ndarray
        Vector of placebo ATT estimates, useful for diagnostics.
    method : str
        Inference method label; currently always ``"placebo"``.
    n_placebo : int
        Number of placebo iterations actually completed (may be smaller
        than the configured ``B`` when some iterations yield NaN).
    """

    att: float
    se: float
    ci: Tuple[float, float]
    p_value: float
    placebo_att: np.ndarray
    method: str
    n_placebo: int


class SDIDResults(BaseEstimatorResults):
    """Public ``SDID.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    it populates the standardized sub-models so the flat accessors (``att`` /
    ``att_ci`` / ``counterfactual`` / ``gap`` / ``pre_rmse``) resolve through
    the base contract. The flat ``counterfactual`` / ``gap`` are the
    treated-unit-weighted aggregate trajectories across cohorts (for a single
    cohort, that cohort's path). The full SDID detail -- the placebo inference,
    the pooled event study, and the per-cohort decomposition -- stays in the
    typed fields below.

    Parameters
    ----------
    inputs : SDIDInputs
        Pre-processed panel.
    inference_detail : SDIDInference
        Overall ATT and placebo inference (Equation 7). Was ``inference``
        before the contract migration; the standardized ``inference`` slot now
        holds the ATT-level :class:`~mlsynth.config_models.InferenceResults`.
    event_study : SDIDEventStudy
        Pooled event-study estimator (Equation 6).
    cohorts : Dict[int, SDIDCohort]
        Per-cohort estimator outputs (Equations 2 and 3), keyed by
        adoption period.
    raw : Dict[str, Any]
        Raw dictionary returned by
        :func:`mlsynth.utils.sdid_helpers.event_study.estimate_event_study_sdid`,
        retained for reproducibility and downstream tooling.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SDIDInputs
    inference_detail: SDIDInference
    event_study: SDIDEventStudy
    cohorts: Dict[int, SDIDCohort]
    raw: Dict[str, Any]


# Resolve forward references (module uses ``from __future__ import annotations``).
SDIDResults.model_rebuild()
