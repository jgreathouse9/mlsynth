"""Typed result containers for Sequential SDiD.

All matrices follow ``mlsynth``'s ``(T, N)`` orientation (rows = time,
columns = cohort).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SeqSDIDInputs:
    """Aggregated cohort-level panel fed into Sequential SDiD.

    Parameters
    ----------
    Y_agg : np.ndarray
        Cohort-level outcome matrix of shape ``(T, A)`` where ``A`` is the
        number of distinct adoption cohorts (treated cohorts in ascending
        order followed by the never-treated cohort, if present).
    pi : np.ndarray
        Cohort shares ``pi_a = n_a / n``, length ``A``, summing to 1 over
        the entire sample.
    cohort_periods : np.ndarray
        Length-``A`` array of cohort adoption periods (1-based time
        indices). The never-treated cohort is encoded as
        ``np.iinfo(np.int64).max`` and lives at index ``A - 1``.
    cohort_labels : Sequence
        Human-readable labels for each cohort (e.g. adoption year), aligned
        with ``cohort_periods``.
    treated_cohort_indices : np.ndarray
        Integer indices into ``Y_agg``'s second axis identifying treated
        (i.e. finitely-adopting) cohorts.
    time_labels : np.ndarray
        Original time labels, length ``T``.
    n_units : int
        Total number of underlying units.
    a_min : int
        Earliest treated cohort to estimate (1-based time index).
    a_max : int
        Latest treated cohort to estimate.
    K : int
        Maximum event-time horizon to estimate.
    """

    Y_agg: np.ndarray
    pi: np.ndarray
    cohort_periods: np.ndarray
    cohort_labels: Sequence
    treated_cohort_indices: np.ndarray
    time_labels: np.ndarray
    n_units: int
    a_min: int
    a_max: int
    K: int


@dataclass(frozen=True)
class SeqSDIDCohortEffect:
    """Single cohort-by-horizon estimate.

    Parameters
    ----------
    cohort_period : int
        1-based time index of the cohort's adoption period.
    k : int
        Horizon (event-time offset), with ``k = 0`` the first treated period.
    tau : float
        Point estimate ``tau_{a,k}^SSDiD``.
    omega : np.ndarray
        Unit weights solving the (a, k) QP, aligned with the slice of
        ``Y_agg`` corresponding to later cohorts ``j > a``.
    lambda_w : np.ndarray
        Time weights solving the (a, k) QP, aligned with pre-event periods
        ``l < a + k``.
    """

    cohort_period: int
    k: int
    tau: float
    omega: np.ndarray
    lambda_w: np.ndarray


@dataclass(frozen=True)
class SeqSDIDEventStudy:
    """Pooled horizon-k effects ``tau_hat_k^SSDiD(mu)``.

    Parameters
    ----------
    horizons : np.ndarray
        Length-``K + 1`` array of event-time horizons ``k = 0, 1, ..., K``.
    tau : np.ndarray
        Pooled effects aligned with ``horizons``.
    se : np.ndarray
        Bootstrap standard errors aligned with ``tau``.
    ci : np.ndarray
        Length-``(K + 1, 2)`` array of Wald confidence intervals.
    bootstrap_draws : np.ndarray
        Bootstrap replicate matrix of shape ``(B, K + 1)``, retained for
        downstream diagnostics or alternative quantile-based intervals.
    alpha : float
        Significance level used for ``ci``.
    """

    horizons: np.ndarray
    tau: np.ndarray
    se: np.ndarray
    ci: np.ndarray
    bootstrap_draws: np.ndarray
    alpha: float


@dataclass(frozen=True)
class SeqSDIDInference:
    """Bayesian bootstrap inference summary."""

    n_bootstrap: int
    method: str  # "bayesian_bootstrap"
    seed: int


@dataclass(frozen=True)
class SeqSDIDResults:
    """Public ``SequentialSDID.fit()`` return container.

    Parameters
    ----------
    inputs : SeqSDIDInputs
        Aggregated panel + cohort metadata.
    cohort_effects : Dict[Tuple[int, int], SeqSDIDCohortEffect]
        Per-(cohort_period, k) effects.
    event_study : SeqSDIDEventStudy
        Pooled horizon-k effects with bootstrap inference.
    inference : SeqSDIDInference
        Bootstrap configuration summary.
    eta : float
        Regularization parameter actually used.
    mode : str
        ``"ssdid"`` (the paper's main estimator) or ``"sdid_imputation"``
        (the ``eta -> infinity`` Borusyak-style limit from Remark 2.2).
    raw_event_study : np.ndarray
        Length-``K + 1`` non-bootstrapped pooled effect vector (the same
        numbers as ``event_study.tau``; kept separately for clarity).
    """

    inputs: SeqSDIDInputs
    cohort_effects: Dict[Tuple[int, int], SeqSDIDCohortEffect]
    event_study: SeqSDIDEventStudy
    inference: SeqSDIDInference
    eta: float
    mode: str
    raw_event_study: np.ndarray
