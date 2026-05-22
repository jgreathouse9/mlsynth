"""Frozen dataclasses for the Synthetic IV (SIV) estimator.

The SIV pipeline of Gulek and Vives-i-Bastida (2024) ("Synthetic IV
Estimation in Panels", arXiv:2412.???) is a two-step procedure:

  1. For each unit ``i``, fit a synthetic control on the pre-period
     using stacked outcome/treatment/instrument predictors and form
     debiased series ``(\\tilde Y_i, \\tilde R_i, \\tilde Z_i)`` over
     the post-period.
  2. Run 2SLS of ``\\tilde Y`` on ``\\tilde R`` with instrument
     ``\\tilde Z`` (or one of the variants that selectively debiases
     only ``Z`` or runs an instrument-space projection first).

The five layers below — inputs, per-unit weights, debiased series,
estimates, inference — keep that pipeline pluggable.

References
----------
Gulek, A. and Vives-i-Bastida, J. (2024). "Synthetic IV Estimation
in Panels."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..fast_scm_helpers.structure import IndexSet


@dataclass(frozen=True)
class SIVInputs:
    """Preprocessed (unit x time) panel for SIV.

    Parameters
    ----------
    Y, R, Z : np.ndarray
        Shape ``(J, T)``. Outcome, treatment intensity, instrument.
    unit_index : IndexSet
        Sorted unit labels in row order of ``Y / R / Z``.
    time_index : IndexSet
        Time labels in column order.
    T0 : int
        Last pre-treatment period (inclusive); intervention starts at
        column ``T0`` (0-indexed), so pre-period is ``[0, T0)`` and
        post-period is ``[T0, T)``.
    T0_train : Optional[int]
        Optional train/blank split inside the pre-period used by the
        ensemble CV and the split-conformal inference. ``None`` falls
        back to a sensible default (``floor(0.75 * T0)``).
    has_pre_treatment : bool
        True iff the treatment ``R`` has any non-zero pre-period value.
        In shift-share designs (like the paper's Syrian application)
        ``R == 0`` for all ``t < T0`` and only outcome columns enter
        the SC design matrix.
    has_pre_instrument : bool
        True iff ``Z`` has any non-zero pre-period value.
    """

    Y: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    unit_index: IndexSet
    time_index: IndexSet
    T0: int
    T0_train: Optional[int] = None
    has_pre_treatment: bool = False
    has_pre_instrument: bool = False

    @property
    def J(self) -> int:
        """Number of units."""
        return int(self.Y.shape[0])

    @property
    def T(self) -> int:
        """Total number of periods."""
        return int(self.Y.shape[1])

    @property
    def T1(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class SIVWeights:
    """Per-unit synthetic control weights and debiased series.

    For each unit ``i``, ``W[i]`` is a length-``J`` vector with
    ``W[i, i] == 0`` and the remaining entries summing to 1 (under
    the simplex constraint) or having ``l1``-norm <= C (under the
    L1-ball constraint). Debiased series are computed for the full
    panel, with the pre-period serving as the in-sample residual
    series used by inference.

    Parameters
    ----------
    W : np.ndarray
        Shape ``(J, J)`` weight matrix.
    Y_sc, R_sc, Z_sc : np.ndarray
        Shape ``(J, T)``. The synthetic-control imputation of each
        series at every period.
    Y_tilde, R_tilde, Z_tilde : np.ndarray
        Shape ``(J, T)``. The debiased series ``X - X_sc``.
    constraint : str
        Either ``"simplex"`` or ``"l1_ball"``; mirrors the
        ``SIVConfig.weight_constraint`` setting that produced the fit.
    """

    W: np.ndarray
    Y_sc: np.ndarray
    R_sc: np.ndarray
    Z_sc: np.ndarray
    Y_tilde: np.ndarray
    R_tilde: np.ndarray
    Z_tilde: np.ndarray
    constraint: str


@dataclass(frozen=True)
class SIVEstimate:
    """A single 2SLS estimate (point + standard error).

    The variant tag identifies which set of debiased series produced
    the estimate so a downstream consumer can tell ``SIV`` from
    ``SIV_Z`` from ``Projected`` apart in a results dict.
    """

    variant: str
    theta_hat: float
    se: float
    pi_hat: float                # reduced-form coefficient
    beta_first_stage: float      # first-stage coefficient
    f_stat: float                # first-stage F statistic
    n_post_obs: int


@dataclass(frozen=True)
class SIVInference:
    """Inferential output: asymptotic Gaussian CI and split-conformal CI.

    Parameters
    ----------
    method : str
        ``"asymptotic"``, ``"conformal"``, or ``"none"``.
    alpha : float
        Two-sided significance level used for the CI.
    theta_hat : float
        Selected estimate (the variant the user asked the orchestrator
        to score; the *other* variants are also retained inside
        :class:`SIVResults`).
    ci_lower, ci_upper : float
        ``(1 - alpha)`` confidence interval.
    p_value : float
        Two-sided test of ``H_0 : theta = 0``.
    event_study_coefs : np.ndarray
        Per-period reduced-form event-study coefficients used by the
        split-conformal test (empty array for ``method != "conformal"``).
    permutation_pvalue : float
        Conformal permutation p-value (NaN for non-conformal methods).
    """

    method: str
    alpha: float
    theta_hat: float
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    p_value: float = float("nan")
    event_study_coefs: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    permutation_pvalue: float = float("nan")


@dataclass(frozen=True)
class SIVResults:
    """Top-level container returned by :meth:`mlsynth.SIV.fit`.

    Holds preprocessed inputs, the SC weights and debiased series for
    both the canonical and projected pipelines, every variant of the
    2SLS estimator, and the inferential output.
    """

    inputs: SIVInputs
    weights: SIVWeights
    weights_projected: Optional[SIVWeights]
    estimates: Dict[str, SIVEstimate]
    selected_variant: str
    inference: SIVInference
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def theta_hat(self) -> float:
        """Point estimate of theta for the selected variant."""
        return self.estimates[self.selected_variant].theta_hat
