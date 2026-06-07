"""Structured containers for the TASC pipeline.

All matrices follow the paper's convention ``Y in R^{N x T}`` with rows = units
(target as the first row, donors below) and columns = time. This is the
transpose of ``datautils.dataprep``'s ``Ywide`` (which is time x unit). The
transpose is performed once in ``setup.prepare_tasc_inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class TASCInputs:
    """Pre-processed panel data fed into the TASC EM and inference loops.

    Parameters
    ----------
    Y_full : np.ndarray
        Outcome matrix of shape ``(N, T)`` with the target in row 0 and the
        ``n = N - 1`` donor units in rows 1..N-1.
    Y_pre : np.ndarray
        Pre-treatment slice ``Y_full[:, :T0]`` of shape ``(N, T0)``.
    Y_post_donors : np.ndarray or None
        Post-treatment donor-only slice ``Y_full[1:, T0:]`` of shape
        ``(n, T - T0)``. ``None`` if ``T0 == T`` (no post period available).
    T0 : int
        Number of pre-treatment periods.
    T : int
        Total number of periods.
    N : int
        Total number of units (target + donors).
    treated_unit_name : str
        Identifier of the treated unit.
    donor_names : Sequence
        Identifiers for the donor units in the order matching rows 1..N-1.
    time_labels : np.ndarray
        Time labels in their original order, length ``T``.
    pre_periods : int
        Alias of ``T0`` kept for compatibility with plotting helpers that
        expect a ``processed_data_dict`` from ``dataprep``.
    post_periods : int
        ``T - T0``.
    Ywide : object
        The wide pandas frame produced by ``dataprep`` (rows = time,
        columns = units). Retained so that ``plot_estimates`` can use it
        directly without re-pivoting.
    y_target : np.ndarray
        Convenience copy of the full observed target series, length ``T``
        (post-treatment values are the *observed* values, used only for plotting
        and effect computation; TASC treats them as missing during filtering).
    """

    Y_full: np.ndarray
    Y_pre: np.ndarray
    Y_post_donors: Optional[np.ndarray]
    T0: int
    T: int
    N: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray
    pre_periods: int
    post_periods: int
    Ywide: object
    y_target: np.ndarray


@dataclass(frozen=True)
class TASCParameters:
    """State-space parameters ``theta = {A, H, Q, R, m0, P0}``.

    Parameters
    ----------
    A : np.ndarray
        Transition matrix, shape ``(d, d)``.
    H : np.ndarray
        Observation matrix, shape ``(N, d)``. Row 0 is ``h_1^T``.
    Q : np.ndarray
        State-noise covariance, shape ``(d, d)``.
    R : np.ndarray
        Observation-noise covariance, shape ``(N, N)``.
    m0 : np.ndarray
        Initial state mean, shape ``(d,)``.
    P0 : np.ndarray
        Initial state covariance, shape ``(d, d)``.
    """

    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray


@dataclass(frozen=True)
class TASCFilteredStates:
    """Output of the forward Kalman pass.

    Parameters
    ----------
    m : np.ndarray
        Filtered means stacked over time, shape ``(T + 1, d)``. Index 0 holds
        the prior ``m0``; index ``k >= 1`` holds the posterior mean
        ``m_{k|k}``.
    P : np.ndarray
        Filtered covariances, shape ``(T + 1, d, d)``. Index 0 holds ``P0``.
    """

    m: np.ndarray
    P: np.ndarray


@dataclass(frozen=True)
class TASCSmoothedStates:
    """Output of the RTS backward pass.

    Parameters
    ----------
    m_s : np.ndarray
        Smoothed means, shape ``(T + 1, d)``. Index 0 is ``m_0^s``.
    P_s : np.ndarray
        Smoothed covariances, shape ``(T + 1, d, d)``. Index 0 is ``P_0^s``.
    G : np.ndarray
        RTS smoother gain matrices, shape ``(T + 1, d, d)``. ``G[k]`` is the
        gain used to smooth time ``k`` from ``k + 1``; ``G[T]`` is unused.
    """

    m_s: np.ndarray
    P_s: np.ndarray
    G: np.ndarray


@dataclass(frozen=True)
class TASCInference:
    """Counterfactual point estimates and posterior confidence intervals.

    Parameters
    ----------
    counterfactual : np.ndarray
        Estimated counterfactual for the target unit across all ``T`` periods,
        ``y_hat_{0, t} = h_1^T m_t^s``.
    ci_lower : np.ndarray
        Lower confidence band, shape ``(T,)``.
    ci_upper : np.ndarray
        Upper confidence band, shape ``(T,)``.
    posterior_variance : np.ndarray
        Posterior variance of the target row, ``h_1^T P_t^s h_1 + R_{1,1}``,
        shape ``(T,)``.
    alpha : float
        Significance level used to build the bands.
    """

    counterfactual: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    posterior_variance: np.ndarray
    alpha: float


@dataclass(frozen=True)
class TASCDesign:
    """Learned model and EM diagnostics.

    Parameters
    ----------
    parameters : TASCParameters
        Final EM-estimated parameters.
    n_em_iter_used : int
        Number of EM iterations actually executed (may be less than the cap
        if ``em_tol`` triggered early stopping).
    em_param_deltas : np.ndarray
        Per-iteration max absolute change in ``(A, H)``. Length equal to
        ``n_em_iter_used``.
    filtered : TASCFilteredStates
        Forward filtered states from the final full pass.
    smoothed : TASCSmoothedStates
        Backward smoothed states from the final full pass.
    """

    parameters: TASCParameters
    n_em_iter_used: int
    em_param_deltas: np.ndarray
    filtered: TASCFilteredStates
    smoothed: TASCSmoothedStates


class TASCResults(BaseEstimatorResults):
    """Public ``TASC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the TASC-specific fields below it exposes the standardized
    sub-models (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``pre_rmse``. TASC is a state-space / EM
    estimator (no donor weights), so the ``weights`` slot records the method
    rather than per-donor weights; the per-period posterior bands live in the
    ``inference`` slot's ``details`` (and on ``inference_detail``).

    Parameters
    ----------
    inputs : TASCInputs
        Pre-processed panel data.
    design : TASCDesign
        Learned model, EM diagnostics, and filtered / smoothed state arrays.
    inference_detail : TASCInference
        The raw counterfactual + posterior-based pointwise confidence bands
        (``counterfactual`` / ``ci_lower`` / ``ci_upper`` / ``posterior_variance``
        / ``alpha``). (Renamed from ``inference``; the standardized
        :class:`~mlsynth.config_models.InferenceResults` is mirrored into the
        ``inference`` slot.)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: TASCInputs
    design: TASCDesign
    inference_detail: TASCInference
