"""Frozen dataclasses for the MSQRT estimator.

Shen, Song & Abadie (2025), *Efficiently Learning Synthetic Control Models
for High-dimensional Disaggregated Data* (arXiv:2510.22828). The estimator
stacks all treated units into one matrix regression ``Y = X Theta + E`` and
fits the donor-weight matrix ``Theta`` by Multivariate Square-root Lasso
(nuclear-norm loss + element-wise L1), then forms the ATT as the mean
post-treatment gap (observed minus synthetic) over the treated cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class MSQRTInputs:
    """Preprocessed block panel for MSQRT (multiple treated units, common timing).

    Matrices are time-major (rows = periods) to match the paper's
    ``Y = X Theta + E`` convention.

    Attributes
    ----------
    Y_pre, Y_post : np.ndarray
        Treated outcomes, shape ``(T0, m)`` and ``(T_post, m)``.
    X_pre, X_post : np.ndarray
        Control (donor) outcomes, shape ``(T0, n)`` and ``(T_post, n)``.
    treated_names, control_names : list
    time_labels : np.ndarray
        All period labels (length ``T0 + T_post``).
    T0 : int
        Number of pre-treatment periods (treatment begins at index ``T0``).
    """

    Y_pre: np.ndarray
    Y_post: np.ndarray
    X_pre: np.ndarray
    X_post: np.ndarray
    treated_names: List[Any]
    control_names: List[Any]
    time_labels: np.ndarray
    T0: int

    @property
    def m(self) -> int:        # number of treated units
        return self.Y_pre.shape[1]

    @property
    def n(self) -> int:        # number of donor units
        return self.X_pre.shape[1]

    @property
    def n_post(self) -> int:
        return self.Y_post.shape[0]


class MSQRTResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.MSQRT.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the MSQRT-specific fields below it exposes the standardized
    sub-models (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``att_ci`` / ``pre_rmse``. The treated
    counterfactual path (``res.counterfactual``) is the cross-treated-unit
    synthetic mean; the full ``(T, m)`` synthetic / gap matrices live in
    ``counterfactual_matrix`` / ``gap_matrix``. The per-treated-unit PCR donor
    weights live in the standardized ``weights`` slot.

    Parameters
    ----------
    inputs : MSQRTInputs
    att_percent : float
        ``att`` as a percentage of the mean synthetic counterfactual on the
        post window.
    theta : np.ndarray
        Estimated donor-weight matrix ``Theta``, shape ``(n, m)``.
    counterfactual_matrix : np.ndarray
        Synthetic (untreated) outcome for the treated units, shape
        ``(T0 + T_post, m)``. (Renamed from ``counterfactual``, which now
        returns the 1-D treated path per the result contract.)
    gap_matrix : np.ndarray
        Observed minus synthetic for the treated units, same shape. (Renamed
        from ``gap``, which now returns the 1-D treated gap path.)
    att_t : np.ndarray
        Mean treated gap at each post-treatment period, shape ``(T_post,)``.
    unit_att : Dict
        ``{treated_name: post-period mean gap}``.
    treated_mean, synthetic_mean : np.ndarray
        Cross-treated-unit means of observed / synthetic over the full timeline
        (the series the plotter draws), length ``T0 + T_post``.
    best_lambda : float
        Selected (or supplied) L1 penalty.
    sparsity : np.ndarray
        Per-treated-unit count of active donors (``|Theta_ij| > tol``).
    inference_intervals : SCPIResults, optional
        CFPT/scpi prediction intervals (Cattaneo, Feng, Palomba & Titiunik
        2025); see :mod:`mlsynth.utils.scpi_helpers`. For MSQRT only the
        out-of-sample error is modelled. (Renamed from ``inference``; the
        standardized :class:`~mlsynth.config_models.InferenceResults` is
        mirrored into the ``inference`` slot so ``res.att_ci`` resolves.)
    metadata : dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MSQRTInputs
    att_percent: float
    theta: np.ndarray
    counterfactual_matrix: np.ndarray
    gap_matrix: np.ndarray
    att_t: np.ndarray
    unit_att: Dict[Any, float]
    treated_mean: np.ndarray
    synthetic_mean: np.ndarray
    best_lambda: float
    sparsity: np.ndarray
    inference_intervals: Optional[Any] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)
