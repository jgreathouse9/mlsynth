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


@dataclass(frozen=True)
class MSQRTResults:
    """Top-level container returned by :meth:`mlsynth.MSQRT.fit`.

    Attributes
    ----------
    inputs : MSQRTInputs
    att : float
        Average treatment effect on the treated (mean post-period gap over all
        treated units).
    att_percent : float
        ``att`` as a percentage of the mean synthetic counterfactual on the
        post window.
    theta : np.ndarray
        Estimated donor-weight matrix ``Theta``, shape ``(n, m)``.
    weights : object
        :class:`mlsynth.config_models.WeightsResults` -- per-treated-unit donor
        weight dicts plus aggregate sparsity stats.
    counterfactual : np.ndarray
        Synthetic (untreated) outcome for the treated units, shape
        ``(T0 + T_post, m)``.
    gap : np.ndarray
        Observed minus synthetic for the treated units, same shape.
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
    pre_rmse : float
        Root-mean-square pre-period gap (overall fit quality).
    inference : SCPIResults, optional
        CFPT/scpi prediction intervals (Cattaneo, Feng, Palomba & Titiunik
        2025); see :mod:`mlsynth.utils.scpi_helpers`. For MSQRT only the
        out-of-sample error is modelled.
    metadata : dict
    """

    inputs: MSQRTInputs
    att: float
    att_percent: float
    theta: np.ndarray
    weights: Any
    counterfactual: np.ndarray
    gap: np.ndarray
    att_t: np.ndarray
    unit_att: Dict[Any, float]
    treated_mean: np.ndarray
    synthetic_mean: np.ndarray
    best_lambda: float
    sparsity: np.ndarray
    pre_rmse: float
    inference: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
