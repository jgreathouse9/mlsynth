"""Frozen dataclasses for the SCD estimator.

SCD (Rincon & Song 2026) operates on grouped microdata: each
``(unit, time)`` cell carries many individual observations with optional
survey weights. :class:`SCDInputs` holds both the aggregated group means /
weighted cell totals (used by the point estimator) and the individual
``(group, time, outcome, weight)`` arrays (used by the influence-function
variance). :class:`InferenceOperators` bundles the fitted weights and the
matrices that drive the confidence-set membership test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass(frozen=True)
class SCDInputs:
    """Preprocessed grouped-microdata inputs for SCD.

    Attributes
    ----------
    treated_name : Any
        Label of the treated unit (group ``0``).
    donor_names : list
        Length-``K`` ordering of donor unit labels (groups ``1..K``).
    group_means : np.ndarray
        Survey-weighted group means, shape ``(K + 1, Ttot)``; row ``0`` is
        the treated group, rows ``1..K`` the donors, columns are periods.
    n_jt : np.ndarray
        Weighted cell totals :math:`\\sum_i \\pi_i \\mathbf 1\\{G_i=j,t\\}`,
        shape ``(K + 1, Ttot)``.
    G, t, Y, weight : np.ndarray
        Per-individual group index (0=treated, 1..K donors), period
        (1-based), outcome, and survey weight; each shape ``(n,)``.
    time_labels : np.ndarray
        Length-``Ttot`` ordering of period labels.
    T0 : int
        Number of pre-treatment periods.
    Tstar : int
        First post-treatment period (1-based; ``Tstar = T0 + 1``).
    Ttot : int
        Total number of periods.
    """

    treated_name: Any
    donor_names: List[Any]
    group_means: np.ndarray
    n_jt: np.ndarray
    G: np.ndarray
    t: np.ndarray
    Y: np.ndarray
    weight: np.ndarray
    time_labels: np.ndarray
    T0: int
    Tstar: int
    Ttot: int

    @property
    def K(self) -> int:
        """Number of donor groups."""
        return len(self.donor_names)

    @property
    def n(self) -> int:
        """Total number of individual observations."""
        return int(self.Y.shape[0])

    @property
    def T1(self) -> int:
        """Number of post-treatment periods."""
        return self.Ttot - self.T0


@dataclass(frozen=True)
class InferenceOperators:
    """Fitted weights and confidence-set machinery for SCD inference.

    Attributes
    ----------
    hat_w : np.ndarray
        Fitted simplex weights, shape ``(K,)``.
    theta : np.ndarray
        Effect path :math:`\\hat\\theta_t` over all periods, shape ``(Ttot,)``.
    gfull, Gfull : np.ndarray
        Differenced treated series ``(Ttot,)`` and donor matrix
        ``(Ttot, K)`` over all periods (``theta = gfull - Gfull @ hat_w``).
    hat_H, hat_h : np.ndarray
        Pre-period Gram ``(1/T0) G'G`` (``(K, K)``) and cross term
        ``(1/T0) G'g`` (``(K,)``); the moment ``hat_H w - hat_h`` is the
        confidence-set deviation.
    precomp : np.ndarray
        ``B2 V^{-1} B2'``, shape ``(K, K)``; the metric of the membership QP.
    sqrtP : np.ndarray
        Symmetric square root of ``precomp``, shape ``(K, K)``.
    sigma2, se : np.ndarray
        Repeated-cross-section pointwise variance ``sigma2_t`` and standard
        error ``se_t = sqrt(sigma2_t / n)``, each shape ``(Ttot,)``.
    hatV_trace : float
        Trace of the (K-1)-dimensional weight variance matrix (a diagnostic).
    K, n, T0, Ttot : int
        Dimensions carried for convenience.
    """

    hat_w: np.ndarray
    theta: np.ndarray
    gfull: np.ndarray
    Gfull: np.ndarray
    hat_H: np.ndarray
    hat_h: np.ndarray
    precomp: np.ndarray
    sqrtP: np.ndarray
    sigma2: np.ndarray
    se: np.ndarray
    hatV_trace: float
    K: int
    n: int
    T0: int
    Ttot: int
