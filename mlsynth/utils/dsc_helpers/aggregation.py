"""Pre-period aggregation weights for DSC.

The DSC weight is

.. math::

   \\widehat w = \\sum_{t \\in \\mathcal T_0} \\lambda_t\\, \\widehat w_t,
   \\qquad
   \\lambda_t \\ge 0, \\quad \\sum_{t \\in \\mathcal T_0} \\lambda_t = 1.

Zhang, Zhang & Zhang (2026, Section 2) point to Arkhangelsky et al.
(2021) for principled choices of :math:`\\lambda_t`. mlsynth currently
ships the uniform rule (default) and a recency-weighted variant; the
SDID-style time weights can be supplied externally via the
``lambda_weights`` argument to :func:`run_dsc`.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError


def build_lambda_weights(
    T0: int,
    method: Literal["uniform", "recency"] = "uniform",
    decay: float = 0.9,
) -> np.ndarray:
    """Return length-``T0`` non-negative weights summing to 1.

    Parameters
    ----------
    T0 : int
        Number of pre-treatment periods.
    method : {"uniform", "recency"}
        ``"uniform"`` returns ``1 / T0`` everywhere. ``"recency"``
        returns a geometrically-decayed schedule
        :math:`\\lambda_t \\propto \\mathrm{decay}^{T_0 - t}` so the
        weight peaks at the most recent pre-period.
    decay : float
        Per-period decay factor for ``"recency"`` (must lie in
        ``(0, 1]``).
    """
    if T0 < 1:
        raise MlsynthEstimationError("T0 must be a positive integer.")
    if method == "uniform":
        return np.full(T0, 1.0 / T0)
    if method == "recency":
        if not (0.0 < decay <= 1.0):
            raise MlsynthEstimationError("decay must lie in (0, 1].")
        ages = np.arange(T0 - 1, -1, -1, dtype=float)
        raw = decay ** ages
        return raw / raw.sum()
    raise MlsynthEstimationError(
        f"Unknown lambda method {method!r}; expected 'uniform' or 'recency'."
    )


def aggregate_period_weights(
    period_weights: np.ndarray,
    lambda_weights: np.ndarray,
) -> np.ndarray:
    """Compute :math:`\\widehat w = \\sum_t \\lambda_t \\widehat w_t`.

    Parameters
    ----------
    period_weights : np.ndarray
        Shape ``(T0, J)`` matrix of per-pre-period donor weights.
    lambda_weights : np.ndarray
        Length-``T0`` aggregation weights.
    """
    if period_weights.ndim != 2:
        raise MlsynthEstimationError(
            "period_weights must be 2-D (T0, J)."
        )
    if lambda_weights.shape != (period_weights.shape[0],):
        raise MlsynthEstimationError(
            f"lambda_weights has shape {lambda_weights.shape}; expected "
            f"({period_weights.shape[0]},)."
        )
    if not np.all(lambda_weights >= -1e-12):
        raise MlsynthEstimationError("lambda_weights must be non-negative.")
    if abs(float(lambda_weights.sum()) - 1.0) > 1e-6:
        raise MlsynthEstimationError("lambda_weights must sum to 1.")
    return lambda_weights @ period_weights
