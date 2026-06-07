"""Configuration for the DSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class DSCConfig(BaseEstimatorConfig):
    """Configuration for the Distributional Synthetic Control (DSC) estimator.

    DSC (Gunsilius 2023; asymptotic theory: Zhang, Zhang & Zhang 2026)
    fits simplex-constrained weights on the *quantile functions* of
    donor outcomes to reconstruct the treated unit's counterfactual
    *outcome distribution*. Unlike the other mlsynth estimators, DSC
    expects micro-level panel data: each ``(unit, time)`` cell carries
    multiple individual observations, supplied as one row per
    individual in the input DataFrame.

    Parameters
    ----------
    M : int, optional
        Number of quantile-grid points used to approximate the
        2-Wasserstein loss. If ``None``, defaults to
        ``max(200, min_cell_size)`` (Zhang et al. 2026 suggest
        :math:`M = C n` for a constant ``C >= 1``).
    grid_method : {"halton", "sobol", "uniform"}
        Sampling rule for the quantile grid. ``"halton"`` (default)
        and ``"sobol"`` are quasi-Monte Carlo with Koksma-Hlawka error
        :math:`O(\\log M / M)`; ``"uniform"`` is i.i.d. with
        :math:`O(M^{-1/2})` error.
    lambda_method : {"uniform", "recency"}
        Default rule for the pre-period aggregation weights
        :math:`\\lambda_t`. Ignored when ``lambda_weights`` is set.
    lambda_decay : float
        Geometric decay factor for ``lambda_method="recency"``.
        Default ``0.9``.
    lambda_weights : sequence of float, optional
        Caller-supplied length-``T0`` aggregation weights (must be
        non-negative and sum to 1). Useful for Arkhangelsky et al.
        (2021) SDiD-style time weights computed externally.
    qte_quantiles : sequence of float, optional
        Quantile grid in ``(0, 1)`` at which to report the QTE. If
        ``None``, an evenly spaced grid of ``n_qte_points`` quantiles
        is used.
    n_qte_points : int
        Length of the default QTE grid when ``qte_quantiles`` is None.
        Default 99 (every percentile from 0.01 to 0.99).
    random_state : int
        Seed forwarded to the QMC quantile-grid sampler.
    """

    M: Optional[int] = Field(
        default=None, ge=50,
        description="Number of quantile-grid points used to approximate the "
                    "2-Wasserstein loss. Defaults to max(200, min cell size).",
    )
    grid_method: Literal["halton", "sobol", "uniform"] = Field(
        default="halton",
        description="Quantile-grid sampling rule (QMC by default).",
    )
    lambda_method: Literal["uniform", "recency"] = Field(
        default="uniform",
        description="Default rule for pre-period aggregation weights lambda_t.",
    )
    lambda_decay: float = Field(
        default=0.9, gt=0.0, le=1.0,
        description="Geometric decay for lambda_method='recency'.",
    )
    lambda_weights: Optional[List[float]] = Field(
        default=None,
        description="Caller-supplied length-T0 aggregation weights "
                    "(non-negative, sum to 1).",
    )
    qte_quantiles: Optional[List[float]] = Field(
        default=None,
        description="Explicit quantile grid in (0, 1) for QTE reporting.",
    )
    n_qte_points: int = Field(
        default=99, ge=1,
        description="Length of the default QTE grid when qte_quantiles is None.",
    )
    random_state: int = Field(
        default=0,
        description="Seed forwarded to the QMC sampler.",
    )
    compute_inference: bool = Field(
        default=False,
        description="Run the Gunsilius (2023) placebo permutation test "
                    "(Algorithm 1): refit DSC with every donor treated as a "
                    "placebo and rank the real unit's post-period Wasserstein "
                    "distance. Costs J extra pre-period refits.",
    )
    inference_grid_points: int = Field(
        default=200, ge=2,
        description="Number of quantiles used to evaluate the squared "
                    "2-Wasserstein distances in the placebo permutation test.",
    )
