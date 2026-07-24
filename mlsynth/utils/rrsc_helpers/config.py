"""Configuration for the RRSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class RRSCConfig(BaseEstimatorConfig):
    """Configuration for RRSC -- robust-regression synthetic control with
    interference (He, Li, Shi & Miao 2026, *A robust regression approach to
    synthetic control with interference*).

    RRSC recovers the direct effect on the treated unit and the interference
    effects on every control unit under an untreated linear factor model,
    without prespecifying which controls are interfered. Stage one estimates
    time-invariant factor loadings from the pre-period panel; stage two treats
    the average direct and interference effects as a sparse-outlier component in
    a robust regression against those loadings.

    Parameters
    ----------
    regime : {"auto", "fixed_n", "large_n"}
        Asymptotic regime. ``"fixed_n"`` (Section 3): few units, long pre- and
        post-periods, high-breakdown LTS under the majority-valid-controls
        condition. ``"large_n"`` (Section 4): many units, short post-period
        allowed, robust M-estimation. ``"auto"`` (default) picks ``"fixed_n"``
        when ``T0 >= 5 N`` (the paper's rule of thumb for reliable pre-period
        factor analysis) and ``"large_n"`` otherwise.
    inference : {"cbb", "conformal", "none"}
        Uncertainty procedure. ``"cbb"`` (default) is the circular block
        bootstrap: the per-unit bootstrap variance drives a normal (Wald)
        confidence interval and p-value, exactly as the authors' analyses
        report. ``"conformal"`` is the moving-block permutation test of
        Chernozhukov et al. (2021), valid with a short post-period.
        ``"none"`` returns point estimates only.
    n_factors : int or None
        Number of common factors ``r``. ``None`` (default) selects ``r`` by the
        Bai & Ng (2002) IC on the pre-period panel.
    max_factors : int
        Upper bound for the Bai-Ng factor search.
    loss : {"huber", "logcosh"}
        Robust loss for the large-N factor regression (Assumption 7).
    huber_delta : float
        Huber tuning constant (large-N).
    lts_alpha : float
        LTS trimming proportion for the fixed-N regime (0.5 = maximal
        breakdown, robustbase default).
    update_dif : bool
        Fixed-N only: regress the post-minus-pre difference (``True``, the
        authors' real-data setting) rather than the raw post mean.
    cbb_block_length : int or None
        Circular-block-bootstrap block length. ``None`` uses ``max(2, T0//10)``.
    cbb_reps : int
        Number of bootstrap replicates.
    cbb_separation : bool
        Resample the pre- and post-periods in separate blocks (the authors'
        default).
    alpha : float
        Two-sided level for confidence intervals and the significance map.
    random_state : int
        Seed for the bootstrap / permutation RNG.
    """

    regime: Literal["auto", "fixed_n", "large_n"] = Field(
        default="auto",
        description="Asymptotic regime: fixed_n, large_n, or auto (by T0 vs 5N).",
    )
    inference: Literal["cbb", "conformal", "none"] = Field(
        default="cbb",
        description="Uncertainty procedure: cbb (bootstrap Wald), conformal, or none.",
    )
    n_factors: Optional[int] = Field(
        default=None, ge=1,
        description="Number of factors r; None selects via Bai-Ng (2002) IC.",
    )
    max_factors: int = Field(
        default=15, ge=1,
        description="Upper bound for the Bai-Ng factor search.",
    )
    loss: Literal["huber", "logcosh"] = Field(
        default="huber",
        description="Robust loss for the large-N factor regression.",
    )
    huber_delta: float = Field(
        default=1.345, gt=0.0,
        description="Huber tuning constant (large-N).",
    )
    lts_alpha: float = Field(
        default=0.5, ge=0.5, le=1.0,
        description="LTS trimming proportion for the fixed-N regime.",
    )
    update_dif: bool = Field(
        default=True,
        description="Fixed-N: regress the post-minus-pre difference.",
    )
    cbb_block_length: Optional[int] = Field(
        default=None, ge=1,
        description="Circular-block-bootstrap block length; None -> max(2, T0//10).",
    )
    cbb_reps: int = Field(
        default=500, ge=50,
        description="Number of circular-block-bootstrap replicates.",
    )
    cbb_separation: bool = Field(
        default=True,
        description="Resample pre- and post-periods in separate blocks.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for CIs and the significance map.",
    )
    random_state: int = Field(
        default=2025,
        description="Seed for the bootstrap / permutation RNG.",
    )

    @model_validator(mode="after")
    def _check_rrsc(self, /) -> "RRSCConfig":
        if self.n_factors is not None and self.n_factors > self.max_factors:
            raise MlsynthConfigError(
                f"n_factors ({self.n_factors}) exceeds max_factors ({self.max_factors})."
            )
        return self
