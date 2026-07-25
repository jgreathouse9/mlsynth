"""Configuration for the CAST estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class CASTConfig(BaseEstimatorConfig):
    """Configuration for CAST -- confidence intervals for treatment effects
    under staggered adoption (Xia, Yan & Wainwright 2025, *Inference under
    Staggered Adoption: Case Study of the Affordable Care Act*).

    CAST estimates the counterfactual outcome matrix by a spectral four-block
    routine and returns a confidence interval for *every* treated unit-period,
    not only for the pooled effect. Aggregates (the average effect on the
    treated, or any weighted version of it) are reported with standard errors
    derived from the same entrywise variances.

    Parameters
    ----------
    rank : int or None
        Rank ``r`` of the underlying mean matrix. ``None`` (default) selects it
        from the spectrum of the observed control block via ``rank_method``.
        The rank is the method's one substantive tuning choice: too small
        under-fits the factor structure, too large absorbs the treatment
        effect.
    rank_method : {"broken_stick", "svht", "ratio"}
        Selector used when ``rank`` is ``None``. ``"broken_stick"`` matches the
        authors' reference package; ``"svht"`` is the Gavish-Donoho optimal
        hard threshold; ``"ratio"`` takes the largest singular-value gap.
    max_rank : int
        Upper bound for the automatic search.
    alpha : float
        Two-sided level for the entrywise and aggregate confidence intervals.
    weights : list or None
        Optional weights for the aggregate effect
        ``tau_w,t = sum_i w_i tau_i,t``, in the order the units appear in the
        panel (sorted unit labels). Either a length-``N`` vector (one weight
        per unit, constant over time) or an ``N x T`` matrix for a weight that
        varies by period -- the paper's general bilinear functional, which is
        what a population-weighted total needs when the population itself moves
        over time. ``None`` gives the equally weighted average over treated
        units.
    renormalize_weights : bool
        Divide the supplied ``weights`` by their total over the units treated
        at each period, so the aggregate is a weighted *mean*. Set ``False``
        to report a population total (the paper's "population effect").
    """

    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Rank r of the mean matrix; None selects it via rank_method.",
    )
    rank_method: Literal["broken_stick", "svht", "ratio"] = Field(
        default="broken_stick",
        description="Automatic rank selector (broken_stick matches the reference).",
    )
    max_rank: int = Field(
        default=10, ge=1,
        description="Upper bound for the automatic rank search.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for entrywise and aggregate intervals.",
    )
    weights: Optional[list] = Field(
        default=None,
        description="Per-unit weights for the aggregate effect; None = equal weights.",
    )
    renormalize_weights: bool = Field(
        default=True,
        description="Normalize weights to sum to one over treated units (mean vs total).",
    )

    @model_validator(mode="after")
    def _check_cast(self, /) -> "CASTConfig":
        if self.rank is not None and self.rank > self.max_rank:
            raise MlsynthConfigError(
                f"rank ({self.rank}) exceeds max_rank ({self.max_rank})."
            )
        if self.weights is not None:
            import numpy as np

            if len(self.weights) == 0:
                raise MlsynthConfigError("weights must be non-empty when supplied.")
            try:
                arr = np.asarray(self.weights, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MlsynthConfigError(
                    f"weights must be numeric and rectangular: {exc}"
                ) from exc
            if arr.ndim not in (1, 2):
                raise MlsynthConfigError(
                    f"weights must be a length-N vector or an N x T matrix; "
                    f"got {arr.ndim} dimensions."
                )
            if not np.all(np.isfinite(arr)):
                raise MlsynthConfigError("weights must all be finite.")
        return self
