"""Configuration for the TWSF estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class TWSFConfig(BaseEstimatorConfig):
    """Configuration for the Two-Way Synthetic Forecasting (TWSF) estimator.

    Implements:

        Shen, D. (2026). "Causal Forecasting in Panel Data: A Two-Way
        Synthetic Forecasting Approach." arXiv:2606.18512.

    TWSF forecasts the treated potential outcome of a unit that has never been
    treated, at dates beyond the end of the panel. Unlike every other estimator
    in mlsynth it does not impute a missing cell inside the observed window, so
    the panel is read differently: ``treat`` flags the *donors*, who receive the
    intervention partway through, while ``target`` names the focal unit, which
    never does.

    Parameters
    ----------
    target : str
        The focal unit to forecast. Must appear in ``unitid`` and must never be
        flagged by ``treat``.
    L : int
        Lag length for the time-side Page matrix. The treated window must
        supply at least two blocks of length ``L + 1``.
    k_y : int
        Spectral rank for the unit-side design (the donors' pre-treatment
        outcomes).
    k_z : int
        Spectral rank for the time-side design (the stacked Page matrix).
    horizon : int
        Number of periods past the end of the panel to forecast.
    multistep : {"recursive", "direct"}
        How the horizon is covered. ``"recursive"`` (default) learns one
        one-step rule and iterates it, keeping the full temporal sample size;
        ``"direct"`` fits a separate rule per lead, which avoids recursive error
        propagation but needs Page blocks of length ``L + h`` and so is often
        infeasible at short treated windows. The two coincide at
        ``horizon = 1``.
    donors : list of str, optional
        Restrict the treated donor pool to these units. ``None`` (default) uses
        every unit that ``treat`` ever flags.
    alpha : float
        Two-sided significance level for the pointwise interval.
    interval : {"confidence", "prediction"}
        ``"confidence"`` (default) is the interval for the expected
        counterfactual trajectory. ``"prediction"`` adds the future innovation
        term for a realised trajectory, which is what a validation exercise
        against subsequently observed outcomes needs.
    """

    target: str = Field(
        ...,
        description="Focal unit to forecast; must never be flagged by `treat`.",
    )
    L: int = Field(
        ..., gt=0,
        description="Lag length for the time-side Page matrix.",
    )
    k_y: int = Field(
        ..., gt=0,
        description="Spectral rank for the unit-side design.",
    )
    k_z: int = Field(
        ..., gt=0,
        description="Spectral rank for the time-side (Page) design.",
    )
    horizon: int = Field(
        default=1, gt=0,
        description="Periods past the end of the panel to forecast.",
    )
    multistep: Literal["recursive", "direct"] = Field(
        default="recursive",
        description="Iterate one rule ('recursive') or fit one per lead "
                    "('direct'). Identical at horizon = 1.",
    )
    donors: Optional[List[str]] = Field(
        default=None,
        description="Restrict the treated donor pool; None uses every unit "
                    "that `treat` ever flags.",
    )
    alpha: float = Field(
        default=0.10, gt=0.0, lt=1.0,
        description="Two-sided significance level for the pointwise interval.",
    )
    interval: Literal["confidence", "prediction"] = Field(
        default="confidence",
        description="Interval for the expected trajectory ('confidence') or "
                    "for a realised one ('prediction', adds future noise).",
    )

    @model_validator(mode="after")
    def _check(self):
        if self.k_z > self.L:
            raise MlsynthConfigError(
                f"k_z ({self.k_z}) cannot exceed the lag length L ({self.L}): "
                "the time-side design has only L rows, so there are at most L "
                "singular directions to retain."
            )
        if self.donors is not None and not self.donors:
            raise MlsynthConfigError(
                "donors was given as an empty list; pass None to use every "
                "unit that `treat` flags, or name at least one donor."
            )
        if self.donors is not None and self.target in self.donors:
            raise MlsynthConfigError(
                f"target {self.target!r} also appears in donors. The focal "
                "unit is forecast, not used to forecast itself."
            )
        return self
