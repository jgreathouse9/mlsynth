"""Configuration for the multi-cell GeoLift analysis (MULTICELLGEOLIFT)."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ....config_models import BaseMAREXConfig
from ....exceptions import MlsynthConfigError


class MultiCellGeoLiftConfig(BaseMAREXConfig):
    """Configuration for the multi-cell GeoLift analysis."""

    cell_column_name: str = Field(
        ..., description="Unit-level column naming each geo's treatment cell "
        "('A', 'B', ...); blank/NaN or `control_label` marks a control geo."
    )
    post_col: str = Field(
        ..., description="0/1 column marking the (shared) post-treatment window."
    )
    control_label: Optional[str] = Field(
        default=None,
        description="Explicit value denoting a control geo (in addition to "
        "blank/NaN), e.g. 'control' or '0'.",
    )

    # --- estimation / inference (mirrors GEOLIFT) ---
    how: str = Field(default="mean", description="Treated aggregation per cell: 'sum' or 'mean'.")
    augment: Optional[str] = Field(default="ridge", description="'ridge' (ASCM) or None (simplex).")
    fixed_effects: bool = Field(default=True, description="Unit fixed effects (GeoLift default).")
    alpha: float = Field(default=0.1, description="Significance level / CI level.")
    cpic: Optional[float] = Field(default=None, description="Cost per incremental conversion (per-cell cost).")
    ns: int = Field(default=1000, description="Conformal permutation count (iid).")
    conformal_type: str = Field(default="iid", description="'iid' or 'block'.")
    seed: int = Field(default=0, description="RNG seed for conformal permutations.")
    display_graphs: bool = Field(default=True, description="Plot the per-cell effects during fit.")

    @model_validator(mode="after")
    def _check(self):
        if self.how not in ("sum", "mean"):
            raise MlsynthConfigError(f"how must be 'sum' or 'mean'; got {self.how!r}.")
        if self.augment not in ("ridge", None):
            raise MlsynthConfigError(f"augment must be 'ridge' or None; got {self.augment!r}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(f"alpha must be in (0, 1); got {self.alpha}.")
        if self.conformal_type not in ("iid", "block"):
            raise MlsynthConfigError(
                f"conformal_type must be 'iid' or 'block'; got {self.conformal_type!r}."
            )
        if self.cpic is not None and self.cpic < 0:
            raise MlsynthConfigError(f"cpic must be >= 0; got {self.cpic}.")
        return self
