"""Configuration for the LEXSCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field, model_validator
from ...config_models import BaseMAREXConfig


class LEXSCMConfig(BaseMAREXConfig):
    """Configuration for LEXSCM - Fast Synthetic Experiment Design pipeline."""

    # =========================================================
    # IDENTIFICATION DESIGN
    # =========================================================

    candidate_col: str = Field(
        ...,
        description="REQUIRED: Column indicating units eligible for treatment selection "
                    "(boolean or 0/1, constant within unit)."
    )

    m: int = Field(
        ...,
        gt=0,
        description="REQUIRED: Number of units selected per treated tuple (m)."
    )

    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 indicator for post-treatment period."
    )
    
    weight_col: Optional[str] = Field(
        default=None,
        description="Weight column."
    )

    unit_cost_col: Optional[str] = Field(
        default=None,
        description="Column name containing the per-unit treatment cost. "
                    "Value must be constant within each unit."
    )

    budget: Optional[float] = Field(
        default=None,
        gt=0,
        description="Hard total budget for the sum of treatment costs of the "
                    "selected treated units."
    )

    # =========================================================
    # SPILLOVER / INTERFERENCE EXCLUSIONS (Vives-i-Bastida 2022)
    # =========================================================

    cluster_col: Optional[str] = Field(
        default=None,
        description="Optional column assigning each unit to a spillover cluster "
                    "(e.g. state/province; constant within unit). Enforces the "
                    "Vives-i-Bastida (2022) interference exclusions: treated units "
                    "may not share a cluster (Stage-1 'no interference'), and a "
                    "treated unit's same-cluster units are barred from its donor "
                    "pool (Stage-2 'exclusion restriction')."
    )

    adjacency: Optional[Any] = Field(
        default=None,
        description="Optional spillover/adjacency matrix: a (J, J) array in sorted "
                    "unit-id order, or a pandas DataFrame indexed and columned by "
                    "unit id (preferred -- order-independent). Two units conflict "
                    "when their entry exceeds `spillover_threshold`. Combined with "
                    "`cluster_col` via logical OR."
    )

    spillover_threshold: float = Field(
        default=0.0, ge=0.0,
        description="Adjacency entries strictly above this value count as a "
                    "spillover conflict (default 0.0: any positive entry)."
    )

    # =========================================================
    # COVERAGE / STRATIFICATION (treated-set quotas across strata)
    # =========================================================

    stratum_col: Optional[str] = Field(
        default=None,
        description="Optional column assigning each unit to a stratum (e.g. "
                    "region/tier/segment; constant within unit) for coverage "
                    "quotas on the treated set. Requires at least one of "
                    "`min_per_stratum` / `max_per_stratum`."
    )

    min_per_stratum: Optional[int] = Field(
        default=None, ge=1,
        description="Require at least this many treated units from EVERY stratum "
                    "that contains a candidate ('test in every region')."
    )

    max_per_stratum: Optional[int] = Field(
        default=None, ge=1,
        description="Allow at most this many treated units from any single "
                    "stratum (a quota)."
    )

    # =========================================================
    # TREATED-UNIT SIZE BANDS (eligibility by scale)
    # =========================================================

    size_col: Optional[str] = Field(
        default=None,
        description="Optional column giving each unit's size (population, revenue, "
                    "...; constant within unit). Units outside [`min_size`, "
                    "`max_size`] are excluded from TREATMENT (they may still serve "
                    "as donors). Required if `min_size`/`max_size` are set."
    )

    min_size: Optional[float] = Field(
        default=None,
        description="Treatment-eligibility floor: a power/operational minimum size."
    )

    max_size: Optional[float] = Field(
        default=None,
        description="Treatment-eligibility ceiling: units too large to be "
                    "reproduced by a convex combination of others (e.g. mega-markets)."
    )

    @model_validator(mode="after")
    def _validate_coverage_and_size(self) -> "LEXSCMConfig":
        if (self.min_per_stratum is not None or self.max_per_stratum is not None) \
                and self.stratum_col is None:
            raise ValueError(
                "min_per_stratum / max_per_stratum require `stratum_col`."
            )
        if (self.min_per_stratum is not None and self.max_per_stratum is not None
                and self.min_per_stratum > self.max_per_stratum):
            raise ValueError(
                f"min_per_stratum ({self.min_per_stratum}) cannot exceed "
                f"max_per_stratum ({self.max_per_stratum})."
            )
        if (self.min_size is not None or self.max_size is not None) \
                and self.size_col is None:
            raise ValueError("min_size / max_size require `size_col`.")
        if (self.min_size is not None and self.max_size is not None
                and self.min_size > self.max_size):
            raise ValueError(
                f"min_size ({self.min_size}) cannot exceed max_size ({self.max_size})."
            )
        return self

    seed: int = 42

    frac_E: float = Field(
        default=0.7,
        description="Fraction of pre-treatment period used for estimation window E."
    )

    # =========================================================
    # SYNTHETIC CONTROL SPECIFICATION
    # =========================================================

    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional covariates included in synthetic control."
    )

    lambda_penalty: float = Field(
        default=0.1,
        description="Penalty for control mismatch in QP."
    )

    # =========================================================
    # SEARCH / COMPUTATIONAL BUDGET
    # =========================================================

    top_K: int = Field(
        default=20,
        description="Number of top candidate treated tuples returned by the "
                    "Stage 1 search (exact enumeration or multi-start local search)."
    )

    top_P: int = Field(
        default=10,
        description="Deprecated and unused by the rebuilt Stage 1 search "
                    "(retained for backward-compatible configs). The multi-start "
                    "local search sets its own seed count internally."
    )

    # =========================================================
    # POWER / INFERENCE (MDE)
    # =========================================================

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    n_post_grid: List[int] = Field(
        default_factory=lambda: list(range(2, 9)),
        description="Post-treatment horizons used for MDE detectability curves."
    )

    n_sims: int = Field(
        default=1000,
        description="Monte Carlo simulations for null distribution in MDE."
    )

    post_imputation: Literal["mean", "max", "double_max"] = Field(default="mean")
    test_statistic: Literal["mean_abs", "mean", "rms"] = Field(default="mean_abs")
    mde_horizon: Literal[
        "early_mean",
        "early_min",
        "late"
    ] = Field(
        default="late",
        description=(
            "Defines how detectability (MDE) is aggregated across post-treatment horizons.\n\n"
            "- 'early_mean': average MDE across early windows (e.g., 2–4 weeks)\n"
            "- 'early_min': minimum MDE across early windows (most optimistic detectability)\n"
            "- 'late': uses longest horizon (e.g., 8-week MDE; recommended default)\n\n"
            "The 'late' option is recommended for operational experiments because it provides a "
            "conservative detectability bound under sustained treatment exposure."
        )
    )
    max_shortlist: int = Field(default=5, gt=0)

    power_target: float = Field(
        default=0.8, gt=0.0, lt=1.0,
        description="Target power for the minimum-detectable-effect search."
    )

    imbalance_tol: float = Field(
        default=0.25, ge=0.0,
        description="Relative slack above the best achievable pre-treatment "
                    "imbalance defining the validity gate for design selection."
    )

    # =========================================================
    # INTERNAL / SYSTEM
    # =========================================================

    display_graph: bool = Field(
        default=False,
        description="Display plot."
    )

    verbose: bool = Field(
        default=True,
        description="Print progress logs."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
