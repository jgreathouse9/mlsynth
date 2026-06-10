"""Configuration for the LEXSCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field
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
