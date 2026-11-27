"""Configuration for the PANGEO estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class PANGEOConfig(BaseModel):
    """Configuration for the PANGEO experimental-design estimator.

    Parallel-trends supergeo design (in the Supergeo / Chen et al. 2023
    lineage): partitions each treatment arm's geos into supergeo pairs
    whose treatment/control halves are maximally parallel over the
    pre-period, via a cvxpy/HiGHS set-partitioning MIP. A prospective
    *design* method -- it returns supergeo pairs + a treatment/control
    assignment, not a treatment effect -- so it takes a single categorical
    ``arm`` column rather than a binary ``treat`` indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Historical (pre-treatment) balanced long panel.
    outcome : str
        Historical outcome column (e.g. sales).
    arm : str
        Single categorical column naming each geo's eligible treatment arm
        (e.g. values ``A``/``B``/``C``). Arms occupy non-overlapping geos;
        the design runs independently within each arm.
    unitid : str
        Unit (geo) identifier column.
    time : str
        Time-period column.
    post_col : str, optional
        0/1 indicator column marking post-treatment periods (0 = pre). When
        given, the design is built on the pre rows alone -- identical to the
        design-only result -- and the realized difference-in-differences ATT
        is additionally computed on the post rows (``results.effects``).
    weight_col : str, optional
        Per-unit aggregation weight (e.g. population), constant within a
        unit. Makes both the supergeo design and the ATT population-weighted.
    max_supergeo_size : int
        Q -- the maximum size of either supergeo within a pair. Set ``1``
        to recover classic matched pairs.
    min_pairs : int
        Minimum number of supergeo pairs per arm.
    """

    class Config:
        arbitrary_types_allowed = True

    df: pd.DataFrame = Field(..., description="Historical pre-treatment panel.")
    outcome: str = Field(..., description="Historical outcome column (e.g. sales).")
    arm: str = Field(..., description="Categorical column of each geo's arm.")
    unitid: str = Field(..., description="Unit (geo) identifier column.")
    time: str = Field(..., description="Time-period column.")
    post_col: Optional[str] = Field(
        default=None,
        description="0/1 indicator of post-treatment periods. If given, the "
                    "design uses pre rows only and a realized DiD ATT is "
                    "computed on the post rows (results.effects).")
    weight_col: Optional[str] = Field(
        default=None,
        description="Per-unit aggregation weight column (e.g. population); "
                    "makes the design and ATT population-weighted.")
    max_supergeo_size: Optional[int] = Field(
        default=None, ge=1,
        description="Q: max size of either supergeo within a pair. If None "
                    "(default), Q is selected automatically by minimising the "
                    "program-level MDE over feasible Q (see metadata "
                    "'q_sweep').")
    min_pairs: int = Field(
        default=1, ge=1,
        description="Minimum number of supergeo pairs per arm.")
    q_selection: Literal["mde_min", "pareto_1se"] = Field(
        default="mde_min",
        description="Rule for automatic Q (only when max_supergeo_size is "
                    "None). 'mde_min' picks the feasible Q of smallest program "
                    "MDE. 'pareto_1se' treats Q as a two-objective choice "
                    "(minimise MDE, maximise pair count K): keep the "
                    "Pareto-efficient Q on (MDE down, K up), then a 1-SE "
                    "tie-break takes the largest K whose MDE is within one "
                    "standard error of the frontier's best -- spending pairs on "
                    "power only when the gain beats its own sampling noise.")
    q_min_pairs: int = Field(
        default=1, ge=1,
        description="Inference floor for q_selection='pareto_1se': prefer a Q "
                    "with at least this many supergeo pairs when the 1-SE band "
                    "admits one (more pairs => more credible permutation / "
                    "t-based inference). Ignored for 'mde_min'.")
    compute_q_sweep: bool = Field(
        default=False,
        description="When max_supergeo_size is fixed, also run the full Q sweep "
                    "and attach it to metadata['q_sweep'] (feasibility, pair "
                    "count, mean program MDE + its SE, randomization p-value "
                    "floor per Q) so the chosen Q can be audited against the "
                    "Pareto/1-SE alternatives. No effect in auto-Q mode, where "
                    "the sweep is always produced.")
    assignment_seed: int = Field(
        default=0,
        description="Seed of the fair coin drawn in each supergeo pair to "
                    "decide which half is treated (Chen et al. 2023, "
                    "arXiv:2301.12044, section 3): an integer from 0 to "
                    "2^63 - 1. Fixed by default, so an analysis run with "
                    "post_col labels the pairs as the design run did; set a "
                    "different seed for a different draw.")
    fast: bool = Field(
        default=True,
        description="Partition solver. Default (True) uses the OSD-style fast "
                    "partition (Shaw 2025 analog): each arm's units are "
                    "level-removed, PCA-embedded, hierarchically clustered into "
                    "size-bounded groups, and split per group -- replacing the "
                    "O(n^{2Q}) candidate enumeration and NP-hard exact-cover "
                    "MIP with an O(n log n) heuristic. Under cluster structure "
                    "it matches the exact optimum (validated against a "
                    "structure-aware oracle) in sub-second time, and is the "
                    "only tractable path at large Q. Set False to use the exact "
                    "enumerate+MIP (the optimum; only feasible for small arms / "
                    "small Q). Both minimise the same de-meaned (DiD) "
                    "pre-period gap-variance objective.")
    fast_candidates: int = Field(
        default=5, ge=1,
        description="Number of candidate groupings tried by the fast partition "
                    "(varying linkage / a small embedding perturbation, as in "
                    "OSD); the lowest-total-cost grouping is kept. Only used "
                    "when fast=True.")
    objective: Literal["ss_res", "r2", "weighted", "supergeo_total"] = Field(
        default="ss_res",
        description="Per-pair parallelism cost: 'ss_res' (absolute DiD "
                    "residual SS; scale-dependent), 'r2' (1-R^2; scale-free), "
                    "'weighted' (recency-weighted residual SS), or "
                    "'supergeo_total', the objective of Chen et al. (2023, "
                    "arXiv:2301.12044): the squared difference of the halves' "
                    "totals over the estimation window, with no level shift "
                    "and no trajectory. 'supergeo_total' is available for "
                    "pairs (max_supergeo_size=1) with fast=False, solved by "
                    "the exact-cover MIP; reach for it to compare designs on "
                    "the paper's objective.")
    mip_backend: Literal["auto", "highspy", "cvxpy"] = Field(
        default="auto",
        description="Backend for the exact-cover MIP (fast=False): 'auto' "
                    "uses highspy when it is installed and cvxpy otherwise; "
                    "'highspy' requires it; 'cvxpy' solves through cvxpy with "
                    "its first installed MILP solver (HiGHS by preference). "
                    "Both state the same program.")
    mip_rel_gap: float = Field(
        default=0.0, ge=0.0, lt=1.0, allow_inf_nan=False,
        description="Relative gap the exact-cover MIP must close (fast=False, "
                    "groups larger than pairs). 0, the default, solves to the "
                    "optimum; HiGHS's own default of 1e-4 returned a design "
                    "0.002% dearer than the optimum on a 20-market panel with "
                    "one outlying market and reported it optimal. The gap "
                    "asked for and the gap proven are on the result "
                    "(mip_reports).")
    mip_abs_gap: float = Field(
        default=0.0, ge=0.0, allow_inf_nan=False,
        description="Absolute gap the exact-cover MIP must close, in the "
                    "objective's units; 0 by default.")
    certify_node_limit: int = Field(
        default=12_000_000, ge=0, strict=True,
        description="Nodes the exact path (fast=False, groups larger than "
                    "pairs) may place after its proof stops at its cap, "
                    "raising a proved lower bound by the iterative cutoff "
                    "(rounds below the root bound times 1.01^i). If the "
                    "rounds prove the design, it is optimal; if the budget "
                    "runs out first, the best design is returned with the gap "
                    "it certified (mip_reports: search 'certified', status "
                    "'certified_gap', dual_bound and proven_gap) and a "
                    "PangeoCertifiedGapWarning. Deterministic: the same panel "
                    "gives the same design and bound on any machine. At 120 "
                    "markets and Q = 2 a node costs about 160 microseconds, so "
                    "the default is about 30 minutes; there it certified a 4.6% "
                    "gap. 0 leaves a capped proof to the MIP loop.")
    popmusic_node_limit: int = Field(
        default=12_000_000, ge=0, strict=True,
        description="Nodes the exact path (fast=False, groups larger than "
                    "pairs) may place improving its design by POPMUSIC after "
                    "its proof stops at its cap and before the iterative "
                    "cutoff: clusters of the design's nearest groups are "
                    "re-covered exactly, a cover kept only when its upper "
                    "bound is lower, so the design never gets worse, and the "
                    "certified gap is then measured from the better design. "
                    "Deterministic. At 120 markets and Q = 2 it lowered the "
                    "design 2.5% with the default budget, and the certified "
                    "gap from 8.4% to 6.1%. Arms whose "
                    "proof finishes never run it. 0 skips it.")
    n_jobs: Optional[int] = Field(
        default=None, ge=1, strict=True,
        description="Threads the exact path's POPMUSIC runs its trees on. "
                    "None (default) uses every core; 1 runs them in this "
                    "thread. The design does not depend on it.")
    recency_decay: float = Field(
        default=0.97, gt=0.0, le=1.0,
        description="Geometric recency-weight decay for objective='weighted' "
                    "(period t weight = recency_decay**(T0-1-t)).")
    frac_E: float = Field(
        default=0.7, gt=0.0, lt=1.0,
        description="Fraction of the pre-period used as the estimation window "
                    "the split is optimised over; the remaining tail is held "
                    "out as a blank window whose residuals give honest, "
                    "out-of-sample variance for the MDE and conformal CIs "
                    "(mirrors LEXSCM / SPCD).")
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional baseline (time-invariant) covariate columns to "
                    "balance across each supergeo pair, in addition to "
                    "pre-period parallelism. Each unit's covariate value is "
                    "its mean over the panel; the per-pair cost gains a "
                    "standardized SMD^2 imbalance term "
                    "sum_m w_m ((cbar_A - cbar_B)/s_m)^2, keeping the outer "
                    "selection a linear MILP.")
    covariate_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-covariate weight on the standardized SMD^2 imbalance "
                    "penalty (default 1.0 each). Larger up-weights balancing "
                    "that covariate relative to raw parallelism.")
    standardize_covariates: bool = Field(
        default=True,
        description="Standardize each covariate by its cross-unit std before "
                    "the SMD^2 imbalance (puts covariates on a common scale).")
    compute_power: bool = Field(
        default=True,
        description="Attach a program- and arm-level MDE / power analysis to "
                    "the result (AR(1) serial-correlation-corrected DiD; the "
                    "program-level MDE is the headline metric).")
    power_target: float = Field(
        default=0.80, gt=0.0, lt=1.0,
        description="Target power the stored MDEs are computed at.")
    power_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the MDE.")
    power_post_periods: List[int] = Field(
        default_factory=lambda: list(range(2, 13)),
        description="Post-period horizons (X) to evaluate the MDE at "
                    "(default 2..12).")
    att_augment: bool = Field(
        default=True,
        description="Use the augmented DiD (free control-scale coefficient "
                    "delta_2, Li & Van den Bulte 2022) for the realized ATT; "
                    "False forces delta_2=1 (plain DiD).")
    att_trend: bool = Field(
        default=True,
        description="Include a linear time-trend regressor in the realized-"
                    "ATT counterfactual (stationarizes a drifting gap).")
    display_graphs: bool = Field(
        default=True,
        description="Plot treatment vs control aggregate trajectories per arm.")
    save: Union[bool, str] = Field(
        default=False, description="Save the plot (True or a filename base).")

    @field_validator("assignment_seed", mode="before")
    @classmethod
    def _seed_is_an_integer(cls, v):
        """The coins' seed: an integer (not a bool) from 0 to 2^63 - 1."""
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= int(v) < 2 ** 63:
            raise ValueError(f"assignment_seed must be an integer from 0 to 2^63 - 1, not {v!r}")
        return int(v)

    @model_validator(mode="after")
    def _supergeo_total_scope(self) -> "PANGEOConfig":
        """The paper's objective is wired for pairs on the exact path only:
        PANGEO's Q caps each half, while the paper caps the group at four
        with any split, and the fast path clusters similar trajectories,
        which this objective does not reward."""
        if self.objective == "supergeo_total":
            if self.fast:
                raise ValueError("objective='supergeo_total' needs fast=False "
                                 "(the exact-cover MIP).")
            if self.max_supergeo_size != 1:
                raise ValueError("objective='supergeo_total' supports pairs "
                                 "only: set max_supergeo_size=1.")
            if self.compute_q_sweep:
                raise ValueError("objective='supergeo_total' supports pairs "
                                 "only, so it has no Q sweep.")
        return self
