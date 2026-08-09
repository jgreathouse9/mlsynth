"""Configuration for the GEOX geo-experiment design.

Co-located with the helper package (mirrors MAREX / lexscm). Inherits the
experimental-design base :class:`BaseMAREXConfig` (df / outcome / unitid / time
plus panel validation) and adds the market-selection knobs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import Field, model_validator

from ...config_models import BaseMAREXConfig
from ...exceptions import MlsynthConfigError


class GEOXConfig(BaseMAREXConfig):
    """Configuration for the GEOX market-selection design."""

    # --- candidate test region ---
    treatment_size: Union[int, List[int]] = Field(
        ...,
        description="Number of markets to treat. A list scans several region "
        "sizes in one run (e.g. [2, 3, 4]) and ranks them against each other, "
        "as GeoLift's N = c(2, 3, 4) does.",
    )
    to_be_treated: Optional[List] = Field(
        default=None, description="Markets forced into every candidate region."
    )
    not_to_be_treated: Optional[List] = Field(
        default=None,
        description="Markets barred from any candidate; they remain donors.",
    )
    # --- design constraints ---
    # Each rule reduces to where the search may look. Cluster / adjacency build a
    # conflict graph, which acts twice: no candidate may contain two conflicting
    # markets, and a treated market's conflict-neighbours are dropped from its
    # own donor pool. Strata and the size band filter the treated side only.
    cluster_col: Optional[str] = Field(
        default=None,
        description="Per-unit column naming each market's cluster (DMA, state). "
        "Markets sharing a cluster conflict: no candidate region may hold two "
        "of them, and each one's cluster-mates leave its donor pool.",
    )
    adjacency: Optional[Any] = Field(
        default=None,
        description="Square DataFrame of pairwise spillover strengths, indexed "
        "and columned by market label. Entries above spillover_threshold mark "
        "a conflict. Combines with cluster_col by logical OR.",
    )
    spillover_threshold: float = Field(
        default=0.0,
        description="Off-diagonal adjacency entries strictly above this mark a "
        "conflicting pair.",
    )
    stratum_col: Optional[str] = Field(
        default=None,
        description="Per-unit column naming each market's stratum, for coverage "
        "quotas. Needs min_per_stratum and/or max_per_stratum to bind.",
    )
    min_per_stratum: Optional[int] = Field(
        default=None,
        description="Treat at least this many markets in every stratum that "
        "holds an eligible market.",
    )
    max_per_stratum: Optional[int] = Field(
        default=None,
        description="Treat at most this many markets in any one stratum.",
    )
    size_col: Optional[str] = Field(
        default=None,
        description="Per-unit column carrying each market's size, for the "
        "treated-unit size band. Needs min_size and/or max_size to bind.",
    )
    min_size: Optional[float] = Field(
        default=None,
        description="Treated-market size floor (inclusive) -- a power or "
        "operational minimum. Smaller markets stay available as donors.",
    )
    max_size: Optional[float] = Field(
        default=None,
        description="Treated-market size ceiling (inclusive). A market far "
        "larger than the donors cannot sit inside their convex hull, so the "
        "ceiling encodes synthesizability. Larger markets stay donors.",
    )

    run_stochastic: bool = Field(
        default=False,
        description="Sample each candidate's neighbours from adjacent "
        "correlation tiers instead of taking the top ones deterministically.",
    )
    stochastic_mode: str = Field(
        default="global",
        description="'global' draws one tier pattern for every anchor; "
        "'per_anchor' draws a fresh one per anchor.",
    )
    how: str = Field(
        default="mean",
        description="Treated aggregation. The SDID fit always runs on the "
        "per-market mean, which keeps the target at donor scale; this sets the "
        "scale the realized report is written in. 'sum' gives the summed "
        "incremental across the treated markets (GeoLift's reporting "
        "convention), 'mean' the per-market effect. Cost is computed from the "
        "summed incremental either way.",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Column flagging post-treatment periods. Absent, the whole "
        "panel is treated as pre-period history to simulate over.",
    )

    # --- scoring engine ---
    engine: str = Field(
        default="sdid",
        description="Estimator that scores each candidate. 'sdid' weights "
        "donors and pre-periods together; 'augsynth' is the ridge-augmented "
        "synthetic control GeoLift uses. The rest of the design -- nomination, "
        "backtests, power, MDE, ranking, constraints -- is the same whichever "
        "engine runs.",
    )
    inference: Optional[str] = Field(
        default=None,
        description="Null the detection test is taken against. 'placebo' "
        "reassigns donors as pretend-treated and needs only a donor-built "
        "counterfactual, so it suits either engine. 'conformal' permutes "
        "pre-period residuals and is available on 'augsynth' alone, since "
        "SDID's time weights exist to say pre-periods are not exchangeable. "
        "Absent, each engine takes its own default: placebo for sdid, "
        "conformal for augsynth, which is GeoLift's choice.",
    )
    ns: int = Field(
        default=1000,
        description="Conformal permutations behind each p-value (augsynth "
        "only). The cost sits inside the effect sweep, since a conformal test "
        "moves with the injected effect and cannot be hoisted out of it.",
    )
    conformal_type: str = Field(
        default="iid",
        description="Conformal permutation scheme (augsynth only): 'iid' or "
        "'block'.",
    )
    fixed_effects: Optional[bool] = Field(
        default=None,
        description="Unit fixed effects (augsynth only). Demeans every unit by "
        "its own pre-period mean so the fit matches shapes and the donor pool "
        "cannot absorb a treated-unit level shift. Defaults to True on the "
        "augsynth path, which is GeoLift's default.",
    )
    augment: Optional[str] = Field(
        default="ridge",
        description="Augmentation layer (augsynth only): 'ridge' for Augmented "
        "SCM, None for plain simplex SCM with an intercept.",
    )

    # --- simulation grid ---
    durations: List[int] = Field(
        ..., description="Treatment durations (periods) to scan."
    )
    effect_sizes: List[float] = Field(
        ..., description="Effect sizes to inject, as proportional lifts."
    )
    n_backtests: int = Field(
        default=5,
        description="backtests per (candidate, "
        "duration).",
    )

    # --- inference and decision rule ---
    alpha: float = Field(
        default=0.1, description="Significance level for detection."
    )
    power_threshold: float = Field(
        default=0.8, description="Power an effect must exceed to count as detected."
    )
    n_validation_backtests: int = Field(
        default=8,
        description="Extra backtests, deeper in history, used only to "
        "re-score the winning region after it has been chosen. The reported "
        "MDE is the smallest in a field of candidates, so it is optimistic for "
        "the same reason the best of many noisy estimates is: the region most "
        "likely to be picked is the one that got lucky. These backtests take "
        "no part in the selection, so the MDE they give for the winner is not "
        "selected on and is the number to plan against. Set to 0 to skip, at "
        "the cost of having only the optimistic figure.",
    )
    n_draws: int = Field(
        default=200,
        description="Placebo draws behind each standard error (Arkhangelsky "
        "et al. Algorithm 4).",
    )

    # --- budgeting ---
    cpic: Optional[float] = Field(
        default=None,
        description="Cost per incremental conversion. When set, the required "
        "investment = cpic x effect_size x summed-treated-volume is reported.",
    )
    budget: Optional[float] = Field(
        default=None,
        description="Spend cap. When set (with cpic), candidates whose "
        "detectable investment exceeds it are dropped from the design.",
    )

    # --- execution ---
    seed: int = Field(
        default=0,
        description="RNG seed for candidate sampling and placebo draws.",
    )
    n_jobs: int = Field(
        default=1,
        description="Worker processes over candidates. Each candidate is pure "
        "at the fixed seed, so any worker count returns the same shortlist.",
    )

    @property
    def treatment_sizes(self) -> List[int]:
        """The requested region sizes, sorted and de-duplicated.

        A scalar ``treatment_size`` reads as a one-element scan, so the rest of
        the pipeline never has to branch on which form was given.
        """
        raw = (self.treatment_size if isinstance(self.treatment_size, list)
               else [self.treatment_size])
        return sorted({int(n) for n in raw})

    @model_validator(mode="after")
    def _validate_grid(self) -> "GEOXConfig":
        sizes = (self.treatment_size if isinstance(self.treatment_size, list)
                 else [self.treatment_size])
        if not sizes:
            raise MlsynthConfigError("treatment_size must not be empty.")
        if any(int(n) < 1 for n in sizes):
            raise MlsynthConfigError(
                f"every treatment_size must be >= 1; got {self.treatment_size}.")
        forced = len(self.to_be_treated or ())
        too_small = [int(n) for n in sizes if int(n) < forced]
        if too_small:
            raise MlsynthConfigError(
                f"to_be_treated names {forced} market(s), which does not fit a "
                f"region of size {too_small}. Drop those sizes or shorten "
                "to_be_treated.")
        if not self.durations or any(d < 1 for d in self.durations):
            raise MlsynthConfigError(
                "durations must be a non-empty list of positive integers.")
        if not self.effect_sizes:
            raise MlsynthConfigError("effect_sizes must be non-empty.")
        if self.n_backtests < 1:
            raise MlsynthConfigError(
                f"n_backtests must be >= 1; got {self.n_backtests}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(
                f"alpha must be in (0, 1); got {self.alpha}.")
        if not 0.0 < self.power_threshold < 1.0:
            raise MlsynthConfigError(
                f"power_threshold must be in (0, 1); got {self.power_threshold}.")
        if self.n_validation_backtests < 0:
            raise MlsynthConfigError(
                "n_validation_backtests must be >= 0; got "
                f"{self.n_validation_backtests}.")
        if self.n_draws < 3:
            raise MlsynthConfigError(
                "n_draws must be at least 3 for a placebo standard error; got "
                f"{self.n_draws}.")
        if self.stochastic_mode not in ("global", "per_anchor"):
            raise MlsynthConfigError(
                "stochastic_mode must be 'global' or 'per_anchor'; got "
                f"{self.stochastic_mode!r}.")
        from .engines import ENGINE_NAMES
        if self.engine not in ENGINE_NAMES:
            raise MlsynthConfigError(
                f"unknown engine {self.engine!r}; available engines are "
                f"{sorted(ENGINE_NAMES)}.")
        if self.inference is not None and self.inference not in (
                "placebo", "conformal"):
            raise MlsynthConfigError(
                f"inference must be 'placebo' or 'conformal'; got "
                f"{self.inference!r}.")
        if self.engine == "sdid" and self.inference == "conformal":
            raise MlsynthConfigError(
                "the sdid engine cannot use conformal inference: the "
                "permutation argument needs the pre-period residuals to be "
                "exchangeable across the matching window, and SDID's time "
                "weights exist precisely to say they are not. Use "
                "engine='augsynth' for conformal, or inference='placebo'.")
        if self.engine != "augsynth":
            for name in ("fixed_effects",):
                if getattr(self, name) is not None:
                    raise MlsynthConfigError(
                        f"{name} is an augsynth-only setting; got engine="
                        f"{self.engine!r}. Drop it or set engine='augsynth'.")
        # Resolve the engine's default null once, so the rest of the pipeline
        # never has to ask which engine it is running.
        if self.inference is None:
            self.inference = ("conformal" if self.engine == "augsynth"
                              else "placebo")
        if self.engine == "augsynth" and self.fixed_effects is None:
            self.fixed_effects = True     # augsynth's fixedeff, GeoLift's default
        if self.conformal_type not in ("iid", "block"):
            raise MlsynthConfigError(
                f"conformal_type must be 'iid' or 'block'; got "
                f"{self.conformal_type!r}.")
        if self.ns < 2:
            raise MlsynthConfigError(f"ns must be >= 2; got {self.ns}.")
        if self.how not in ("sum", "mean"):
            raise MlsynthConfigError(
                f"how must be 'sum' or 'mean'; got {self.how!r}.")
        if self.spillover_threshold < 0.0:
            raise MlsynthConfigError(
                "spillover_threshold must be >= 0; got "
                f"{self.spillover_threshold}.")
        size_bounds = self.min_size is not None or self.max_size is not None
        if size_bounds and self.size_col is None:
            raise MlsynthConfigError(
                "min_size/max_size need size_col: without a size column there "
                "is no market size to band.")
        if (self.min_size is not None and self.max_size is not None
                and self.min_size > self.max_size):
            raise MlsynthConfigError(
                f"min_size ({self.min_size}) exceeds max_size ({self.max_size}); "
                "the size band is empty.")
        quota = self.min_per_stratum is not None or self.max_per_stratum is not None
        if quota and self.stratum_col is None:
            raise MlsynthConfigError(
                "min_per_stratum/max_per_stratum need stratum_col: without a "
                "stratum column there is nothing to count coverage over.")
        for name, value in (("min_per_stratum", self.min_per_stratum),
                            ("max_per_stratum", self.max_per_stratum)):
            if value is not None and value < 0:
                raise MlsynthConfigError(f"{name} must be >= 0; got {value}.")
        if (self.min_per_stratum is not None and self.max_per_stratum is not None
                and self.min_per_stratum > self.max_per_stratum):
            raise MlsynthConfigError(
                f"min_per_stratum ({self.min_per_stratum}) exceeds "
                f"max_per_stratum ({self.max_per_stratum}); no count satisfies both.")
        if self.budget is not None and self.cpic is None:
            raise MlsynthConfigError(
                "budget needs cpic: without a cost per incremental conversion "
                "there is no investment to compare the budget against.")
        return self
