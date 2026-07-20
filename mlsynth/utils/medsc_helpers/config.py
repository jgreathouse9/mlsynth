"""Configuration for the MEDSC (Mediation Analysis SC) estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility. MEDSC implements the
Mellace & Pasquini (2022) causal-mediation synthetic control: it decomposes the
treatment effect on a panel outcome into a direct effect and an indirect effect
that runs through a single mediator series.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class MEDSCConfig(BaseEstimatorConfig):
    """Configuration for the MEDSC estimator (mediation-analysis synthetic control).

    MEDSC (Mellace & Pasquini 2022) splits the synthetic-control treatment
    effect into two channels:

    * the direct effect -- what the intervention would have done holding the
      mediator at its treated (post-intervention) path, estimated by a
      cross-world synthetic control that also matches the treated unit's
      post-treatment mediator; and
    * the indirect effect -- everything that flows through the mediator,
      recovered as total minus direct.

    The total effect is an ordinary synthetic control (the mediator does not
    enter). The direct effect is re-estimated separately for each post period,
    matching the mediator path up to that period (paper Section 3.2), on a donor
    pool that may differ from the total pool so the cross-world control can span
    the treated unit's post-treatment mediator values.

    Parameters
    ----------
    mediator : str
        Name of the mediator column in ``df`` (e.g. a price series). The direct
        effect matches the treated unit's post-treatment path of this variable.
    total_donors, direct_donors : list, optional
        Explicit donor pools for the total and direct fits. ``None`` (default)
        uses every non-treated unit. The two pools may differ: the direct pool
        typically adds units the total pool excludes, so the cross-world
        control can reach the treated unit's post-treatment mediator values
        (the paper adds the high-mediator states back for the direct effect).
        Every listed unit must be a donor (a non-treated unit) in ``df``.
    covariates : list of str, optional
        Predictor columns matched (alongside the outcome path) via the
        cross-validated bilevel predictor weights (mscmt backend). ``None``
        (default) -> outcome-path matching, which reproduces the paper's
        decomposition; covariates are an option, not the default.
    predictor_lags : list, optional
        Time labels at which the outcome (and pre-treatment mediator) enter as
        matching predictors. ``None`` (default) matches the full pre-treatment
        outcome path. Set this to reproduce a paper's exact special-predictor
        lag spec.
    pre_weight : float
        Weight the direct fit places on the pre-treatment constraints; the
        post-treatment mediator constraints share the remaining
        ``1 - pre_weight`` equally (paper's 3/4 -- 1/4 split; default 0.75).
    backend : {"auto", "outcome-only", "mscmt"}
        Predictor-weight backend. ``"auto"`` (default) is ``"outcome-only"``
        with no covariates and ``"mscmt"`` (cross-validated global V search)
        with covariates.
    inference : bool
        If True (default), run Abadie in-space placebo inference on the total
        effect (refit treating each donor as pseudo-treated; the p-value ranks
        the treated unit's post/pre RMSPE ratio). Placebo donors with
        pre-treatment RMSPE above ``placebo_cutoff`` times the treated unit's
        are dropped.
    placebo_cutoff : float
        Multiple of the treated pre-RMSPE above which a placebo unit is
        discarded (Abadie's ill-fitting-placebo screen; default 5.0).
    seed : int
        RNG seed for the mscmt differential-evolution search.
    mscmt_maxiter, mscmt_popsize : int
        Differential-evolution budget for the mscmt backend.
    """

    mediator: str = Field(
        ..., description="Mediator column name in df (the channel variable).")
    total_donors: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="Donor pool for the total effect; None -> all non-treated units.")
    direct_donors: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="Donor pool for the direct effect; None -> same as total_donors.")
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Predictor columns; None -> outcome-path matching (default).")
    predictor_lags: Optional[List[Any]] = Field(
        default=None,
        description="Time labels for the outcome/mediator lag predictors; None -> "
                    "full pre-treatment outcome path.")
    pre_weight: float = Field(
        default=0.75, gt=0.0, lt=1.0,
        description="Direct-fit weight on the pre-treatment constraints "
                    "(post-mediator constraints share 1 - pre_weight).")
    backend: Literal["auto", "outcome-only", "mscmt"] = Field(
        default="auto",
        description="Predictor-weight backend (see class docstring).")
    inference: bool = Field(
        default=True,
        description="Run in-space placebo inference on the total effect.")
    placebo_cutoff: float = Field(
        default=5.0, gt=0.0,
        description="Drop placebo units with pre-RMSPE above this multiple of "
                    "the treated unit's.")
    seed: int = Field(default=0, description="RNG seed for the mscmt DE search.")
    mscmt_maxiter: int = Field(
        default=300, ge=1, description="mscmt differential-evolution max iterations.")
    mscmt_popsize: int = Field(
        default=15, ge=1, description="mscmt differential-evolution population size.")

    @field_validator("mediator")
    @classmethod
    def _mediator_nonempty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("mediator must be a non-empty column name.")
        return v

    @model_validator(mode="after")
    def _check_columns_and_pools(self) -> "MEDSCConfig":
        cols = set(self.df.columns)
        if self.mediator not in cols:
            raise MlsynthConfigError(
                f"mediator column '{self.mediator}' is not in df.")
        if self.mediator == self.outcome:
            raise MlsynthConfigError(
                "mediator and outcome must be different columns.")
        if self.covariates:
            missing = [c for c in self.covariates if c not in cols]
            if missing:
                raise MlsynthConfigError(
                    f"covariate column(s) {missing} are not in df.")
        for name, pool in (("total_donors", self.total_donors),
                           ("direct_donors", self.direct_donors)):
            if pool is not None and len(pool) == 0:
                raise MlsynthConfigError(
                    f"{name} must be non-empty when provided (use None for all donors).")
        return self
