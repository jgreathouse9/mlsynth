"""Configuration for the SRC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SRCConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Regressing Control estimator (Zhu 2023).

    SRC matches each donor to the treated unit by a univariate regression, then
    synthesises the matched controls with box-``[0, 1]`` weights chosen by a
    Mallows/Cp unbiased-risk criterion. The combined donor coefficients may be
    negative (controlled extrapolation), and the Cp penalty -- not a predictor
    (``V``) search -- identifies the weights, so the default estimator is
    deterministic.

    The covariate + predictor-weight (``V``) variant of the paper's Basque
    application (Algorithm 3 / Table 5) is available as an explicit, seeded
    opt-in (``covariates=[...]`` with ``v_search="de"``); see ``v_search``.

    References
    ----------
    Zhu, Rong J. B. (2023). *Synthetic Regressing Control.*
    arXiv:2306.02584.
    """

    ridge: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Tikhonov stabiliser added to the Cp Gram so the box QP is strictly "
            "convex (the reference's 0.001*I). Not part of the paper's criterion; "
            "keep small. Larger values shrink the synthesis weights toward zero."
        ),
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional predictor columns (Algorithm 3). Each covariate is "
            "aggregated over its window (``covariate_windows``, else the whole "
            "pre-period), standardised to the outcome rows' scale, and added as "
            "one extra matching row."
        ),
    )
    covariate_windows: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-covariate inclusive ``(start, end)`` aggregation window of time "
            "labels, e.g. {'invest': (1964, 1969)}. Covariates not listed are "
            "averaged over the full pre-treatment period. Mirrors Abadie's "
            "predictor specification. Requires ``covariates``."
        ),
    )
    fit_window: Optional[Tuple[Any, Any]] = Field(
        default=None,
        description=(
            "Inclusive ``(start, end)`` window of pre-treatment periods whose "
            "outcomes form the outcome matching rows (Abadie's "
            "``time.optimize.ssr``). Default None uses every pre-treatment "
            "period. Set it to reproduce a paper's fit window."
        ),
    )
    screen: Literal["none", "sirs"] = Field(
        default="none",
        description=(
            "Donor screening before the fit (Algorithm 2). 'none' (default) "
            "keeps every donor. 'sirs' ranks donors by the SIRS marginal utility "
            "(Zhu et al. 2011) on the pre-period outcomes and keeps the top "
            "``min(floor(T0/log(T0/2)), T0-1)`` (override with ``n_screen``). "
            "Use it when donors are not few relative to pre-periods "
            "(``J >= 4*T0/5``): screening restores a well-posed, non-degenerate "
            "Cp fit. It does not recover the paper's exact Basque weight cells "
            "(those are on the non-identified V manifold)."
        ),
    )
    n_screen: Optional[int] = Field(
        default=None, ge=1,
        description=(
            "Override the number of donors kept by ``screen='sirs'``. None "
            "(default) uses the paper's ``min(floor(T0/log(T0/2)), T0-1)``."
        ),
    )
    v_search: Literal["none", "de"] = Field(
        default="none",
        description=(
            "Predictor-weight (V) optimisation for the covariate variant. "
            "'none' (default) holds V = I -- the deterministic, Cp-identified "
            "estimator. 'de' runs a seeded global differential-evolution search "
            "over V (the paper's Algorithm 3 / Table 5 spec). WARNING: the V "
            "optimum is not identified -- a manifold of V fits the pre-period "
            "equally well but disagrees out of sample -- so the resulting weights "
            "are seed-dependent and must not be read as identified quantities. "
            "Requires ``covariates``."
        ),
    )
    v_seed: int = Field(
        default=0,
        description="Seed for the ``v_search='de'`` search (makes a call reproducible).",
    )
    v_maxiter: int = Field(
        default=60, ge=1,
        description="Max generations for the ``v_search='de'`` differential evolution.",
    )
    v_popsize: int = Field(
        default=12, ge=2,
        description="Population-size multiplier for the ``v_search='de'`` search.",
    )

    @model_validator(mode="after")
    def _check(self) -> "SRCConfig":
        if self.covariates is not None and len(self.covariates) == 0:
            raise MlsynthConfigError(
                "SRC 'covariates' must be a non-empty list or None."
            )
        if self.covariate_windows and not self.covariates:
            raise MlsynthConfigError(
                "SRC 'covariate_windows' requires 'covariates'."
            )
        if self.v_search == "de" and not self.covariates:
            raise MlsynthConfigError(
                "SRC v_search='de' optimises predictor weights and therefore "
                "requires 'covariates'."
            )
        return self
