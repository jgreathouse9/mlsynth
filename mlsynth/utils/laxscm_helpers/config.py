"""Configuration for the RESCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class RESCMConfig(BaseEstimatorConfig):
    """Configuration for the Relaxed/Balanced SCM (RESCM) estimator.

    Pick one or more named corner-case estimators of the RESCM convex program
    via ``methods`` (e.g. ``["SC", "LINF", "RELAX_L2"]``); the first listed
    drives the convenience aliases on the returned ``RESCMResults``. Valid
    names and aliases come from the registry in
    :mod:`mlsynth.utils.laxscm_helpers.specs` (``METHOD_SPECS``).
    """

    methods: List[str] = Field(
        default_factory=lambda: ["SC", "LINF", "RELAX_L2"],
        description=(
            "Named RESCM estimators to fit (e.g. 'SC', 'LASSO', 'L2', 'ENET', "
            "'LINF', 'L1LINF', 'RELAX_L2', 'RELAX_ENTROPY', 'RELAX_EL'; aliases "
            "allowed). The first listed drives the result aliases."
        ),
    )
    tau: Optional[Union[float, Literal["heuristic"]]] = Field(
        default=None,
        description=(
            "Relaxation parameter for the SCM-relaxation methods. ``None`` "
            "selects it by time-series cross-validation (slow). ``\"heuristic\"`` "
            "skips CV and uses the Bickel-Ritov-Tsybakov universal penalty "
            "``sd(y) * sqrt(2 T0 log 2J)`` (with 2x/4x feasibility fallbacks); "
            "much faster, at the cost of a defensible-but-not-optimal tau."
        ),
    )
    n_splits: Optional[int] = Field(
        default=None, ge=2,
        description="Number of CV folds for tau selection (relaxation methods).",
    )
    n_taus: Optional[int] = Field(
        default=None, ge=1,
        description="Grid size for the cross-validated tau search.",
    )
    solver: Any = Field(
        default="CLARABEL",
        description="CVXPY solver name (e.g. 'CLARABEL', 'ECOS', 'OSQP', 'SCS').",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Significance level for confidence intervals and ATE inference.",
    )
    standardize: bool = Field(
        default=True,
        description=(
            "Per-donor scale standardization for the relaxation branch "
            "(``RELAX_*``). The relaxed-balance L-infinity FOC is scale-"
            "sensitive: ``True`` (default) standardizes donor columns (the "
            "paper's Appendix-B recommendation, preferable for heterogeneous "
            "scales); ``False`` solves on the raw series, matching the authors' "
            "``scmrelax`` reference implementation."
        ),
    )
    @model_validator(mode="after")
    def _validate_methods(self):
        from mlsynth.utils.laxscm_helpers.specs import normalize_method

        if not self.methods:
            raise MlsynthConfigError("`methods` must list at least one RESCM estimator.")
        normalized = []
        for m in self.methods:
            try:
                normalized.append(normalize_method(m))
            except ValueError as e:
                raise MlsynthConfigError(str(e)) from e
        object.__setattr__(self, "methods", normalized)
        return self

    class Config:
        extra = "forbid"  # Unknown fields will raise a validation error
