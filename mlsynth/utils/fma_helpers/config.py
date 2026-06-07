"""Configuration for the FMA estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class FMAConfig(BaseEstimatorConfig):
    """Configuration for the Factor Model Approach (FMA) estimator.

    Implements Li & Sonnier (2023), *"Statistical Inference for the
    Factor Model Approach to Estimate Causal Effects in
    Quasi-Experimental Settings"*, JMR 60(3):449-472. FMA estimates
    the ATT for a single treated unit by extracting principal-
    component factors from the control panel, projecting the treated
    unit's pre-period outcomes onto those factors, and using the
    fitted loading to predict the untreated potential outcome in the
    post-period.

    Parameters
    ----------
    stationarity : {"stationary", "nonstationary"}
        Selects the factor-selection criterion: ``"stationary"`` uses
        the paper's modified Bai-Ng (MBN) criterion (Web Appendix
        D.1); ``"nonstationary"`` uses Bai (2004) IPC1 with a
        log-log adjustment for non-stationary factors. Default:
        ``"nonstationary"`` (the paper's recommendation for general
        applied settings).
    preprocessing : {"demean", "standardize"}
        Preprocessing applied to the control panel before PCA.
    n_factors : int or None
        Override the data-driven factor count. ``None`` triggers the
        criterion in ``stationarity``.
    max_factors : int
        Upper bound passed to the factor-selection routine.
    alpha : float
        Two-sided significance level for CIs.
    inference_methods : list of {"asymptotic", "bootstrap", "placebo"}
        Inference procedures to run. Defaults to ``["asymptotic"]``,
        which gives the paper's Theorem 3.1 normal CI for the ATT.
        Add ``"bootstrap"`` to get per-period ATT_t CIs via the Web
        Appendix F residual bootstrap, and ``"placebo"`` to get the
        Web Appendix G control-as-pseudo-treated band.
    n_bootstrap : int
        Number of bootstrap replicates (Web Appendix F). Ignored when
        ``"bootstrap"`` is not in ``inference_methods``.
    bootstrap_seed : int
        Seed for the bootstrap RNG.
    """

    stationarity: Literal["stationary", "nonstationary"] = Field(
        default="nonstationary",
        description="Stationarity assumption for factor selection.",
    )
    preprocessing: Literal["demean", "standardize"] = Field(
        default="demean",
        description="Preprocessing applied to the control panel before PCA.",
    )
    n_factors: Optional[int] = Field(
        default=None, ge=1,
        description="Optional override of the data-driven factor count.",
    )
    max_factors: int = Field(
        default=10, ge=1,
        description="Upper bound on the factor-selection routine.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for CIs.",
    )
    inference_methods: List[str] = Field(
        default_factory=lambda: ["asymptotic"],
        description="Inference procedures: asymptotic / bootstrap / placebo.",
    )
    n_bootstrap: int = Field(
        default=1000, ge=100,
        description="Number of bootstrap replicates (Web Appendix F).",
    )
    bootstrap_seed: int = Field(
        default=0,
        description="Seed for the bootstrap RNG.",
    )

    @model_validator(mode="after")
    def check_fma_params(cls, values: Any) -> Any:
        allowed = {"asymptotic", "bootstrap", "placebo"}
        for m in values.inference_methods:
            if m not in allowed:
                raise MlsynthConfigError(
                    f"inference_methods entry {m!r} is not one of "
                    f"{sorted(allowed)}."
                )
        return values
