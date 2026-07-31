"""Configuration for the Two-Step Synthetic Control (TSSC) estimator.

Co-located with the TSSC helper package. The shared
:class:`~mlsynth.config_models.BaseEstimatorConfig` remains central; only the
per-estimator config lives here. Re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError

#: The SC-class variants of Li & Shankar (2023) Table 1, in the order TSSC
#: reports them. Spelled exactly as the paper does; see ``method`` below for why
#: the casing is not relaxed.
TSSC_METHODS = ("SC", "MSCa", "MSCb", "MSCc")


class TSSCConfig(BaseEstimatorConfig):
    """Configuration for the Two-Step Synthetic Control (TSSC) estimator.

    Implements:

        Li, K. T., & Shankar, V. (2023). "A Two-Step Synthetic Control
        Approach for Estimating Causal Effects of Marketing Events."
        Management Science. https://doi.org/10.1287/mnsc.2023.4878

    Parameters
    ----------
    alpha : float
        Two-sided significance level for the Step-1 restriction tests
        (the SC-pretrends test and the two single-restriction tests).
        Default 0.05.
    subsample_size : int or None
        Subsample size ``m`` for the Step-1 subsampling procedure. When
        ``None`` (default) it is set to ``T_1`` (the bootstrap special
        case the paper's simulations validate). For genuine subsampling,
        the paper's rule of thumb is ``m`` between ``T_1/2`` and ``T_1``
        for moderate ``T_1`` (and smaller for large ``T_1``).
    draws : int
        Number of subsampling replications ``B`` for the Step-1 tests and
        bootstrap replications for the per-variant ATT confidence
        intervals. Default 500.
    ci : float
        Confidence level for the per-variant ATT confidence interval.
        Default 0.95.
    seed : int or None
        Seed for the subsampling RNG (reproducibility). Default None.
    """

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0,
                         description="Significance level for Step-1 restriction tests.")
    subsample_size: Optional[int] = Field(default=None, ge=2,
                         description="Subsample size m; None uses T_1 (bootstrap).")
    draws: int = Field(default=500, ge=1, description="Subsampling/bootstrap replications.")
    method: Optional[Literal["SC", "MSCa", "MSCb", "MSCc"]] = Field(
        default=None,
        description=(
            "Fit one named SC-class variant instead of all four. This is an "
            "override, not a preference: Step 1 is skipped entirely rather than "
            "run and ignored, so no recommendation is produced and "
            "``results.selection`` is None. Reach for it when the variant is "
            "already decided -- 'MSCa' is the demeaned synthetic control of "
            "Ferman & Pinto (2021) -- or when fitting all four is wasteful, as "
            "in a simulation. None (default) runs the full two-step procedure. "
            "Names are case-sensitive and spelled as in Li & Shankar (2023) "
            "Table 1."
        ),
    )
    inference: bool = Field(
        default=True,
        description=(
            "Compute the per-variant subsampling confidence intervals. Set "
            "False to skip them when only the point estimate is wanted; at the "
            "default 500 draws the subsampling dominates the runtime, so a "
            "Monte Carlo that never reads an interval pays for it many times "
            "over. Requires ``method``, since Step 1 is itself a subsampling "
            "procedure and cannot recommend a variant without it. The point "
            "estimate is unaffected either way."
        ),
    )
    ci: float = Field(default=0.95, gt=0.0, lt=1.0, description="ATT confidence level.")
    seed: Optional[int] = Field(default=None, description="RNG seed for subsampling.")
    compute_scpi_pi: bool = Field(
        default=False,
        description=(
            "Also compute model-based prediction intervals through VanillaSC's "
            "generalized scpi engine (Cattaneo-Feng-Palomba-Titiunik 2025) for "
            "each SC-class variant, mapped to scpi's weight-constraint family by "
            "the variant's restrictions: SC -> simplex, MSCa -> simplex + "
            "constant, MSCb -> ols, MSCc -> ols + constant. The sum-constrained "
            "variants map exactly; the no-adding-up variants (MSCb / MSCc) carry "
            "bare non-negativity, which scpi does not have, so they use scpi's "
            "ols set (the band does not re-impose w >= 0, so it is slightly "
            "conservative). Each variant fit gains a ``scpi`` band and the "
            "recommended variant's band is surfaced on ``res.scpi``."
        ),
    )
    scpi_sims: int = Field(
        default=200, ge=10,
        description="Gaussian draws for the scpi in-sample QCQP simulation.",
    )
    scpi_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the scpi prediction intervals.",
    )
    scpi_e_method: Literal["gaussian", "ls", "empirical"] = Field(
        default="gaussian",
        description="Out-of-sample tabulation for the scpi prediction intervals.",
    )

    @model_validator(mode="after")
    def _check_variant_selection(self) -> "TSSCConfig":
        # Pydantic's Literal already rejects unknown values, but its message
        # does not read as an mlsynth error and buries the valid options in a
        # type dump. The estimator translates ValidationError to
        # MlsynthConfigError; this validator only handles the combination rule,
        # which pydantic cannot express.
        if not self.inference and self.method is None:
            raise MlsynthConfigError(
                "inference=False requires method to be set. Step 1 is itself a "
                "subsampling procedure, so with inference off there is no way "
                "to recommend a variant and no way to know which variant the "
                f"summary should describe. Pass one of {list(TSSC_METHODS)}, or "
                "leave inference=True to run the full two-step procedure."
            )
        return self
