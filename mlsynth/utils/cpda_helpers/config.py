"""Configuration for the CPDA estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class CPDAConfig(BaseEstimatorConfig):
    """Configuration for the covariate-adjusted panel data approach (CPDA)."""

    covariates: List[str] = Field(default_factory=list, description="Time-varying covariates x_it, the ones Equation 2 gives a common slope. CPDA is the covariate-adjusted panel data approach, so at least one is required: with none, Equation 12 leaves v_t = y_t and the estimator is PDA under another name. Pass the covariates or use PDA directly.")
    beta_method: Literal["cce", "bai"] = Field(default="cce", description="Which estimator supplies the covariate slope. 'cce' (default) is Pesaran's (2006) common correlated effects, Equation 16, consistent as N grows with T fixed, which is the regime Remark 1 says most panels are in. 'bai' is Bai's (2009) interactive fixed effects, which needs both N and T large. The paper says only 'Bai's (2009) or Pesaran's (2006) method' and never which produced which published column.")
    r: int = Field(default=2, ge=1, description="Number of common factors, read only when beta_method='bai'. Clamped to the panel's shape when it exceeds it.")
    selector: Literal["lasso_cv", "lasso_bic", "aicc", "all"] = Field(default="lasso_cv", description="Which rule chooses the donor subset for Equation 13. The paper says only that it 'can be chosen using a model selection criterion as in Hsiao, Ching, and Wan (2012), or the LASSO method ... as suggested by Li and Bell (2017)', which does not pin the answer: on the paper's own Table 9 panel these four span a mean absolute effect of 3.79 to 14.04 around a published 9.56. 'lasso_cv' is the default because it has the lowest leave-one-pre-period-out error when the selection is repeated inside every fold, not because it agrees with any published number. 'all' keeps every donor, which is Equation 11 with w unrestricted. Set sensitivity=True to see the spread.")
    standardize_selection: bool = Field(default=False, description="Rescale the selection design before the penalty. Off by default because that design is donor residuals -- one measurement, where a donor's own spread carries information and rescaling would assert that a quiet donor should be as easy to select as a volatile one. Turn it on when the columns are different measurements, since the LASSO's penalty is not scale invariant and cannot reach a small-scale column at any weight.")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Two-sided significance level for the ATT interval and p-value.")
    seed: int = Field(default=0, description="Seed for the cross-validated penalty, where the selector uses one. Fixed by default so a fit is reproducible.")
    lrvar_lag: Optional[int] = Field(default=None, description="Bartlett-kernel truncation lag for the ATT's long-run variance. None (default) takes the usual floor(4 (T2/100)^(2/9)) rule.")
    sensitivity: bool = Field(default=False, description="Also fit under every other selector and record each one's ATT and subset size on the result. Off by default because it costs one extra fit per selector, but reaching for it is the honest move whenever the number is going to be reported: the selector is not pinned by the paper and it moves the estimate, so a point estimate alone claims an identification the method does not have.")

    @model_validator(mode="after")
    def _needs_covariates(self):
        if not self.covariates:
            raise MlsynthConfigError(
                "CPDA needs at least one covariate. Equation 12 residualises "
                "the outcome by x'beta, so with no covariate there is nothing "
                "to residualise and the estimator reduces to PDA. Pass "
                "`covariates=[...]`, or use PDA if that is what you want."
            )
        if len(set(self.covariates)) != len(self.covariates):
            raise MlsynthConfigError(
                f"Duplicate covariates: {self.covariates}. A repeated column "
                f"makes the slope design singular."
            )
        return self
