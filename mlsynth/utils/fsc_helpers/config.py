"""Configuration for FSC (Okano & Kurisu 2026).

One input beyond the usual four columns, and it is what makes the estimator
possible rather than a convenience. ``argument`` names the column holding the
point at which each outcome is observed -- the age, the quantile level, the
matrix cell -- because an FSC outcome is a whole object per ``(unit, time)``
cell rather than a number, and the panel has to say where along that object
each row sits.

``space`` then says which metric space the objects live in, which fixes two
things the method needs and nothing else: whether a basis expansion is used in
the augmentation, and how the augmented estimate is projected back into the
image of the isometry. The three values correspond to Examples 1, 2 and 3 of the
paper, which are also its three empirical applications.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class FSCConfig(BaseEstimatorConfig):
    """Typed configuration for :class:`mlsynth.FSC`."""

    argument: str = Field(
        ...,
        description=(
            "Column giving the point at which each outcome value is observed: "
            "age for a fertility curve, quantile level for a distribution, "
            "cell index for a matrix. Every (unit, time) cell must be observed "
            "on the same grid."),
    )
    space: Literal["function", "distribution", "matrix"] = Field(
        default="function",
        description=(
            "Metric space the outcomes live in, following the paper's "
            "examples. 'function' -- curves in L2, the identity isometry and "
            "no projection (Example 1). 'distribution' -- quantile functions "
            "in the 2-Wasserstein space, projected back by increasing "
            "rearrangement (Example 2). 'matrix' -- symmetric positive "
            "semidefinite matrices under the Frobenius metric, supplied as the "
            "row-major lower triangle and projected back by clipping negative "
            "eigenvalues (Example 3). The matrix case is Euclidean, so it uses "
            "the standard basis and skips the spline expansion."),
    )
    augment: bool = Field(
        default=True,
        description=(
            "Apply the ridge augmentation of section 3.2. The plain FSC "
            "weights are always reported; augmentation adds the bias "
            "correction on top, which closes pre-treatment imbalance the "
            "simplex cannot at the cost of allowing negative weights."),
    )
    n_basis: int = Field(
        default=50,
        description=(
            "Number of cubic B-spline basis functions K used to expand the "
            "curves before the ridge correction. The paper uses 50. Ignored "
            "when space='matrix', where the standard basis is already "
            "orthonormal. The floor of 4 is structural: a cubic basis needs at "
            "least two knots."),
    )
    ridge_lambda: Optional[float] = Field(
        default=None,
        description=(
            "Ridge penalty in eq. (9). None (the default) selects it by "
            "leave-one-pre-period-out cross-validation over `lambda_bounds` "
            "(Remark 5). Note the objective is typically flat near its "
            "minimum, so the penalty is weakly identified and the estimate is "
            "insensitive to it over a wide range."),
    )
    lambda_bounds: Tuple[float, float] = Field(
        default=(0.0, 10.0),
        description=(
            "Search interval for the cross-validated penalty. The paper's "
            "scripts use (0, 10) for the two functional applications and "
            "(0, 1) for the covariance one."),
    )
    value_bounds: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Optional (low, high) truncation applied before the increasing "
            "rearrangement, for space='distribution' only -- the support of "
            "the distribution, e.g. (0, 110) for an age-at-death quantile "
            "function. Left open when None."),
    )
    inference: Literal["conformal", "placebo", "both", "none"] = Field(
        default="conformal",
        description=(
            "'conformal' returns a pointwise prediction band for the "
            "counterfactual curve by inverting the sharp null against the "
            "pre-treatment residuals (section 4.3). 'placebo' returns a "
            "per-period p-value by re-running the estimator with each donor "
            "cast as treated -- accurate but N times the work. 'both' does "
            "both; 'none' skips inference."),
    )
    conformal_alpha: float = Field(
        default=0.10,
        description=(
            "Nominal level of the prediction band. Must exceed 1 / (T0 + 1): "
            "below that the inverted test never excludes anything and the "
            "band is unbounded."),
    )

    @field_validator("space", "inference", mode="before")
    @classmethod
    def _check_choice(cls, value, info):
        """Reject an unknown choice before pydantic's Literal check fires.

        Without this the caller sees pydantic's ValidationError; the library
        contract is that a bad config raises MlsynthConfigError.
        """
        allowed = {"space": ("function", "distribution", "matrix"),
                   "inference": ("conformal", "placebo", "both", "none")}
        options = allowed[info.field_name]
        if value not in options:
            raise MlsynthConfigError(
                f"{info.field_name}={value!r} is not one of "
                f"{', '.join(map(repr, options))}.")
        return value

    @model_validator(mode="after")
    def _check(self) -> "FSCConfig":
        # Bounds live here rather than on the Field so a bad value surfaces as
        # MlsynthConfigError. Field constraints raise pydantic's own
        # ValidationError, which would leak the wrong exception type to callers
        # constructing the config directly.
        if self.n_basis < 4:
            raise MlsynthConfigError(
                f"n_basis={self.n_basis} is below 4; a cubic B-spline basis "
                "needs at least two knots, so K = 4 is the smallest usable "
                "value.")
        if self.ridge_lambda is not None and self.ridge_lambda < 0.0:
            raise MlsynthConfigError(
                f"ridge_lambda={self.ridge_lambda} is negative; the penalty in "
                "eq. (9) must be non-negative. Pass None to select it by "
                "cross-validation.")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise MlsynthConfigError(
                f"conformal_alpha={self.conformal_alpha} is not in (0, 1); it "
                "is a nominal coverage level.")
        reserved = {"outcome": self.outcome, "treat": self.treat,
                    "unitid": self.unitid, "time": self.time}
        clash = [k for k, v in reserved.items() if v == self.argument]
        if clash:
            raise MlsynthConfigError(
                f"argument={self.argument!r} also names the {clash[0]} column; "
                "the five column roles must be distinct.")
        lo, hi = self.lambda_bounds
        if not lo < hi:
            raise MlsynthConfigError(
                f"lambda_bounds must be (low, high) with low < high; got "
                f"({lo}, {hi}).")
        if lo < 0.0:
            raise MlsynthConfigError(
                f"lambda_bounds must be non-negative; got ({lo}, {hi}).")
        if self.value_bounds is not None:
            if self.space != "distribution":
                raise MlsynthConfigError(
                    "value_bounds truncates the support of a quantile function "
                    f"and only applies to space='distribution'; got "
                    f"space={self.space!r}.")
            blo, bhi = self.value_bounds
            if not blo < bhi:
                raise MlsynthConfigError(
                    f"value_bounds must be (low, high) with low < high; got "
                    f"({blo}, {bhi}).")
        return self
