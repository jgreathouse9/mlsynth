"""Configuration for the SL estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError
from .experts import EXPERTS


class SLConfig(BaseEstimatorConfig):
    """Configuration for the synthetic learner (SL)."""

    experts: List[Literal["lasso", "factor", "forest", "did"]] = Field(default_factory=lambda: list(EXPERTS), description="The expert library, in the column order it should occupy. The default is the paper's own four. Their value comes from erring differently, not from each being good: measured on two panels the four-member library has an error participation ratio of 1.03 to 1.08 out of 4, meaning the members miss in the same direction at the same times, and an equal-weight ensemble of them is 1.17 to 2.92 times the best single member's in-window RMSE. 'forest' is the only member with an information set of its own, which is what covariates reach.")
    covariates: List[str] = Field(default_factory=list, description="Time-varying covariates, read by the 'forest' expert only. Every unit's series for each named column enters the forest's design, mirroring the authors' external_covariates. The other three experts see donor outcomes alone, so covariates are what makes the forest err differently from them.")
    train_periods: Optional[int] = Field(default=None, gt=0, description="How many of the pre-treatment periods fit the experts (Algorithm 1's first split). The rest fit the weights. None (default) takes 60 percent, which is the 30-of-50 split the paper's own scripts use. The experts never see the weighting window, which is what makes the weighting out of sample.")
    eta: Optional[float] = Field(default=None, ge=0.0, description="Learning rate for the exponential weights of Equation 12. None (default) takes the paper's 1/(sqrt(T) var(y)), which evaluates to 48.25 on their panel against the best_eta <- 50 their .Rhistory hard-codes. Their own script computes 51.43 from 1/(sqrt(88) var(med_ts)) on a series of length 100, the same 88 their measured window ends at. All three average instead of selecting: effective_k comes out at 3.69, 3.65 and 3.67 of 4, and concentrating on the best expert needs eta 17 to 60 times larger. Set it explicitly to move along that interpolation; 0 is the simple average. Read effective_k on the result to see which regime a fit is in.")
    post_skip: int = Field(default=0, ge=0, description="Drop this many periods from the start of the post-treatment window before measuring. The paper's Table 4 rows m = 0, 1yr, 2yr, 3yr are this knob, which lets an effect that takes time to arrive be measured away from the switch-on.")
    n_boot: int = Field(default=10000, gt=0, description="Bootstrap replicates for Algorithm 2's critical values and p-value. The paper uses 10000.")
    block: int = Field(default=3, gt=0, description="Block length for the moving-block bootstrap, clamped to the resampling pool. The paper uses 3.")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Level the fit is called against. Recorded on the fit; Algorithm 2 reports critical values at 1, 5, 10 and 20 percent regardless, since a level is a choice a reader makes and the whole null distribution is already computed.")
    seed: int = Field(default=0, description="Seed for the forest expert and the bootstrap resampling. The lasso, factor and did experts are deterministic by construction, so a fit does not move with this except through the forest.")
    lasso_folds: int = Field(default=5, ge=2, description="Cross-validation folds for the lasso and factor experts' penalty. The folds are contiguous and unshuffled, so the penalty is a function of the data: the authors' cv.glmnet(nfolds=5) draws its folds from the RNG, which on their 30-period window puts lambda.min in one of three places, one keeping no donors at all, and moves the reported effect 40 percent.")
    factor_rank: int = Field(default=1, ge=1, description="How many leading factors the 'factor' expert extracts. The paper takes one, the leading eigenvector of the donor Gram matrix.")
    forest_trees: int = Field(default=500, ge=1, description="Trees in the 'forest' expert. 500 is randomForest's default, which the authors leave alone.")
    forest_max_leaf_nodes: int = Field(default=20, ge=2, description="Leaf cap for the 'forest' expert, the authors' maxnodes = 20.")

    @model_validator(mode="after")
    def _check_library(self) -> "SLConfig":
        if not self.experts:
            raise MlsynthConfigError(
                "SL needs at least one expert. The estimator is a weighted "
                "combination over a library; with an empty library there is "
                "nothing to weight.")
        if len(set(self.experts)) != len(self.experts):
            raise MlsynthConfigError(
                f"Duplicate experts: {self.experts}. A repeated member gets "
                f"counted twice in the weighting, which silently doubles its "
                f"influence.")
        if len(set(self.covariates)) != len(self.covariates):
            raise MlsynthConfigError(
                f"Duplicate covariates: {self.covariates}.")
        return self
