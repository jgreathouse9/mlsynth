"""Configuration for the BPSCS estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class BPSCSConfig(BaseEstimatorConfig):
    """Configuration for Bayesian Penalized Synthetic Control under Spillovers (BPSCS).

    Implements the utility-based shrinkage priors of:

        Fernandez-Morales, E., Oganisian, A., & Lee, Y. (2026). "Bayesian
        shrinkage priors for penalized synthetic control estimators in the
        presence of spillovers." Biometrics 82(2), ujag054.

    BPSCS models the treated unit's no-intervention outcome with an
    autoregressive linear synthetic control whose donor coefficients carry a
    shrinkage prior. The prior scale for each donor is set by a utility that
    blends covariate similarity and spatial distance to the treated unit, so
    donors that are spatially close -- and therefore likely contaminated by
    spillovers -- are shrunk toward zero rather than excluded outright. Two
    priors are available: ``dhs`` (distance-horseshoe, continuous shrinkage) and
    ``ds2`` (distance-spike-and-slab, a hard utility cutoff). The posterior is
    drawn with NUTS (NumPyro, in double precision), so this estimator requires
    the ``[bayes]`` optional dependency (``pip install mlsynth[bayes]``).

    Parameters
    ----------
    covariates : list of str
        Baseline (time-invariant) covariate columns used for the covariate
        similarity term of the utility. At least one is required.
    coords : list of str
        Coordinate columns (e.g. ``["lat", "lon"]``) giving each unit's spatial
        location; the utility's distance term uses the Euclidean distance from
        each donor to the treated unit. At least one is required.
    prior : {"dhs", "ds2"}
        Shrinkage prior. ``dhs`` is the distance-horseshoe (continuous);
        ``ds2`` is the distance-spike-and-slab (hard cutoff at the inclusion
        radius).
    kappa_d : float
        Importance weight in [0, 1] trading off covariate similarity against
        spatial distance in the utility. ``0`` uses spatial distance only (the
        paper's emphasized spillover regime); ``1`` uses covariate similarity
        only.
    inclusion_quantile : float
        For ``ds2``, the quantile of the utility scores that sets the inclusion
        radius; donors below it fall in the spike (excluded). Paper default
        0.25.
    n_warmup : int
        NUTS warm-up (adaptation) iterations per chain.
    n_samples : int
        Retained NUTS samples per chain.
    n_chains : int
        Number of NUTS chains.
    target_accept : float
        NUTS target acceptance probability. The paper uses 0.995; the default
        here (0.95) trades a little robustness for speed.
    max_tree_depth : int
        NUTS maximum tree depth (paper 15; default 12 here).
    ci_alpha : float
        Two-sided level for the credible interval (0.05 -> 95% band).
    seed : int
        PRNG seed (makes a fit reproducible).
    display_graphs : bool
        Plot the observed-vs-counterfactual path with the credible band.
    verbose : bool
        Show the NUTS progress bar.
    """

    covariates: List[str] = Field(..., min_length=1)
    coords: List[str] = Field(..., min_length=1)
    prior: Literal["dhs", "ds2"] = Field(default="dhs")
    kappa_d: float = Field(default=0.0, ge=0.0, le=1.0)
    inclusion_quantile: float = Field(default=0.25, gt=0.0, lt=1.0)
    n_warmup: int = Field(default=2000, ge=1)
    n_samples: int = Field(default=1000, ge=1)
    n_chains: int = Field(default=4, ge=1)
    target_accept: float = Field(default=0.95, gt=0.0, lt=1.0)
    max_tree_depth: int = Field(default=12, ge=1)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = Field(default=0)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)

    @model_validator(mode="after")
    def _check(self) -> "BPSCSConfig":
        if self.n_samples * self.n_chains < 2:
            raise MlsynthConfigError(
                "BPSCS needs at least two posterior draws "
                "(n_samples * n_chains >= 2)."
            )
        return self
