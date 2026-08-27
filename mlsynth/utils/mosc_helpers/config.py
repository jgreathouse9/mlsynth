"""Configuration for the MOSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class MOSCConfig(BaseEstimatorConfig):
    """Configuration for many-outcomes synthetic control (MOSC).

    Implements:

        Wang, Y., Schein, A., Shou, J., & Blei, D. M. "A Many-outcomes
        Perspective on the Synthetic Control Method."

    MOSC justifies synthetic control through negative control outcomes instead
    of a linear factor model. A probabilistic factor model is fit to the
    pre-intervention panel; the per-unit latent loadings it returns are the
    estimated confounding structure, and they enter a downstream outcome
    regression as though they were observed confounders. Choosing the likelihood
    is the point: a count panel gets a gamma-Poisson model where the rest of the
    library's factor estimators assume a Gaussian one.

    The working set is three fields. ``factor_model`` picks the likelihood,
    ``n_factors`` its rank, and ``outcome_scale`` whether the panel is modelled
    as given or in first differences. The rest carry defaults that most fits
    leave alone.

    Parameters
    ----------
    factor_model : {"gap", "ppca"}
        Likelihood for the factor model. ``"gap"`` is the gamma-Poisson model of
        the paper's case study, drawn by conjugate Gibbs; it needs a non-negative
        outcome. ``"ppca"`` is probabilistic PCA (Tipping & Bishop 1999), the
        Gaussian arm, and admits any real outcome.
    n_factors : int
        Rank ``K`` of the factor model. Must be smaller than both the number of
        units and the number of pre-intervention periods.
    outcome_scale : {"level", "difference"}
        Whether the factor model sees the outcome as supplied or its first
        difference. ``"level"`` is the default because transforming a caller's
        outcome without being asked would change what the estimate means. Reach
        for ``"difference"`` when the reported ``pearson_dispersion`` or
        ``residual_autocorrelation`` says the level series does not satisfy the
        model -- a cumulative series is the case that forces it. A differenced
        fit is re-integrated, so the counterfactual comes back on the outcome's
        own scale.
    n_samples : int
        Posterior draws retained. The counterfactual is a posterior mean, so this
        does not need to be large.
    n_warmup : int
        Gibbs sweeps discarded before collection. Ignored by the ``"ppca"`` arm,
        which is fit by EM.
    ridge_alphas : tuple of float
        Ridge penalties cross-validated over in the outcome regression.
    heldout_fraction : float
        Share of pre-period cells held out to score the fit. Reported as a
        predictive log density, which is a score and carries no size guarantee.
    inference : {"bootstrap", "posterior"}
        How the interval is formed. ``"bootstrap"`` is the paper's Section 3.4
        procedure: resample units with replacement, re-run the algorithm on each
        resample, and take quantiles of the resulting estimates. It targets the
        sampling uncertainty in the g-formula of Theorem 4, which comes from
        having finitely many units. ``"posterior"`` returns the spread of the
        factor model's own draws, which is what the paper's figures plot; it is a
        band on the counterfactual's conditional mean and is narrower, because it
        conditions on the units that happened to be observed.
    n_bootstrap : int
        Resamples drawn when ``inference="bootstrap"``.
    ci_alpha : float
        Two-sided level for the interval (0.05 -> 95%).
    seed : int
        PRNG seed, which makes a fit reproducible.

    Notes
    -----
    Three deviations from the paper are deliberate, each established by the
    spike in ``benchmarks/reference/mosc_spike/``.

    The effect follows equation 43, ``Y - f(0)``. The authors' code computes
    ``f(0) - Y``, which inverts it; on their null result the difference is
    invisible, and on any real effect it flips the sign.

    Inference follows the paper's text and not its figures. Section 3.4
    prescribes a nonparametric bootstrap over units and states that its coverage
    is evaluated in Section 5; that evaluation does not appear, and the case
    study plots the posterior spread instead. On the authors' own control panels
    -- teams that never admitted fans, where the effect is zero by construction
    -- the posterior band covered zero four times in ten at a nominal 95%. The
    bootstrap is therefore the default and the posterior band is available under
    a name that says what it is.

    The paper's ``p_pop`` model check is not offered. Its stated false rejection
    rate is 0.05 and its measured rate on a correctly specified model is 0.40,
    because equation 36 sums the discrepancy over held-out cells until the
    comparison stops being random. ``heldout_log_density`` is the same
    comparison reported as a score.

    The lagged pre-intervention outcome that the authors' code adds to the
    design is absent. It appears nowhere in equations 40-41, and the baseline it
    is compared against gets no equivalent term.
    """

    factor_model: Literal["gap", "ppca"] = Field(
        default="gap",
        description="Likelihood for the factor model: gamma-Poisson for counts, PPCA for real-valued outcomes.",
    )
    n_factors: int = Field(
        default=10, ge=1,
        description="Rank K of the factor model; must be below the unit count and the pre-period length.",
    )
    outcome_scale: Literal["level", "difference"] = Field(
        default="level",
        description="Model the outcome as supplied, or its first difference (re-integrated before reporting).",
    )
    n_samples: int = Field(
        default=200, ge=1,
        description="Posterior draws retained; the counterfactual is a posterior mean over them.",
    )
    n_warmup: int = Field(
        default=200, ge=0,
        description="Gibbs sweeps discarded before collection; unused by the PPCA arm.",
    )
    ridge_alphas: Tuple[float, ...] = Field(
        default=(0.0, 1e-4, 1e-3, 1e-2),
        description="Ridge penalties cross-validated over in the downstream outcome regression.",
    )
    heldout_fraction: float = Field(
        default=0.10, gt=0.0, lt=1.0,
        description="Share of pre-period cells held out to score the factor model's fit.",
    )
    inference: Literal["bootstrap", "posterior"] = Field(
        default="bootstrap",
        description="Interval from the paper's unit bootstrap (Section 3.4), or from the factor model's posterior spread.",
    )
    n_bootstrap: int = Field(
        default=200, ge=2,
        description="Resamples drawn when inference='bootstrap'; each re-runs the algorithm on a resampled donor pool.",
    )
    ci_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the credible interval (0.05 gives a 95% band).",
    )
    seed: int = Field(
        default=0,
        description="PRNG seed; a fit with the same seed and inputs is reproducible.",
    )

    @model_validator(mode="after")
    def _check(self) -> "MOSCConfig":
        if not self.ridge_alphas:
            raise MlsynthConfigError("MOSC needs at least one ridge penalty in ridge_alphas.")
        if any(a < 0 for a in self.ridge_alphas):
            raise MlsynthConfigError("MOSC ridge penalties must be non-negative.")
        return self
