"""Configuration for the SPOTSYNTH estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SPOTSYNTHConfig(BaseEstimatorConfig):
    """Configuration for the SPOTSYNTH estimator.

    O'Riordan & Gilligan-Lee (2025), *"Spillover detection for donor selection in
    synthetic control models"* (Journal of Causal Inference 13:20240036,
    doi:10.1515/jci-2024-0036). Screens each candidate donor for spillover
    contamination via a pre-intervention forecast test (Algorithm 1), excludes
    the contaminated donors, and fits a simplex synthetic control on the valid
    set. Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` interface and expects a single treated unit.

    Parameters
    ----------
    selection : {"S1", "S2", "all"}
        Donor-selection rule. ``S1`` keeps the ``n_donors`` donors with the
        smallest forecast error (the analyst fixes how many to keep); ``S2``
        keeps donors whose realised post-intervention value falls inside the
        ``ppi`` posterior predictive interval (controls the false-positive rate);
        ``all`` keeps every donor (the unscreened baseline).
    forecast : {"loo", "lag"}
        Forecast anchor for the screen. ``loo`` (default) is the leave-one-out
        anchor: each donor's whole post-intervention trajectory is forecast from
        the *other* donors' common factors and scored by the mean absolute
        deviation. It is robust to the *onset speed* of the spillover (it detects
        gradually-arriving contamination that the first-post-point misses) and is
        the right choice whenever the donor pool is mostly valid -- the typical
        applied setting. ``lag`` is the paper's literal Algorithm 1: forecast the
        *first* post-intervention point from the last clean pre-intervention
        cross-section. It is the correct anchor only in the paper's stress regime
        -- a *mostly-invalid* donor pool with a *sharp* (immediate) spillover --
        where ``loo`` inverts because the contaminated majority defines the
        "consensus". See the forecast-anchor discussion in the estimator docs.
    n_donors : int, optional
        Number of donors to keep under ``S1`` (default: half the pool).
    ppi : float
        Posterior-predictive-interval level for ``S2`` (default 0.8).
    n_factors : int
        Number of donor factors used to regularise the forecast (default 5).
    time_average : int, optional
        Bucket width for time-averaging the data before screening (``lag`` only;
        reduces false negatives with very noisy donors).
    inference : {"bayes", "frequentist"}
        Synthetic-control weight model. ``bayes`` (default) is the authors'
        Bayesian simplex SC -- weights with a ``Dirichlet(dirichlet_alpha)``
        prior, a half-normal prior on the residual sd, pre-period standardisation
        of target and donors, and 95% posterior-predictive credible intervals --
        fit with NumPyro's NUTS (the same Hamiltonian-Monte-Carlo family as the
        authors' Stan). It requires the optional ``numpyro`` package.
        ``frequentist`` is a fast, dependency-free simplex least-squares point
        estimate (no intervals), useful for large simulations or when NumPyro is
        unavailable.
    dirichlet_alpha : float
        Dirichlet concentration on the donor weights (paper uses 0.4; ``< 1``
        favours sparse weights).
    ci_level : float
        Credible-interval level for the Bayesian fit (paper uses 0.95).
    n_samples, n_warmup : int
        Posterior draws and warm-up iterations for the Bayesian sampler.
    debias : bool
        If True, also compute the proximal (two-stage / GMM) debiased ATT
        (equation 5), using the screen-excluded donors as proximal controls to
        correct errors-in-variables bias when the kept donors are noisy proxies.
    seed : int
        RNG seed for the Bayesian sampler.
    """

    selection: Literal["S1", "S2", "all"] = Field(
        default="S1", description="Donor-selection rule (S1, S2, or all).")
    forecast: Literal["loo", "lag"] = Field(
        default="loo",
        description="Forecast anchor: loo (default, onset-robust) or lag "
                    "(paper Algorithm 1, for mostly-invalid sharp-spillover pools).")
    n_donors: Optional[int] = Field(
        default=None, ge=1, description="Number of donors to keep under S1.")
    ppi: float = Field(
        default=0.8, gt=0.0, lt=1.0,
        description="Posterior predictive interval level for S2.")
    n_factors: int = Field(
        default=5, ge=1, description="Number of donor factors in the forecast.")
    time_average: Optional[int] = Field(
        default=None, ge=1, description="Bucket width for time-averaging (lag).")
    inference: Literal["bayes", "frequentist"] = Field(
        default="bayes",
        description="SC weight model (bayes = NumPyro NUTS, or frequentist simplex LS).")
    dirichlet_alpha: float = Field(
        default=0.4, gt=0.0, description="Dirichlet concentration on donor weights.")
    ci_level: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Credible-interval level.")
    n_samples: int = Field(
        default=4000, ge=200, description="Posterior draws for the Bayesian fit.")
    n_warmup: int = Field(
        default=2000, ge=100, description="Warm-up iterations for the Bayesian fit.")
    debias: bool = Field(
        default=False, description="Compute the proximal (GMM) debiased ATT.")
    seed: int = Field(
        default=0, description="RNG seed for the Bayesian sampler.")
