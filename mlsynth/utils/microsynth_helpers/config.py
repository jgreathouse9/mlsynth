"""Configuration for the MicroSynth estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class MicroSynthConfig(BaseEstimatorConfig):
    """Configuration for the MicroSynth estimator.

    Implements Robbins & Davenport (2021, *J. Stat. Software*),
    "microsynth: Synthetic Control Methods for Disaggregated and
    Micro-Level Data in R". A user-level balancing estimator: solve a
    constrained QP for non-negative simplex weights on control users
    that exactly balance covariate moments against the treated group's
    moments, then read off the ATT as the weighted-mean outcome
    difference.

    Unlike aggregate-unit SCM estimators in :mod:`mlsynth`, MicroSynth
    operates at the individual-user level with many treated units and
    a large donor pool of controls. The dual ascent solver scales with
    the number of balancing constraints (``d + 1``), not with the
    number of controls, making it tractable for ``N_C`` in the
    millions on a single machine.
    """

    covariates: List[str] = Field(
        ...,
        description=(
            "Column names in ``df`` to use as balancing covariates. "
            "These must be time-invariant per unit (a single value "
            "per user); time-varying features should be collapsed by "
            "the caller (e.g., to pre-treatment means) before passing."
        ),
    )
    outcome_lag_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional list of pre-treatment time labels (as found in "
            "the ``time`` column) whose outcome values become "
            "additional balancing constraints -- the canonical lagged-"
            "outcome predictors. Appended after ``covariates``."
        ),
    )
    standardize_covariates: bool = Field(
        default=True,
        description=(
            "Standardize each covariate to unit SD across all units "
            "before fitting. Improves numerical conditioning of the "
            "dual problem; does not change the final weights."
        ),
    )
    balance_tol: float = Field(
        default=1e-4,
        gt=0.0,
        description=(
            "Maximum absolute standardized mean difference per "
            "covariate accepted as 'balanced' after weighting. Used "
            "for the feasibility diagnostic."
        ),
    )
    max_iter: int = Field(
        default=500,
        ge=10,
        description="L-BFGS-B maximum iterations for the dual problem.",
    )
    gtol: float = Field(
        default=1e-8,
        gt=0.0,
        description="L-BFGS-B gradient tolerance.",
    )
    weight_method: Literal["simplex", "panel"] = Field(
        default="simplex",
        description=(
            "Control-weight scheme. ``'simplex'`` (default) is mlsynth's "
            "min-variance simplex balancing (weights >= 0 summing to 1; the "
            "weighted-mean ATT used for holdout-style studies). ``'panel'`` is "
            "the microsynth panel method (Robbins et al.): a non-negative QP "
            "that exactly balances the treated group's column TOTALS on the "
            "covariates (an intercept makes the weights sum to the treated "
            "count) and least-squares-fits each pre-period outcome, reporting "
            "per-period TOTAL treatment effects over the post-window. A "
            "strictly-convex ridge (``panel_ridge``) selects the unique "
            "minimum-norm / maximum-ESS optimum. Pair with "
            "``outcome_lag_periods`` set to the full pre-period window to "
            "balance the outcome trajectory."
        ),
    )
    panel_ridge: float = Field(
        default=1e-6,
        gt=0.0,
        description=(
            "Strictly-convex ridge weight for ``weight_method='panel'``. The "
            "microsynth QP is rank-deficient over a large control pool (the "
            "counterfactual is not identified by the constraints alone); this "
            "ridge pins the unique minimum-norm / maximum-ESS optimum. Keep "
            "small so the lagged-outcome fit and exact covariate balance "
            "dominate."
        ),
    )
    propensity_mode: bool = Field(
        default=False,
        description=(
            "Propensity-score-type weighting (microsynth's ``match.out=FALSE`` "
            "cross-sectional usage). When ``True`` the weights are computed from "
            "the **covariates only** (lagged outcomes are ignored), the data may "
            "be a single-period cross-section, and the balancing weights "
            "(``donor_weights`` / ``design.w``) are the deliverable -- "
            "covariate-balancing weights on the controls usable as inverse-"
            "propensity-style weights downstream. Forces the panel QP."
        ),
    )
    run_inference: bool = Field(
        default=True,
        description=(
            "Whether to run inference. For ``weight_method='simplex'`` this is a "
            "paired stratified bootstrap CI; for ``weight_method='panel'`` it is "
            "a placebo-permutation test (see ``n_permutations`` / "
            "``permutation_test``)."
        ),
    )
    n_bootstrap: int = Field(
        default=500,
        ge=2,
        description="Bootstrap replications for the simplex CI.",
    )
    n_permutations: int = Field(
        default=250,
        ge=0,
        description=(
            "Number of placebo permutation groups for the panel-method "
            "inference (microsynth's ``perm``). Each draws a random set of "
            "``n_T`` controls as a placebo treated area and refits the QP, so "
            "cost scales with this times the QP solve; 0 disables it."
        ),
    )
    permutation_test: Literal["lower", "upper", "twosided"] = Field(
        default="twosided",
        description=(
            "Tail of the placebo-permutation p-value for the panel method "
            "(microsynth's ``test``): ``'lower'`` (effect below placebos), "
            "``'upper'`` (above), or ``'twosided'`` (magnitude)."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the bootstrap.",
    )
