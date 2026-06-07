"""Configuration for the SPCD estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
import pandas as pd
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseMAREXConfig


class SPCDConfig(BaseMAREXConfig):
    """
    Configuration for the Synthetic Principal Component Design (SPCD) estimator.

    Implements Lu, Li, Ying, Blanchet (2022), "Synthetic Principal Component
    Design: Fast Covariate Balancing with Synthetic Controls",
    arXiv:2211.15241v1.

    Parameters
    ----------
    variant : {"spcd", "norm_spcd"}
        Iteration-box choice. ``"spcd"`` uses Eq. (4)/(7) of the paper;
        ``"norm_spcd"`` uses Eq. (5)/(8). The paper's Section 4
        experiments use ``"norm_spcd"`` with closed-form weights.
    weights : {"empirical", "exact"}
        Final-weight-step choice. ``"empirical"`` uses Eq. (9) — the
        closed-form approximation used in all of the paper's
        experiments. ``"exact"`` solves Eq. (6) via cvxpy.
    alpha_ridge : float or None, optional
        Ridge term ``alpha`` in Eq. (2), playing the role of the noise
        variance ``sigma``. If ``None``, it is chosen by out-of-sample
        pre-period balance over a noise-scale grid
        (``select_alpha_by_holdout``), since the post-period RMSE is a
        jumpy function of ``alpha`` when ``N > T_pre``. Pass a value
        (e.g. a known noise variance) to bypass selection.
    lam_balance : float or None, optional
        Sum-zero penalty ``lambda`` in Eq. (2). Auto-estimated if
        ``None``. Theorem 1 requires this to be "large enough".
    beta : float or None, optional
        Iteration step parameter ``beta`` in Eqs. (4)/(5)/(7)/(8).
        Auto-estimated from the spectrum if ``None``.
    max_iter : int
        Maximum iterations for the SPCD/NormSPCD while loop.
    T0 : int or None, optional
        Number of pre-treatment periods.
    post_col : str or None, optional
        Column indicating post-treatment periods.
    solver : Any, optional
        CVXPY-compatible solver. Used only when ``weights="exact"``.
    display_graph : bool
        Whether to display the synthetic treated/control plot.
    verbose : bool
        Solver verbosity.

    Notes
    -----
    Algorithms 3 and 4 of the paper (Appendix 3.2) are abstract
    meta-versions used in the proof of Theorem 3 (global convergence).
    They correspond to the same iterations as Algorithms 1 and 2 acting
    on a generic Hermitian perturbed rank-1 matrix and are not exposed
    as separate user options here.
    """

    variant: Literal["spcd", "norm_spcd"] = Field(
        default="norm_spcd",
        description="SPCD iteration variant. 'spcd' uses Eq. (4)/(7); "
                    "'norm_spcd' uses Eq. (5)/(8).",
    )
    weights: Literal["empirical", "exact"] = Field(
        default="empirical",
        description="Final-weight-step choice. 'empirical' uses Eq. (9); "
                    "'exact' solves Eq. (6) via cvxpy.",
    )
    alpha_ridge: Optional[float] = Field(
        default=None,
        ge=0,
        description="Ridge term alpha in Eq. (2) (the noise variance sigma). "
                    "If None, selected by out-of-sample pre-period balance "
                    "over a noise-scale grid (select_alpha_by_holdout).",
    )
    lam_balance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sum-zero penalty lambda in Eq. (2). Auto-estimated if None.",
    )
    beta: Optional[float] = Field(
        default=None,
        ge=0,
        description="Iteration step parameter beta in Eqs. (4)/(5)/(7)/(8). "
                    "Auto-estimated if None.",
    )
    max_iter: int = Field(
        default=200,
        gt=0,
        description="Maximum iterations for the SPCD/NormSPCD while loop.",
    )
    T0: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of pre-treatment periods when post_col is not supplied.",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 or boolean column identifying post-treatment periods.",
    )
    arm: Optional[str] = Field(
        default=None,
        description="Optional categorical column naming each unit's treatment "
                    "arm. When given, SPCD solves its design independently "
                    "within each arm's units and returns SPCDMultiArmResults "
                    "(a dict of per-arm SPCDResults); when None (default), a "
                    "single SPCDResults is returned.",
    )
    solver: Any = Field(
        default=None,
        description="CVXPY-compatible solver, only used when weights='exact'.",
    )
    display_graph: bool = Field(default=False, description="Whether to display SPCD plots.")
    verbose: bool = Field(default=False, description="Whether to print solver progress.")

    # ------------------------------------------------------------------
    # Inference and power analysis (LEXSCM-style E/B holdout split).
    # ------------------------------------------------------------------
    enable_inference: bool = Field(
        default=True,
        description="Run conformal inference + Monte Carlo power analysis. "
                    "Trains the design on the first holdout_frac_E of pretreatment "
                    "and uses the remaining periods as out-of-sample residuals.",
    )
    holdout_frac_E: float = Field(
        default=0.7,
        ge=0.1,
        le=0.95,
        description="Fraction of pretreatment periods used for the SPCD design fit. "
                    "The remaining 1 - holdout_frac_E periods form the holdout window.",
    )
    inference_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Two-sided significance level for conformal CI and MDE.",
    )
    power_target: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Target statistical power for the MDE search.",
    )
    mde_n_sims: int = Field(
        default=5000, gt=0,
        description="Monte Carlo draws for the null distribution in the MDE.",
    )
    mde_n_trials: int = Field(
        default=400, gt=0,
        description="Trials per tau grid point for empirical power estimation.",
    )
    mde_horizon_grid: Optional[List[int]] = Field(
        default=None,
        description="Optional list of post-treatment horizons for the "
                    "detectability curve. If None, no curve is computed.",
    )
    inference_seed: int = Field(
        default=1400,
        description="Seed for the Monte Carlo MDE machinery.",
    )
    min_blank_size: int = Field(
        default=5, gt=0,
        description="Minimum holdout-window size below which inference is "
                    "skipped with a warning (design is still fit on the "
                    "estimation window).",
    )
    pooled_weights: Literal["size", "equal"] = Field(
        default="size",
        description="Weighting for the multi-arm pooled average-effect MDE "
                    "(only used when an 'arm' column is set). 'size' weights "
                    "each arm by its unit count (population-average effect); "
                    "'equal' weights arms equally.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional covariate columns to balance on *in addition* "
                    "to the pre-treatment outcomes. Each unit's per-covariate "
                    "pre-period mean is z-scored across units and folded into "
                    "the SPCD iteration matrix as a covariate-balance term "
                    "(M += covariate_weight * scale * X X^T). None (default) "
                    "balances on outcomes only. Time-invariant covariates "
                    "(e.g. last year's market share) collapse to their value.",
    )
    covariate_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of covariate balance vs outcome balance "
                    "in the SPCD Gram matrix. 0 ignores covariates; 1 gives "
                    "covariates equal 'energy' to the outcomes; >1 upweights "
                    "covariate balance. Only used when 'covariates' is set.",
    )

    @model_validator(mode="after")
    def check_spcd_params(cls, values: Any) -> Any:
        df = values.df
        n_periods = df[values.time].nunique()

        if values.post_col is not None and values.post_col not in df.columns:
            raise MlsynthConfigError(f"post_col '{values.post_col}' is not present in df.")

        if values.T0 is not None and values.T0 > n_periods:
            raise MlsynthConfigError("T0 cannot exceed the number of unique time periods in df.")

        if values.covariates:
            missing = [c for c in values.covariates if c not in df.columns]
            if missing:
                raise MlsynthConfigError(
                    f"covariates not present in df: {missing}."
                )
            non_numeric = [
                c for c in values.covariates
                if not pd.api.types.is_numeric_dtype(df[c])
            ]
            if non_numeric:
                raise MlsynthConfigError(
                    f"SPCD covariates must be numeric; non-numeric: {non_numeric}."
                )

        return values
