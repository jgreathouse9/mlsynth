"""Configuration for the SparseSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SparseSCConfig(BaseEstimatorConfig):
    """Configuration for the Sparse Synthetic Control (SparseSC) estimator.

    Implements the L1-penalized predictor-weighting SCM variant of
    Vives-i-Bastida and collaborators (port of the MATLAB
    ``sparse_synth.m`` driver) for the canonical Abadie, Diamond, and
    Hainmueller (2010) framework.

    Like every other ``mlsynth`` estimator this one is fed a single
    long-format ``df`` with one row per (unit, time). Predictors are
    constructed under the hood from the long frame: each column listed
    in ``covariates`` is collapsed to its pre-treatment mean per unit,
    and each entry of ``outcome_lag_periods`` adds the outcome at that
    specific pre-treatment period as a predictor.
    """

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Names of columns in ``df`` to use as predictors. For each "
            "covariate, the per-unit pre-treatment mean is taken as the "
            "predictor value. The first covariate is the *anchor*: its "
            "V-weight is pinned to 1."
        ),
    )
    outcome_lag_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional list of pre-treatment time labels (as found in the "
            "``time`` column) whose outcome values become additional "
            "predictors -- the canonical ADH lagged-outcome predictors. "
            "Appended after ``covariates`` in the predictor matrix."
        ),
    )
    T0_train: Optional[int] = Field(
        default=None,
        ge=2,
        description=(
            "End of the training block within the pre-treatment period "
            "(exclusive). Validation runs on [T0_train, T0_total). "
            "Defaults to floor(T0_total * 0.75)."
        ),
    )
    lambda_grid: Optional[List[float]] = Field(
        default=None,
        description=(
            "L1 penalty grid for predictor selection. Defaults to "
            "[0] + numpy.logspace(-4, 0, 50) -- the MATLAB default."
        ),
    )
    standardize: bool = Field(
        default=True,
        description=(
            "Standardize each predictor across all units before fitting."
        ),
    )
    outer_loss_window: str = Field(
        default="training",
        description=(
            "Which pre-treatment block the outer V-objective evaluates "
            "the outcome MSE over. 'training' (default) matches the "
            "paper's Algorithm 1 line 4 ('for the training data') and "
            "the MATLAB driver sparse_synth.m; this gives the pre-fit "
            "shown in paper Figure 3 and the Table 1 estimates. "
            "'validation' takes the page-4 L_V definition literally "
            "(Y_val in the outer objective); useful for ablation but "
            "produces much worse in-sample fit."
        ),
    )
    solver: Any = Field(
        default=None,
        description="CVXPY solver for the inner W-weight QP. Defaults to OSQP.",
    )
    max_outer_iter: int = Field(
        default=500,
        ge=10,
        description=(
            "Max iterations of the outer L-BFGS-B optimization of "
            "V-weights per lambda. With the analytical envelope-theorem "
            "gradient and ftol=1e-12 (the analytical-mode default), "
            "L-BFGS-B may need several hundred iterations to converge "
            "fully on hard predictor sets; each iteration is microseconds "
            "without the finite-difference multiplier, so 500 is cheap."
        ),
    )
    run_inference: bool = Field(
        default=True,
        description="Whether to run the post-estimation inference procedure.",
    )
    inference_method: Literal["conformal", "placebo", "none"] = Field(
        default="conformal",
        description=(
            "Which inference procedure to run when ``run_inference`` "
            "is True. ``conformal`` (default) builds a moving-block "
            "conformal CI for the ATT in the spirit of Chernozhukov, "
            "Wuethrich and Zhu (2021), calibrated on the validation "
            "residuals; ``placebo`` runs the Abadie-style placebo "
            "permutation; ``none`` skips inference entirely (equivalent "
            "to ``run_inference=False``)."
        ),
    )
    conformal_window: Literal["validation", "pre"] = Field(
        default="validation",
        description=(
            "Residual block used to calibrate the conformal CI. "
            "``validation`` uses only the held-out validation periods "
            "[T0_train, T0_total); ``pre`` uses the full pre-treatment "
            "block [0, T0_total). Validation is smaller but truly "
            "out-of-sample under the chosen V."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Two-sided significance level for the ATT CI.",
    )
    n_placebo: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of placebo donors to use. ``None`` uses every donor."
        ),
    )
    placebo_resweep: bool = Field(
        default=False,
        description=(
            "If True, re-run the full lambda sweep for each placebo. "
            "Slow but most faithful to the actual fit; ``False`` reuses "
            "the lambda selected on the actual treated unit."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the placebo subsample.",
    )
    use_analytical_grad: bool = Field(
        default=False,
        description=(
            "Use the envelope-theorem closed-form Jacobian inside the "
            "outer L-BFGS-B sweep. The analytical Jacobian is exact "
            "(verified against finite differences to ~1e-7) and yields "
            "a 5-10x speedup, but the clean gradient lets L-BFGS-B "
            "settle at the first critical point near the cold init "
            "on the non-convex L1-penalized V-objective; the FD path's "
            "implicit gradient noise tends to find better local optima "
            "at non-zero lambda. Off by default for correctness; opt in "
            "when running large placebo sweeps where exact local optimum "
            "matters less than throughput."
        ),
    )
    warm_start: bool = Field(
        default=False,
        description=(
            "Reuse the previous lambda's V-solution as the initialiser "
            "for the next lambda in the sweep. Warm-starting can save "
            "outer iterations, but on rank-deficient designs it can "
            "also push v into a poorly-conditioned region that breaks "
            "the inner Clarabel solve and can cause L-BFGS-B to settle "
            "in a different local optimum than the canonical "
            "MATLAB-style cold init. Off by default."
        ),
    )
