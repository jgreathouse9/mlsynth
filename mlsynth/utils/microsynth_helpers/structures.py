"""Typed result containers for MicroSynth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MicroSynthInputs:
    """Pre-processed user-level matrices for MicroSynth.

    Parameters
    ----------
    X_T : np.ndarray
        Treated-user covariate matrix, shape ``(n_T, d)``. Already
        standardized if ``standardize_covariates=True``.
    X_C : np.ndarray
        Control-user covariate matrix, shape ``(n_C, d)``. Same
        standardization as ``X_T``.
    Y_T : np.ndarray
        Treated-user post-treatment outcomes, shape
        ``(n_T, T_post)`` where ``T_post`` is the number of
        post-treatment periods. If ``T_post = 1`` this is collapsed
        to ``(n_T,)``.
    Y_C : np.ndarray
        Control-user post-treatment outcomes, shape ``(n_C, T_post)``
        or ``(n_C,)`` matching ``Y_T``.
    treated_unit_names : Sequence
        Identifiers of the treated users, in row order of ``X_T``.
    control_unit_names : Sequence
        Identifiers of the control users, in row order of ``X_C``.
    covariate_names : Sequence[str]
        Labels of the balancing constraints in column order of ``X_T``
        / ``X_C``. Includes both the user-supplied ``covariates`` and
        any ``outcome_lag_periods`` columns.
    n_T, n_C, d, T_post : int
        Cached shapes.
    cohort_time : Any
        The treatment-onset time inferred from ``df``.
    covariate_sd : np.ndarray
        Pooled SD used for standardization, shape ``(d,)``. ``None``
        if standardization was disabled.
    outcome : str
        Outcome column name.
    """

    X_T: np.ndarray
    X_C: np.ndarray
    Y_T: np.ndarray
    Y_C: np.ndarray
    treated_unit_names: Sequence
    control_unit_names: Sequence
    covariate_names: Sequence
    cohort_time: Any
    covariate_sd: Optional[np.ndarray]
    outcome: str

    @property
    def n_T(self) -> int:
        return int(self.X_T.shape[0])

    @property
    def n_C(self) -> int:
        return int(self.X_C.shape[0])

    @property
    def d(self) -> int:
        return int(self.X_T.shape[1])

    @property
    def T_post(self) -> int:
        return 1 if self.Y_T.ndim == 1 else int(self.Y_T.shape[1])


@dataclass(frozen=True)
class MicroSynthDesign:
    """Outputs of the dual ascent + balance diagnostics.

    Parameters
    ----------
    w : np.ndarray
        Control-side weights on the simplex, shape ``(n_C,)``.
        ``sum(w) == 1``, ``w >= 0``.
    dual_lambda : np.ndarray
        Lagrange multipliers for the covariate balance constraints,
        shape ``(d,)``.
    dual_nu : float
        Lagrange multiplier for the sum-to-one constraint.
    smd_before : np.ndarray
        Per-covariate standardized mean difference between treated
        and unweighted controls, shape ``(d,)``.
    smd_after : np.ndarray
        Per-covariate SMD after applying ``w``, shape ``(d,)``.
        Should be near zero on every constraint.
    ess : float
        Effective sample size of the weighted control group,
        ``1 / sum(w^2)``.
    max_weight : float
        Largest single control-user weight.
    feasible : bool
        ``True`` if every ``|smd_after_k| < balance_tol``. ``False``
        signals that the QP did not achieve balance and the treated
        group may lie outside the convex hull of controls.
    feasibility_message : str
        Human-readable diagnostic.
    n_iterations : int
        L-BFGS-B iterations to convergence.
    converged : bool
        Whether the optimizer reported success.
    """

    w: np.ndarray
    dual_lambda: np.ndarray
    dual_nu: float
    smd_before: np.ndarray
    smd_after: np.ndarray
    ess: float
    max_weight: float
    feasible: bool
    feasibility_message: str
    n_iterations: int
    converged: bool


@dataclass(frozen=True)
class MicroSynthInference:
    """Bootstrap confidence interval and standard error."""

    method: str                     # "paired_bootstrap" or "none"
    att: float                       # point estimate
    se: float                        # bootstrap SE
    ci: np.ndarray                   # [low, high]
    n_bootstrap: int                 # successful reps
    bootstrap_atts: np.ndarray       # full distribution


@dataclass(frozen=True)
class MicroSynthResults:
    """Public return container for ``MicroSynth.fit()``.

    Parameters
    ----------
    inputs : MicroSynthInputs
        Pre-processed inputs.
    design : MicroSynthDesign
        Weights, dual variables, balance diagnostics.
    inference : MicroSynthInference
        Bootstrap CI on the ATT (or ``method = "none"`` if disabled).
    counterfactual : np.ndarray
        Weighted-control outcomes per post-treatment period,
        shape matches ``Y_T``.
    gap : np.ndarray
        Treated mean minus counterfactual, per post-treatment period.
        Shape matches ``Y_T``.
    gap_trajectory : np.ndarray
        Per-post-period gap, always 1-D (length ``T_post``).
    att : float
        Mean of ``gap_trajectory``.
    donor_weights : Dict[Any, float]
        ``{control_user_name: w_i}`` for all controls with
        ``w_i > 0``.
    """

    inputs: MicroSynthInputs
    design: MicroSynthDesign
    inference: MicroSynthInference
    counterfactual: np.ndarray
    gap: np.ndarray
    gap_trajectory: np.ndarray
    att: float
    donor_weights: Dict[Any, float]
