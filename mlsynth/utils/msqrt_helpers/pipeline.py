"""Orchestration for the MSQRT estimator (Shen, Song & Abadie 2025).

Coordinates the steps end to end: choose the L1 penalty (or take the supplied
one), fit the donor-weight matrix by Multivariate Square-root Lasso, build the
synthetic counterfactual, form the ATT and per-period/per-unit effects,
optionally attach a block-conformal band, and assemble :class:`MSQRTResults`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...config_models import InferenceResults, WeightsResults
from ..results_helpers import build_effect_submodels
from ..scpi_helpers import out_of_sample_intervals
from .optimization import fit_msqrt_weights, select_lambda_cv
from .structures import MSQRTInputs, MSQRTResults

_TOL = 1e-2


def _default_lambda_grid(n_lambda: int) -> np.ndarray:
    return np.logspace(-2, 2, int(n_lambda))


def _build_weights(theta: np.ndarray, inputs: MSQRTInputs) -> WeightsResults:
    """Per-treated-unit donor-weight dicts plus aggregate sparsity summary."""
    controls = [str(c) for c in inputs.control_names]
    per_unit = {}
    for j, tname in enumerate(inputs.treated_names):
        col = theta[:, j]
        per_unit[str(tname)] = {controls[k]: float(col[k])
                                for k in range(len(controls))
                                if abs(col[k]) > _TOL}
    if inputs.m == 1:
        donor_weights = next(iter(per_unit.values()))
    else:
        donor_weights = {}
        for d in controls:
            donor_weights[d] = float(np.mean(
                [theta[controls.index(d), j] for j in range(inputs.m)]))
        donor_weights = {d: w for d, w in donor_weights.items() if abs(w) > _TOL}

    nonzero = np.sum(np.abs(theta) > _TOL, axis=0)
    summary = {
        "weights_are": "msqrt_donor_matrix",
        "note": ("Columns of Theta give each treated unit's donor weights from "
                 "the Multivariate Square-root Lasso fit (Shen, Song & Abadie "
                 "2025, eq. 5). Weights are not constrained to the simplex."),
        "n_treated": int(inputs.m),
        "n_donors": int(inputs.n),
        "avg_active_donors_per_treated": float(np.mean(nonzero)),
        "max_active_donors": int(nonzero.max()) if nonzero.size else 0,
        "per_unit_donor_weights": per_unit,
    }
    return WeightsResults(donor_weights=donor_weights, summary_stats=summary)


def run_msqrt(
    inputs: MSQRTInputs,
    *,
    lambda_: Optional[float] = None,
    n_lambda: int = 15,
    lambdas: Optional[Sequence[float]] = None,
    cv_initial_train: Optional[int] = None,
    cv_val_window: Optional[int] = None,
    cv_step: Optional[int] = None,
    cv_folds: Optional[int] = None,
    inference: bool = False,
    alpha: float = 0.1,
    time_dependence: str = "iid",
) -> MSQRTResults:
    """Fit MSQRT and assemble :class:`MSQRTResults`.

    Parameters
    ----------
    inputs : MSQRTInputs
    lambda_ : float, optional
        Fixed L1 penalty. If None, chosen by rolling-origin CV.
    n_lambda : int
        Size of the default log-spaced CV grid (used when ``lambdas`` is None).
    lambdas : sequence, optional
        Explicit CV grid (overrides ``n_lambda``).
    cv_* : int, optional
        Rolling-origin CV schedule overrides; adaptive defaults otherwise.
    inference : bool
        Attach CFPT/scpi prediction intervals (Cattaneo, Feng, Palomba &
        Titiunik 2025) on all four predictands + simultaneous bands. For MSQRT
        only the *out-of-sample* (post-treatment noise) error is modelled --
        see :mod:`mlsynth.utils.scpi_helpers`. Default False.
    alpha : float
        Total miscoverage level for the intervals.
    time_dependence : {"iid", "general"}
        Time-averaging assumption for the time-averaged predictands.
    """
    Y_pre, X_pre = inputs.Y_pre, inputs.X_pre
    Y_post, X_post = inputs.Y_post, inputs.X_post

    if lambda_ is not None:
        best_lambda = float(lambda_)
    else:
        grid = list(lambdas) if lambdas is not None else _default_lambda_grid(n_lambda)
        best_lambda = select_lambda_cv(
            Y_pre, X_pre, grid,
            initial_train=cv_initial_train, val_window=cv_val_window,
            step=cv_step, n_folds=cv_folds,
        )

    theta, Y_pre_hat, _ = fit_msqrt_weights(Y_pre, X_pre, best_lambda, tol=_TOL)
    Y_post_hat = X_post @ theta

    # Full-timeline observed / synthetic for the treated units (T x m).
    observed = np.vstack([Y_pre, Y_post])
    synthetic = np.vstack([Y_pre_hat, Y_post_hat])
    gap = observed - synthetic

    pre_gap = Y_pre - Y_pre_hat
    post_gap = Y_post - Y_post_hat

    att = float(post_gap.mean())
    synth_post_mean = float(Y_post_hat.mean())
    att_percent = float(100.0 * att / synth_post_mean) if synth_post_mean else float("nan")
    att_t = post_gap.mean(axis=1)                        # (T_post,)
    unit_att = {str(inputs.treated_names[j]): float(post_gap[:, j].mean())
                for j in range(inputs.m)}

    treated_mean = observed.mean(axis=1)                 # (T,)
    synthetic_mean = synthetic.mean(axis=1)
    sparsity = np.sum(np.abs(theta) > _TOL, axis=0)
    pre_rmse = float(np.sqrt(np.mean(pre_gap ** 2)))

    weights = _build_weights(theta, inputs)

    inf = None
    if inference:
        post_labels = list(inputs.time_labels[inputs.T0:])
        inf = out_of_sample_intervals(
            effects=post_gap, pre_residuals=pre_gap,
            unit_names=[str(u) for u in inputs.treated_names],
            period_labels=post_labels, alpha=alpha,
            time_dependence=time_dependence,
        )

    metadata = {
        "n_treated": int(inputs.m),
        "n_control": int(inputs.n),
        "T0": int(inputs.T0),
        "n_post": int(inputs.n_post),
        "best_lambda": best_lambda,
        "estimator": "MSQRT",
        "objective": "(1/sqrt(T0))||Y - X Theta||_* + lambda ||Theta||_1",
    }
    # Standardized InferenceResults mirrored from the SCPI overall ATT band.
    std_inference = None
    if inf is not None:
        std_inference = InferenceResults(
            method=inf.method,
            ci_lower=float(inf.taua.lower), ci_upper=float(inf.taua.upper),
            confidence_level=float(1.0 - alpha),
            details=inf,
        )
    submodels = build_effect_submodels(
        observed_outcome=treated_mean,
        counterfactual_outcome=synthetic_mean,
        n_pre_periods=int(inputs.T0),
        n_post_periods=int(inputs.n_post),
        time_periods=np.asarray(inputs.time_labels),
        weights=weights,
        inference=std_inference,
        method_name="MSQRT",
        effects_overrides={"att": float(att), "att_percent": float(att_percent)},
        fit_overrides={"rmse_pre": float(pre_rmse)},
        intervention_time=inputs.time_labels[inputs.T0],
    )
    return MSQRTResults(
        **submodels,
        inputs=inputs, att_percent=att_percent, theta=theta,
        counterfactual_matrix=synthetic, gap_matrix=gap, att_t=att_t,
        unit_att=unit_att, treated_mean=treated_mean,
        synthetic_mean=synthetic_mean, best_lambda=best_lambda,
        sparsity=sparsity, inference_intervals=inf, metadata=metadata,
    )
