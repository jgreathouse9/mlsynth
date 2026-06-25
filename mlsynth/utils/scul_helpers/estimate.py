"""Core estimation for Synthetic Control Using Lasso (SCUL).

Hollingsworth, A., & Wing, C. (2022). *"Tactics for design and inference in
synthetic control studies: An applied example using high-dimensional data."*
(Working paper; reference implementation ``github.com/hollina/scul``.)

SCUL builds the synthetic control as a **lasso** regression of the treated
unit's pre-treatment outcome on a (possibly high-dimensional, multi-type) donor
pool. Unlike convex synthetic control, the weights are unrestricted -- negative
weights and an intercept (extrapolation) are allowed -- which lets the donor
pool be larger than the pre-period. The penalty is chosen by a **rolling-origin
expanding-window cross-validation** that respects the time ordering (standard
k-fold CV would shuffle time and is inappropriate here).

This is a faithful port of the reference ``SCUL.R``. The lasso is solved with
``sklearn.linear_model.Lasso`` under glmnet's standardisation convention (each
donor column divided by its population standard deviation ``mysd``), so at a
given penalty the two solvers target the same objective. For continuously
distributed donors the lasso solution is unique (Tibshirani 2013, *The Lasso
Problem and Uniqueness*), so a well-converged ``sklearn`` fit and a
well-converged ``glmnet`` fit agree to solver tolerance.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import Lasso, lasso_path

# Tight convergence: the lasso optimum is unique (continuous donors), and
# glmnet's default threshold under-converges this correlated p>>n problem, so we
# solve tightly to land on the unique solution.
_LASSO_TOL = 1e-12
_LASSO_MAXITER = 2_000_000


def _mysd(x: np.ndarray) -> np.ndarray:
    """glmnet's population standard deviation (``1/N`` variance), per ``mysd.R``."""
    return np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))


def _glmnet_lambda_path(x: np.ndarray, y: np.ndarray, n_lambda: int = 100) -> np.ndarray:
    """glmnet's default penalty grid for a training fit (``SCUL.R`` ``lambdapath``).

    ``lambda_max = max_j |<x_j^std, y - ybar>| / n`` (the smallest penalty zeroing
    every coefficient), then ``n_lambda`` points log-spaced down to
    ``lambda_max * eps`` with ``eps = 0.01`` when ``n < p`` (glmnet's rule).
    """
    n, p = x.shape
    sd = _mysd(x)
    sd[sd == 0] = 1.0
    xs = (x - x.mean(0)) / sd
    lam_max = float(np.max(np.abs(xs.T @ (y - y.mean()))) / n)
    eps = 0.01 if n < p else 1e-4
    return np.exp(np.linspace(np.log(lam_max), np.log(lam_max * eps), n_lambda))


def _lasso_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, lam: float, x_new: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a glmnet-standardised lasso at penalty ``lam`` and predict ``x_new``.

    Standardises the training donors by ``mysd`` (glmnet convention) so the
    penalty ``lam`` matches glmnet's, fits ``sklearn.Lasso`` with an intercept,
    then maps the coefficients back to the original donor scale. Returns
    ``(prediction, coef_original_scale, intercept_original_scale)``.
    """
    mu = x_train.mean(0)
    sd = _mysd(x_train)
    sd[sd == 0] = 1.0
    model = Lasso(alpha=float(lam), fit_intercept=True,
                  max_iter=_LASSO_MAXITER, tol=_LASSO_TOL)
    model.fit((x_train - mu) / sd, y_train)
    coef = model.coef_ / sd
    intercept = float(model.intercept_ - mu @ coef)
    return x_new @ coef + intercept, coef, intercept


def rolling_cv_lambda(
    x_pre: np.ndarray, y_pre: np.ndarray,
    number_initial_periods: int, training_post_length: int,
    cv_option: str = "median",
) -> float:
    """Rolling-origin expanding-window CV penalty (``SCUL.R`` ``lambda.median``).

    For each expanding training window (the first ``i`` pre-periods, ``i`` from
    ``number_initial_periods`` up), fit a lasso over glmnet's penalty grid and
    score it on the next ``training_post_length`` pre-periods (an out-of-sample
    window that mimics the post-treatment forecast). ``"median"`` returns the
    median across windows of each window's MSE-minimising penalty -- the
    reference's robust default; ``"min"`` returns the penalty with the smallest
    mean CV MSE.

    Raises
    ------
    ValueError
        If the pre-period is too short for even one CV window.
    """
    pre_len = x_pre.shape[0]
    max_cv = pre_len - number_initial_periods - training_post_length + 1
    if max_cv < 1:
        raise ValueError(
            f"Pre-period length {pre_len} too short for SCUL cross-validation with "
            f"number_initial_periods={number_initial_periods} and "
            f"training_post_length={training_post_length} (need at least one window)."
        )
    per_window_best = []
    all_mse = []
    path_ref = None
    for i in range(number_initial_periods, number_initial_periods + max_cv):
        x_tr, y_tr = x_pre[:i], y_pre[:i]
        path = _glmnet_lambda_path(x_tr, y_tr)
        if path_ref is None:
            path_ref = path
        x_te, y_te = x_pre[i:i + training_post_length], y_pre[i:i + training_post_length]
        # Whole glmnet-standardised path in one warm-started solve (far faster
        # than refitting per penalty), then test MSE at each penalty.
        mu = x_tr.mean(0)
        sd = _mysd(x_tr)
        sd[sd == 0] = 1.0
        ybar = y_tr.mean()
        _, coefs, _ = lasso_path((x_tr - mu) / sd, y_tr - ybar, alphas=path,
                                 tol=_LASSO_TOL, max_iter=_LASSO_MAXITER)
        coefs = coefs / sd[:, None]                      # (P, n_lambda), original scale
        intercepts = ybar - mu @ coefs                   # (n_lambda,)
        preds = x_te @ coefs + intercepts                # (T_te, n_lambda)
        mses = np.mean((preds - y_te[:, None]) ** 2, axis=0)
        per_window_best.append(float(path[int(np.argmin(mses))]))
        all_mse.append(mses)
    if cv_option == "median":
        return float(np.median(per_window_best))
    # "min": smallest mean CV MSE over the (shared) first-window grid.
    mean_mse = np.mean(np.vstack(all_mse), axis=0)
    return float(path_ref[int(np.argmin(mean_mse))])


def fit_scul(
    outcome_vector: np.ndarray,
    donor_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    *,
    number_initial_periods: int = 5,
    training_post_length: int = 7,
    cv_option: str = "median",
) -> dict:
    """Fit SCUL and return weights, the synthetic series, and the fit measure.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated-unit outcome over all ``T`` periods, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor pool, shape ``(T, P)`` -- the (possibly high-dimensional,
        multi-type) candidate predictors. ``P`` may exceed the pre-period.
    num_pre_treatment_periods : int
        Number of pre-treatment periods ``T0``.
    number_initial_periods : int, default 5
        Training-window length for the first cross-validation run
        (``SCUL.input$NumberInitialTimePeriods``).
    training_post_length : int, default 7
        Out-of-sample window length scored in each CV run
        (``SCUL.input$TrainingPostPeriodLength``).
    cv_option : {"median", "min"}, default "median"
        Penalty-selection rule across CV windows (the reference's
        ``lambda.median`` / ``lambda.min``).

    Returns
    -------
    dict with keys ``counterfactual`` (T,), ``weights`` (P,), ``intercept``
    (float), ``ridge_lambda`` (the selected penalty), ``cohens_d`` (mean
    pre-period \\|gap\\|/sd, the unit-free fit measure), ``support`` (bool P,).
    """
    y = np.asarray(outcome_vector, dtype=float).ravel()
    X = np.asarray(donor_matrix, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T0 = int(num_pre_treatment_periods)
    x_pre, y_pre = X[:T0], y[:T0]

    lam = rolling_cv_lambda(x_pre, y_pre, number_initial_periods,
                            training_post_length, cv_option)
    counterfactual, coef, intercept = _lasso_fit_predict(x_pre, y_pre, lam, X)

    pre_sd = float(np.std(y_pre, ddof=1)) or 1.0
    cohens_d = float(np.mean(np.abs(counterfactual[:T0] - y_pre)) / pre_sd)
    return {
        "counterfactual": counterfactual,
        "weights": coef,
        "intercept": intercept,
        "ridge_lambda": float(lam),
        "cohens_d": cohens_d,
        "support": np.abs(coef) > 1e-10,
    }
