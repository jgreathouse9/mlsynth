import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold
import cvxpy as cp



def l1linf_sc(Y1, Y0, alpha=0.5, lam=1e-4, std=False, intercept=True):
    """
    L1 + L∞ SC, CVXPY version fully matching CVXOPT 'solve_w'.
    """
    Y1 = np.ravel(Y1)
    T, J = Y0.shape

    # Standardize donors if requested
    if std:
        scale = Y0.std(axis=0)
        scale[scale == 0] = 1.0
        Y0_scaled = Y0 / scale
    else:
        Y0_scaled = Y0
        scale = np.ones(J)

    # Minimum lam for numerical stability (match CVXOPT)
    lam = max(lam, np.std(Y1) * 1e-8)

    # Build Y0_plus with intercept as first column (like CVXOPT)
    if intercept:
        Y0_plus = np.hstack([np.ones((T, 1)), Y0_scaled])
        n_w = J + 1  # number of weights including intercept
    else:
        Y0_plus = Y0_scaled
        n_w = J

    # Define CVXPY variables in exact CVXOPT order: [w (incl intercept), u, t]
    w = cp.Variable(n_w)
    u = cp.Variable(J)
    t = cp.Variable()

    # Residual
    residual = Y1 - Y0_plus @ w

    # Objective: squared error + L1/Linf penalty
    objective = cp.Minimize((1/T) * cp.sum_squares(residual) +
                            lam * (alpha * cp.sum(u) + (1 - alpha) * t))

    # Constraints: w-u <=0, -w-u<=0, w-t<=0, -w-t<=0
    # Only apply constraints to donor weights, exclude intercept (first element)
    w_idx = slice(1, n_w) if intercept else slice(0, n_w)
    constraints = [
        w[w_idx] - u <= 0,
        -w[w_idx] - u <= 0,
        w[w_idx] - t <= 0,
        -w[w_idx] - t <= 0
    ]

    # Solve using CVXOPT for exact match
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    # Collect weights
    w_hat = w.value
    if std:
        # Undo scaling for donor weights only
        if intercept:
            w_hat[1:] /= scale
        else:
            w_hat /= scale

    return w_hat

def inf_sc(Y1, Y0, lam=1e-4, std=False, intercept=True, nonneg=True):
    """
    L-infinity penalized SCM using CVXPY with cp.norm(..., "inf").

    Parameters
    ----------
    Y1 : array-like, shape (T,)
        Treated unit pre-treatment outcomes
    Y0 : array-like, shape (T, J)
        Donor units pre-treatment outcomes
    lam : float
        Regularization parameter for L∞ penalty
    std : bool
        Standardize donor columns if True
    intercept : bool
        Include an intercept in the model
    nonneg : bool
        If True, constrain donor weights to be non-negative
    """
    Y1 = np.ravel(Y1)
    T, J = Y0.shape

    # Standardize donors if requested
    if std:
        scale = Y0.std(axis=0)
        scale[scale == 0] = 1.0
        Y0_scaled = Y0 / scale
    else:
        Y0_scaled = Y0
        scale = np.ones(J)

    # Include intercept column if requested
    if intercept:
        Y0_plus = np.hstack([np.ones((T, 1)), Y0_scaled])
        n_w = J + 1
        w_idx = slice(1, n_w)  # only apply L∞ to donor weights
    else:
        Y0_plus = Y0_scaled
        n_w = J
        w_idx = slice(0, n_w)

    # Variables
    w = cp.Variable(n_w)

    # Residuals
    residual = Y1 - Y0_plus @ w

    # Objective: squared error + L∞ penalty on donor weights
    objective = cp.Minimize((1 / T) * cp.sum_squares(residual) + lam * cp.norm(w[w_idx], "inf"))

    # Constraints: optional non-negativity
    constraints = []
    if nonneg:
        constraints.append(w[w_idx] >= 0)

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)  # or SCS/OSQP

    # Undo standardization if needed
    w_hat = w.value
    if std:
        if intercept:
            w_hat[1:] /= scale
        else:
            w_hat /= scale

    return w_hat


def sc_en(Y1, Y0, alpha=0.5, lam=1e-4, std=False, intercept=True, nonneg=True):
    """
    L1 + L2 penalized SCM using CVXPY.

    Parameters
    ----------
    Y1 : array-like, shape (T,)
        Treated unit pre-treatment outcomes
    Y0 : array-like, shape (T, J)
        Donor units pre-treatment outcomes
    alpha : float
        Weight between L1 and L2 penalties (0 ≤ alpha ≤ 1)
    lam : float
        Regularization parameter
    std : bool
        Standardize donor columns if True
    intercept : bool
        Include an intercept in the model
    nonneg : bool
        If True, constrain donor weights to be non-negative
    """
    Y1 = np.ravel(Y1)
    T, J = Y0.shape

    # Standardize donors if requested
    if std:
        scale = Y0.std(axis=0)
        scale[scale == 0] = 1.0
        Y0_scaled = Y0 / scale
    else:
        Y0_scaled = Y0
        scale = np.ones(J)

    # Include intercept if requested
    if intercept:
        Y0_plus = np.hstack([np.ones((T, 1)), Y0_scaled])
        n_w = J + 1
        w_idx = slice(1, n_w)  # L1/L2 penalties apply only to donors
    else:
        Y0_plus = Y0_scaled
        n_w = J
        w_idx = slice(0, n_w)

    # Variables
    w = cp.Variable(n_w)

    # Residuals
    residual = Y1 - Y0_plus @ w

    # Objective: squared error + L2 + L1 penalties
    # L2 on weights (except intercept), L1 on weights (except intercept)
    l2_penalty = lam * (1 - alpha) * cp.sum_squares(w[w_idx])
    l1_penalty = lam * alpha * cp.norm(w[w_idx], 1)
    objective = cp.Minimize((1 / T) * cp.sum_squares(residual) + l2_penalty + l1_penalty)

    # Constraints: optional non-negativity
    constraints = []
    if nonneg:
        constraints.append(w[w_idx] >= 0)

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)  # or SCS/OSQP

    # Undo standardization if needed
    w_hat = w.value
    if std:
        if intercept:
            w_hat[1:] /= scale
        else:
            w_hat /= scale

    return w_hat


def generate_lambda_seq(Y1, Y0, alpha, epsilon=1e-4, num=30):
    """
    Generate lambda sequence exactly like the original authors.
    """
    # Standardize donor matrix
    sY0 = (Y0 - np.mean(Y0, axis=0)) / np.std(Y0, axis=0)
    sY0[:, np.std(Y0, axis=0) == 0] = 0  # avoid divide by zero
    alpha = max(alpha, 0.01)  # numerical stability

    # Compute lambda_max
    lam_max = np.max(np.abs(sY0.T @ Y1)) / (Y0.shape[0] * alpha)
    lam_min = lam_max * epsilon

    lam_max = min(lam_max, 20)
    lam_min = max(lam_min, 1e-4)

    lam_seq = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), num))
    return lam_seq

def ts_cv(Y1_pre, Y0_pre, method_func,
                   n_folds=5, n_repeats=2,
                   std=False, intercept=True,
                   alphas=None, lam_num=30):
    """
    Time-series aware cross-validation fully matching original authors.
    Automatically generates lambda sequences per alpha.
    """
    T, J = Y0_pre.shape

    # Default alpha grid
    if alphas is None:
        alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    best_sspe = np.inf
    best_result = {}

    for alpha in alphas:
        # Generate lambda grid for this alpha
        lams = generate_lambda_seq(Y1_pre, Y0_pre, alpha, num=lam_num)

        for lam in lams:
            total_sspe = 0

            for r in range(n_repeats):
                # repeated KFold forward-chaining CV
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=r)
                Tau = np.zeros_like(Y1_pre)

                for train_index, test_index in kf.split(Y0_pre):
                    if method_func.__name__ == 'l1linf_sc':
                        w = method_func(Y1_pre[train_index], Y0_pre[train_index, :],
                                        alpha=alpha, lam=lam, std=std, intercept=intercept)
                    elif method_func.__name__ == 'inf_sc':
                        w = method_func(Y1_pre[train_index], Y0_pre[train_index, :],
                                        lam=lam, std=std, intercept=intercept, nonneg=True)
                    elif method_func.__name__ == 'sc_en':
                        w = method_func(Y1_pre[train_index], Y0_pre[train_index, :],
                                        alpha=alpha, lam=lam, std=std, intercept=intercept, nonneg=True)
                    else:
                        raise ValueError(f"Unknown method: {method_func.__name__}")

                    # Compute residuals
                    if intercept:
                        Y0_plus = np.hstack([np.ones((T, 1)), Y0_pre])
                        Tau[test_index] = Y1_pre[test_index] - Y0_plus[test_index] @ w
                    else:
                        Tau[test_index] = Y1_pre[test_index] - Y0_pre[test_index] @ w

                total_sspe += np.sum(np.square(Tau))

            average_sspe = total_sspe / n_repeats

            if average_sspe < best_sspe:
                best_sspe = average_sspe
                best_result = {
                    "alpha": alpha if 'alpha' in method_func.__code__.co_varnames else None,
                    "lambda": lam,
                    "weights": w,
                    "cv_score": best_sspe
                }

    return best_result


