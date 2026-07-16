from mlsynth.utils.estutils import Opt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from joblib import Parallel, delayed
from typing import Optional
import cvxpy as cp
from scipy.stats import norm
from itertools import combinations
import pandas as pd
from typing import List, Dict, Set
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro as npy
from numpyro import distributions as npy_dist
from numpyro.infer import MCMC, NUTS, Predictive
from typing import Optional, Dict
import pickle


def fit_scopt_cv(Y0, y, T0, donor_names=None, p_list=[1,2], lam_grid=None, include_ols=True,
                 initial_train=None, step=1):
    """
    Fit synthetic control using Opt.SCopt with automatic lambda selection via rolling-origin CV.

    Penalized models (Lasso/Ridge) are implemented by using scm_model_type="OLS" and setting
    p=1 (Lasso) or p=2 (Ridge) with a lambda_penalty.

    Parameters
    ----------
    Y0 : np.ndarray
        Donor matrix (T x N)
    y : np.ndarray
        Treated unit outcomes (T,)
    T0 : int
        Number of pre-treatment periods
    donor_names : list or pandas Index, optional
        Donor unit names
    p_list : list of int
        Norms to use (1=Lasso, 2=Ridge)
    lam_grid : list or np.ndarray, optional
        Grid of lambda values to test for each penalized model
    include_ols : bool
        Whether to include OLS
    initial_train : int, optional
        Size of the first training set for rolling CV (default: 1/3 of T0)
    step : int
        Step size to advance training window in rolling CV

    Returns
    -------
    dict
        Dictionary with keys:
        - weights: {label: np.ndarray}
        - y_hat: {label: np.ndarray}
        - donor_weights: {label: dict of donor_name -> weight}
        - pre_rmse: {label: float}
        - best_lambda: {method: λ}
    """
    results = {"weights": {}, "y_hat": {}, "donor_weights": {}, "pre_rmse": {}, "best_lambda": {}}
    Y0_pre, y_pre = Y0[:T0, :], y[:T0]

    if initial_train is None:
        initial_train = max(1, T0 // 4)
    if lam_grid is None:
        lam_grid = np.linspace(0.2, 1.0, 50)

    # Helper to compute RMSE
    def _rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    # Helper to map weights to donor names safely
    def map_weights(w, names):
        return dict(zip(list(names), w)) if names is not None else None

    # --- OLS ---
    if include_ols:
        scm = Opt.SCopt(
            num_control_units=Y0.shape[1],
            target_outcomes_pre_treatment=y_pre,
            num_pre_treatment_periods=T0,
            donor_outcomes_pre_treatment=Y0,
            scm_model_type="OLS",
            donor_names=donor_names,
            lambda_penalty=0.0,
            p=2,
            q=2,
        )
        w = scm.solution.primal_vars[next(iter(scm.solution.primal_vars))]
        y_hat = Y0 @ w
        results["weights"]["OLS"] = w
        results["y_hat"]["OLS"] = y_hat
        results["donor_weights"]["OLS"] = map_weights(w, donor_names)
        results["pre_rmse"]["OLS"] = _rmse(y_pre, y_hat[:T0])

    # --- Penalized models with rolling-origin CV ---
    for p in p_list:
        method_name = "LASSO" if p == 1 else "Ridge"
        best_rmse, best_w, best_lam = np.inf, None, None

        for lam in lam_grid:
            rmse_vals = []
            train_start = 0
            train_end = initial_train

            while train_end < T0:
                val_start, val_end = train_end, min(train_end + step, T0)
                Y_train, Y_val = Y0_pre[train_start:train_end, :], Y0_pre[val_start:val_end, :]
                y_train, y_val = y_pre[train_start:train_end], y_pre[val_start:val_end]

                scm = Opt.SCopt(
                    num_control_units=Y0.shape[1],
                    target_outcomes_pre_treatment=y_train,
                    num_pre_treatment_periods=train_end - train_start,
                    donor_outcomes_pre_treatment=Y_train,
                    scm_model_type="OLS",
                    donor_names=donor_names,
                    lambda_penalty=lam,
                    p=p,
                    q=p
                )
                w_val = scm.solution.primal_vars[next(iter(scm.solution.primal_vars))]
                y_val_hat = Y_val @ w_val
                rmse_vals.append(_rmse(y_val, y_val_hat))
                train_end += step

            avg_rmse = np.mean(rmse_vals)
            if avg_rmse < best_rmse:
                best_rmse, best_w, best_lam = avg_rmse, w_val, lam

        # Save best penalized fit
        label = f"{method_name} (lambda={best_lam:.3f})"
        y_hat_full = Y0 @ best_w
        results["weights"][label] = best_w
        results["y_hat"][label] = y_hat_full
        results["donor_weights"][label] = map_weights(best_w, donor_names)
        results["pre_rmse"][label] = _rmse(y_pre, y_hat_full[:T0])
        results["best_lambda"][method_name] = best_lam

    return results





def fit_penalized(Y0_full, y_full, T0, donor_names=None, p_list=[1, 2], lam=0.2, include_ols=True):
    """
    Fit penalized synthetic control on pre-treatment data and predict full counterfactuals.
    Also returns donor weights mapped to names if donor_names is provided,
    and RMSE on the pre-treatment period.

    Parameters
    ----------
    Y0_full : np.ndarray
        Full donor matrix (T x N)
    y_full : np.ndarray
        Full treated vector (T,)
    T0 : int
        Number of pre-treatment periods
    donor_names : list of str, optional
        Names of donor units (length N)
    p_list : list of int
        Norms to use (1 for LASSO, 2 for Ridge)
    lam : float
        Regularization parameter (>0 for penalized)
    include_ols : bool
        Whether to also return OLS weights (p=2, lam=0)

    Returns
    -------
    dict
        Dictionary with keys:
        - "weights": {label: np.ndarray of weights}
        - "y_hat": {label: np.ndarray of counterfactual predictions (full T)}
        - "donor_weights": {label: dict of donor_name -> weight}
        - "pre_rmse": {label: RMSE on pre-treatment period}
    """
    results = {"weights": {}, "y_hat": {}, "donor_weights": {}, "pre_rmse": {}}

    # Slice pre-treatment data
    Y0_pre = Y0_full[:T0, :]
    y_pre = y_full[:T0]

    # Helper to map weights to donor names if provided
    def __map_weights(w):
        if donor_names is None:
            return None
        return dict(zip(donor_names, w))

    # Helper to compute pre-treatment RMSE
    def __compute_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    # --- OLS ---
    if include_ols:
        w_ols = np.linalg.lstsq(Y0_pre, y_pre, rcond=None)[0]
        y_hat_ols = Y0_full @ w_ols
        results["weights"]["OLS"] = w_ols
        results["y_hat"]["OLS"] = y_hat_ols
        results["donor_weights"]["OLS"] = __map_weights(w_ols)
        results["pre_rmse"]["OLS"] = __compute_rmse(y_pre, y_hat_ols[:T0])

    # --- Penalized versions ---
    for p in p_list:
        if p not in [1, 2]:
            raise ValueError("Only p=1 or p=2 supported.")
        w = cp.Variable(Y0_pre.shape[1])
        residual = y_pre - Y0_pre @ w
        penalty = lam * cp.norm1(w) if p == 1 else lam * cp.sum_squares(w)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(residual) + penalty))
        problem.solve(solver=cp.CLARABEL)
        name = "LASSO" if p == 1 else "Ridge"
        w_val = w.value
        y_hat = Y0_full @ w_val
        results["weights"][f"{name} (lambda={lam})"] = w_val
        results["y_hat"][f"{name} (lambda={lam})"] = y_hat
        results["donor_weights"][f"{name} (lambda={lam})"] = __map_weights(w_val)
        results["pre_rmse"][f"{name} (lambda={lam})"] = __compute_rmse(y_pre, y_hat[:T0])

    return results

def plot_placebo(
        y, Y0, T0, time,
        criterion=None,
        threshold=0.1,
        rmse_multiplier=1.5,
        show_rank_test=True
):
    """
    Plot SCM placebo gaps with donor filtering, return rank metrics and
    the matrix of weights (Firpo & Possebom style).

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Outcome vector for the treated unit.
    Y0 : np.ndarray, shape (T, N0)
        Outcome matrix for donor units.
    T0 : int
        Number of pre-treatment periods.
    time : array-like, shape (T,)
        Labels for each time period (e.g., years or months).
    criterion : {'cohen', 'rmse'}, optional
        Metric used to filter donor units. Default is None (all donors retained).
    threshold : float, optional
        Threshold for donor filtering (used for 'cohen').
    rmse_multiplier : float, optional
        Multiplier for RMSE-based donor filtering.
    show_rank_test : bool, optional
        If True, a placebo rank test is plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the subplots.
    placebo_effects_filtered : np.ndarray, shape (T, N_filtered)
        Placebo gap matrix for filtered donors.
    rank_metrics : dict
        Dictionary containing:
        - 'all_ratios': RMSE Post/Pre ratios
        - 'labels': unit labels
        - 'order': ranking indices
        - 'p_values': rank-based p-values
    weights_matrix : np.ndarray, shape (N0, N0+1)
        Each column contains the SCM weights for a unit (first column = treated unit).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mlsynth.utils.estutils import Opt

    N0 = Y0.shape[1]
    T = Y0.shape[0]

    # --- Prepare weights matrix ---
    weights_matrix = np.zeros((N0, N0 + 1))  # first column = treated weights

    # --- Compute synthetic controls for each donor as placebo ---
    synthetic_controls = np.zeros((T, N0))
    placebo_effects = np.zeros((T, N0))

    for j in range(N0):
        mask = np.ones(N0, dtype=bool)
        mask[j] = False
        y_placebo = Y0[:, j]
        Y0_placebo = Y0[:, mask]

        result = Opt.SCopt(
            Y0_placebo.shape[1],
            y_placebo[:T0],
            T0,
            Y0_placebo,
            scm_model_type="SIMPLEX"
        )
        weights = np.array(list(result.solution.primal_vars.values())[0])
        # Fill weights matrix (skip j-th donor)
        weights_full = np.zeros(N0)
        weights_full[mask] = weights
        weights_matrix[:, j + 1] = weights_full  # +1 because 0-th column = treated

        y_synth = Y0_placebo @ weights
        synthetic_controls[:, j] = y_synth
        placebo_effects[:, j] = y_placebo - y_synth

    # --- Compute treated unit gap ---
    treated_result = Opt.SCopt(Y0.shape[1], y[:T0], T0, Y0, scm_model_type="SIMPLEX")
    treated_weights = np.array(list(treated_result.solution.primal_vars.values())[0])
    y_synth_treated = Y0 @ treated_weights
    treated_gap = y - y_synth_treated
    weights_matrix[:, 0] = treated_weights  # first column = treated

    # --- Filtering ---
    Y0_pre = Y0[:T0, :]
    Y_synth_pre = synthetic_controls[:T0, :]

    criterion = criterion.lower() if criterion is not None else None

    if criterion == "cohen":
        Y0_bar = np.mean(Y0_pre, axis=0)
        sigma_s = np.mean((Y0_pre - Y0_bar) ** 2, axis=0)
        D_values = np.mean(np.abs((Y0_pre - Y_synth_pre) / sigma_s), axis=0)
        keep_mask = D_values <= threshold
    elif criterion == "rmse":
        treated_rmse = np.sqrt(np.mean((y[:T0] - y_synth_treated[:T0]) ** 2))
        donor_rmses = np.sqrt(np.mean((Y0_pre - Y_synth_pre) ** 2, axis=0))
        keep_mask = donor_rmses <= rmse_multiplier * treated_rmse
    else:
        keep_mask = np.ones(N0, dtype=bool)

    placebo_effects_filtered = placebo_effects[:, keep_mask]
    filtered_labels = [f"D{i+1}" for i, keep in enumerate(keep_mask) if keep]

    # --- Plot ---
    n_cols = 2 if show_rank_test else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(14, 6))
    if n_cols == 1:
        axes = [axes]

    ax = axes[0]
    for j in range(placebo_effects_filtered.shape[1]):
        ax.plot(time, placebo_effects_filtered[:, j], color="grey", alpha=0.6)
    ax.plot(time, treated_gap, color="black", linewidth=2.5, label="Treated unit gap")
    ax.axvline(x=time[T0 - 1], color="red", linestyle="--", linewidth=1.5, label="Intervention")
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome Gap")
    ax.set_title("Treated Unit vs In-space Placebo Gaps")
    ax.legend()

    # --- Rank test ---
    rank_metrics = None
    if show_rank_test:
        ax = axes[1]
        post_period = slice(T0, T)
        pre_period = slice(0, T0)

        rmse_pre = np.sqrt(np.mean(placebo_effects_filtered[pre_period, :] ** 2, axis=0))
        rmse_post = np.sqrt(np.mean(placebo_effects_filtered[post_period, :] ** 2, axis=0))
        rmse_ratio = rmse_post / rmse_pre

        treated_rmse_pre = np.sqrt(np.mean(treated_gap[pre_period] ** 2))
        treated_rmse_post = np.sqrt(np.mean(treated_gap[post_period] ** 2))
        treated_ratio = treated_rmse_post / treated_rmse_pre

        all_ratios = np.concatenate([[treated_ratio], rmse_ratio])
        labels = ["T"] + filtered_labels
        order = np.argsort(all_ratios)[::-1]
        ranks = np.argsort(np.argsort(-all_ratios)) + 1
        p_values = ranks / len(all_ratios)

        ax.bar(np.array(labels)[order], all_ratios[order],
               color=["black" if i == 0 else "grey" for i in order])
        ax.set_ylabel("RMSE Post / RMSE Pre")
        ax.set_title("Placebo Rank Test")
        ax.set_xticklabels(np.array(labels)[order], rotation=45)

        rank_metrics = {
            "all_ratios": all_ratios,
            "labels": labels,
            "order": order,
            "p_values": p_values
        }

    plt.tight_layout()
    plt.show()

    return {
    "fig": fig,
    "placebo_effects_filtered": placebo_effects_filtered,
    "rank_metrics": rank_metrics,
    "weights_matrix": weights_matrix}


def in_time_placebos(y, Y0, T0, time, lags, scm_model_type="SIMPLEX"):
    """
    Compute in-time placebos for SCM by varying the pre-treatment cutoff.

    Parameters
    ----------
    y : np.ndarray (T,)
        Treated unit outcomes.
    Y0 : np.ndarray (T, N0)
        Donor matrix (all control units).
    T0 : int
        Original pre-treatment period length.
    time : array-like
        Time labels for plotting.
    lags : list[int]
        List of integers specifying how many periods to move the treatment date earlier.
        For example, [5, 10] means we try T0-5 and T0-10 as placebo cutoffs.
    scm_model_type : str
        Model type for Opt.SCopt ("SIMPLEX", "QUADPROG", etc.)

    Returns
    -------
    dict
        Dictionary with keys:
            - "T0_list": list of adjusted pre-periods
            - "placebo_gaps": list of gap arrays (each of length T)
            - "y_synth_list": list of synthetic control series
    """
    placebo_gaps = []
    y_synth_list = []
    T0_list = []

    for lag in lags:
        T0_new = T0 - lag
        if T0_new <= 5:
            print(f"Skipping lag={lag}: not enough pre-treatment periods.")
            continue

        result = Opt.SCopt(
            Y0.shape[1],
            y[:T0_new],
            T0_new,
            Y0,
            scm_model_type=scm_model_type
        )

        weights = list(result.solution.primal_vars.values())[0]
        y_synth = Y0 @ weights
        y_synth_list.append(y_synth)

        tau = y - y_synth
        placebo_gaps.append(tau)
        T0_list.append(T0_new)

    return {
        "T0_list": T0_list,
        "placebo_gaps": placebo_gaps,
        "y_synth_list": y_synth_list
    }


def scm_estimator(y_pre, Y0_pre):
    """
    Wrapper around Opt.SCopt to estimate SCM weights.
    """
    num_donors = Y0_pre.shape[1]
    result = Opt.SCopt(num_donors, y_pre, len(y_pre), Y0_pre, scm_model_type="SIMPLEX")
    weights = list(result.solution.primal_vars.values())[0]
    return np.array(weights)


def subsampling_inference(treated_pre, donors_pre, treated_post, donors_post,
                          m=None, J=1000, alpha=0.05):
    """
    Subsampling-based inference for SCM.

    Parameters
    ----------
    treated_pre : np.array (T1,)
        Pre-treatment outcomes
    donors_pre : np.array (T1, N0)
        Pre-treatment donor outcomes
    treated_post : np.array (T2,)
        Post-treatment outcomes
    donors_post : np.array (T2, N0)
        Post-treatment donor outcomes
    m : int, optional
        Subsample size (default = sqrt(T1))
    J : int
        Number of subsampling iterations
    alpha : float
        Significance level

    Returns
    -------
    delta_hat : float
        Estimated average treatment effect
    ci_lower, ci_upper : float
        Confidence interval bounds
    """

    T1 = treated_pre.shape[0]
    T2 = treated_post.shape[0]
    N0 = donors_pre.shape[1]

    # Default subsample size
    if m is None:
        m = int(np.sqrt(T1))
    if m >= T1 or m <= 1:
        raise ValueError("Invalid subsample size m.")

    # --- Step 1: Fit SCM on full pre-treatment ---
    w_hat = scm_estimator(treated_pre, donors_pre)

    # --- Step 2: Post-treatment counterfactual ---
    y0_post_hat = donors_post @ w_hat
    delta_t_hat = treated_post - y0_post_hat
    delta_hat = np.mean(delta_t_hat)

    # --- Step 3: Residuals and variance estimate ---
    v1t_hat = delta_t_hat - delta_hat
    sigma2_hat = np.mean(v1t_hat ** 2)

    avg_donor_post = np.mean(donors_post, axis=0)

    # --- Step 4: Subsampling ---
    deviations = np.zeros(J)
    for j in range(J):
        # Random subsample from pre-treatment periods
        idx = np.random.choice(T1, size=m, replace=False)
        y_sub = treated_pre[idx]
        Y_sub = donors_pre[idx, :]

        # Refit SCM on subsample
        w_sub = scm_estimator(y_sub, Y_sub)

        # Simulate residual noise
        eps_sub = np.random.normal(0, np.sqrt(sigma2_hat), T2)

        term1 = -np.sqrt(T2 / T1) * (avg_donor_post @ (np.sqrt(m) * (w_sub - w_hat)))
        term2 = np.sum(eps_sub) / np.sqrt(T2)
        deviations[j] = term1 + term2

    # --- Step 5: Construct confidence interval ---
    deviations.sort()
    lower_q = deviations[int(J * (alpha / 2))]
    upper_q = deviations[int(J * (1 - alpha / 2)) - 1]

    scale = 1 / np.sqrt(T2)
    ci_lower = delta_hat - scale * upper_q
    ci_upper = delta_hat - scale * lower_q

    return delta_hat, ci_lower, ci_upper


def subsampling_inference_timewise(treated_pre, donors_pre, treated_post, donors_post,
                                   m=None, J=2000, alpha=0.05):
    """
    Subsampling-based inference for SCM with time-varying uncertainty bands.
    """
    T1 = treated_pre.shape[0]
    T2 = treated_post.shape[0]
    T = T1 + T2
    N0 = donors_pre.shape[1]

    if m is None:
        m = int(np.sqrt(T1))

    # --- Full SCM fit ---
    w_hat = scm_estimator(treated_pre, donors_pre)
    y0_hat = np.concatenate([donors_pre, donors_post]) @ w_hat
    tau_hat = np.concatenate([treated_pre, treated_post]) - y0_hat

    # --- Pre-treatment residual variance ---
    delta_post = treated_post - donors_post @ w_hat
    v_hat = delta_post - np.mean(delta_post)
    sigma2_hat = np.mean(v_hat**2)

    # --- Storage for subsample paths ---
    tau_s_paths = np.zeros((J, T))

    for j in range(J):
        idx = np.random.choice(T1, size=m, replace=False)
        y_sub = treated_pre[idx]
        Y_sub = donors_pre[idx, :]

        w_sub = scm_estimator(y_sub, Y_sub)
        y0_sub_hat = np.concatenate([donors_pre, donors_post]) @ w_sub
        tau_s_paths[j, :] = np.concatenate([treated_pre, treated_post]) - y0_sub_hat

    # --- Compute pointwise quantiles ---
    lower_band = np.percentile(tau_s_paths, 100 * (alpha / 2), axis=0)
    upper_band = np.percentile(tau_s_paths, 100 * (1 - alpha / 2), axis=0)

    results = {
        "tau_hat": tau_hat,
        "lower_band": lower_band,
        "upper_band": upper_band,
        "w_hat": w_hat
    }
    return results



# ---- wrapper to get SCM weights using your Opt.SCopt ----
def scm_weights_from_pre(y_pre, Y_pre, scm_model_type="SIMPLEX"):
    """
    Fit SCM weights using Opt.SCopt on the pre-sample y_pre (length m) and Y_pre (m x N0)
    Returns a 1-D array of weights length N0.
    """
    num_donors = Y_pre.shape[1]
    result = Opt.SCopt(num_donors, y_pre, len(y_pre), Y_pre, scm_model_type=scm_model_type)
    w = list(result.solution.primal_vars.values())[0]
    return np.array(w)

# ---- debiased t-test implementation ----
def debiased_ttest_scm(y, Y0, T0, K=3, scm_model_type="SIMPLEX", alpha=0.05, use_first_K_blocks=True):
    """
    Implements K-fold cross-fit debiased SCM t-test (Chernozhukov, Wuthrich & Zhu).
    Returns: dict with tau_hat, tau_k array, sigma_hat_tau_hat, T_stat, CI_lower, CI_upper.
    """
    T = y.shape[0]
    T1 = T - T0
    if T1 <= 0:
        raise ValueError("No post-treatment periods (T1 <= 0).")

    # determine r as in paper: r = min(floor(T0 / K), T1)
    r = min(T0 // K if K > 0 else T0, T1)
    if r < 1:
        raise ValueError("T0 too small relative to K and T1. Choose smaller K or provide more pre-periods.")

    # Build K consecutive blocks H1..HK over pre-treatment indices 0..T0-1.
    # We'll use the first K blocks by default (paper does this), but if T0 not divisible by K some remainder is left.
    H_blocks = []
    start = 0
    for k in range(K):
        end = start + r
        # for last block, if not enough remaining, include the rest to avoid empty block
        if k == K - 1:
            end = min(T0, end + (T0 - (r * K)))  # absorb remainder in last block
        H_blocks.append(np.arange(start, end))
        start = end

    # Pre-check: ensure none empty
    for i, H in enumerate(H_blocks):
        if H.size == 0:
            raise ValueError(f"Block {i+1} is empty. Reduce K or increase T0.")

    # Precompute full pre/post slices
    pre_idx = np.arange(0, T0)
    post_idx = np.arange(T0, T)

    y_pre = y[:T0]
    y_post = y[T0:]
    donors_pre = Y0[:T0, :]
    donors_post = Y0[T0:, :]

    tau_k = np.zeros(K)

    # For each k: estimate w^(k) on H(-k), compute component estimator (5)
    for k in range(K):
        Hk = H_blocks[k]
        H_minus_k_mask = np.ones(T0, dtype=bool)
        H_minus_k_mask[Hk] = False

        # data used for weight estimation
        y_pre_minus_k = y_pre[H_minus_k_mask]
        donors_pre_minus_k = donors_pre[H_minus_k_mask, :]

        # fit SCM weights on H(-k)
        w_k = scm_weights_from_pre(y_pre_minus_k, donors_pre_minus_k, scm_model_type=scm_model_type)

        # post-term: average over post-treatment of (Y0t - sum_i w_k Yit)
        y0_post_pred_k = donors_post @ w_k
        post_avg = np.mean(y_post - y0_post_pred_k)  # 1/T1 sum_{t in post} (Y0t - sum_i w_k Yit)

        # pre-bias-term: average over Hk of (Y0t - sum_i w_k Yit)
        y0_pre_pred_on_Hk = donors_pre[Hk, :] @ w_k
        pre_avg_Hk = np.mean(y_pre[Hk] - y0_pre_pred_on_Hk)  # 1/|Hk| sum_{t in Hk} ...

        tau_k[k] = post_avg - pre_avg_Hk

    # aggregate
    tau_hat = np.mean(tau_k)

    # compute sigma_hat_tau_hat as in paper:
    # sigma_hat_tau_hat = sqrt( 1 + K*r / T1 ) * sqrt( 1/(K-1) * sum_k (tau_k - tau_hat)^2 )
    # note: paper uses r = min(floor(T0/K), T1) — we used that
    s2 = np.sum((tau_k - tau_hat) ** 2) / max(1, (K - 1))
    sigma_hat_tau_hat = np.sqrt(1.0 + (K * r) / T1) * np.sqrt(s2)

    # t-statistic and CI
    T_stat = (np.sqrt(K) * (tau_hat - 0.0)) / (sigma_hat_tau_hat + 1e-16)  # test H0: tau = 0
    t_crit = t.ppf(1 - alpha / 2, df=K - 1)
    ci_half_width = t_crit * (sigma_hat_tau_hat / np.sqrt(K))
    ci_lower = tau_hat - ci_half_width
    ci_upper = tau_hat + ci_half_width

    return {
        "tau_hat": tau_hat,
        "tau_k": tau_k,
        "sigma_hat_tau_hat": sigma_hat_tau_hat,
        "T_stat": T_stat,
        "df": K - 1,
        "ci": (ci_lower, ci_upper),
        "H_blocks": H_blocks,
        "r": r

    }


def _fit_scm(target_pre, controls_pre):
    """Solve SCM weights with simplex constraints."""
    N_controls = controls_pre.shape[1]
    w = cp.Variable(N_controls, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(target_pre - controls_pre @ w))
    constraints = [cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver failed: {problem.status}")
    return w.value


def _standard_placebo_one(idx, y_pre, donors_pre, y_post, donors_post, pre_rmse_orig, rmse_threshold):
    """Compute R for one standard placebo unit."""
    N0 = donors_pre.shape[1]
    mask = np.ones(N0, dtype=bool)
    mask[idx] = False
    controls_pre = np.column_stack((y_pre, donors_pre[:, mask]))
    controls_post = np.column_stack((y_post, donors_post[:, mask]))
    target_pre = donors_pre[:, idx]
    target_post = donors_post[:, idx]

    try:
        w = _fit_scm(target_pre, controls_pre)
        hat_pre = controls_pre @ w
        hat_post = controls_post @ w
        pre_mse = np.mean((target_pre - hat_pre) ** 2)
        pre_rmse = np.sqrt(pre_mse)
        if pre_rmse > rmse_threshold * pre_rmse_orig:
            return None
        post_mse = np.mean((target_post - hat_post) ** 2)
        post_rmse = np.sqrt(post_mse)
        R = post_rmse / pre_rmse if pre_rmse > 0 else np.inf
        return R
    except:
        return None


def _lto_one_pair(k, l, y_pre, donors_pre, y_post, donors_post, pre_rmse_orig, rmse_threshold):
    """Single LTO computation for placebo test."""
    N0 = donors_pre.shape[1]
    mask = np.ones(N0, dtype=bool)
    mask[[k, l]] = False
    donors_sub_pre = donors_pre[:, mask]
    donors_sub_post = donors_post[:, mask]

    try:
        # Fit for treated unit I
        w_I = _fit_scm(y_pre, donors_sub_pre)
        hat_pre_I = donors_sub_pre @ w_I
        hat_post_I = donors_sub_post @ w_I
        pre_mse_I = np.mean((y_pre - hat_pre_I) ** 2)
        post_mse_I = np.mean((y_post - hat_post_I) ** 2)
        pre_rmspe_I = np.sqrt(pre_mse_I)
        post_rmspe_I = np.sqrt(post_mse_I)
        R_I = post_rmspe_I / pre_rmspe_I if pre_rmspe_I > 0 else np.inf

        # Check pre-fit quality for treated
        if pre_rmspe_I > rmse_threshold * pre_rmse_orig:
            return None

        # Function to compute R and tau_avg for a placebo unit
        def compute_R_placebo(idx):
            target_pre = donors_pre[:, idx]
            try:
                w_p = _fit_scm(target_pre, donors_sub_pre)
            except:
                return np.inf, np.nan  # Treat as poor fit
            hat_pre_p = donors_sub_pre @ w_p
            target_post = donors_post[:, idx]
            hat_post_p = donors_sub_post @ w_p
            pre_mse_p = np.mean((target_pre - hat_pre_p) ** 2)
            post_mse_p = np.mean((target_post - hat_post_p) ** 2)
            pre_rmspe_p = np.sqrt(pre_mse_p)
            post_rmspe_p = np.sqrt(post_mse_p)
            R = post_rmspe_p / pre_rmspe_p if pre_rmspe_p > 0 else np.inf
            tau = target_post - hat_post_p
            tau_avg = np.mean(tau)
            return R, tau_avg

        R_k, tau_avg_k = compute_R_placebo(k)
        R_l, tau_avg_l = compute_R_placebo(l)
        RLTO = max(R_k, R_l)
        not_win = 1 if R_I <= RLTO else 0
        return dict(not_win=not_win, pre_rmse=pre_rmspe_I, tau_avg_k=tau_avg_k, tau_avg_l=tau_avg_l)
    except:
        return None


def _f(N, a):
    """Compute f(N, a) from the paper."""
    if a >= 2 / 3:
        raise ValueError("a must be less than 2/3")
    term1 = 9 * (1 - 1 / N) ** 2
    term2 = 12 * (-4 / 3 / N ** 2 + 1 / N + a * (1 - 1 / N) * (1 - 2 / N))
    sqrt_part = np.sqrt(term1 - term2)
    return (3 - 3 / N - sqrt_part) / 2


def _compute_c(N, alpha, tol=1e-10):
    """Compute c(N, alpha) using binary search."""
    f_alpha = _f(N, alpha)
    floor_Nf = np.floor(N * f_alpha)
    target = (floor_Nf + 1) / N

    low = 0
    high = 2 / 3 - alpha - tol  # Upper limit for alpha + c < 2/3
    while high - low > tol:
        mid = (low + high) / 2
        if _f(N, alpha + mid) >= target:
            high = mid
        else:
            low = mid
    return high


def lto_scm_inference(
        y_pre, donors_pre, y_post, donors_post,
        alpha=0.05, rmse_threshold=2.0, n_jobs=1, powered=False
):
    """
    Leave-Two-Out placebo test for SCM (Lei & Sudijono, 2025).
    """
    N0 = donors_pre.shape[1]
    N = N0 + 1  # Total units

    # --- Baseline SCM ---
    w_orig = _fit_scm(y_pre, donors_pre)
    y_hat_pre = donors_pre @ w_orig
    y_hat_post = donors_post @ w_orig

    pre_rmse_orig = np.sqrt(np.mean((y_pre - y_hat_pre) ** 2))
    post_rmse_orig = np.sqrt(np.mean((y_post - y_hat_post) ** 2))
    R_I = post_rmse_orig / pre_rmse_orig if pre_rmse_orig > 0 else np.inf
    tau_orig = y_post - y_hat_post
    tau_orig_avg = np.mean(tau_orig)

    # --- Standard Placebo for papp and pexact ---
    placebo_results = Parallel(n_jobs=n_jobs)(
        delayed(_standard_placebo_one)(idx, y_pre, donors_pre, y_post, donors_post, pre_rmse_orig, rmse_threshold)
        for idx in range(N0)
    )
    valid_R_placebos = [r for r in placebo_results if r is not None]
    if valid_R_placebos:
        R_placebos_arr = np.array(valid_R_placebos)
        num_ge = np.sum(R_placebos_arr >= R_I)
    else:
        num_ge = 0
    M = len(valid_R_placebos)
    N_eff = M + 1
    p_exact_placebo = (num_ge + 1) / N_eff if N_eff > 0 else 1.0
    p_app_placebo = num_ge / N_eff if N_eff > 0 else 1.0

    # --- All LTO combinations ---
    pairs = list(combinations(range(N0), 2))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_lto_one_pair)(k, l, y_pre, donors_pre, y_post, donors_post, pre_rmse_orig, rmse_threshold)
        for k, l in pairs
    )
    results = [r for r in results if r is not None]

    if len(results) == 0:
        return {
            "tau_orig": tau_orig,
            "p_value": 1.0,
            "num_lto": 0,
            "ci": (np.nan, np.nan),
            "p_app_placebo": 1.0,
            "p_exact_placebo": 1.0,
            "p_naive_lto": 1.0,
            "p_powered_lto": 1.0
        }

    # --- Filter on pre-fit quality (optional, as per original code) ---
    valid = [r for r in results if r["pre_rmse"] <= rmse_threshold * pre_rmse_orig]
    if len(valid) == 0:
        return {
            "tau_orig": tau_orig,
            "p_value": 1.0,
            "num_lto": 0,
            "ci": (np.nan, np.nan),
            "p_app_placebo": p_app_placebo,
            "p_exact_placebo": p_exact_placebo,
            "p_naive_lto": 1.0,
            "p_powered_lto": 1.0
        }

    not_wins = np.array([r["not_win"] for r in valid])
    p_naive = np.mean(not_wins)
    num_lto = len(valid)
    c = _compute_c(N, alpha)
    p_powered = p_naive - c + 1e-10
    if powered:
        p_value = p_powered
    else:
        p_value = p_naive

    # --- Collect placebo tau_avg for CI ---
    tau_lto_list = []
    for r in valid:
        if not np.isnan(r["tau_avg_k"]):
            tau_lto_list.append(r["tau_avg_k"])
        if not np.isnan(r["tau_avg_l"]):
            tau_lto_list.append(r["tau_avg_l"])

    tau_lto = np.array(tau_lto_list)
    if len(tau_lto) > 0:
        ci_lower = np.quantile(tau_lto, alpha / 2)
        ci_upper = np.quantile(tau_lto, 1 - alpha / 2)
    else:
        ci_lower, ci_upper = np.nan, np.nan

    return {
        "tau_orig": tau_orig,
        "tau_orig_avg": tau_orig_avg,
        "pre_rmse_orig": pre_rmse_orig,
        "num_lto": num_lto,
        "p_value": p_value,
        "ci": (ci_lower, ci_upper),
        "tau_lto": tau_lto,
        "p_app_placebo": p_app_placebo,
        "p_exact_placebo": p_exact_placebo,
        "p_naive_lto": p_naive,
        "p_powered_lto": p_powered
    }

def block_permutation_inference(
    Y_treated, Y_control, T0, alpha=0.05, max_combinations=1000,
    acf_threshold=0.1, random_state=None
):
    """
    Compute synthetic control inference using block permutation.

    Parameters
    ----------
    Y_treated : np.ndarray
        Treated unit outcomes (1D array, length T).
    Y_control : np.ndarray
        Synthetic control outcomes (1D array, length T).
    T0 : int
        Index separating pre-treatment (0..T0-1) from post-treatment (T0..T-1).
    alpha : float
        Significance level for split-conformal confidence intervals.
    max_combinations : int
        Number of permutations to run.
    acf_threshold : float
        Threshold for autocorrelation to determine block size.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing:
        - treated_effects: post-treatment treatment effects
        - fulltreated_effects: all-period treatment effects
        - S_obs: observed global statistic
        - global_p_value: p-value from block permutation
        - per_period_pvals: per-period p-values
        - CI: pointwise confidence intervals
        - global_CI: global confidence interval (constant band)
        - block_size: estimated block size
    """
    rng = np.random.default_rng(random_state)
    T = len(Y_treated)

    # --- Compute residuals and treatment effects ---
    residuals = Y_treated[:T0] - Y_control[:T0]
    treated_effects = Y_treated[T0:] - Y_control[T0:]
    fulltreated_effects = Y_treated - Y_control

    # --- Estimate block size from autocorrelation ---
    def estimate_block_size(residuals, threshold):
        n = len(residuals)
        mean_resid = np.mean(residuals)
        var_resid = np.var(residuals)
        for lag in range(1, n):
            cov = np.mean((residuals[:n - lag] - mean_resid) * (residuals[lag:] - mean_resid))
            if abs(cov / var_resid) < threshold:
                return lag
        return n // 2

    block_size = estimate_block_size(residuals, acf_threshold)

    # --- Block permutation test ---
    n_blocks = len(residuals) // block_size
    block_indices = np.arange(n_blocks)
    T_post = len(treated_effects)
    S_obs = np.mean(np.abs(treated_effects))

    S_perm = []
    for _ in range(max_combinations):
        sampled_blocks = rng.choice(block_indices, size=(T_post // block_size + 1), replace=True)
        permuted = np.concatenate([
            residuals[b * block_size:(b + 1) * block_size] for b in sampled_blocks
        ])[:T_post]
        S_perm.append(np.mean(np.abs(permuted)))
    S_perm = np.array(S_perm)

    global_p_value = np.mean(S_perm >= S_obs)

    # --- Per-period p-values ---
    per_period_pvals = np.mean(
        np.abs(residuals)[:, None] >= np.abs(treated_effects)[None, :],
        axis=0
    )

    # --- Pointwise CIs (split-conformal) ---
    q = np.quantile(np.abs(residuals), 1 - alpha)
    interval = np.column_stack([treated_effects - q, treated_effects + q])
    CI = np.vstack([np.full((T0, 2), np.nan), interval])

    # --- Global CI (scalar tuple) ---
    delta = np.quantile(S_perm, 1 - alpha) - S_obs
    global_CI = (np.mean(treated_effects) - delta, np.mean(treated_effects) + delta)

    return {
        "treated_effects": treated_effects,
        "fulltreated_effects": fulltreated_effects,
        "S_obs": S_obs,
        "global_p_value": global_p_value,
        "per_period_pvals": per_period_pvals,
        "CI": CI,
        "global_CI": global_CI,
        "block_size": block_size

    }




def preprocess_hotel_prices(url: str, drop_date: str = '2017-08-21') -> pd.DataFrame:
    """
    Load and preprocess hotel prices data.

    Args:
        url (str): URL to the CSV data.
        drop_date (str): Last date to include in the dataset (format 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns ['id', 'Date', 'fullname', 'mediterranean',
                                                          'Hotel Prices', 'Moratorium'].
    """
    # Load the data
    df = pd.read_csv(url, delimiter=',', thousands='.', decimal=',')
    df.columns = df.columns.str.lower()

    # Convert 'yyyy_mm_dd' column to datetime
    df['yrweek'] = pd.to_datetime(df['yyyy_mm_dd'])

    # Extract year and week number
    df['year'] = df['yrweek'].dt.year
    df['week'] = df['yrweek'].dt.isocalendar().week

    # Create proper weekly date (Monday of the given week)
    df['yrweek'] = pd.to_datetime(df['year'].astype(str) + df['week'].astype(str) + '1', format='%G%V%u')

    # Filter by drop date
    drop_date = pd.to_datetime(drop_date)
    df = df[df['yrweek'] <= drop_date]

    # Create numeric id per city
    df['id'] = df.groupby(['city_id']).ngroup()

    # Add donor name
    df['name'] = 'Donor'
    df['fullname'] = df['name'].map(str) + ' ' + df['id'].map(str)
    df.loc[df["is_barcelona"] == 1, "fullname"] = "Barcelona"

    # Calculate the average indexed_price by id and week
    df = df.groupby(['id', 'yrweek', 'fullname', 'mediterranean'])['indexed_price'].mean().reset_index()
    df = df.sort_values(by=['id', 'yrweek'])

    # Define Moratorium indicator
    moratorium_start = pd.to_datetime("2015-07-06")
    df['Moratorium'] = ((df['fullname'] == "Barcelona") & (df['yrweek'] > moratorium_start)).astype(int)

    # Rename columns
    df.rename(columns={'indexed_price': 'Hotel Prices', 'yrweek': "Date"}, inplace=True)

    return df


import pandas as pd
import numpy as np
from IPython.display import display
from mlsynth import FSCM, FDID
from mlsynth.utils.estutils import Opt
from mlsynth.utils.datautils import dataprep


def run_barcelona_analysis(df_full, treat="Moratorium", outcome="Hotel Prices",
                            unitid="fullname", time="Date"):
    """
    Runs SCM, FDID, and FSCM on full and Mediterranean donor pools,
    returns a summary DataFrame with ATT, R2, and T0 RMSE, and displays it.
    """

    donor_pools = {
        "Full Donor Results": df_full,
        "Mediterranean Donor Results": df_full[df_full["mediterranean"] == 1]
    }

    results = {}

    for pool_name, pool_df in donor_pools.items():
        prepped_data = dataprep(pool_df, unitid, time, outcome, treat)

        y = prepped_data["y"]
        Y0 = prepped_data["donor_matrix"]
        T0 = prepped_data["pre_periods"]
        donor_names = prepped_data["donor_names"]

        # SCM / Simplex
        simplex_solution = Opt.SCopt(
            num_control_units=Y0.shape[1],
            target_outcomes_pre_treatment=y[:T0],
            num_pre_treatment_periods=T0,
            donor_outcomes_pre_treatment=Y0,
            scm_model_type="SIMPLEX",
            donor_names=donor_names,
        )
        # Extract simplex weights
        simplex_weights = np.array(list(simplex_solution.solution.primal_vars.values())[0]).flatten()

        # Store all models
        results[pool_name] = {
            "SCM": simplex_solution,
            "FDID": FDID({
                "df": pool_df,
                "treat": treat,
                "time": time,
                "outcome": outcome,
                "unitid": unitid,
                "display_graphs": False,
                "counterfactual_color": ["blue"]
            }).fit(),
            "FSCM": FSCM({
                "df": pool_df,
                "treat": treat,
                "time": time,
                "outcome": outcome,
                "unitid": unitid,
                "display_graphs": False,
                "counterfactual_color": ["blue"]
            }).fit()
        }

    # Extract summary stats
    summary_stats = {}
    for pool_name in donor_pools.keys():
        pool_results = results[pool_name]

        fdid_obj = pool_results["FDID"][0]  # FDID returns a list
        fdid_effects = fdid_obj.raw_results["Effects"]
        fdid_fit = fdid_obj.raw_results["Fit"]
        fdid_weights = fdid_obj.raw_results["Weights"]

        fscm_obj = pool_results["FSCM"]
        fscm_effects = fscm_obj.raw_results["Effects"]
        fscm_fit = fscm_obj.raw_results["Fit"]
        fscm_weights = fscm_obj.raw_results["Weights"][0]
        fscm_cardinality = fscm_obj.raw_results["Weights"][1]

        summary_stats[pool_name] = {
            "FDID": {
                "ATT": fdid_effects["ATT"],
                "R2": fdid_fit["R-Squared"],
                "T0_RMSE": fdid_fit["T0 RMSE"],
                "Donors": fdid_weights
            },
            "FSCM": {
                "ATT": fscm_effects["ATT"],
                "R2": fscm_fit["R-Squared"],
                "T0_RMSE": fscm_fit["T0 RMSE"],
                "Donors": fscm_weights,
                "Total_Donor_Cardinality": fscm_cardinality
            }
        }

    # Build summary DataFrame
    rows = []
    for pool_name, pool_data in summary_stats.items():
        for model_name, model_data in pool_data.items():
            rows.append({
                "Donor Pool": pool_name,
                "Model": model_name,
                "ATT": model_data["ATT"],
                "R2": model_data["R2"],
                "T0_RMSE": model_data["T0_RMSE"]
            })

    summary_df = pd.DataFrame(rows)
    return summary_df




# ============================================================
# BAYESIAN SYNTHETIC CONTROL
# ============================================================



# ============================================================
# BAYESIAN SYNTHETIC CONTROL
# ============================================================


def bayesian_scm(target: jnp.ndarray, donors: jnp.ndarray, T0: int, key_seed: int = 123):
    """
    Fit a Bayesian Synthetic Control Model (SCM) for a treated unit using a
    Dirichlet prior over donor weights. This function is used within the
    donor-screening stage described in:

        O’Riordan, Michael and Gilligan-Lee, Ciarán M. (2025).
        "Spillover Detection for Donor Selection in Synthetic Control Models."
        Journal of Causal Inference, 13(1), 20240036.
        https://doi.org/10.1515/jci-2024-0036

    Model specification:
        y_pre ~ Normal(intercept + donors_pre @ w, σ)
        w ~ Dirichlet(α · 1_K),     α ~ Gamma(0.5, 0.5)
        intercept ~ Normal(0, 10)
        σ ~ HalfNormal(1)

    Posterior predictive draws are returned for both pre- and post-treatment
    periods.

    Parameters
    ----------
    target : (T,) array
        Outcome series for the treated unit.
    donors : (T, K) array
        Outcome matrix for K donor units.
    T0 : int
        Pre-treatment cutoff.
    key_seed : int
        Seed for the JAX PRNGKey.

    Returns
    -------
    trace : dict
        Posterior samples of weights, intercept, and noise.
    ppc_pre : dict
        Posterior predictive distribution for the pre-treatment period.
    ppc_post : dict
        Posterior predictive distribution for the post-treatment period.
    model : callable
        NumPyro model function for further predictive sampling.
    """
    key = jr.PRNGKey(key_seed)

    # Split pre- and post-treatment
    target_pre = target[:T0]
    donors_pre = donors[:T0, :]
    donors_post = donors[T0:, :]

    # Define the model
    def model(control_units, treated_unit=None):
        num_units = jnp.shape(control_units)[1]
        concentration = npy.sample("concentration", npy_dist.Gamma(0.5, 0.5)) * jnp.ones(num_units)
        weights = npy.sample("weights", npy_dist.Dirichlet(concentration=concentration))
        intercept = npy.sample("intercept", npy_dist.Normal(0, 10))
        counterfactual = intercept + jnp.matmul(control_units, weights)
        noise = npy.sample("noise", npy_dist.HalfNormal(scale=1.0))
        with npy.handlers.condition(data={"obs": treated_unit}):
            npy.sample("obs", npy_dist.Normal(counterfactual, noise))

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1500,
        num_chains=1,
        progress_bar=False
    )
    mcmc.run(key, control_units=donors_pre, treated_unit=target_pre)
    trace = mcmc.get_samples()

    # Posterior predictive on both pre and post
    predictive = Predictive(model, trace)
    ppc_pre = predictive(key, control_units=donors_pre, treated_unit=None)
    ppc_post = predictive(key, control_units=donors_post, treated_unit=None)

    return trace, ppc_pre, ppc_post, model


def spotify_donor_screen(
        donors,
        y,
        T0,
        k=5,
        threshold=0.8,
        lambda_decay=0.5,
        key=jr.PRNGKey(0),
        donor_names=None,
        bayesian_scm_func=bayesian_scm,
        save=False,
        save_path="scscm_screened_results.pkl"
):
    """
    Bayesian donor screening using forward predictive validation.

    This function implements the Bayesian donor-screening method used by
    Spotify (O’Riordan & Gilligan-Lee, 2025). For each donor, a Bayesian
    SCM is fit in which that donor is treated as the pseudo-treated unit.
    The model generates a k-step-ahead posterior predictive distribution,
    and the donor is retained only if the realized post-treatment values
    fall inside the predictive credible intervals with sufficient
    exponentially-weighted frequency.

    The function returns all objects needed for subsequent proximal
    debiasing (as in Equation (5) of O’Riordan & Gilligan-Lee, 2025),
    including partitions of included donors (X_inc) and excluded donors
    (Z_exc), the Bayesian SCM fit on the final set of clean donors, and
    posterior predictive summaries.

    Parameters
    ----------
    donors : array_like, shape (T, D)
        Matrix of donor unit outcomes over time.
    y : array_like, shape (T,)
        Outcome series of the treated unit.
    T0 : int
        Last pre-treatment time index.
    k : int, optional
        Number of post-treatment periods used for predictive checking.
    threshold : float, optional
        Minimum weighted proportion of post-treatment periods that must
        fall inside the predictive interval to keep a donor.
    lambda_decay : float, optional
        Exponential decay parameter for time weights (more recent k
        periods weighted more heavily).
    key : jax.random.PRNGKey, optional
        Base PRNG key for all model fits.
    donor_names : list of str, optional
        Optional list of donor labels. Auto-generated if None.
    bayesian_scm_func : callable, optional
        Function implementing the Bayesian SCM fit (default: bayesian_scm).
    save : bool, optional
        If True, save a pickle containing all screening outputs.
    save_path : str, optional
        Path to save the pickle file.

    Returns
    -------
    dict
        Dictionary containing:
        - Clean Donors: retained donors and their indices.
        - Excluded Donors: screened-out donors and indices.
        - Partitions: y_pre, y_post, X_inc_pre/post, Z_exc_pre/post.
        - SCM: posterior predictive means and 95% intervals for the
          Bayesian SCM fit using only the retained donors.

    References
    ----------
    O’Riordan, M., & Gilligan-Lee, C. M. (2025).
        *Spillover detection for donor selection in synthetic control models*.
        Journal of Causal Inference, 13(1), 20240036.
        https://doi.org/10.1515/jci-2024-0036
    """

    T, D = donors.shape
    assert T0 + k <= T, "Not enough post-treatment periods for k"
    if donor_names is None:
        donor_names = [f"Unit{i}" for i in range(D)]

    valid = jnp.ones(D, dtype=bool)

    # Exponential-decay weights
    raw_weights = jnp.exp(-lambda_decay * jnp.arange(k))
    weights = raw_weights / raw_weights.sum()

    # -------------------------------------------------------------
    # DONOR SCREENING
    # -------------------------------------------------------------
    for i in range(D):
        key, subkey = jr.split(key)

        fake_target = donors[:, i]
        fake_donors = jnp.delete(donors, i, axis=1)

        trace, _, _, model = bayesian_scm_func(
            target=fake_target,
            donors=fake_donors,
            T0=T0,
            key_seed=int(jr.randint(subkey, (), 0, 1_000_000))
        )

        predictive = Predictive(model, trace)

        post_pred = predictive(
            subkey,
            control_units=fake_donors[T0:T0+k, :],
            treated_unit=None
        )["obs"]

        actual_vec = fake_target[T0:T0+k]

        lo = jnp.quantile(post_pred, (1 - threshold) / 2, axis=0)
        hi = jnp.quantile(post_pred, 1 - (1 - threshold) / 2, axis=0)

        inside = (actual_vec >= lo) & (actual_vec <= hi)

        weighted_score = jnp.sum(weights * inside)

        valid_i = weighted_score >= threshold
        valid = valid.at[i].set(valid_i)

        print(
            f"{donor_names[i]:<20} | "
            f"score={weighted_score:.3f} | "
            f"{'KEPT' if valid_i else 'EXCLUDED'}"
        )

    kept_indices = [i for i in range(D) if bool(valid[i])]
    excluded_indices = [i for i in range(D) if not bool(valid[i])]

    kept_names = [donor_names[i] for i in kept_indices]
    excluded_names = [donor_names[i] for i in excluded_indices]

    X_inc = donors[:, kept_indices]
    Z_exc = donors[:, excluded_indices] if len(excluded_indices) > 0 else None

    # Partition everything pre/post for debiasing
    y_pre = y[:T0]
    y_post = y[T0:]

    X_inc_pre = X_inc[:T0, :]
    X_inc_post = X_inc[T0:, :]

    if Z_exc is not None:
        Z_exc_pre = Z_exc[:T0, :]
        Z_exc_post = Z_exc[T0:, :]
    else:
        Z_exc_pre = Z_exc_post = None

    # -------------------------------------------------------------
    # FIT SCM USING ONLY KEPT DONORS
    # -------------------------------------------------------------
    sc_trace, sc_ppc_pre, sc_ppc_post, sc_model = bayesian_scm_func(
        target=y,
        donors=X_inc,
        T0=T0,
        key_seed=int(jr.randint(key, (), 0, 500_000))
    )

    in_sample = sc_ppc_pre["obs"] if "obs" in sc_ppc_pre else sc_ppc_pre
    out_sample = sc_ppc_post["obs"] if "obs" in sc_ppc_post else sc_ppc_post

    in_mean = jnp.mean(in_sample, axis=0)
    in_lower = jnp.percentile(in_sample, 2.5, axis=0)
    in_upper = jnp.percentile(in_sample, 97.5, axis=0)

    out_mean = jnp.mean(out_sample, axis=0)
    out_lower = jnp.percentile(out_sample, 2.5, axis=0)
    out_upper = jnp.percentile(out_sample, 97.5, axis=0)

    results = {
        "Clean Donors": {
            "names": kept_names,
            "indices": kept_indices,
            "matrix": X_inc,
        },
        "Excluded Donors": {
            "names": excluded_names,
            "indices": excluded_indices,
            "matrix": Z_exc,
        },
        "Partitions": {
            "y_pre": y_pre,
            "y_post": y_post,
            "X_inc_pre": X_inc_pre,
            "X_inc_post": X_inc_post,
            "Z_exc_pre": Z_exc_pre,
            "Z_exc_post": Z_exc_post,
        },
        "SCM": {
            "trace": sc_trace,
            "ppc_pre": sc_ppc_pre,
            "ppc_post": sc_ppc_post,
            "model": sc_model,
            "in_mean": in_mean,
            "in_lower": in_lower,
            "in_upper": in_upper,
            "out_mean": out_mean,
            "out_lower": out_lower,
            "out_upper": out_upper,
        }
    }

    # -------------------------------------------------------------
    # SAVE OPTION
    # -------------------------------------------------------------
    if save:
        safe_results = results.copy()
        safe_results["SCM"] = {
            "in_mean": in_mean,
            "in_lower": in_lower,
            "in_upper": in_upper,
            "out_mean": out_mean,
            "out_lower": out_lower,
            "out_upper": out_upper,
            # optionally: convert samples to CPU numpy
            "in_samples": np.array(in_sample),
            "out_samples": np.array(out_sample),
        }

        # Remove non-pickleable objects
        # (models, traces, predictive kernels)
        for key_ in ["trace", "ppc_pre", "ppc_post", "model"]:
            if key_ in results["SCM"]:
                del results["SCM"][key_]

        with open(save_path, "wb") as f:
            pickle.dump(safe_results, f)

    return results



def bscm_proximal(
        y_pre: jnp.ndarray,
        y_post: jnp.ndarray,
        X_donors_pre: jnp.ndarray,  # included ("clean") donors
        X_donors_post: jnp.ndarray,
        Z_proxies_pre: jnp.ndarray,  # excluded ("spillover") donors → proxies
        Z_proxies_post: jnp.ndarray,
        concentration: float = 0.4,
        num_warmup: int = 1000,
        num_samples: int = 3000,
        num_chains: int = 4,
        seed: int = 0,
) -> Dict[str, jnp.ndarray]:
    """
    Bayesian Proximal Synthetic Control.

    Implements Equation (5) of O’Riordan & Gilligan-Lee (2025), which
    combines (i) a Bayesian synthetic control outcome model with Dirichlet
    weights and (ii) a proximal debiasing model using excluded donors as
    proxies. All estimation is performed in z-scored space, and posterior
    counterfactuals are returned on the original outcome scale.

    Parameters
    ----------
    y_pre : jnp.ndarray
        Treated unit's pre-treatment outcomes (T0,).
    y_post : jnp.ndarray
        Treated unit's post-treatment outcomes (T1,).
    X_donors_pre : jnp.ndarray
        Included donor outcomes, pre-treatment (T0, J).
    X_donors_post : jnp.ndarray
        Included donor outcomes, post-treatment (T1, J).
    Z_proxies_pre : jnp.ndarray
        Excluded donor outcomes (proxies), pre-treatment (T0, K).
    Z_proxies_post : jnp.ndarray
        Excluded donor outcomes, post-treatment (T1, K).
    concentration : float
        Dirichlet concentration for donor weight prior.
    num_warmup : int
        Number of NUTS warmup iterations.
    num_samples : int
        Number of posterior draws per chain.
    num_chains : int
        Number of MCMC chains.
    seed : int
        Random seed for NumPyro.

    Returns
    -------
    dict
        Dictionary containing:
        - counterfactual_mean : posterior mean counterfactual path.
        - counterfactual_lower/upper : 95% credible interval.
        - treatment_effect_mean : posterior mean treatment effect.
        - treatment_effect_lower/upper : treatment effect CI.
        - counterfactual_samples : all posterior draws.
        - weights : posterior draws of donor weights β.
        - posterior_samples : all MCMC samples from NumPyro.

    References
    ----------
    O’Riordan, M., & Gilligan-Lee, C. M. (2025).
        *Spillover detection for donor selection in synthetic control models*.
        Journal of Causal Inference, 13(1), 20240036.
        https://doi.org/10.1515/jci-2024-0036
    """

    T0 = y_pre.shape[0]
    J = X_donors_pre.shape[1]  # number of clean donors
    K = Z_proxies_pre.shape[1]  # number of proxy donors

    # =================================================================
    # 1. Z-SCORE STANDARDIZATION (pre-treatment only)
    # =================================================================
    y_mean = y_pre.mean()
    y_std = y_pre.std() + 1e-8
    y_pre_std = (y_pre - y_mean) / y_std
    y_post_std = (y_post - y_mean) / y_std

    X_mean = X_donors_pre.mean(axis=0, keepdims=True)
    X_std = X_donors_pre.std(axis=0, keepdims=True) + 1e-8
    X_pre_std = (X_donors_pre - X_mean) / X_std
    X_post_std = (X_donors_post - X_mean) / X_std

    Z_mean = Z_proxies_pre.mean(axis=0, keepdims=True)
    Z_std = Z_proxies_pre.std(axis=0, keepdims=True) + 1e-8
    Z_pre_std = (Z_proxies_pre - Z_mean) / Z_std
    Z_post_std = (Z_proxies_post - Z_mean) / Z_std

    # =================================================================
    # 2. NUMPYRO MODEL (all in standardized space)
    # =================================================================
    def model():
        # Outcome model
        alpha = npy.sample("alpha", npy_dist.Normal(0, 10))
        beta = npy.sample("beta", npy_dist.Dirichlet(jnp.full(J, concentration)))
        sigma_y = npy.sample("sigma_y", npy_dist.HalfNormal(5.0))

        mu_y = alpha + X_pre_std @ beta
        npy.sample("y_obs", npy_dist.Normal(mu_y, sigma_y), obs=y_pre_std)

        # Proximal debiasing: excluded donors predict included donors
        gamma = npy.sample("gamma", npy_dist.Normal(0, 10).expand([J]))
        lam = npy.sample("lam", npy_dist.Normal(0, 10).expand([K, J]))
        sigma_x = npy.sample("sigma_x", npy_dist.HalfNormal(5.0).expand([J]))

        mu_x = gamma + Z_pre_std @ lam
        npy.sample("X_obs", npy_dist.Normal(mu_x, sigma_x), obs=X_pre_std)

        # Counterfactual (standardized → original scale)
        cf_pre_std = alpha + X_pre_std @ beta
        cf_post_std = alpha + X_post_std @ beta

        npy.deterministic("cf_pre", cf_pre_std * y_std + y_mean)
        npy.deterministic("cf_post", cf_post_std * y_std + y_mean)
        npy.deterministic("treatment_effect", y_post - (cf_post_std * y_std + y_mean))
        npy.deterministic("cf_full", jnp.concatenate([cf_pre_std * y_std + y_mean,
                                                      cf_post_std * y_std + y_mean]))

    # =================================================================
    # 3. MCMC
    # =================================================================
    rng_key = jr.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()

    # =================================================================
    # 4. Results (all in original units)
    # =================================================================
    cf_post = samples["cf_post"]
    te = samples["treatment_effect"]
    cf_full = samples["cf_full"]

    return {
        "counterfactual_mean": cf_full.mean(axis=0),
        "counterfactual_lower": jnp.percentile(cf_full, 2.5, axis=0),
        "counterfactual_upper": jnp.percentile(cf_full, 97.5, axis=0),
        "treatment_effect_mean": samples["treatment_effect"].mean(axis=0),
        "treatment_effect_lower": jnp.percentile(samples["treatment_effect"], 2.5, axis=0),
        "treatment_effect_upper": jnp.percentile(samples["treatment_effect"], 97.5, axis=0),
        "counterfactual_samples": cf_full,
        "weights": samples["beta"],
        "posterior_samples": samples,
    }




import cvxpy as cp
import numpy as np

def scscm_pi_intercept(Y, W, Z0, T0, t1, lag=0):
    """
    Proximal Synthetic Control using only proxies, with convex hull constraint on donor weights,
    and an added intercept.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector of treated unit (shape: T, ).
    W : np.ndarray
        Donor matrix (shape: T, N_donors).
    Z0 : np.ndarray
        Proxy matrix (shape: T, N_proxies).
    T0 : int
        Number of pre-treatment periods.
    t1 : int
        Number of post-treatment periods for averaging tau.
    lag : int, default 0
        HAC lag length for covariance estimation (not used in this version).

    Returns
    -------
    tau : float
        Estimated treatment effect.
    taut : np.ndarray
        Residual vector (Y - W*alpha - intercept).
    alpha : np.ndarray
        Estimated donor weights (convex hull constrained).
    alpha0 : float
        Estimated intercept.
    counterfactual : np.ndarray
        Fitted pre/post-treatment values without post-treatment mean adjustment.
    """
    T, N_donors = W.shape

    # --- Optimization variables ---
    alpha_var = cp.Variable(N_donors)
    alpha0_var = cp.Variable()
    tau_var = cp.Variable(1)

    # --- Pre-treatment moment condition ---
    U0 = Z0[:T0, :].T @ (Y[:T0][:, None] - W[:T0, :] @ alpha_var - alpha0_var)

    # --- Post-treatment residuals ---
    U1 = Y[T0:T0+t1, None] - tau_var - W[T0:T0+t1, :] @ alpha_var - alpha0_var

    # --- Objective: sum of squared moments ---
    objective = cp.Minimize(cp.sum_squares(U0) + cp.sum_squares(U1))

    # --- Constraints ---
    constraints = [alpha_var >= 0, cp.sum(alpha_var) == 1]

    # --- Solve ---
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    alpha = alpha_var.value
    alpha0 = alpha0_var.value

    # --- Fitted values ---
    counterfactual = W @ alpha + alpha0
    taut = Y - counterfactual
    tau = np.mean(taut[T0:T0+t1])

    return {
        'tau': tau,
        'taut': taut,
        'alpha': alpha,
        'alpha0': alpha0,
        'Proximal_SC': counterfactual
    }


