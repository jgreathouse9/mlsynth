import numpy as np
import cvxpy as cp
from collections import defaultdict
from copy import deepcopy
from scipy import stats


def estimate_cohort_sdid_effects(a: int, data: dict, pooled_tau_ell: dict) -> dict:
    """
    Estimate SDID effects for a specific cohort.

    Args:
        a (int): Adoption period (treatment start time).
        data (dict): Cohort-specific data.
        pooled_tau_ell (dict): Dictionary to store pooled event-time effects.

    Returns:
        dict: Contains effects, actual, counterfactual, ATT, and event-time series.
    """
    y_mat = data["y"]  # (T, Na)
    y_mean = y_mat.mean(axis=1)  # (T,)
    Y0 = data["donor_matrix"]  # (T, N0)
    T = data["total_periods"]
    T0 = data["pre_periods"]
    T_post = data["post_periods"]

    # Estimate weights
    Y0_pre = Y0[:T0, :]
    y_pre = y_mean[:T0]
    zeta = compute_regularization(Y0_pre, T_post)
    _, unit_w = unit_weights(Y0_pre, y_pre, zeta)
    Y0_post_mean = Y0[T0:, :].mean(axis=0)
    _, time_w = fit_time_weights(Y0_pre, Y0_post_mean)

    # Compute bias and synthetic outcome
    bias = (time_w @ y_pre) - (time_w @ (Y0_pre @ unit_w))
    y_synth = Y0 @ unit_w  # (T,)

    # Event-time effects
    ell = np.arange(T) - (a - 1)  # Event time relative to treatment
    tau = y_mean - (y_synth + bias)  # Treatment effects
    mask = ell != 0  # Exclude treatment start period
    ell_valid = ell[mask]
    tau_valid = tau[mask]
    cohort_effects = np.column_stack((ell_valid, tau_valid))

    # Categorize pre- and post-intervention effects
    pre_mask = ell < 0
    post_mask = ell > 0
    pre_effects = np.column_stack((ell[pre_mask], tau[pre_mask])) if np.any(pre_mask) else np.array([])
    post_effects = np.column_stack((ell[post_mask], tau[post_mask])) if np.any(post_mask) else np.array([])

    # Store effects for pooling
    for e, t in zip(ell[post_mask], tau[post_mask]):
        pooled_tau_ell[e].append((len(data["treated_indices"]), t))
    for e, t in zip(ell[pre_mask], tau[pre_mask]):
        pooled_tau_ell[e].append((len(data["treated_indices"]), t))

    # Treatment effects series
    treatment_effects_series = np.zeros(T)
    treatment_effects_series[mask] = tau_valid

    # ATT (post-treatment average)
    post_treatment_diffs = tau[post_mask]
    att = np.mean(post_treatment_diffs) if post_treatment_diffs.size else np.nan

    return {
        "effects": cohort_effects,
        "pre_effects": pre_effects,  # Added: Pre-intervention effects
        "post_effects": post_effects,  # Added: Post-intervention effects
        "actual": y_mean,
        "counterfactual": y_synth,
        "fitted_counterfactual": y_synth + bias,
        "att": att,
        "treatment_effects_series": treatment_effects_series,
        "ell": ell  # Added: Event times for reference
    }

def estimate_event_study_sdid(prepped: dict, placebo_iterations: int = 1000, seed: int = 1400):
    """
    Estimate event-study SDID effects with placebo inference for variance, SE, and 95% CI.

    Args:
        prepped (dict): Preprocessed data with cohort-specific information.
        placebo_iterations (int): Number of placebo resamples (B).
        seed (int): Random seed for reproducibility.

    Returns:
        dict:
            - tau_a_ell: Per-cohort effects, actual, counterfactual, pre/post effects.
            - tau_ell: Pooled effects for all ℓ.
            - cohort_estimates: Per-cohort ATT, SE, CI, and event-time estimates.
            - pooled_estimates: Pooled event-time estimates with SE and CI.
    """
    cohorts = prepped["cohorts"]
    tau_a_ell = {}
    pooled_tau_ell = defaultdict(list)

    # Estimate SDID for actual data
    for a, data in cohorts.items():
        tau_a_ell[a] = estimate_cohort_sdid_effects(a, data, pooled_tau_ell)

    # Aggregate pooled effects
    tau_ell = {
        ell: sum(n * tau for n, tau in values) / sum(n for n, _ in values)
        for ell, values in pooled_tau_ell.items()
    }

    # Compute overall ATT
    total_post_periods = sum(data["post_periods"] for data in cohorts.values())
    att = sum(
        cohorts[a]["post_periods"] / total_post_periods * tau_a_ell[a]["att"]
        for a in cohorts if not np.isnan(tau_a_ell[a]["att"])
    )

    # Perform placebo inference
    placebo_results = estimate_placebo_variance(prepped, placebo_iterations, seed)

    # Prepare cohort-specific estimates
    cohort_estimates = {
        a: {
            "att": tau_a_ell[a]["att"],
            "att_se": np.sqrt(placebo_results["cohort_variances"][a]) if a in placebo_results["cohort_variances"] else np.nan,
            "att_ci": [
                tau_a_ell[a]["att"] - stats.norm.ppf(0.975) * np.sqrt(placebo_results["cohort_variances"][a]),
                tau_a_ell[a]["att"] + stats.norm.ppf(0.975) * np.sqrt(placebo_results["cohort_variances"][a])
            ] if a in placebo_results["cohort_variances"] else [np.nan, np.nan],
            "event_estimates": {
                int(ell): {
                    "tau": tau,
                    "se": np.sqrt(placebo_results["event_variances"][a][ell]) if ell in placebo_results["event_variances"][a] else np.nan,
                    "ci": [
                        tau - stats.norm.ppf(0.975) * np.sqrt(placebo_results["event_variances"][a][ell]),
                        tau + stats.norm.ppf(0.975) * np.sqrt(placebo_results["event_variances"][a][ell])
                    ] if ell in placebo_results["event_variances"][a] else [np.nan, np.nan]
                }
                for ell, tau in zip(tau_a_ell[a]["pre_effects"][:, 0], tau_a_ell[a]["pre_effects"][:, 1])
            } | {
                int(ell): {
                    "tau": tau,
                    "se": np.sqrt(placebo_results["event_variances"][a][ell]) if ell in placebo_results["event_variances"][a] else np.nan,
                    "ci": [
                        tau - stats.norm.ppf(0.975) * np.sqrt(placebo_results["event_variances"][a][ell]),
                        tau + stats.norm.ppf(0.975) * np.sqrt(placebo_results["event_variances"][a][ell])
                    ] if ell in placebo_results["event_variances"][a] else [np.nan, np.nan]
                }
                for ell, tau in zip(tau_a_ell[a]["post_effects"][:, 0], tau_a_ell[a]["post_effects"][:, 1])
            }
        }
        for a in cohorts
    }

    # Prepare pooled event-time estimates
    pooled_estimates = {
        ell: {
            "tau": tau_ell[ell],
            "se": np.sqrt(placebo_results["pooled_event_variances"][ell]) if ell in placebo_results["pooled_event_variances"] else np.nan,
            "ci": [
                tau_ell[ell] - stats.norm.ppf(0.975) * np.sqrt(placebo_results["pooled_event_variances"][ell]),
                tau_ell[ell] + stats.norm.ppf(0.975) * np.sqrt(placebo_results["pooled_event_variances"][ell])
            ] if ell in placebo_results["pooled_event_variances"] else [np.nan, np.nan]
        }
        for ell in tau_ell
    }

    return {
        "tau_a_ell": tau_a_ell,
        "tau_ell": tau_ell,
        "att": att,
        "att_se": np.sqrt(placebo_results["att_variance"]),
        "att_ci": [att - stats.norm.ppf(0.975) * np.sqrt(placebo_results["att_variance"]), att + stats.norm.ppf(0.975) * np.sqrt(placebo_results["att_variance"])],
        "cohort_estimates": cohort_estimates,
        "pooled_estimates": pooled_estimates
    }

def estimate_placebo_variance(prepped: dict, B: int, seed: int) -> dict:
    """
    Estimate variance of ATT and event-time effects using placebo inference.

    Args:
        prepped (dict): Preprocessed data with cohort-specific information.
        B (int): Number of placebo iterations.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Variance estimates for ATT, cohort-specific effects, and event-time effects.
    """
    np.random.seed(seed)
    cohorts = prepped["cohorts"]
    placebo_att_estimates = []
    placebo_cohort_estimates = defaultdict(list)
    placebo_event_estimates = defaultdict(lambda: defaultdict(list))  # a -> ell -> estimates
    placebo_pooled_event_estimates = defaultdict(list)  # ell -> estimates

    # Identify control units (never-treated)
    control_units = []
    for a, data in cohorts.items():
        control_indices = [i for i in range(data["donor_matrix"].shape[1]) if i not in data["treated_indices"]]
        control_units.extend(control_indices)
    control_units = list(set(control_units))  # Unique control units

    if len(control_units) <= sum(len(data["treated_indices"]) for data in cohorts.values()):
        raise ValueError("Placebo inference requires more control units than treated units.")

    for b in range(B):
        # Create placebo treatment structure
        placebo_cohorts = deepcopy(cohorts)
        for a, data in placebo_cohorts.items():
            n_treated = len(data["treated_indices"])
            placebo_treated = np.random.choice(control_units, size=n_treated, replace=False)
            data["treated_indices"] = list(placebo_treated)
            data["y"] = data["donor_matrix"][:, placebo_treated]

        # Estimate SDID for placebo data
        pooled_tau_ell = defaultdict(list)
        cohort_atts = {}
        for a, data in placebo_cohorts.items():
            result = estimate_cohort_sdid_effects(a, data, pooled_tau_ell)
            cohort_atts[a] = result["att"]

            # Collect event-time effects
            for ell, tau in zip(result["pre_effects"][:, 0], result["pre_effects"][:, 1]):
                placebo_event_estimates[a][int(ell)].append(tau)
            for ell, tau in zip(result["post_effects"][:, 0], result["post_effects"][:, 1]):
                placebo_event_estimates[a][int(ell)].append(tau)

        # Compute placebo ATT
        total_post_periods = sum(data["post_periods"] for data in cohorts.values())
        att = sum(
            cohorts[a]["post_periods"] / total_post_periods * cohort_atts[a]
            for a in cohorts if not np.isnan(cohort_atts[a])
        )
        placebo_att_estimates.append(att)

        # Store adoption-specific estimates
        for a in cohort_atts:
            if not np.isnan(cohort_atts[a]):
                placebo_cohort_estimates[a].append(cohort_atts[a])

        # Store pooled event-time estimates
        for ell, values in pooled_tau_ell.items():
            pooled_tau = sum(n * tau for n, tau in values) / sum(n for n, _ in values)
            placebo_pooled_event_estimates[ell].append(pooled_tau)

    # Compute variances
    placebo_att_variance = np.var(placebo_att_estimates, ddof=1)
    cohort_variances = {a: np.var(estimates, ddof=1) for a, estimates in placebo_cohort_estimates.items()}
    event_variances = {
        a: {ell: np.var(estimates, ddof=1) for ell, estimates in ell_dict.items()}
        for a, ell_dict in placebo_event_estimates.items()
    }
    pooled_event_variances = {
        ell: np.var(estimates, ddof=1) for ell, estimates in placebo_pooled_event_estimates.items()
    }

    return {
        "att_variance": placebo_att_variance,
        "cohort_variances": cohort_variances,
        "event_variances": event_variances,
        "pooled_event_variances": pooled_event_variances
    }

def fit_time_weights(Y0: np.ndarray, Y0_post_mean: np.ndarray):
    T0, N = Y0.shape
    beta = cp.Variable()
    w = cp.Variable(T0, nonneg=True)
    prediction = beta + (w.T @ Y0)
    constraints = [cp.sum(w) == 1]
    objective = cp.Minimize(cp.sum_squares(prediction - Y0_post_mean))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    return beta.value, w.value

def compute_regularization(Y0: np.ndarray, n_treated_post: int) -> float:
    lambda_reg = (n_treated_post ** 0.25) * np.std(np.diff(Y0, axis=0).flatten(), ddof=1)
    return lambda_reg

def unit_weights(Y0: np.ndarray, Y1: np.ndarray, zeta: float):
    T0, N = Y0.shape
    beta = cp.Variable()
    w = cp.Variable(N, nonneg=True)
    prediction = beta + Y0 @ w
    penalty = T0 * zeta**2 * cp.sum_squares(w)
    objective = cp.Minimize(cp.sum_squares(prediction - Y1) + penalty)
    constraints = [cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    return beta.value, w.value
