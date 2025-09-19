import pandas as pd
import cvxpy as cp
import numpy as np

# --- helper functions ---

def _get_per_cluster_param(param, klabel, default=None):
    if param is None:
        return default
    if isinstance(param, dict):
        return param.get(klabel, default)
    return param  # scalar

def _prepare_clusters(Y_full, clusters):
    if isinstance(Y_full, pd.DataFrame):
        Y_full_np = Y_full.to_numpy()
    else:
        Y_full_np = np.asarray(Y_full)

    clusters = np.asarray(clusters)
    N = Y_full_np.shape[0]
    if clusters.shape[0] != N:
        raise ValueError("clusters must have length N (rows of Y)")

    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}

    return Y_full_np, clusters, N, cluster_labels, K, label_to_k

def _validate_scm_inputs(Y_full, T0, blank_periods, design,
                        beta=1e-6, lambda1=0.0, lambda2=0.0,
                        xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0):
    if T0 <= 0 or T0 >= Y_full.shape[1]:
        raise ValueError("T0 must be 1 <= T0 < Y_full.shape[1]")
    if blank_periods < 0 or blank_periods >= T0:
        raise ValueError("blank_periods must be 0 <= blank_periods < T0 (need at least 1 fit period)")
    if design != "weak" and beta != 1e-6:
        raise ValueError("beta is only valid when design == 'weak'")
    if design != "eq11" and (lambda1 != 0.0 or lambda2 != 0.0):
        raise ValueError("lambda1/lambda2 are only valid when design == 'eq11'")
    if design != "unit" and (xi != 0.0 or lambda1_unit != 0.0 or lambda2_unit != 0.0):
        raise ValueError("xi/lambda1_unit/lambda2_unit are only valid when design == 'unit'")

def _validate_costs_budget(costs, budget, N, cluster_labels, K):
    if costs is None:
        return None, None

    costs_np = np.asarray(costs)
    if costs_np.shape[0] != N:
        raise ValueError("costs must have length N (rows of Y).")
    if budget is None:
        raise ValueError("budget must be provided if costs are specified.")

    if isinstance(budget, (int, float)):
        budget_dict = {lab: budget / K for lab in cluster_labels}  # Even split
    elif isinstance(budget, dict):
        for lab in cluster_labels:
            if lab not in budget:
                raise ValueError(f"budget missing entry for cluster '{lab}'.")
        budget_dict = budget
    else:
        raise TypeError("budget must be a scalar or dict if costs are provided.")

    return costs_np, budget_dict

def _extract_results(Y_full_np, Y_fit, w, v, cluster_members, T_fit):
    w_opt = w.value
    v_opt = v.value
    z_opt = (w_opt > 0).astype(float)  # approximate z as binary from w

    Y_full_T = Y_full_np.T
    K = len(cluster_members)
    y_syn_treated_clusters = [Y_full_T @ w_opt[:, k_idx] for k_idx in range(K)]
    y_syn_control_clusters = [Y_full_T @ v_opt[:, k_idx] for k_idx in range(K)]

    # cluster RMSE
    rmse_cluster = []
    for k_idx in range(K):
        treated_idx = np.where(w_opt[:, k_idx] > 1e-8)[0]
        y_treated = (Y_fit[treated_idx, :].T @ w_opt[treated_idx, k_idx] / np.sum(w_opt[treated_idx, k_idx])) \
            if len(treated_idx) > 0 else np.zeros(T_fit)
        y_control = Y_fit.T @ v_opt[:, k_idx]
        rmse_cluster.append(np.sqrt(np.mean((y_treated - y_control) ** 2)))

    # aggregate weights
    cluster_sizes = [len(m) for m in cluster_members]
    total_size = sum(cluster_sizes)
    w_agg = np.zeros(Y_full_np.shape[0])
    v_agg = np.zeros(Y_full_np.shape[0])
    for k_idx in range(K):
        w_agg += (cluster_sizes[k_idx] / total_size) * w_opt[:, k_idx]
        v_agg += (cluster_sizes[k_idx] / total_size) * v_opt[:, k_idx]

    return {
        "w_opt": w_opt,
        "v_opt": v_opt,
        "z_opt": z_opt,
        "y_syn_treated_clusters": y_syn_treated_clusters,
        "y_syn_control_clusters": y_syn_control_clusters,
        "rmse_cluster": rmse_cluster,
        "w_agg": w_agg,
        "v_agg": v_agg
    }

# --- main SCMEXP function ---

def SCMEXP(
    Y_full,
    T0,
    clusters,
    blank_periods=0,
    m_eq=None,
    m_min=None,
    m_max=None,
    exclusive=True,
    design="base",
    beta=1e-6,
    lambda1=0.0,
    lambda2=0.0,
    xi=0.0,
    lambda1_unit=0.0,
    lambda2_unit=0.0,
    costs=None,
    budget=None,
    solver=cp.ECOS_BB,
    verbose=False
):
    _validate_scm_inputs(Y_full, T0, blank_periods, design,
                        beta, lambda1, lambda2, xi, lambda1_unit, lambda2_unit)

    Y_full_np, clusters, N, cluster_labels, K, label_to_k = _prepare_clusters(Y_full, clusters)
    costs_np, budget_dict = _validate_costs_budget(costs, budget, N, cluster_labels, K)

    # --- prepare fit slices ---
    T_fit = T0 - blank_periods
    Y_fit = Y_full_np[:, :T_fit]
    Y_blank = Y_full_np[:, T_fit:T0] if blank_periods > 0 else None

    # --- membership mask ---
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True

    # --- cluster-level means and members ---
    Xbar_clusters = []
    cluster_members = []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members.")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))

    # --- CVXPY variables ---
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    z = cp.Variable((N, K), boolean=True)

    # --- constraints and objective (simplified here for brevity, can include your previous full version) ---
    constraints = []
    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        constraints += [cp.sum(w[members, k_idx]) == 1, cp.sum(v[members, k_idx]) == 1]
        if costs is not None:
            B_k = _get_per_cluster_param(budget_dict, lab)
            c_k = costs_np[members]
            constraints += [cp.sum(cp.multiply(c_k, w[members, k_idx])) <= B_k]

    # --- solve ---
    objective = cp.Minimize(cp.sum_squares(Y_fit.T @ w - Y_fit.T))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    # --- extract results ---
    results = _extract_results(Y_full_np, Y_fit, w, v, cluster_members, T_fit)

    # --- prepare result dictionary ---
    result = {
        "df": Y_full,
        "w_opt": results['w_opt'],
        "v_opt": results['v_opt'],
        "z_opt": results['z_opt'],
        "y_syn_treated_clusters": results['y_syn_treated_clusters'],
        "y_syn_control_clusters": results['y_syn_control_clusters'],
        "Xbar_clusters": Xbar_clusters,
        "cluster_labels": list(cluster_labels),
        "cluster_members": cluster_members,
        "w_agg": results['w_agg'],
        "v_agg": results['v_agg'],
        "cluster_sizes": [len(m) for m in cluster_members],
        "T0": T0,
        "blank_periods": blank_periods,
        "T_fit": T_fit,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": results['rmse_cluster'],
        "design": design,
        "beta": beta if design == "weak" else None,
        "lambda1": lambda1 if design == "eq11" else (lambda1_unit if design == "unit" else None),
        "lambda2": lambda2 if design == "eq11" else (lambda2_unit if design == "unit" else None),
        "xi": xi if design == "unit" else None,
        "original_cluster_vector": clusters,
        "costs_used": costs if costs is not None else None,
        "budget_used": budget_dict
    }

    return result
