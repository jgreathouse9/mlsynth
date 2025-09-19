import pandas as pd
import cvxpy as cp
import numpy as np


def _get_per_cluster_param(param, klabel, default=None):
    """
    Retrieve a parameter value specific to a cluster or apply a default.
    """
    if param is None:
        return default
    if isinstance(param, dict):
        return param.get(klabel, default)
    return param


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
    """
    Clustered Synthetic Control for Experimental Design (SCMEXP).
    """

    # Preserve original for output
    Y_input = Y_full
    if hasattr(Y_full, "to_numpy"):
        Y_full = Y_full.to_numpy()
    N, T_total = Y_full.shape

    # --- Validation ---
    if T0 <= 0 or T0 >= T_total:
        raise ValueError("T0 must satisfy 1 <= T0 < Y_full.shape[1]")
    if blank_periods < 0 or blank_periods >= T0:
        raise ValueError("blank_periods must satisfy 0 <= blank_periods < T0")
    if design not in {"base", "weak", "eq11", "unit"}:
        raise ValueError(f"Invalid design '{design}'")
    if design != "weak" and beta != 1e-6:
        raise ValueError("beta only valid for design='weak'")
    if design != "eq11" and (lambda1 != 0.0 or lambda2 != 0.0):
        raise ValueError("lambda1/lambda2 only valid for design='eq11'")
    if design != "unit" and (xi != 0.0 or lambda1_unit != 0.0 or lambda2_unit != 0.0):
        raise ValueError("xi/lambda1_unit/lambda2_unit only valid for design='unit'")

    # --- Clusters ---
    clusters = np.asarray(clusters)
    if clusters.shape[0] != N:
        raise ValueError("clusters must have length N (rows of Y)")
    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}

    # --- Costs & budget ---
    if costs is not None:
        costs = np.asarray(costs)
        if costs.shape[0] != N:
            raise ValueError("costs must have length N")
        if budget is None:
            raise ValueError("budget must be provided if costs are specified")
        if isinstance(budget, (int, float)):
            budget = {lab: budget / K for lab in cluster_labels}
        elif isinstance(budget, dict):
            for lab in cluster_labels:
                if lab not in budget:
                    raise ValueError(f"budget missing entry for cluster '{lab}'")

    # --- Membership mask ---
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True

    # --- Pre-treatment slices ---
    T_fit = T0 - blank_periods
    Y_fit = Y_full[:, :T_fit]
    Y_blank = Y_full[:, T_fit:T0] if blank_periods > 0 else None

    # --- Cluster means ---
    Xbar_clusters = []
    cluster_members = []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))

    # --- Distance precomputation (if needed later) ---
    D1 = np.zeros((N, K))
    for k_idx in range(K):
        D1[:, k_idx] = np.sum((Y_fit - Xbar_clusters[k_idx][None, :]) ** 2, axis=1)

    # --- CVXPY variables ---
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    z = cp.Variable((N, K), boolean=True)

    # --- Constraints ---
    constraints = []
    for k in range(K):
        for j in range(N):
            if not M[j, k]:
                constraints += [w[j, k] == 0, v[j, k] == 0, z[j, k] == 0]

    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        constraints += [cp.sum(w[members, k_idx]) == 1]
        constraints += [cp.sum(v[members, k_idx]) == 1]

        m_eq_k = _get_per_cluster_param(m_eq, lab)
        m_min_k = _get_per_cluster_param(m_min, lab)
        m_max_k = _get_per_cluster_param(m_max, lab)

        if m_eq_k is not None:
            constraints += [cp.sum(z[members, k_idx]) == int(m_eq_k)]
        if m_min_k is not None:
            constraints += [cp.sum(z[members, k_idx]) >= int(m_min_k)]
        if m_max_k is not None:
            constraints += [cp.sum(z[members, k_idx]) <= int(m_max_k)]

        for j in members:
            constraints += [w[j, k_idx] <= z[j, k_idx]]
            constraints += [v[j, k_idx] <= 1 - z[j, k_idx]]

        if costs is not None:
            B_k = _get_per_cluster_param(budget, lab)
            c_k = costs[members]
            constraints += [cp.sum(cp.multiply(c_k, w[members, k_idx])) <= B_k]

    if exclusive:
        for j in range(N):
            constraints += [cp.sum(z[j, :]) <= 1]

    # --- Objective ---
    Y_T = Y_fit.T
    obj_terms = []
    for k_idx, lab in enumerate(cluster_labels):
        Xbar_k = Xbar_clusters[k_idx]
        syn_treated_k = Y_T @ w[:, k_idx]
        syn_control_k = Y_T @ v[:, k_idx]
        obj_terms.append(cp.sum_squares(Xbar_k - syn_treated_k))
        obj_terms.append(cp.sum_squares(Xbar_k - syn_control_k))

        if design == "weak":
            obj_terms.append(beta * cp.sum_squares(syn_treated_k - syn_control_k))
        elif design == "eq11":
            if lambda1 > 0:
                obj_terms.append(lambda1 * cp.norm1(w[:, k_idx] - v[:, k_idx]))
            if lambda2 > 0:
                obj_terms.append(lambda2 * cp.sum_squares(w[:, k_idx] - v[:, k_idx]))
        elif design == "unit":
            obj_terms.append(xi * cp.sum_squares(syn_treated_k - syn_control_k))
            if lambda1_unit > 0:
                obj_terms.append(lambda1_unit * cp.norm1(w[:, k_idx] - v[:, k_idx]))
            if lambda2_unit > 0:
                obj_terms.append(lambda2_unit * cp.sum_squares(w[:, k_idx] - v[:, k_idx]))

    objective = cp.Minimize(cp.sum(obj_terms))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Optimization failed with status {problem.status}")

    # --- Extract results ---
    w_val = w.value
    v_val = v.value
    z_val = z.value

    w_agg = {}
    v_agg = {}
    syn_trajs_treated = {}
    syn_trajs_control = {}

    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        w_cluster = w_val[members, k_idx]
        v_cluster = v_val[members, k_idx]
        w_agg[lab] = w_cluster
        v_agg[lab] = v_cluster

        syn_trajs_treated[lab] = Y_full.T @ w_val[:, k_idx]
        syn_trajs_control[lab] = Y_full.T @ v_val[:, k_idx]

    results = {
        "weights_treated": w_val,
        "weights_control": v_val,
        "selection": z_val,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "synthetic_treated": syn_trajs_treated,
        "synthetic_control": syn_trajs_control,
        "cluster_labels": cluster_labels,
        "cluster_members": cluster_members,
        "objective_value": problem.value,
        "status": problem.status,
        "costs": costs,
        "budget": budget,
        "design": design,
        "df": Y_input,
    }

    return results
