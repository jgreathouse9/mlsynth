import pandas as pd
import cvxpy as cp
import numpy as np


def _get_per_cluster_param(param, klabel, default=None):
    """
    Retrieve a parameter value specific to a cluster or apply a default.

    Parameters
    ----------
    param : None, scalar, or dict
        Parameter specification. If dict, maps cluster label to value.
    klabel : hashable
        Cluster label to look up.
    default : any
        Fallback if param is None or key not in dict.

    Returns
    -------
    value for the given cluster
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

    Parameters
    ----------
    Y_full : pd.DataFrame or ndarray, shape (N_units, T_total)
        Outcome matrix.
    T0 : int
        Number of pre-treatment periods for fitting.
    clusters : array-like, length N_units
        Cluster labels for each unit.
    blank_periods : int, default=0
        Initial periods to ignore in fitting.
    m_eq, m_min, m_max : int, dict, or None
        Exact/min/max cardinality per cluster.
    exclusive : bool
        Each unit assigned to at most one cluster if True.
    design : {'base','weak','eq11','unit'}
        Type of synthetic control design.
    beta, lambda1, lambda2, xi, lambda1_unit, lambda2_unit : float
        Design-specific penalties.
    costs : ndarray, optional
        Cost per unit.
    budget : float, int, or dict, optional
        Budget constraint; must be provided if costs are set.
    solver : cvxpy solver
    verbose : bool

    Returns
    -------
    dict
        Includes weights, synthetic trajectories, cluster info, costs/budget,
        and original input in 'df'.
    """
    # Preserve original for output
    Y_input = Y_full
    if hasattr(Y_full, "to_numpy"):
        Y_full = Y_full.to_numpy()
    N, T_total = Y_full.shape

    # Basic validation
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

    # Clusters
    clusters = np.asarray(clusters)
    if clusters.shape[0] != N:
        raise ValueError("clusters must have length N (rows of Y)")
    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}

    # Costs & budget
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

    # Membership mask
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True

    # Pre-treatment slices
    T_fit = T0 - blank_periods
    Y_fit = Y_full[:, :T_fit]
    Y_blank = Y_full[:, T_fit:T0] if blank_periods > 0 else None

    # Cluster means & members
    Xbar_clusters = []
    cluster_members = []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))

    # Precompute distances
    D1 = np.zeros((N, K))
    for k_idx in range(K):
        D1[:, k_idx] = np.sum((Y_fit - Xbar_clusters[k_idx][None, :]) ** 2, axis=1)

    D2_list = []
    for k_idx in range(K):
        members = cluster_members[k_idx]
        Xm = Y_fit[members, :]
        diff = Xm[:, None, :] - Xm[None, :, :]
        D2_list.append(np.sum(diff**2, axis=2))

    # CVXPY variables
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    z = cp.Variable((N, K), boolean=True)

    # Constraints
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

    # Objective
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
                obj_terms.append(lambda1 * cp.sum(cp.multiply(w[:, k_idx], D1[:, k_idx])))
            if lambda2 > 0:
                obj_terms.append(lambda2 * cp.sum(cp.multiply(v[:, k_idx], D1[:, k_idx])))
        elif design == "unit":
            members = cluster_members[k_idx]
            if xi > 0:
                for j in members:
                    X_j = Y_fit[j, :]
                    obj_terms.append(xi * w[j, k_idx] * cp.sum_squares(X_j - syn_control_k))
            if lambda1_unit > 0:
                obj_terms.append(lambda1_unit * cp.sum(cp.multiply(w[members, k_idx], D1[members, k_idx])))
            if lambda2_unit > 0 and len(members) > 1:
                Dmat = D2_list[k_idx]
                v_m = v[members, k_idx]
                w_m = w[members, k_idx]
                obj_terms.append(lambda2_unit * cp.sum(cp.multiply(w_m, Dmat @ v_m)))

    # Solve
    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    prob.solve(solver=cp.SCIP, verbose=verbose)

    # Extract results
    w_opt = w.value
    v_opt = v.value
    z_opt = z.value
    Y_full_T = Y_full.T
    y_syn_treated_clusters = [Y_full_T @ w_opt[:, k] for k in range(K)]
    y_syn_control_clusters = [Y_full_T @ v_opt[:, k] for k in range(K)]

    rmse_cluster = []
    for k_idx in range(K):
        treated_idx = np.where(w_opt[:, k_idx] > 1e-8)[0]
        y_treated = (Y_fit[treated_idx, :].T @ w_opt[treated_idx, k_idx]) / np.sum(w_opt[treated_idx, k_idx]) \
            if len(treated_idx) > 0 else np.zeros(T_fit)
        y_control = Y_fit.T @ v_opt[:, k_idx]
        rmse_cluster.append(np.sqrt(np.mean((y_treated - y_control) ** 2)))

    cluster_sizes = [len(m) for m in cluster_members]
    total_size = sum(cluster_sizes)
    w_agg = sum((cluster_sizes[k]/total_size) * w_opt[:, k] for k in range(K))
    v_agg = sum((cluster_sizes[k]/total_size) * v_opt[:, k] for k in range(K))

    result = {
        "Y_Full": Y_full,
        "w_opt": w_opt,
        "v_opt": v_opt,
        "z_opt": z_opt,
        "y_syn_treated_clusters": y_syn_treated_clusters,
        "y_syn_control_clusters": y_syn_control_clusters,
        "Xbar_clusters": Xbar_clusters,
        "cluster_labels": list(cluster_labels),
        "cluster_members": cluster_members,
        "w_agg": w_agg,
        "v_agg": v_agg,
        "cluster_sizes": cluster_sizes,
        "T0": T0,
        "blank_periods": blank_periods,
        "T_fit": T_fit,
        "Y_fit": Y_fit,
        "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster,
        "design": design,
        "beta": beta if design=="weak" else None,
        "lambda1": lambda1 if design=="eq11" else (lambda1_unit if design=="unit" else None),
        "lambda2": lambda2 if design=="eq11" else (lambda2_unit if design=="unit" else None),
        "xi": xi if design=="unit" else None,
        "df": Y_input,  # preserve original for indexing
        "original_cluster_vector": clusters,
        "costs_used": costs if costs is not None else None,
        "budget_used": budget
    }

    return result

