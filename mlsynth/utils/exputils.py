import pandas as pd
import cvxpy as cp
import numpy as np


def _get_per_cluster_param(param, klabel, default=None):
    """Helper: param may be None, scalar, or dict {klabel: val}."""
    if param is None:
        return default
    if isinstance(param, dict):
        return param.get(klabel, default)
    return param  # scalar


def SCMEXP(
    Y_full,
    T0,
    clusters,
    blank_periods=0,
    m_eq=None,
    m_min=None,
    m_max=None,
    exclusive=True,
    design="base",      # mutually exclusive: 'base', 'weak', 'eq11', 'unit'
    beta=1e-6,          # weak-targeted
    lambda1=0.0,        # eq11
    lambda2=0.0,        # eq11
    xi=0.0,             # unit OA.1
    lambda1_unit=0.0,   # unit OA.2
    lambda2_unit=0.0,   # unit OA.3
    solver=cp.ECOS_BB,
    verbose=False,
):
    """Clustered Synthetic Control for Experimental Design (SCMEXP)."""

    # --- preserve input type ---
    if isinstance(Y_full, pd.DataFrame):
        widedf = Y_full.copy()
        Y_full = Y_full.to_numpy()
    else:
        widedf = np.array(Y_full, copy=True)

    valid_designs = {"base", "weak", "eq11", "unit"}
    if design not in valid_designs:
        raise ValueError(f"design must be one of {valid_designs}; got '{design}'")

    # --- basic shape checks ---
    if T0 <= 0 or T0 >= Y_full.shape[1]:
        raise ValueError("T0 must be 1 <= T0 < Y_full.shape[1]")
    if blank_periods < 0 or blank_periods >= T0:
        raise ValueError("blank_periods must be 0 <= blank_periods < T0")

    # parameter consistency
    if design != "weak" and beta != 1e-6:
        raise ValueError("beta is only valid when design == 'weak'")
    if design != "eq11" and (lambda1 != 0.0 or lambda2 != 0.0):
        raise ValueError("lambda1/lambda2 are only valid when design == 'eq11'")
    if design != "unit" and (xi != 0.0 or lambda1_unit != 0.0 or lambda2_unit != 0.0):
        raise ValueError("xi/lambda1_unit/lambda2_unit are only valid when design == 'unit'")

    # --- prepare data slices ---
    T_fit = T0 - blank_periods
    Y_fit = Y_full[:, :T_fit]
    Y_blank = Y_full[:, T_fit:T0] if blank_periods > 0 else None

    N, _ = Y_fit.shape
    clusters = np.asarray(clusters)
    if clusters.shape[0] != N:
        raise ValueError("clusters must have length N (rows of Y).")

    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}

    # membership mask
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True

    # cluster-level means and membership
    Xbar_clusters, cluster_members = [], []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members.")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))

    # precompute distances
    D1 = np.zeros((N, K))
    for k_idx in range(K):
        diffs = Y_fit - Xbar_clusters[k_idx][None, :]
        D1[:, k_idx] = np.sum(diffs**2, axis=1)

    D2_list = []
    for k_idx in range(K):
        members = cluster_members[k_idx]
        Xm = Y_fit[members, :]
        diff = Xm[:, None, :] - Xm[None, :, :]
        D2_list.append(np.sum(diff**2, axis=2))

    # --- variables ---
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    z = cp.Variable((N, K), boolean=True)

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

    if exclusive:
        for j in range(N):
            constraints += [cp.sum(z[j, :]) <= 1]

    # --- objective ---
    Y_T = Y_fit.T
    obj_terms = []

    if design == "base":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            obj_terms += [
                cp.sum_squares(Xbar_k - Y_T @ w[:, k_idx]),
                cp.sum_squares(Xbar_k - Y_T @ v[:, k_idx]),
            ]

    elif design == "weak":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]
            obj_terms += [
                cp.sum_squares(Xbar_k - syn_treated),
                beta * cp.sum_squares(syn_treated - syn_control),
            ]

    elif design == "eq11":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            obj_terms += [
                cp.sum_squares(Xbar_k - Y_T @ w[:, k_idx]),
                cp.sum_squares(Xbar_k - Y_T @ v[:, k_idx]),
            ]
            if lambda1 > 0:
                obj_terms.append(lambda1 * cp.sum(cp.multiply(w[:, k_idx], D1[:, k_idx])))
            if lambda2 > 0:
                obj_terms.append(lambda2 * cp.sum(cp.multiply(v[:, k_idx], D1[:, k_idx])))

    elif design == "unit":
        for k_idx in range(K):
            members = cluster_members[k_idx]
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]

            obj_terms += [
                cp.sum_squares(Xbar_k - syn_treated),
                cp.sum_squares(Xbar_k - syn_control),
            ]
            if xi > 0:
                for j in members:
                    obj_terms.append(xi * w[j, k_idx] * cp.sum_squares(Y_fit[j, :] - syn_control))
            if lambda1_unit > 0:
                obj_terms.append(lambda1_unit * cp.sum(cp.multiply(w[:, k_idx], D1[:, k_idx])))
            if lambda2_unit > 0 and len(members) > 0:
                Dmat = D2_list[k_idx]
                v_m = v[members, k_idx]
                inner_vec = Dmat.dot(v_m)
                obj_terms.append(lambda2_unit * cp.sum(cp.multiply(w[members, k_idx], inner_vec)))

    objective = cp.Minimize(cp.sum(obj_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    # --- extract results ---
    w_opt, v_opt, z_opt = w.value, v.value, z.value

    Y_full_T = Y_full.T
    y_syn_treated_clusters = [Y_full_T @ w_opt[:, k] for k in range(K)]
    y_syn_control_clusters = [Y_full_T @ v_opt[:, k] for k in range(K)]

    rmse_cluster = []
    for k_idx in range(K):
        treated_idx = np.where(w_opt[:, k_idx] > 1e-8)[0]
        if len(treated_idx) > 0:
            y_treated = (Y_fit[treated_idx, :].T @ w_opt[treated_idx, k_idx]) / np.sum(w_opt[treated_idx, k_idx])
        else:
            y_treated = np.zeros(T_fit)
        y_control = Y_fit.T @ v_opt[:, k_idx]
        rmse_cluster.append(np.sqrt(np.mean((y_treated - y_control) ** 2)))

    # aggregate weights
    cluster_sizes = [len(m) for m in cluster_members]
    total_size = sum(cluster_sizes)
    agg_weights = np.array(cluster_sizes) / total_size
    w_agg = np.zeros(N)
    v_agg = np.zeros(N)
    for k_idx in range(K):
        w_agg += agg_weights[k_idx] * w_opt[:, k_idx]
        v_agg += agg_weights[k_idx] * v_opt[:, k_idx]

    # --- return full dict ---
    return {
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
        "beta": beta if design == "weak" else None,
        "lambda1": (lambda1 if design == "eq11" else (lambda1_unit if design == "unit" else None)),
        "lambda2": (lambda2 if design == "eq11" else (lambda2_unit if design == "unit" else None)),
        "xi": (xi if design == "unit" else None),
        "df": widedf,  # preserved input
        "original_cluster_vector": clusters,
    }
