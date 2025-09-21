import pandas as pd
import cvxpy as cp
import numpy as np

def _get_per_cluster_param(param, klabel, default=None):
    if param is None:
        return default
    if isinstance(param, dict):
        return param.get(klabel, default)
    return param  # scalar

def _prepare_clusters(Y_full, clusters):
    """
    Convert Y and clusters to NumPy arrays, check dimensions, and return cluster info.

    Parameters
    ----------
    Y_full : pd.DataFrame or np.ndarray
        Outcome matrix with units as rows and time periods as columns.
    clusters : array-like
        Cluster labels for each unit.

    Returns
    -------
    Y_full_np : np.ndarray
        Converted outcome matrix.
    clusters : np.ndarray
        Cluster labels as NumPy array.
    N : int
        Number of units (rows of Y_full).
    cluster_labels : np.ndarray
        Unique cluster labels.
    K : int
        Number of clusters.
    label_to_k : dict
        Mapping from cluster label to index (0..K-1).
    """
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
    """
    Validate SCM input arguments for shape, design compatibility, and parameter usage.
    Raises ValueError if any check fails.
    """
    # --- basic shape checks ---
    if T0 <= 0 or T0 >= Y_full.shape[1]:
        raise ValueError("T0 must be 1 <= T0 < Y_full.shape[1]")
    if blank_periods < 0 or blank_periods >= T0:
        raise ValueError("blank_periods must be 0 <= blank_periods < T0 (need at least 1 fit period)")

    # --- incompatible parameter usage ---
    if design != "weak" and beta != 1e-6:
        raise ValueError("beta is only valid when design == 'weak'")
    if design != "eq11" and (lambda1 != 0.0 or lambda2 != 0.0):
        raise ValueError("lambda1/lambda2 are only valid when design == 'eq11'")
    if design != "unit" and (xi != 0.0 or lambda1_unit != 0.0 or lambda2_unit != 0.0):
        raise ValueError("xi/lambda1_unit/lambda2_unit are only valid when design == 'unit'")


def _validate_costs_budget(costs, budget, N, cluster_labels, K):
    """
    Validate and process cost and budget inputs for SCM.

    Parameters
    ----------
    costs : array-like or None
        Vector of costs per unit, length N.
    budget : scalar, dict, or None
        Total budget constraint.
    N : int
        Number of units.
    cluster_labels : array-like
        Unique cluster labels.
    K : int
        Number of clusters.

    Returns
    -------
    costs_np : np.ndarray or None
        Costs as a NumPy array.
    budget_dict : dict or None
        Budget per cluster.
    """
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

def _prepare_fit_slices(Y_full_np, T0, blank_periods):
    T_fit = T0 - blank_periods
    Y_fit = Y_full_np[:, :T_fit]
    Y_blank = Y_full_np[:, T_fit:T0] if blank_periods > 0 else None
    return Y_fit, Y_blank, T_fit


def _build_membership_mask(clusters, label_to_k, N, K):
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True
    return M


def _compute_cluster_means_members(Y_fit, M, cluster_labels):
    Xbar_clusters = []
    cluster_members = []
    K = len(cluster_labels)
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members.")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))
    return Xbar_clusters, cluster_members


def _precompute_distances(Y_fit, Xbar_clusters, cluster_members):
    N, K = Y_fit.shape[0], len(Xbar_clusters)
    D1 = np.zeros((N, K))
    for k_idx in range(K):
        diffs = Y_fit - Xbar_clusters[k_idx][None, :]
        D1[:, k_idx] = np.sum(diffs**2, axis=1)

    D2_list = []
    for k_idx, members in enumerate(cluster_members):
        Xm = Y_fit[members, :]
        diff = Xm[:, None, :] - Xm[None, :, :]
        D2_list.append(np.sum(diff**2, axis=2))
    return D1, D2_list


def _init_cvxpy_variables(N, K):
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    z = cp.Variable((N, K), boolean=True)
    return w, v, z


def _build_constraints(w, v, z, M, cluster_members, cluster_labels,
                       m_eq, m_min, m_max, costs, budget_dict, exclusive):
    N, K = M.shape
    constraints = []

    # enforce zeros outside cluster
    for k in range(K):
        for j in range(N):
            if not M[j, k]:
                constraints += [w[j, k] == 0, v[j, k] == 0, z[j, k] == 0]

    # per-cluster normalization, cardinality, linking, cost
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
            B_k = _get_per_cluster_param(budget_dict, lab)
            c_k = costs[members]
            constraints += [cp.sum(cp.multiply(c_k, w[members, k_idx])) <= B_k]

    # exclusive assignment
    if exclusive:
        for j in range(N):
            constraints += [cp.sum(z[j, :]) <= 1]

    return constraints


def _build_objective(Y_fit, Xbar_clusters, cluster_members, w, v, z,
                     design, beta=1e-6, lambda1=0.0, lambda2=0.0,
                     xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0,
                     D1=None, D2_list=None):
    Y_T = Y_fit.T
    obj_terms = []
    K = len(cluster_members)

    if design == "base":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - Y_T @ w[:, k_idx]))
            obj_terms.append(cp.sum_squares(Xbar_k - Y_T @ v[:, k_idx]))

    elif design == "weak":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated))
            obj_terms.append(beta * cp.sum_squares(syn_treated - syn_control))

    elif design == "eq11":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated))
            obj_terms.append(cp.sum_squares(Xbar_k - syn_control))
            if lambda1 > 0:
                obj_terms.append(lambda1 * cp.sum(cp.multiply(w[:, k_idx], D1[:, k_idx])))
            if lambda2 > 0:
                obj_terms.append(lambda2 * cp.sum(cp.multiply(v[:, k_idx], D1[:, k_idx])))

    elif design == "unit":
        for k_idx, lab in enumerate(cluster_members):
            members = cluster_members[k_idx]
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]

            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated))
            obj_terms.append(cp.sum_squares(Xbar_k - syn_control))

            if xi > 0:
                for j in members:
                    obj_terms.append(xi * w[j, k_idx] * cp.sum_squares(Y_fit[j, :] - syn_control))
            if lambda1_unit > 0:
                obj_terms.append(lambda1_unit * cp.sum(cp.multiply(w[members, k_idx], D1[members, k_idx])))
            if lambda2_unit > 0 and len(members) > 1:
                Dmat = D2_list[k_idx]
                v_m = v[members, k_idx]
                w_m = w[members, k_idx]
                obj_terms.append(lambda2_unit * cp.sum(cp.multiply(w_m, Dmat @ v_m)))

    return cp.Minimize(cp.sum(obj_terms))



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

    Constructs synthetic control weights per cluster for experimental design,
    optionally enforcing cost and budget constraints, cardinality constraints,
    and design-specific penalization terms. Supports a range of design choices
    including baseline, weak-targeted, eq11, and unit-level objectives.

    Parameters
    ----------
    Y_full : pd.DataFrame or np.ndarray
        Outcome matrix with units as rows and time periods as columns.
    T0 : int
        Number of pre-treatment periods to use for fitting.
    clusters : array-like
        Cluster label for each unit.
    blank_periods : int, default=0
        Number of "blank" pre-treatment periods excluded from fitting for validation.
    m_eq : int or dict, optional
        Number of units to select per cluster for exact cardinality constraints.
        If dict, keys correspond to cluster labels.
    m_min : int or dict, optional
        Minimum number of units to select per cluster.
    m_max : int or dict, optional
        Maximum number of units to select per cluster.
    exclusive : bool, default=True
        If True, each unit may belong to at most one cluster for selection purposes.
    design : str, default='base'
        Choice of SCM objective function:
            - 'base' : standard squared error between cluster mean and synthetic units
            - 'weak' : weakly-targeted design with beta penalization
            - 'eq11' : design with lambda1/lambda2 cluster-level penalization
            - 'unit' : unit-level penalization with xi, lambda1_unit, lambda2_unit
    beta : float, default=1e-6
        Penalization for weak-targeted design (only used if design='weak').
    lambda1, lambda2 : float, default=0.0
        Cluster-level penalization weights (used if design='eq11').
    xi : float, default=0.0
        Unit-level squared-error penalization (used if design='unit').
    lambda1_unit, lambda2_unit : float, default=0.0
        Unit-level penalization weights (used if design='unit').
    costs : array-like, optional
        Vector of costs per unit (length N). If provided, budget must be specified.
    budget : float or dict, optional
        Scalar total budget or dictionary of cluster budgets. Required if `costs` is provided.
    solver : cvxpy solver, default=cp.ECOS_BB
        CVXPY solver used to solve the optimization problem.
    verbose : bool, default=False
        If True, print solver output.

    Returns
    -------
    result : dict
        Dictionary containing:
            - 'df' : original Y_full DataFrame/array
            - 'w_opt', 'v_opt', 'z_opt' : optimized CVXPY variables (treated/control weights, selection mask)
            - 'y_syn_treated_clusters', 'y_syn_control_clusters' : synthetic outcomes per cluster
            - 'Xbar_clusters' : cluster-level means of pre-treatment outcomes
            - 'cluster_labels', 'cluster_members' : cluster info
            - 'w_agg', 'v_agg' : aggregated synthetic weights across clusters
            - 'cluster_sizes' : number of units per cluster
            - 'T0', 'blank_periods', 'T_fit', 'Y_fit', 'Y_blank' : fitting info
            - 'rmse_cluster' : root-mean-squared-error per cluster
            - 'design', 'beta', 'lambda1', 'lambda2', 'xi' : design and penalization parameters
            - 'original_cluster_vector' : input cluster vector
            - 'costs_used', 'budget_used' : cost and budget info if provided

    Notes
    -----
    - The function enforces zeros outside clusters, cardinality constraints,
      linking constraints between weights and selection masks, and budget limits.
    - Supports multiple design options for experimental SCM, allowing
      flexible penalization at the cluster or unit level.
    - Blank periods allow evaluation on holdout pre-treatment periods.
    - CVXPY Boolean variables (z) are used for unit selection; may require
      a mixed-integer solver such as `cp.SCIP` or `cp.ECOS_BB`.
    """

    # --- validate inputs ---
    _validate_scm_inputs(Y_full, T0, blank_periods, design,
                         beta, lambda1, lambda2, xi, lambda1_unit, lambda2_unit)

    Y_full_np, clusters, N, cluster_labels, K, label_to_k = _prepare_clusters(Y_full, clusters)
    costs_np, budget_dict = _validate_costs_budget(costs, budget, N, cluster_labels, K)

    # --- prepare fit slices ---
    Y_fit, Y_blank, T_fit = _prepare_fit_slices(Y_full_np, T0, blank_periods)

    # --- membership mask ---
    M = _build_membership_mask(clusters, label_to_k, N, K)

    # --- cluster-level means and members ---
    Xbar_clusters, cluster_members = _compute_cluster_means_members(Y_fit, M, cluster_labels)

    # --- precompute D1/D2 for penalized designs ---
    D1, D2_list = _precompute_distances(Y_fit, Xbar_clusters, cluster_members)

    # --- CVXPY variables ---
    w, v, z = _init_cvxpy_variables(N, K)

    # --- constraints ---
    constraints = _build_constraints(w, v, z, M, cluster_members, cluster_labels,
                                     m_eq, m_min, m_max, costs_np, budget_dict, exclusive)

    # --- objective ---
    objective = _build_objective(Y_fit, Xbar_clusters, cluster_members, w, v, z,
                                 design, beta, lambda1, lambda2, xi,
                                 lambda1_unit, lambda2_unit, D1, D2_list)

    # --- solve problem ---
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIP, verbose=verbose)

    # --- extract ---
    w_opt = w.value
    v_opt = v.value
    z_opt = z.value

    Y_full_T = Y_full_np.T
    y_syn_treated_clusters = [Y_full_T @ w_opt[:, k_idx] for k_idx in range(K)]
    y_syn_control_clusters = [Y_full_T @ v_opt[:, k_idx] for k_idx in range(K)]

    # cluster RMSE
    rmse_cluster = []
    for k_idx in range(K):
        treated_idx = np.where(w_opt[:, k_idx] > 1e-8)[0]
        y_treated = (Y_fit[treated_idx, :].T @ w_opt[treated_idx, k_idx] / np.sum(w_opt[treated_idx, k_idx])) if len(treated_idx) > 0 else np.zeros(T_fit)
        y_control = Y_fit.T @ v_opt[:, k_idx]
        rmse_cluster.append(np.sqrt(np.mean((y_treated - y_control) ** 2)))

    # aggregate weights
    cluster_sizes = [len(m) for m in cluster_members]
    total_size = sum(cluster_sizes)
    w_agg = np.zeros(N)
    v_agg = np.zeros(N)
    for k_idx in range(K):
        w_agg += (cluster_sizes[k_idx] / total_size) * w_opt[:, k_idx]
        v_agg += (cluster_sizes[k_idx] / total_size) * v_opt[:, k_idx]

    result = {
        "df": Y_full,
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
        "xi": xi if design == "unit" else None,
        "original_cluster_vector": clusters,
        "costs_used": costs if costs is not None else None,
        "budget_used": budget
    }

    return result

def _compute_placebo_ci_vectorized(tau_hat, Y_blank_T, w, v, rmspe_pre, rmse_cluster=None, alpha=0.05):
    """
    Vectorized computation of placebo CI and p-values for global or cluster-level SCM inference.

    Args:
        tau_hat : np.ndarray
            Per-period treatment effects. Shape (T_post,) for global or (K, T_post) for clusters.
        Y_blank_T : np.ndarray
            Transposed blank periods, shape (blank_periods, N_units)
        w : np.ndarray
            Weight vectors, shape (N_units,) for global or (N_units, K) for clusters
        v : np.ndarray
            Control weight vectors, same shape as w
        rmspe_pre : float
            Pre-treatment RMSPE for scaling
        rmse_cluster : np.ndarray or None
            Cluster-specific RMSPE for scaling, shape (K,)
        alpha : float
            Significance level for CI

    Returns:
        ci_lower, ci_upper, p_values : np.ndarray
            Confidence intervals and p-values with same shape as tau_hat
    """
    assert Y_blank_T.shape[1] == w.shape[0], "Mismatch between Y_blank_T and weight vector dimensions."

    if np.isnan(Y_blank_T).any():
    raise ValueError("Y_blank contains NaNs, cannot compute placebo inference.")


    if w.ndim == 1:
        # Global inference
        u_blank = Y_blank_T @ w - Y_blank_T @ v
        scale = np.sqrt(np.mean(u_blank**2)) / rmspe_pre if rmspe_pre > 0 else 1
        q = np.quantile(np.abs(u_blank), 1 - alpha)
        ci_lower = tau_hat - q * scale
        ci_upper = tau_hat + q * scale
        p_values = np.array([np.mean(np.abs(u_blank) >= np.abs(t)) for t in tau_hat])
    else:
        # Cluster-level inference
        u_blank = Y_blank_T @ w - Y_blank_T @ v  # (blank_periods, K)
        abs_u_blank = np.abs(u_blank)
        q = np.quantile(abs_u_blank, 1 - alpha, axis=0)  # (K,)
        tau_hat = np.atleast_2d(tau_hat)
        safe_rmse = np.where(rmse_cluster > 0, rmse_cluster, 1.0)
        scale = np.sqrt(np.mean(u_blank**2, axis=0)) / safe_rmse
        ci_lower = tau_hat - q[:, None] * scale[:, None]
        ci_upper = tau_hat + q[:, None] * scale[:, None]
        # p-values
        T_post = tau_hat.shape[1]
        p_values = np.zeros_like(tau_hat)
        for t in range(T_post):
            p_values[:, t] = np.mean(abs_u_blank >= np.abs(tau_hat[:, t]), axis=0)

    return ci_lower, ci_upper, p_values


def inference_scm_vectorized(result, Y_full, T_post, alpha=0.05, method='placebo'):
    """
    Vectorized SCM inference for global and cluster-specific effects.

    Args:
        result : dict
            SCM output including 'w_opt', 'v_opt', 'w_agg', 'v_agg', 'Y_fit', 'Y_blank', 'rmse_cluster', 'T0'
        Y_full : np.ndarray
            Full outcome data (N_units x T_total)
        T_post : int
            Number of post-intervention periods
        alpha : float
            Significance level
        method : str
            Only 'placebo' supported

    Returns:
        dict with global and cluster-specific inference
    """
    if method != 'placebo':
        raise ValueError("Only 'placebo' method is supported.")

    T0 = result["T0"]
    w_agg = result["w_agg"]
    v_agg = result["v_agg"]
    w_opt = result["w_opt"]
    v_opt = result["v_opt"]
    Y_fit = result["Y_fit"]
    Y_blank = result["Y_blank"]
    rmse_cluster = result["rmse_cluster"]

    # Post-intervention outcomes
    Y_post_T = Y_full[:, T0:T0 + T_post].T  # (T_post, N_units)

    # Global effects
    tau_hat = Y_post_T @ w_agg - Y_post_T @ v_agg
    avg_tau_hat = np.mean(tau_hat) if T_post > 0 else 0.0

    # Cluster effects
    tau_hat_cluster = (Y_post_T @ w_opt - Y_post_T @ v_opt).T  # (K, T_post)
    avg_tau_cluster = np.mean(tau_hat_cluster, axis=1) if T_post > 0 else np.zeros(w_opt.shape[1])

    # Pre-fit RMSPE (global)
    syn_fit_treated = Y_fit.T @ w_agg
    syn_fit_control = Y_fit.T @ v_agg
    rmspe_pre = np.sqrt(np.mean((syn_fit_treated - syn_fit_control) ** 2))

    # Placebo inference
    Y_blank_T = Y_blank.T
    ci_lower, ci_upper, p_values = _compute_placebo_ci_vectorized(
        tau_hat, Y_blank_T, w_agg, v_agg, rmspe_pre, alpha=alpha
    )
    ci_lower_cluster, ci_upper_cluster, p_values_cluster = _compute_placebo_ci_vectorized(
        tau_hat_cluster, Y_blank_T, w_opt, v_opt, rmspe_pre, rmse_cluster, alpha=alpha
    )

    return {
        "tau_hat": tau_hat,
        "avg_tau_hat": avg_tau_hat,
        "tau_hat_cluster": tau_hat_cluster,
        "avg_tau_cluster": avg_tau_cluster,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_values_cluster": p_values_cluster,
        "ci_lower_cluster": ci_lower_cluster,
        "ci_upper_cluster": ci_upper_cluster,
        "rmspe_pre": rmspe_pre,
        "rmse_cluster": rmse_cluster
    }
