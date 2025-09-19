import pandas as pd
import cvxpy as cp
import numpy as np



def _get_per_cluster_param(param, klabel, default=None):
    """
    Retrieve a parameter value specific to a cluster or apply a default.

    This helper function processes a parameter that may be provided as None, a scalar
    value, or a dictionary mapping cluster labels to values. It returns the value
    associated with the given cluster label if a dictionary is provided, falls back
    to a default value if the label is not found, uses the scalar value if provided,
    or returns the default if the parameter is None.

    Parameters
    ----------
    param : None, int, float, or dict
        The parameter to process. If None, returns the default. If a scalar (int or
        float), returns that value for all clusters. If a dict, maps cluster labels
        to specific values (e.g., {"Willemstad": 12, "non-Willemstad": 8}).
    klabel : hashable
        The cluster label (e.g., string or integer) to look up in the param dict.
    default : any, optional (default=None)
        The fallback value to return if param is None or if klabel is not found in
        a param dict.

    Returns
    -------
    any
        The parameter value for the given cluster label. Returns default if param
        is None, param.get(klabel, default) if param is a dict, or param if it is
        a scalar.

    Examples
    --------
    >>> _get_per_cluster_param(None, "Willemstad")
    None
    >>> _get_per_cluster_param(5, "Willemstad")
    5
    >>> _get_per_cluster_param({"Willemstad": 12, "non-Willemstad": 8}, "Willemstad")
    12
    >>> _get_per_cluster_param({"Willemstad": 12}, "non-Willemstad", default=5)
    5
    """
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
    # design selector (mutually exclusive modes): 'base', 'weak', 'eq11', 'unit'
    design="base",
    # weak-targeted param (only used when design == "weak")
    beta=1e-6,
    # eq11 params (only used when design == "eq11")
    lambda1=0.0,
    lambda2=0.0,
    # unit-level params (only used when design == "unit")
    xi=0.0,        # OA.1
    lambda1_unit=0.0,  # OA.2 (treated side)
    lambda2_unit=0.0,  # OA.3 pairwise (treated-control pairwise term)
    # New: cost and budget constraints
    costs=None,    # Vector of costs per unit, shape (N,)
    budget=None,   # Total budget, scalar or dict {klabel: val}
    solver=cp.ECOS_BB,
    verbose=False
):
    """
    Clustered Synthetic Control for Experimental Design (SCMEXP).

    Constructs synthetic treated and control units within clusters to approximate
    pre-treatment trajectories, enabling targeted experimental design when full
    randomization is infeasible or unethical. Supports optional cost and budget
    constraints to limit treated unit selection based on financial feasibility.

    Parameters
    ----------
    Y_full : ndarray, shape (N_units, T_total)
        Outcome matrix with units as rows and time periods as columns.
    T0 : int
        Number of pre-treatment periods used for fitting synthetic controls. Must
        satisfy 1 <= T0 < T_total.
    clusters : array-like, shape (N_units,)
        Cluster labels for each unit. Units within the same cluster are used
        together in the synthetic control construction.
    blank_periods : int, default=0
        Number of initial periods within T0 to ignore in fitting.
    m_eq : int, dict, or None
        Exact cardinality of selected units per cluster. Can be scalar or dict
        mapping cluster label to value.
    m_min : int, dict, or None
        Minimum number of units to select per cluster. Can be scalar or dict.
    m_max : int, dict, or None
        Maximum number of units to select per cluster. Can be scalar or dict.
    exclusive : bool, default=False
        If True, ensures each unit is assigned to at most one cluster's synthetic control.
    design : {'base', 'weak', 'eq11', 'unit'}, default='base'
        Type of synthetic control design:
            - 'base' : cluster-targeted fit to cluster mean.
            - 'weak' : weakly-targeted; adds penalty (beta) to align control with treated.
            - 'eq11' : penalized design with lambda1/lambda2 weighting distances from cluster means.
            - 'unit' : unit-level penalized design (xi, lambda1_unit, lambda2_unit).
    beta : float, default=1e-6
        Weak-targeting penalty (used only if design='weak').
    lambda1 : float, default=0.0
        Penalization of treated units’ distance from cluster mean (used if design='eq11').
    lambda2 : float, default=0.0
        Penalization of control units’ distance from cluster mean (used if design='eq11').
    xi : float, default=0.0
        Unit-level OA.1 penalty (used if design='unit'), penalizes treated deviation from synthetic control.
    lambda1_unit : float, default=0.0
        Unit-level OA.2 penalty, penalizes treated deviation from cluster mean (used if design='unit').
    lambda2_unit : float, default=0.0
        Unit-level OA.3 penalty, penalizes pairwise differences between treated and control (used if design='unit').
    costs : ndarray, shape (N_units,), optional
        Vector of costs per unit (e.g., audit fees or implementation expenses).
        If provided, enables budget-constrained optimization for treated units.
    budget : float, int, or dict, optional
        Total budget constraint. If scalar, splits evenly across clusters. If dict,
        maps cluster labels to specific budgets (e.g., {"Willemstad": 12, "non-Willemstad": 8}).
        Requires `costs` to be provided. Enforces ∑ c_j * w_j <= budget per cluster.
    solver : cvxpy Solver, default=cp.ECOS_BB
        Solver used for the optimization problem.
    verbose : bool, default=False
        If True, prints solver progress.

    Returns
    -------
    result : dict
        Dictionary containing synthetic control solution and diagnostics:
            w_opt : ndarray, shape (N_units, K_clusters)
                Optimized weights for synthetic treated units per cluster.
            v_opt : ndarray, shape (N_units, K_clusters)
                Optimized weights for synthetic control units per cluster.
            z_opt : ndarray, shape (N_units, K_clusters)
                Binary selection indicators for units in each cluster.
            y_syn_treated_clusters : list of ndarray
                Synthetic treated trajectories for each cluster.
            y_syn_control_clusters : list of ndarray
                Synthetic control trajectories for each cluster.
            Xbar_clusters : list of ndarray
                Cluster-level mean trajectories (pre-treatment).
            cluster_labels : list
                Unique cluster labels.
            cluster_members : list of ndarray
                Indices of units in each cluster.
            w_agg, v_agg : ndarray
                Aggregated weights across clusters.
            rmse_cluster : list of float
                RMSE between synthetic treated and control fit to cluster mean (pre-treatment).
            design-specific parameters (lambda1, lambda2, beta, xi) returned as applicable.
            costs_used : ndarray or None
                The costs vector if provided, else None.
            budget_used : dict or float
                The effective budget(s) applied per cluster.

    Notes
    -----
    - This function allows experimenters to select treated units in a way that mimics
      randomization by ensuring pre-treatment similarity between treated and control.
    - The optimization enforces similarity while allowing cardinality constraints
      and penalized deviations.
    - Cost/budget constraints apply only to treated weights (w) per cluster, ensuring
      ∑ c_j * w_j <= budget_k for feasibility in resource-limited designs.
    - Realized post-treatment differences between synthetic treated and control
      units provide an estimate of the ATT.
    - If costs and budget are provided, the solver prioritizes affordable units
      while maintaining feature matching; may select fewer than m_max if over budget.

    References
    ----------
    Abadie, A., & Zhao, J. (2025). Synthetic Controls for Experimental Design.
    arXiv:2108.02196. https://arxiv.org/abs/2108.02196
    """
    # --- validation of mutually exclusive design selection ---
    widedf= Y_full
    Y_full= Y_full.to_numpy()
    valid_designs = {"base", "weak", "eq11", "unit"}
    if design not in valid_designs:
        raise ValueError(f"design must be one of {valid_designs}; got '{design}'")

    # --- basic shape checks ---
    if T0 <= 0 or T0 >= Y_full.shape[1]:
        raise ValueError("T0 must be 1 <= T0 < Y_full.shape[1]")
    if blank_periods < 0 or blank_periods >= T0:
        raise ValueError("blank_periods must be 0 <= blank_periods < T0 (need at least 1 fit period)")

    # check incompatible parameter usage (help the user avoid accidental mixes)
    if design != "weak" and beta != 1e-6:
        raise ValueError("beta is only valid when design == 'weak'")
    if design != "eq11" and (lambda1 != 0.0 or lambda2 != 0.0):
        raise ValueError("lambda1/lambda2 are only valid when design == 'eq11'")
    if design != "unit" and (xi != 0.0 or lambda1_unit != 0.0 or lambda2_unit != 0.0):
        raise ValueError("xi/lambda1_unit/lambda2_unit are only valid when design == 'unit'")

    # New: Validate costs and budget
    if costs is not None:
        costs = np.asarray(costs)
        N, _ = Y_full.shape
        if costs.shape[0] != N:
            raise ValueError("costs must have length N (rows of Y).")
        if budget is None:
            raise ValueError("budget must be provided if costs are specified.")
        if isinstance(budget, (int, float)):
            budget = {lab: budget / K for lab in cluster_labels}  # Even split; but K not defined yet - fix below
        elif isinstance(budget, dict):
            for lab in cluster_labels:
                if lab not in budget:
                    raise ValueError(f"budget missing entry for cluster '{lab}'.")

    # --- prepare data slices ---
    T_fit = T0 - blank_periods
    Y_fit = Y_full[:, :T_fit]  # shape (N, T_fit)
    Y_blank = Y_full[:, T_fit:T0] if blank_periods > 0 else None

    N, T_fit_actual = Y_fit.shape
    clusters = np.asarray(clusters)
    if clusters.shape[0] != N:
        raise ValueError("clusters must have length N (rows of Y).")

    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}

    # membership mask M: shape (N, K)
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True

    # cluster-level means and membership lists
    Xbar_clusters = []
    cluster_members = []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members.")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))  # shape (T_fit,)

    # New: Fix budget split if scalar (now K is defined)
    if costs is not None and isinstance(budget, (int, float)):
        budget = {lab: budget / K for lab in cluster_labels}

    # Precompute D1 = || Xbar_k - X_j ||^2 (N x K) if needed for eq11 or unit
    D1 = np.zeros((N, K))
    for k_idx in range(K):
        diffs = Y_fit - Xbar_clusters[k_idx][None, :]  # (N, T_fit)
        D1[:, k_idx] = np.sum(diffs**2, axis=1)

    # Precompute D2 per cluster (pairwise squared distances among members) used by 'unit' design
    D2_list = []
    for k_idx in range(K):
        members = cluster_members[k_idx]
        Xm = Y_fit[members, :]  # (m, T_fit)
        diff = Xm[:, None, :] - Xm[None, :, :]  # (m, m, T_fit)
        D2 = np.sum(diff**2, axis=2)  # (m, m)
        D2_list.append(D2)

    # CVXPY variables
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    z = cp.Variable((N, K), boolean=True)

    constraints = []
    # enforce membership zeros
    for k in range(K):
        for j in range(N):
            if not M[j, k]:
                constraints += [w[j, k] == 0, v[j, k] == 0, z[j, k] == 0]

    # per-cluster normalization + cardinality + linking constraints + cost
    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        constraints += [cp.sum(w[members, k_idx]) == 1]
        constraints += [cp.sum(v[members, k_idx]) == 1]

        m_eq_k = _get_per_cluster_param(m_eq, lab, default=None)
        m_min_k = _get_per_cluster_param(m_min, lab, default=None)
        m_max_k = _get_per_cluster_param(m_max, lab, default=None)

        if m_eq_k is not None:
            constraints += [cp.sum(z[members, k_idx]) == int(m_eq_k)]
        if m_min_k is not None:
            constraints += [cp.sum(z[members, k_idx]) >= int(m_min_k)]
        if m_max_k is not None:
            constraints += [cp.sum(z[members, k_idx]) <= int(m_max_k)]

        for j in members:
            constraints += [w[j, k_idx] <= z[j, k_idx]]
            constraints += [v[j, k_idx] <= 1 - z[j, k_idx]]

        # New: Add cost constraint per cluster
        if costs is not None:
            B_k = _get_per_cluster_param(budget, lab)
            c_k = costs[members]  # Costs for members of this cluster
            constraints += [cp.sum(cp.multiply(c_k, w[members, k_idx])) <= B_k]

    if exclusive:
        for j in range(N):
            constraints += [cp.sum(z[j, :]) <= 1]

    # Build objective depending on design
    Y_T = Y_fit.T  # (T_fit, N)
    obj_terms = []

    if design == "base":
        # Plain cluster-targeted: both treated & control fit to cluster mean
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated_k = Y_T @ w[:, k_idx]
            syn_control_k = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated_k))
            obj_terms.append(cp.sum_squares(Xbar_k - syn_control_k))

    elif design == "weak":
        # Weakly targeted: control is fit to treated (beta * ||treated - control||^2)
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated_k = Y_T @ w[:, k_idx]
            syn_control_k = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated_k))
            obj_terms.append(beta * cp.sum_squares(syn_treated_k - syn_control_k))

    elif design == "eq11":
        # Equation (11) style penalized: base fits + lambda1 on w distances, lambda2 on v distances
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated_k = Y_T @ w[:, k_idx]
            syn_control_k = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated_k))
            obj_terms.append(cp.sum_squares(Xbar_k - syn_control_k))
            if lambda1 > 0:
                obj_terms.append(lambda1 * cp.sum(cp.multiply(w[:, k_idx], D1[:, k_idx])))
            if lambda2 > 0:
                obj_terms.append(lambda2 * cp.sum(cp.multiply(v[:, k_idx], D1[:, k_idx])))

    elif design == "unit":
        # Unit-level penalized design (OA.1 + OA.2 + OA.3)
        # OA.1 (xi): sum_j w_j * || X_j - (Y_fit^T v) ||^2
        # OA.2 (lambda1_unit): sum_j w_j ||Xbar - X_j||^2
        # OA.3 (lambda2_unit): sum_j w_j * sum_i v_i ||X_j - X_i||^2  (pairwise inside cluster)
        for k_idx in range(K):
            members = cluster_members[k_idx]
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated_k = Y_T @ w[:, k_idx]    # (T_fit,)
            syn_control_k = Y_T @ v[:, k_idx]    # (T_fit,)

            # cluster-level fits
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated_k))
            obj_terms.append(cp.sum_squares(Xbar_k - syn_control_k))

            # OA.1: xi * sum_{j in members} w_jk * || X_j - syn_control_k ||^2
            if xi > 0:
                for local_idx, j in enumerate(members):
                    X_j = Y_fit[j, :]  # numpy (T_fit,)
                    # cp expression: xi * w[j,k] * || X_j - syn_control_k ||^2
                    obj_terms.append(xi * w[j, k_idx] * cp.sum_squares(X_j - syn_control_k))

            # OA.2: lambda1_unit * sum_j w_jk * || Xbar_k - X_j ||^2
            if lambda1_unit > 0:
                obj_terms.append(lambda1_unit * cp.sum(cp.multiply(w[:, k_idx], D1[:, k_idx])))

            # OA.3: lambda2_unit * sum_j w_jk * ( Dmat @ v_m )_j
            if lambda2_unit > 0:
                # D2_list[k_idx] is (m,m) for members
                if len(members) > 0:
                    Dmat = D2_list[k_idx]               # numpy (m, m)
                    v_m = v[members, k_idx]            # cp (m,)
                    inner_vec = Dmat @ v_m.value if v_m.value is None else Dmat.dot(v_m)  # Fix for cvxpy compatibility
                    w_m = w[members, k_idx]            # cp (m,)
                    obj_terms.append(lambda2_unit * cp.sum(cp.multiply(w_m, inner_vec)))

    else:
        raise RuntimeError("unhandled design branch (this should not happen)")

    objective = cp.Minimize(cp.sum(obj_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIP, verbose=verbose)

    # extract
    w_opt = w.value
    v_opt = v.value
    z_opt = z.value

    # full predictions on Y_full
    Y_full_T = Y_full.T
    y_syn_treated_clusters = []
    y_syn_control_clusters = []
    for k_idx in range(K):
        w_k = w_opt[:, k_idx]
        v_k = v_opt[:, k_idx]
        y_syn_treated_clusters.append(Y_full_T @ w_k)
        y_syn_control_clusters.append(Y_full_T @ v_k)

    # cluster rmse on fit period (treated vs control)
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
        "beta": beta if design == "weak" else None,
        "lambda1": (lambda1 if design == "eq11" else (lambda1_unit if design == "unit" else None)),
        "lambda2": (lambda2 if design == "eq11" else (lambda2_unit if design == "unit" else None)),
        "xi": (xi if design == "unit" else None),
        "df": widedf,
        "original_cluster_vector":  clusters,
        # New: Cost-related outputs
        "costs_used": costs if costs is not None else None,
        "budget_used": budget
    }
    return result
