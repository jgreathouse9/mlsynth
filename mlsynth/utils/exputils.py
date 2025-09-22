import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
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
    if T0 <= 0 or T0 > Y_full.shape[1]:  # allow equality
        raise ValueError(f"T0 must be 1 <= T0 <= Y_full.shape[1] (got T0={T0}, Y_full.shape[1]={Y_full.shape[1]})")
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

def marexinf(
    Y_treated, Y_control, T0, TcE, Tb, alpha=0.05, max_combinations=1000, random_state=None
):
    """
    Approximate inference for synthetic control using random subsampling.

    This function computes treatment effects, split-conformal confidence intervals,
    and permutation-based p-values for both global and per-period tests using
    pre-treatment "blank" periods as placebos. It follows the approach described
    in Abadie and Zhao (2025).

    Parameters
    ----------
    Y_treated : array-like
        Outcomes for the treated unit (length = T0 + post-treatment).
    Y_control : array-like
        Outcomes for the synthetic control (same length as Y_treated).
    T0 : int
        Number of pre-treatment periods.
    TcE : int
        Number of pre-treatment periods used for fitting the synthetic control.
    Tb : int
        Number of blank pre-treatment periods (not used in fitting).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    max_combinations : int, default=1000
        Number of random subsets sampled for the global permutation test.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'treated_effects': array of treatment effects for post-treatment periods
        - 'placebo_effects': array of effects in blank pre-treatment periods
        - 'S_obs': observed global test statistic
        - 'global_p_value': p-value for the global test
        - 'per_period_pvals': p-values for each post-treatment period
        - 'CI': array of split-conformal confidence intervals (lower, upper)

    References
    ----------
    Abadie, A., & Zhao, J. (2025). Synthetic Controls for Experimental Design.
    arXiv:2108.02196 [stat.ME]. Retrieved from https://arxiv.org/abs/2108.02196
    """

    rng = np.random.default_rng(random_state)
    T_post = len(Y_treated) - T0

    # Effects
    blank_indices = np.arange(TcE, TcE + Tb)
    post_indices = np.arange(T0, T0 + T_post)

    placebo_effects = Y_treated[blank_indices] - Y_control[blank_indices]
    treated_effects = Y_treated[post_indices] - Y_control[post_indices]

    # Combine for permutation test
    all_effects = np.concatenate([placebo_effects, treated_effects])
    n_total = len(all_effects)

    # Random subsampling of subsets
    pi_list = np.array([
        rng.choice(n_total, size=T_post, replace=False)
        for _ in range(max_combinations)
    ])

    # Global test statistic
    S_obs = np.mean(np.abs(treated_effects))
    S_perm = np.mean(np.abs(all_effects[pi_list]), axis=1)
    global_p_value = np.mean(S_perm >= S_obs)

    # Per-period p-values
    per_period_pvals = np.mean(
        np.abs(placebo_effects)[:, None] >= np.abs(treated_effects)[None, :],
        axis=0
    )

    # Confidence intervals (split conformal)
    q = np.quantile(np.abs(placebo_effects), 1 - alpha)
    CI = np.column_stack([treated_effects - q, treated_effects + q])

    return {
        "treated_effects": treated_effects,
        "placebo_effects": placebo_effects,
        "S_obs": S_obs,
        "global_p_value": global_p_value,
        "per_period_pvals": per_period_pvals,
        "CI": CI
    }






def plot_cluster_full(df, marex_results, show=True, save_path=None):
    """
    Plots all units in each cluster (thin gray), synthetic treated (black),
    and synthetic control (blue). Blank periods (predicted but not fitted) are shaded in orange.
    """
    style_params = {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 14,
        "font.family": "sans-serif",
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": "large",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "legend.fontsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d3d3d3",
        "grid.linestyle": ":",
        "grid.linewidth": 1.0,
        "lines.linewidth": 1.0,
        "lines.marker": "",
        "lines.markersize": 0,
    }

    clusters = marex_results.clusters
    n_clusters = len(clusters)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    # Determine blank period indices from global Y_blank
    Y_blank = marex_results.globres.Y_blank
    if Y_blank is not None:
        blank_mask = ~np.isnan(Y_blank[0])  # True for blank periods
        if blank_mask.any():
            blank_start = marex_results.study.T0 - Y_blank.shape[1]
            blank_end = blank_start + blank_mask.sum()
        else:
            blank_start = blank_end = None
    else:
        blank_start = blank_end = None

    with plt.rc_context(style_params):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(style_params["figure.figsize"][0], style_params["figure.figsize"][1] * n_rows)
        )
        axes = axes.flatten() if n_clusters > 1 else [axes]

        for i, (cluster_id, cluster_res) in enumerate(clusters.items()):
            ax = axes[i]

            # Plot all units as thin gray lines
            for member in cluster_res.members:
                member_data = df[df[df.columns[0]] == member].sort_values('time')  # first col is unitid
                ax.plot(member_data['time'], member_data['Y_obs'], color='lightgray', linewidth=0.8)

            # Plot synthetic treated and control
            x = range(1, len(cluster_res.synthetic_treated) + 1)
            ax.plot(x, cluster_res.synthetic_treated, color='black', linewidth=2, label='Synthetic Treated')
            ax.plot(x, cluster_res.synthetic_control, color='blue', linewidth=2, label='Synthetic Control')

            # Shade blank periods
            if blank_start is not None and blank_end is not None:
                ax.axvspan(blank_start + 1, blank_end, color='orange', alpha=0.2, label='Blank Periods')

            ax.set_title(f'Cluster {cluster_id}')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Outcome')
            ax.grid(True)
            ax.legend(fontsize=8)

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
