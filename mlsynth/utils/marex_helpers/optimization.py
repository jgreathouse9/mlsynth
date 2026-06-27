"""MAREX design optimizers (Abadie & Zhao 2026).

``solve_design`` solves the exact mixed-integer design (binary selection ``z``);
``solve_design_relaxed`` relaxes ``z`` to ``[0, 1]``, solves the QP, then
discretizes post hoc. Both return a raw result dict consumed by the
orchestrator.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError
from .formulation import (
    build_constraints,
    build_objective,
    compute_cluster_means_members,
    get_per_cluster_param,
    init_cvxpy_variables,
    precompute_distances,
    prepare_clusters,
    prepare_fit_slices,
    build_membership_mask,
    validate_costs_budget,
    validate_scm_inputs,
)


def _augment_fit(Y_fit, covariates, covariate_weight, standardize=False):
    """Build the design predictor matrix ``X = [Y^E ; Z]`` (units x predictors).

    The paper's predictor vector is ``X_j = [Y^E_j ; Z_j]`` -- pre-period
    outcomes plus covariates -- so the design matches on both. Covariates are
    appended as extra predictor columns; the synthetic outcome series still uses
    outcomes only. With ``standardize=True`` each predictor column is scaled to
    unit variance across units (the paper's Walmart normalisation), so the match
    is not dominated by high-level series.
    """
    X = Y_fit
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov[:, None]
        X = np.hstack([Y_fit, covariate_weight * cov])
    if standardize:
        sd = X.std(axis=0)
        X = X / np.where(sd == 0, 1.0, sd)
    return X


def _aggregate_weights(w_opt, v_opt, cluster_members, N, K):
    cluster_sizes = [len(m) for m in cluster_members]
    total = sum(cluster_sizes)
    w_agg = np.zeros(N)
    v_agg = np.zeros(N)
    for k_idx in range(K):
        w_agg += (cluster_sizes[k_idx] / total) * w_opt[:, k_idx]
        v_agg += (cluster_sizes[k_idx] / total) * v_opt[:, k_idx]
    return w_agg, v_agg, cluster_sizes


def solve_design(
    Y_full, T0, clusters, blank_periods=0, m_eq=None, m_min=None, m_max=None,
    exclusive=True, design="standard", beta=1e-6, lambda1=0.0, lambda2=0.0, xi=0.0,
    lambda1_unit=0.0, lambda2_unit=0.0, costs=None, budget=None,
    covariates=None, covariate_weight=1.0, standardize=False,
    solver=cp.SCIP, verbose=False, restrictions=None, forbidden=None,
):
    """Exact mixed-integer MAREX design (was ``SCMEXP``).

    ``forbidden`` is an optional list of previously-chosen assignments, each a
    list of ``(unit_idx, cluster_idx)`` pairs; for every one a no-good cut
    ``sum z[pairs] <= |pairs| - 1`` is added, forbidding that exact design so a
    re-solve yields the next-best distinct one (used to build a solution pool).
    """
    validate_scm_inputs(Y_full, T0, blank_periods, design, beta, lambda1,
                        lambda2, xi, lambda1_unit, lambda2_unit)
    Y_full_np, clusters, N, cluster_labels, K, label_to_k = prepare_clusters(Y_full, clusters)
    costs_np, budget_dict = validate_costs_budget(costs, budget, N, cluster_labels, K)
    Y_fit, Y_blank, T_fit = prepare_fit_slices(Y_full_np, T0, blank_periods)
    X_fit = _augment_fit(Y_fit, covariates, covariate_weight, standardize)   # predictors = [Y^E ; Z]
    M = build_membership_mask(clusters, label_to_k, N, K)
    Xbar_clusters, cluster_members = compute_cluster_means_members(X_fit, M, cluster_labels)
    D1, D2_list = precompute_distances(X_fit, Xbar_clusters, cluster_members)

    w, v, z = init_cvxpy_variables(N, K)
    constraints = build_constraints(w, v, z, M, cluster_members, cluster_labels,
                                    m_eq, m_min, m_max, costs_np, budget_dict,
                                    exclusive, restrictions=restrictions)
    # no-good cuts: forbid each previously chosen assignment so a re-solve finds
    # the next-best distinct design (solution pool).
    for pairs in (forbidden or []):
        if pairs:
            constraints = constraints + [
                cp.sum([z[int(j), int(k)] for (j, k) in pairs]) <= len(pairs) - 1]
    objective = build_objective(X_fit, Xbar_clusters, cluster_members, w, v, z,
                               design, beta, lambda1, lambda2, xi,
                               lambda1_unit, lambda2_unit, D1, D2_list)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if z.value is None or w.value is None:
        msg = f"MAREX optimization failed: {prob.status}"
        if restrictions is not None and "infeasible" in str(prob.status).lower():
            msg = ("MAREX design is infeasible under the active restrictions "
                   "(forced/forbidden units, border conflicts, size band) and "
                   "the cluster cardinality (m_min/m_max). Relax them.")
        raise MlsynthEstimationError(msg)
    w_opt, v_opt, z_opt = w.value, v.value, z.value
    if w_opt is None or z_opt is None:
        raise MlsynthEstimationError(
            f"MAREX design is infeasible (solver status: {prob.status}).")
    Y_full_T = Y_full_np.T
    rmse_cluster = []
    for k_idx in range(K):
        treated_idx = np.where(w_opt[:, k_idx] > 1e-8)[0]
        y_treated = (Y_fit[treated_idx, :].T @ w_opt[treated_idx, k_idx]
                     / np.sum(w_opt[treated_idx, k_idx])) if len(treated_idx) > 0 else np.zeros(T_fit)
        y_control = Y_fit.T @ v_opt[:, k_idx]
        rmse_cluster.append(np.sqrt(np.mean((y_treated - y_control) ** 2)))

    w_agg, v_agg, cluster_sizes = _aggregate_weights(w_opt, v_opt, cluster_members, N, K)
    return {
        "df": Y_full, "w_opt": w_opt, "v_opt": v_opt, "z_opt": z_opt,
        "Xbar_clusters": Xbar_clusters, "cluster_labels": list(cluster_labels),
        "cluster_members": cluster_members, "w_agg": w_agg, "v_agg": v_agg,
        "cluster_sizes": cluster_sizes, "T0": T0, "blank_periods": blank_periods,
        "T_fit": T_fit, "Y_fit": Y_fit, "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster, "design": design,
        "original_cluster_vector": clusters,
        "objective": float(prob.value) if prob.value is not None else float("nan"),
    }


def solve_design_pool(Y_full, T0, clusters, *, top_K=1, **kwargs):
    """Enumerate up to ``top_K`` distinct exact designs via no-good cuts.

    Calls :func:`solve_design` repeatedly, forbidding each chosen assignment so
    the next solve returns the next-best distinct design. Stops early when the
    feasible region is exhausted (the solver becomes infeasible).
    """
    designs: list = []
    forbidden: list = []
    for _ in range(int(top_K)):
        try:
            raw = solve_design(Y_full=Y_full, T0=T0, clusters=clusters,
                               forbidden=forbidden, **kwargs)
        except MlsynthEstimationError:
            break
        designs.append(raw)
        z_opt = np.asarray(raw["z_opt"])
        pairs = [(int(j), int(k)) for j, k in zip(*np.where(z_opt > 0.5))]
        if not pairs:                       # nothing selected -> cannot cut further
            break
        forbidden.append(pairs)
    return designs


def post_hoc_discretize(w_opt, v_opt, cluster_members, cluster_labels,
                        m_eq=None, m_min=None, m_max=None, trim_threshold=1e-2,
                        Y_fit=None, Y_blank=None):
    """Round relaxed weights to a feasible integer design (was internal)."""
    K = len(cluster_members)
    w_discrete = np.zeros_like(w_opt)
    v_discrete = np.zeros_like(v_opt)
    selected_treated = [[] for _ in range(K)]
    selected_control = [[] for _ in range(K)]
    rmse_blank = []

    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        w_k = w_opt[members, k_idx].copy()
        v_k = v_opt[members, k_idx].copy()
        w_k[w_k < trim_threshold] = 0
        v_k[v_k < trim_threshold] = 0

        m_eq_k = get_per_cluster_param(m_eq, lab)
        m_min_k = get_per_cluster_param(m_min, lab, default=1)
        m_max_k = get_per_cluster_param(m_max, lab, default=len(members) // 2)
        if m_eq_k is not None:
            m_select = int(m_eq_k)
        else:
            nonzero = np.count_nonzero(w_k) or m_min_k
            m_select = min(max(nonzero, m_min_k), m_max_k)

        top_indices = np.argsort(-w_k)[:m_select]
        treated_idx = members[top_indices]
        selected_treated[k_idx] = treated_idx.tolist()
        w_sel = w_k[top_indices]
        w_k = np.zeros_like(w_k)
        w_k[top_indices] = w_sel / w_sel.sum() if w_sel.sum() > 0 else 1.0 / len(top_indices)
        w_discrete[members, k_idx] = w_k

        control_idx = np.setdiff1d(members, treated_idx)
        selected_control[k_idx] = control_idx.tolist()
        mask = np.isin(members, control_idx)
        v_sel = v_k[mask]
        if v_sel.sum() > 0:
            v_k = np.zeros_like(v_k); v_k[mask] = v_sel / v_sel.sum()
        elif len(control_idx) > 0:
            v_k = np.zeros_like(v_k); v_k[mask] = 1.0 / len(control_idx)
        v_discrete[members, k_idx] = v_k

        if Y_blank is not None and len(treated_idx) > 0 and len(control_idx) > 0:
            y_treated_blank = Y_blank[treated_idx, :].mean(axis=0)
            y_control_blank = Y_blank.T @ v_discrete[:, k_idx]
            rmse_blank.append(np.sqrt(np.mean((y_treated_blank - y_control_blank) ** 2)))
        else:
            rmse_blank.append(None)

    return w_discrete, v_discrete, selected_treated, selected_control, rmse_blank


def solve_design_relaxed(
    Y_full, T0, clusters, blank_periods=0, m_eq=None, m_min=None, m_max=None,
    exclusive=True, design="standard", beta=1e-6, lambda1=0.0, lambda2=0.0, xi=0.0,
    lambda1_unit=0.0, lambda2_unit=0.0, costs=None, budget=None,
    covariates=None, covariate_weight=1.0, standardize=False, solver=None,
    verbose=False, zeta=0.0, trim_threshold=1e-2,
):
    """Relaxed (continuous-``z``) design with post-hoc discretization (was ``SCMEXP_REL``)."""
    validate_scm_inputs(Y_full, T0, blank_periods, design, beta, lambda1,
                        lambda2, xi, lambda1_unit, lambda2_unit)
    Y_full_np, clusters, N, cluster_labels, K, label_to_k = prepare_clusters(Y_full, clusters)
    costs_np, budget_dict = validate_costs_budget(costs, budget, N, cluster_labels, K)
    Y_fit, Y_blank, T_fit = prepare_fit_slices(Y_full_np, T0, blank_periods)
    X_fit = _augment_fit(Y_fit, covariates, covariate_weight, standardize)
    M = build_membership_mask(clusters, label_to_k, N, K)
    Xbar_clusters, cluster_members = compute_cluster_means_members(X_fit, M, cluster_labels)
    D1, D2_list = precompute_distances(X_fit, Xbar_clusters, cluster_members)

    w, v, z = init_cvxpy_variables(N, K, boolean=False)   # continuous z in [0, 1]
    constraints = build_constraints(w, v, z, M, cluster_members, cluster_labels,
                                    m_eq, m_min, m_max, costs_np, budget_dict, exclusive)
    constraints += [z <= 1]
    solver = solver or cp.CLARABEL
    objective = build_objective(X_fit, Xbar_clusters, cluster_members, w, v, z,
                               design, beta, lambda1, lambda2, xi,
                               lambda1_unit, lambda2_unit, D1, D2_list, zeta=zeta)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    w_opt_rel, v_opt_rel, z_opt_rel = w.value, v.value, z.value
    w_opt, v_opt, sel_t, sel_c, rmse_blank = post_hoc_discretize(
        w_opt_rel, v_opt_rel, cluster_members, cluster_labels, m_eq, m_min, m_max,
        trim_threshold=trim_threshold, Y_fit=Y_fit, Y_blank=Y_blank)

    rmse_cluster = []
    for k_idx in range(K):
        treated_idx = sel_t[k_idx]
        y_treated = (Y_fit[treated_idx, :].T @ w_opt[treated_idx, k_idx]
                     / np.sum(w_opt[treated_idx, k_idx])) if len(treated_idx) > 0 else np.zeros(T_fit)
        y_control = Y_fit.T @ v_opt[:, k_idx]
        rmse_cluster.append(np.sqrt(np.mean((y_treated - y_control) ** 2)))

    w_agg, v_agg, cluster_sizes = _aggregate_weights(w_opt, v_opt, cluster_members, N, K)
    return {
        "df": Y_full, "w_opt": w_opt, "v_opt": v_opt, "z_opt": None,
        "w_opt_rel": w_opt_rel, "v_opt_rel": v_opt_rel, "z_opt_rel": z_opt_rel,
        "selected_treated": sel_t, "selected_control": sel_c,
        "Xbar_clusters": Xbar_clusters, "cluster_labels": list(cluster_labels),
        "cluster_members": cluster_members, "w_agg": w_agg, "v_agg": v_agg,
        "cluster_sizes": cluster_sizes, "T0": T0, "blank_periods": blank_periods,
        "T_fit": T_fit, "Y_fit": Y_fit, "Y_blank": Y_blank,
        "rmse_cluster": rmse_cluster, "rmse_blank": rmse_blank, "design": design,
        "original_cluster_vector": clusters, "zeta": zeta,
    }
