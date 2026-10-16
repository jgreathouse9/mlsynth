"""MAREX design optimizers (Abadie & Zhao 2026).

``solve_design`` solves the exact mixed-integer design (binary selection ``z``);
``solve_design_relaxed`` relaxes ``z`` to ``[0, 1]``, solves the QP, then
discretizes post hoc. Both return a raw result dict consumed by the
orchestrator.

SCIP accepts a binary within its integrality tolerance of an integer, so a
selection left near ``1e-6`` rounds to "not selected" while still admitting a
treated weight of that size, and a market can carry weight on both sides of the
design. ``solve_design`` therefore takes only the selection from SCIP: it fixes
``z`` at its rounding and refits the weights by :func:`refit_selection` with
the selection held constant, so the weights and objective reported are those of
the design the selection defines.
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

    ``covariate_weight`` multiplies the covariate columns after the
    standardisation, as in Vives-i-Bastida (2022, Algorithm 1): standardise
    every predictor, then weight it. Weighting first would be undone by the
    division, leaving every weight with the same design. A column that is
    constant across units, up to rounding, is left unscaled: under simplex
    weights it adds nothing to the objective, and dividing its rounding noise
    would turn it into a column of order 1e15.
    """
    X = Y_fit
    n_out = Y_fit.shape[1]
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov[:, None]
        X = np.hstack([Y_fit, cov])
    if standardize:
        sd = X.std(axis=0)
        flat = sd <= 1e-12 * np.abs(X).max(axis=0, initial=0.0)
        X = X / np.where(flat, 1.0, sd)
    if covariates is not None:
        X = np.hstack([X[:, :n_out], covariate_weight * X[:, n_out:]])
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


def finish_design(Y_full, Y_full_np, w_opt, v_opt, z_opt, Xbar_clusters,
                  cluster_labels, cluster_members, N, K, T0, blank_periods,
                  T_fit, Y_fit, Y_blank, design, clusters, objective):
    """The design dict both backends return, from their solved weights.

    ``solve_design`` (the MIQP) and
    :func:`mlsynth.utils.marex_helpers.exact.solve_design_exact` (the search)
    reach ``w``, ``v`` and ``z`` differently and hand them here, so everything
    downstream -- inference, post-fit diagnostics, the result object and the
    solution pool -- reads one structure whichever backend produced it.
    """
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
        "objective": objective,
    }


#: Designs whose refit is two exact simplex fits per cluster (see refit_selection).
_SEPARABLE = frozenset({"standard", "penalized", "unit_penalized"})


def refit_selection(X_fit, Xbar_clusters, cluster_members, cluster_labels, M,
                    z_sel, *, m_eq=None, m_min=None, m_max=None, costs=None,
                    budget_dict=None, exclusive=True, restrictions=None,
                    design="standard", beta=1e-6, lambda1=0.0, lambda2=0.0,
                    xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0, D1=None,
                    D2_list=None, solver=cp.SCIP, control_penalty=0.0):
    """The weights of a fixed selection ``z_sel``, and their objective.

    ``z_sel`` is rounded at 0.5 before anything else reads it.

    For the ``standard``, ``penalized`` and ``unit_penalized`` designs with no
    cost budget and exclusive arms, fixing the selection splits each cluster
    into a treated fit and a control fit, and both are solved exactly by the
    active-set simplex solver: the computation the exact backend's search runs
    on each candidate, so a weight the optimum puts at zero is exactly zero.
    Otherwise the MIQP's own constraints and objective are rebuilt with ``z``
    continuous and pinned to ``z_sel``, a convex QP, and handed to ``solver``,
    the solver the MIQP used.

    Raises
    ------
    MlsynthEstimationError
        When the selection breaks the cardinality or a restriction, leaves a
        cluster with no control market, or the QP is not solved to optimality.
    """
    if (design in _SEPARABLE and costs is None and budget_dict is None
            and exclusive):
        return _refit_by_simplex_fits(
            X_fit, Xbar_clusters, cluster_members, cluster_labels, M, z_sel,
            m_eq=m_eq, m_min=m_min, m_max=m_max, restrictions=restrictions,
            design=design, lambda1=lambda1, lambda2=lambda2, xi=xi,
            lambda1_unit=lambda1_unit, D1=D1)
    return _refit_by_qp(
        X_fit, Xbar_clusters, cluster_members, cluster_labels, M, z_sel,
        m_eq=m_eq, m_min=m_min, m_max=m_max, costs=costs,
        budget_dict=budget_dict, exclusive=exclusive, restrictions=restrictions,
        design=design, beta=beta, lambda1=lambda1, lambda2=lambda2, xi=xi,
        lambda1_unit=lambda1_unit, lambda2_unit=lambda2_unit, D1=D1,
        D2_list=D2_list, solver=solver, control_penalty=control_penalty)


def _refit_by_simplex_fits(X_fit, Xbar_clusters, cluster_members, cluster_labels,
                           M, z_sel, *, m_eq, m_min, m_max, restrictions, design,
                           lambda1, lambda2, xi, lambda1_unit, D1):
    from .exact import _local_rules, _penalties, side_fit   # exact imports this module

    N, K = M.shape
    selected = np.asarray(z_sel, dtype=float) > 0.5
    cluster_of = np.array([int(np.argmax(M[j])) for j in range(N)])
    w_fit, v_fit, total = np.zeros((N, K)), np.zeros((N, K)), 0.0
    for k_idx, lab in enumerate(cluster_labels):
        members = np.asarray(cluster_members[k_idx])
        treated = [i for i, j in enumerate(members) if selected[j, k_idx]]
        me = get_per_cluster_param(m_eq, lab)
        mn = get_per_cluster_param(m_min, lab)
        mx = get_per_cluster_param(m_max, lab)
        if ((me is not None and len(treated) != int(me))
                or (mn is not None and len(treated) < int(mn))
                or (mx is not None and len(treated) > int(mx))):
            raise MlsynthEstimationError(
                f"MAREX cannot refit the selected design: cluster {lab!r} has "
                f"{len(treated)} selected markets (m_eq={me}, m_min={mn}, "
                f"m_max={mx}).")
        filt, donors = _local_rules(restrictions, members, cluster_of)
        if not treated or not filt.accepts(tuple(treated)):
            raise MlsynthEstimationError(
                f"MAREX cannot refit the selected design: the selection in "
                f"cluster {lab!r} breaks a forced, forbidden, conflict or "
                f"stratum restriction.")
        drop = set(treated)
        for i, j in donors:
            if i in drop:
                drop.add(j)
        control = [i for i in range(len(members)) if i not in drop]
        if not control:
            raise MlsynthEstimationError(
                f"MAREX cannot refit the selected design: cluster {lab!r} has "
                f"no control market left.")
        B, A = X_fit[members].T, Xbar_clusters[k_idx]
        a, b = _penalties(design, k_idx, members, X_fit, D1, lambda1, lambda2,
                          xi, lambda1_unit)
        g, w = side_fit(B, A, treated, None if a is None else a[treated])
        h, v = side_fit(B, A, control, None if b is None else b[control])
        w_fit[members[treated], k_idx] = w
        v_fit[members[control], k_idx] = v
        total += g + h
    return w_fit, v_fit, float(total)


def _refit_by_qp(X_fit, Xbar_clusters, cluster_members, cluster_labels, M, z_sel,
                 *, m_eq, m_min, m_max, costs, budget_dict, exclusive,
                 restrictions, design, beta, lambda1, lambda2, xi, lambda1_unit,
                 lambda2_unit, D1, D2_list, solver, control_penalty=0.0):
    N, K = M.shape
    # Rounded here as well as by the caller: a selection pinned at a binary a
    # solver left near 1e-6 would admit a treated weight of that size, the
    # defect the refit exists to remove.
    z_fixed = (np.asarray(z_sel, dtype=float) > 0.5).astype(float)
    w, v, z = init_cvxpy_variables(N, K, boolean=False)
    constraints = build_constraints(w, v, z, M, cluster_members, cluster_labels,
                                    m_eq, m_min, m_max, costs, budget_dict,
                                    exclusive, restrictions=restrictions)
    constraints = constraints + [z == z_fixed]
    objective = build_objective(X_fit, Xbar_clusters, cluster_members, w, v, z,
                                design, beta, lambda1, lambda2, xi,
                                lambda1_unit, lambda2_unit, D1, D2_list,
                                control_penalty=control_penalty)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)
    if prob.status != cp.OPTIMAL or w.value is None or v.value is None:
        raise MlsynthEstimationError(
            f"MAREX cannot refit the selected design (solver status: "
            f"{prob.status}); the selection is infeasible under the design's "
            f"constraints or the QP did not solve.")
    # Entries the constraints force to zero are set to exactly zero: the
    # orchestrator classifies markets by their weights, and a solver leaves
    # them at its own tolerance.
    selected = z_fixed > 0.5
    w_fit = np.where(selected, np.maximum(w.value, 0.0), 0.0)
    v_fit = np.where(np.asarray(M) > 0, np.maximum(v.value, 0.0), 0.0)
    if exclusive:
        v_fit = np.where(selected, 0.0, v_fit)
    return w_fit, v_fit, float(prob.value)


def solve_design(
    Y_full, T0, clusters, blank_periods=0, m_eq=None, m_min=None, m_max=None,
    exclusive=True, design="standard", beta=1e-6, lambda1=0.0, lambda2=0.0, xi=0.0,
    lambda1_unit=0.0, lambda2_unit=0.0, costs=None, budget=None,
    covariates=None, covariate_weight=1.0, standardize=False,
    solver=cp.SCIP, verbose=False, restrictions=None, forbidden=None,
    warm_start=None, time_limit=None, control_penalty=0.0,
):
    """Exact mixed-integer MAREX design (was ``SCMEXP``).

    ``forbidden`` is an optional list of previously-chosen assignments, each a
    list of ``(unit_idx, cluster_idx)`` pairs; for every one a no-good cut
    ``sum z[pairs] <= |pairs| - 1`` is added, forbidding that exact design so a
    re-solve yields the next-best distinct one (used to build a solution pool).

    ``warm_start`` is an optional iterable of treated unit indices (row order of
    ``Y_full``); each is seeded ``z[j, cluster(j)] = 1`` as a SCIP partial-solution
    MIP start. It is a hint -- the proven optimum is unchanged -- but gives the
    branch-and-bound a strong incumbent immediately. ``time_limit`` (seconds) caps
    the SCIP solve and returns the best incumbent found; with both set, MAREX can
    refine a seed design under a budget instead of proving optimality. Defaults of
    ``None`` leave the original solve path untouched.
    """
    validate_scm_inputs(Y_full, T0, blank_periods, design, beta, lambda1,
                        lambda2, xi, lambda1_unit, lambda2_unit,
                        control_penalty=control_penalty)
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
                               lambda1_unit, lambda2_unit, D1, D2_list,
                               control_penalty=control_penalty)
    prob = cp.Problem(objective, constraints)
    if warm_start is None and time_limit is None:
        prob.solve(solver=solver, verbose=verbose)          # original solve path
    else:
        scip_params = {"limits/time": float(time_limit)} if time_limit else None
        if warm_start is not None:
            from .warmstart import solve_warmstarted
            z_warm = np.zeros((N, K))
            for j in warm_start:
                z_warm[int(j), label_to_k[clusters[int(j)]]] = 1.0
            solve_warmstarted(
                prob, z, z_warm, verbose=verbose,
                solver_opts={"scip_params": scip_params} if scip_params else None)
        else:
            eff_solver = solver if solver is not None else cp.SCIP
            kw = {"solver": eff_solver, "verbose": verbose}
            if scip_params:
                kw["scip_params"] = scip_params
            prob.solve(**kw)

    if z.value is None or w.value is None:
        if time_limit is not None:
            raise MlsynthEstimationError(
                f"MAREX found no feasible design within time_limit={time_limit}s "
                f"(solver status: {prob.status}); increase the budget.")
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

    # Guard against a solver that reports "optimal" but returns a point that
    # violates the integer cardinality (SCIP can do this intermittently on the
    # harder constrained re-solves of the no-good-cut pool, yielding a degenerate
    # all-units design). Reject such a solve so it never poisons the pool; the
    # pool loop treats this as the feasible region being exhausted.
    z_round = np.asarray(z_opt) > 0.5
    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        sel = int(z_round[members, k_idx].sum())
        me = get_per_cluster_param(m_eq, lab)
        mn = get_per_cluster_param(m_min, lab)
        mx = get_per_cluster_param(m_max, lab)
        if ((me is not None and sel != int(me))
                or (mn is not None and sel < int(mn))
                or (mx is not None and sel > int(mx))):
            raise MlsynthEstimationError(
                f"MAREX solver returned a cardinality-violating design for "
                f"cluster {lab!r} (selected {sel}; expected m_eq={me}, "
                f"m_min={mn}, m_max={mx}); treating the feasible region as "
                f"exhausted."
            )
    z_opt = z_round.astype(float)
    w_opt, v_opt, objective = refit_selection(
        X_fit, Xbar_clusters, cluster_members, cluster_labels, M, z_opt,
        m_eq=m_eq, m_min=m_min, m_max=m_max, costs=costs_np,
        budget_dict=budget_dict, exclusive=exclusive, restrictions=restrictions,
        design=design, beta=beta, lambda1=lambda1, lambda2=lambda2, xi=xi,
        lambda1_unit=lambda1_unit, lambda2_unit=lambda2_unit, D1=D1,
        D2_list=D2_list, solver=solver if solver is not None else cp.SCIP,
        control_penalty=control_penalty)
    return finish_design(Y_full, Y_full_np, w_opt, v_opt, z_opt, Xbar_clusters,
                         cluster_labels, cluster_members, N, K, T0,
                         blank_periods, T_fit, Y_fit, Y_blank, design, clusters,
                         objective)


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
    verbose=False, zeta=0.0, trim_threshold=1e-2, control_penalty=0.0,
):
    """Relaxed (continuous-``z``) design with post-hoc discretization (was ``SCMEXP_REL``)."""
    validate_scm_inputs(Y_full, T0, blank_periods, design, beta, lambda1,
                        lambda2, xi, lambda1_unit, lambda2_unit,
                        control_penalty=control_penalty)
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
                               lambda1_unit, lambda2_unit, D1, D2_list, zeta=zeta,
                               control_penalty=control_penalty)
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
