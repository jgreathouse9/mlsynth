"""Design-formulation primitives for MAREX (Abadie & Zhao 2026).

The experimenter chooses treated weights ``w`` and control weights ``v`` per
cluster on the simplex, with a binary selection mask ``z`` linking them
(``w_j <= z_j``, ``v_j <= 1 - z_j``) so a unit is either treated or a control,
never both (the disjointness ``w_j v_j = 0``). These helpers build the cvxpy
variables, constraints, and the design-specific objective; the objective form
is selected by ``design``:

* ``"standard"`` -- match each cluster mean with both synthetic units;
* ``"weakly_targeted"`` -- match the treated synthetic to the mean and softly
  tie the control synthetic to it (weight ``beta``);
* ``"penalized"`` -- ``standard`` plus cluster-level distance penalties
  (``lambda1`` / ``lambda2``);
* ``"unit_penalized"`` -- ``standard`` plus unit-level penalties (``xi`` /
  ``lambda1_unit``).

Two further penalties appear in the paper's text but have no convex form in
these variables, and are refused with a reason: ``lambda2_unit`` pairs ``w``
against ``v`` through a distance matrix (bilinear in two decision variables),
and ``zeta`` minimises ``z(1 - z)`` (concave). The relaxed path reaches
integrality by discretising after the solve, not by penalising during it.

``xi`` is the eq. (10) unit-level term. It weights each candidate's own
reproducibility by that candidate's treated weight, so the inner quantity is a
minimum over the per-unit control weights and is therefore constant in ``w``:
:func:`per_unit_reproducibility` evaluates those minima once and the penalty
enters linearly. Multiplying ``w`` by a convex function of the shared joint
``v`` instead is not the same objective and is not convex in ``(w, v)``.
"""

from __future__ import annotations

import numbers

import cvxpy as cp

from ...exceptions import MlsynthConfigError
from ..solvers.active_set import solve_simplex_qp
import numpy as np
import pandas as pd


def get_per_cluster_param(param, klabel, default=None):
    """Resolve a possibly-per-cluster parameter to its value for ``klabel``."""
    if param is None:
        return default
    if isinstance(param, dict):
        return param.get(klabel, default)
    return param  # scalar


def prepare_clusters(Y_full, clusters):
    """Coerce ``Y_full``/``clusters`` to arrays and return cluster bookkeeping."""
    Y_full_np = Y_full.to_numpy() if isinstance(Y_full, pd.DataFrame) else np.asarray(Y_full)
    clusters = np.asarray(clusters)
    N = Y_full_np.shape[0]
    if clusters.shape[0] != N:
        raise ValueError("clusters must have length N (rows of Y)")
    cluster_labels = np.unique(clusters)
    K = len(cluster_labels)
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}
    return Y_full_np, clusters, N, cluster_labels, K, label_to_k


def validate_scm_inputs(Y_full, T0, blank_periods, design,
                        beta=1e-6, lambda1=0.0, lambda2=0.0,
                        xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0):
    """Validate shapes and design/parameter compatibility (raises ValueError)."""
    if T0 <= 0 or T0 > Y_full.shape[1]:
        raise ValueError(f"T0 must be 1 <= T0 <= Y_full.shape[1] (got T0={T0}, "
                         f"Y_full.shape[1]={Y_full.shape[1]})")
    if blank_periods < 0 or blank_periods >= T0:
        raise ValueError("blank_periods must be 0 <= blank_periods < T0 (need at least 1 fit period)")
    if design != "weakly_targeted" and beta != 1e-6:
        raise ValueError("beta is only valid when design == 'weakly_targeted'")
    if design != "penalized" and (lambda1 != 0.0 or lambda2 != 0.0):
        raise ValueError("lambda1/lambda2 are only valid when design == 'penalized'")
    if design != "unit_penalized" and (xi != 0.0 or lambda1_unit != 0.0 or lambda2_unit != 0.0):
        raise ValueError("xi/lambda1_unit/lambda2_unit are only valid when design == 'unit_penalized'")


def validate_costs_budget(costs, budget, N, cluster_labels, K):
    """Validate cost/budget inputs; return ``(costs_np, budget_dict)``."""
    if costs is None:
        return None, None
    costs_np = np.asarray(costs)
    if costs_np.shape[0] != N:
        raise ValueError("costs must have length N (rows of Y).")
    if budget is None:
        raise ValueError("budget must be provided if costs are specified.")
    # numbers.Real so a numpy scalar budget takes the scalar branch rather than
    # falling through to "must be a scalar or dict" -- see issue #320.
    if isinstance(budget, numbers.Real) and not isinstance(budget, bool):
        budget_dict = {lab: budget / K for lab in cluster_labels}
    elif isinstance(budget, dict):
        for lab in cluster_labels:
            if lab not in budget:
                raise ValueError(f"budget missing entry for cluster '{lab}'.")
        budget_dict = budget
    else:
        raise TypeError("budget must be a scalar or dict if costs are provided.")
    return costs_np, budget_dict


def prepare_fit_slices(Y_full_np, T0, blank_periods):
    """Split the pre-period into a fitting slice and a held-out blank slice."""
    T_fit = T0 - blank_periods
    Y_fit = Y_full_np[:, :T_fit]
    Y_blank = Y_full_np[:, T_fit:T0] if blank_periods > 0 else None
    return Y_fit, Y_blank, T_fit


def build_membership_mask(clusters, label_to_k, N, K):
    """Boolean ``(N, K)`` mask of unit-to-cluster membership."""
    M = np.zeros((N, K), dtype=bool)
    for j in range(N):
        M[j, label_to_k[clusters[j]]] = True
    return M


def compute_cluster_means_members(Y_fit, M, cluster_labels):
    """Per-cluster predictor means and member index arrays."""
    Xbar_clusters, cluster_members = [], []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.where(M[:, k_idx])[0]
        if members.size == 0:
            raise ValueError(f"Cluster '{lab}' has no members.")
        cluster_members.append(members)
        Xbar_clusters.append(Y_fit[members, :].mean(axis=0))
    return Xbar_clusters, cluster_members


def precompute_distances(Y_fit, Xbar_clusters, cluster_members):
    """Unit-to-cluster-mean distances ``D1`` and within-cluster pairwise ``D2``."""
    N, K = Y_fit.shape[0], len(Xbar_clusters)
    D1 = np.zeros((N, K))
    for k_idx in range(K):
        diffs = Y_fit - Xbar_clusters[k_idx][None, :]
        D1[:, k_idx] = np.sum(diffs ** 2, axis=1)
    D2_list = []
    for members in cluster_members:
        Xm = Y_fit[members, :]
        diff = Xm[:, None, :] - Xm[None, :, :]
        D2_list.append(np.sum(diff ** 2, axis=2))
    return D1, D2_list


def init_cvxpy_variables(N, K, boolean=True):
    """Treated (``w``), control (``v``) weights and selection (``z``).

    ``z`` is binary for the exact MIQP (``boolean=True``) or continuous in
    ``[0, 1]`` for the relaxed QP (``boolean=False``).
    """
    w = cp.Variable((N, K), nonneg=True)
    v = cp.Variable((N, K), nonneg=True)
    if boolean:
        z = cp.Variable((N, K), boolean=True)
    else:
        z = cp.Variable((N, K), nonneg=True)
    return w, v, z


def build_constraints(w, v, z, M, cluster_members, cluster_labels,
                      m_eq, m_min, m_max, costs, budget_dict, exclusive,
                      restrictions=None):
    """Simplex, disjointness, cardinality, cost, exclusivity, and (optional)
    geographic-restriction constraints."""
    N, K = M.shape
    constraints = []
    for k in range(K):
        for j in range(N):
            if not M[j, k]:
                constraints += [w[j, k] == 0, v[j, k] == 0, z[j, k] == 0]

    for k_idx, lab in enumerate(cluster_labels):
        members = cluster_members[k_idx]
        constraints += [cp.sum(w[members, k_idx]) == 1]
        constraints += [cp.sum(v[members, k_idx]) == 1]

        m_eq_k = get_per_cluster_param(m_eq, lab)
        m_min_k = get_per_cluster_param(m_min, lab)
        m_max_k = get_per_cluster_param(m_max, lab)
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
            B_k = get_per_cluster_param(budget_dict, lab)
            c_k = costs[members]
            constraints += [cp.sum(cp.multiply(c_k, w[members, k_idx])) <= B_k]

    if exclusive:
        for j in range(N):
            constraints += [cp.sum(z[j, :]) <= 1]

    if restrictions is not None:
        from .restrictions import apply_restrictions_marex
        constraints += apply_restrictions_marex(z, v, M, restrictions)
    return constraints



def per_unit_reproducibility(Y_fit, members):
    r"""``min_v ||x_j - sum_i v_i x_i||^2`` for each unit, over the simplex.

    Equation (10)'s penalty is ``xi * sum_j w_j * min_{v_.j} ||x_j - sum_i
    v_ij x_i||^2``. The inner weights appear in exactly one term of the
    objective, multiplied by the non-negative scalar ``w_j``, and a positive
    scalar cannot move an argmin -- so ``v*_.j`` does not depend on ``w`` and
    the minimum is a constant. That is what makes the term linear in ``w``, and
    therefore something a convex program can carry.

    Written instead as ``w_j`` times an *evaluated* squared norm of a joint
    variable, the term is a product of two variables: the Hessian of ``w v^2``
    is ``[[0, 2v], [2v, 2w]]`` with determinant ``-4 v^2``, so it is genuinely
    non-convex and no solver will take it.

    Each unit is fitted against every other unit in the panel. Equation (10)
    excludes the treated set from one another's donor pools, which cannot be
    imposed here because the assignment is itself a decision variable; the
    difference is the ``m - 1`` donors a chosen tuple would remove, measured
    elsewhere at a median 0.0 to 0.7 per cent in the penalty with the same
    design selected in 11 of 12 cells.
    """
    N = Y_fit.shape[0]
    out = np.zeros(N, dtype=float)
    for j in range(N):
        others = [i for i in range(N) if i != j]
        if not others:
            continue
        D = Y_fit[others, :].T
        a = solve_simplex_qp(D, Y_fit[j, :])
        residual = Y_fit[j, :] - D @ a
        out[j] = float(residual @ residual)
    return out


def build_objective(Y_fit, Xbar_clusters, cluster_members, w, v, z,
                    design, beta=1e-6, lambda1=0.0, lambda2=0.0,
                    xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0,
                    D1=None, D2_list=None, zeta=0.0):
    """Design-specific cvxpy objective (see module docstring).

    ``lambda2_unit`` and a non-zero ``zeta`` raise
    :class:`~mlsynth.exceptions.MlsynthConfigError`: neither has a convex form
    in these variables. The relaxed (continuous-``z``) solve discretises after
    the fact via ``post_hoc_discretize``.
    """
    Y_T = Y_fit.T
    obj_terms = []
    K = len(cluster_members)

    if design == "standard":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - Y_T @ w[:, k_idx]))
            obj_terms.append(cp.sum_squares(Xbar_k - Y_T @ v[:, k_idx]))

    elif design == "weakly_targeted":
        for k_idx in range(K):
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated))
            obj_terms.append(beta * cp.sum_squares(syn_treated - syn_control))

    elif design == "penalized":
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

    elif design == "unit_penalized":
        for k_idx in range(K):
            members = cluster_members[k_idx]
            Xbar_k = Xbar_clusters[k_idx]
            syn_treated = Y_T @ w[:, k_idx]
            syn_control = Y_T @ v[:, k_idx]
            obj_terms.append(cp.sum_squares(Xbar_k - syn_treated))
            obj_terms.append(cp.sum_squares(Xbar_k - syn_control))
            if xi > 0:
                # Equation (10) multiplies w_j by a MINIMUM over v_.j, which is
                # a constant in w, so the term is linear. Multiplying w_j by an
                # evaluated squared norm of the shared control variable instead
                # makes it a product of two variables -- non-convex, and
                # rejected before any solver sees it.
                d = per_unit_reproducibility(Y_fit, members)
                obj_terms.append(
                    xi * cp.sum(cp.multiply(w[members, k_idx], d[members])))
            if lambda1_unit > 0:
                obj_terms.append(lambda1_unit * cp.sum(cp.multiply(w[members, k_idx], D1[members, k_idx])))
            if lambda2_unit > 0 and len(members) > 1:
                # w' D v is bilinear in two decision variables. Unlike the xi
                # term there is no inner minimum to precompute, so there is
                # nothing to linearise and the program cannot be made convex.
                raise MlsynthConfigError(
                    "lambda2_unit pairs the treated weights against the control "
                    "weights through a distance matrix, which is bilinear in two "
                    "decision variables and leaves no convex program to solve. "
                    "Use lambda1_unit, which is linear in w, or the 'penalized' "
                    "design, whose lambda1 and lambda2 weigh each side against "
                    "the population target separately."
                )

    if zeta and zeta > 0:
        # z(1 - z) is concave, so minimising it is non-convex. Nothing in the
        # package sets this -- the relaxed solver's default is 0 -- so the
        # parameter has never run, and its docstring claimed otherwise.
        raise MlsynthConfigError(
            "zeta minimises z(1 - z), which is concave, so the program is not "
            "convex and no solver will accept it. The relaxed path discretises "
            "after solving (post_hoc_discretize) instead of penalising "
            "fractional assignments during the solve."
        )

    return cp.Minimize(cp.sum(obj_terms))
