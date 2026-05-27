"""Top-level MAREX solve: run the design optimizer and assemble frozen results."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .inference import compute_inference
from .optimization import solve_design, solve_design_relaxed
from .structures import (
    MAREXClusterDesign,
    MAREXGlobalDesign,
    MAREXInference,
    MAREXResults,
    MAREXStudy,
)


def solve_marex(
    Y_full,
    T0,
    clusters,
    design="standard",
    blank_periods=0,
    m_eq=None,
    m_min=None,
    m_max=None,
    exclusive=True,
    beta=1e-6,
    lambda1=0.0,
    lambda2=0.0,
    xi=0.0,
    lambda1_unit=0.0,
    lambda2_unit=0.0,
    costs=None,
    budget=None,
    covariates=None,
    covariate_weight=1.0,
    solver=None,
    verbose=False,
    relaxed=False,
    inference=False,
    alpha=0.05,
    max_combinations=1000,
    random_state=42,
) -> MAREXResults:
    """Solve the MAREX design and return a frozen :class:`MAREXResults`.

    With ``relaxed=True`` the continuous-``z`` QP with post-hoc discretization is
    used; otherwise the exact MIQP. With ``inference=True``, blank-period
    placebo inference is computed for every cluster and the aggregate.
    """
    solve = solve_design_relaxed if relaxed else solve_design
    raw = solve(
        Y_full=Y_full, T0=T0, clusters=clusters, blank_periods=blank_periods,
        m_eq=m_eq, m_min=m_min, m_max=m_max, exclusive=exclusive, design=design,
        beta=beta, lambda1=lambda1, lambda2=lambda2, xi=xi,
        lambda1_unit=lambda1_unit, lambda2_unit=lambda2_unit,
        costs=costs, budget=budget, covariates=covariates,
        covariate_weight=covariate_weight, solver=solver, verbose=verbose,
    )

    df = raw["df"]
    Y_full_np = df.to_numpy() if hasattr(df, "to_numpy") else np.asarray(df)
    unit_labels = list(df.index) if hasattr(df, "index") else list(range(Y_full_np.shape[0]))
    w_opt, v_opt, z_opt = raw["w_opt"], raw["v_opt"], raw.get("z_opt")
    clusters_vec = raw["original_cluster_vector"]
    cluster_labels = raw["cluster_labels"]
    label_to_k = {lab: i for i, lab in enumerate(cluster_labels)}
    T0_eff = raw["T0"]
    TcE = T0_eff - blank_periods

    def _maybe_infer(syn_t, syn_c):
        if not (inference and blank_periods > 0):
            return None
        return compute_inference(syn_t, syn_c, T0_eff, TcE, blank_periods,
                                 alpha=alpha, max_combinations=max_combinations,
                                 random_state=random_state)

    # swap labels so the *treated* group is the smaller-support set (Abadie &
    # Zhao's convention: treat as few units as possible), ties broken by the
    # earlier first-treated index.
    def _first_pos(x):
        nz = np.where(x > 1e-8)[0]
        return int(nz[0]) if nz.size else len(x)

    w_sw, v_sw = w_opt.copy(), v_opt.copy()
    for lab in np.unique(clusters_vec):
        k = label_to_k[lab]
        tw, cw = w_opt[:, k], v_opt[:, k]
        n_t, n_c = int((tw > 1e-8).sum()), int((cw > 1e-8).sum())
        if (n_t > n_c) or (n_t == n_c and _first_pos(tw) > _first_pos(cw)):
            w_sw[:, k], v_sw[:, k] = cw, tw

    # per-cluster designs
    clusters_out = {}
    for lab in np.unique(clusters_vec):
        k = label_to_k[lab]
        idx = np.where(clusters_vec == lab)[0]
        members = [unit_labels[i] for i in idx]
        treated_w = w_sw[:, k]
        control_w = v_sw[:, k]
        sel = (treated_w[idx] > 1e-8).astype(float)
        syn_t = treated_w @ Y_full_np
        syn_c = control_w @ Y_full_np
        uwm = {
            "Treated": {unit_labels[i]: float(treated_w[i]) for i in range(len(unit_labels)) if treated_w[i] > 1e-8},
            "Control": {unit_labels[i]: float(control_w[i]) for i in range(len(unit_labels)) if control_w[i] > 1e-8},
        }
        clusters_out[str(lab)] = MAREXClusterDesign(
            label=str(lab), members=members, cardinality=len(members),
            treated_weights=treated_w, control_weights=control_w,
            selection_indicators=sel, synthetic_treated=syn_t,
            synthetic_control=syn_c, pre_treatment_means=raw["Xbar_clusters"][k],
            rmse=float(raw["rmse_cluster"][k]), unit_weight_map=uwm,
            inference=_maybe_infer(syn_t, syn_c),
        )

    # aggregate (global) design
    sizes = raw["cluster_sizes"]; total = sum(sizes)
    w_agg = sum((sizes[label_to_k[lab]] / total) * w_sw[:, label_to_k[lab]]
                for lab in np.unique(clusters_vec))
    v_agg = sum((sizes[label_to_k[lab]] / total) * v_sw[:, label_to_k[lab]]
                for lab in np.unique(clusters_vec))
    is_treated = np.zeros(len(unit_labels), dtype=bool)
    for i, lab in enumerate(clusters_vec):
        is_treated[i] = w_sw[i, label_to_k[lab]] > 1e-8
    adj_treated = np.where(is_treated, w_agg, 0.0)
    adj_control = np.where(is_treated, 0.0, v_agg)
    g_syn_t = adj_treated @ Y_full_np
    g_syn_c = adj_control @ Y_full_np
    globres = MAREXGlobalDesign(
        Y_full=Y_full_np, Y_fit=raw["Y_fit"], Y_blank=raw["Y_blank"],
        treated_weights_agg=adj_treated, control_weights_agg=adj_control,
        synthetic_treated=g_syn_t, synthetic_control=g_syn_c,
        inference=_maybe_infer(g_syn_t, g_syn_c),
    )

    study = MAREXStudy(
        design=design, T0=T0_eff, blank_periods=blank_periods,
        beta=beta, lambda1=lambda1 or lambda1_unit, lambda2=lambda2 or lambda2_unit, xi=xi,
    )
    return MAREXResults(clusters=clusters_out, study=study, globres=globres)
