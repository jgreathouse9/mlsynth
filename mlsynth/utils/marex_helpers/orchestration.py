"""Top-level MAREX solve: run the design optimizer and assemble frozen results."""

from __future__ import annotations

from typing import Optional

import numpy as np

from dataclasses import replace as _dc_replace

from ...config_models import WeightsResults
from ..post_fit import compute_post_fit, compute_power_analysis, to_effect_result
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
    covariate_names=(),
    covariate_weight=1.0,
    standardize=False,
    solver=None,
    verbose=False,
    relaxed=False,
    inference=False,
    alpha=0.05,
    max_combinations=1000,
    random_state=42,
    unit_index=None,
    time_index=None,
    restrictions=None,
    warm_start=None,
    time_limit=None,
) -> MAREXResults:
    """Solve the MAREX design and return a frozen :class:`MAREXResults`.

    With ``relaxed=True`` the continuous-``z`` QP with post-hoc discretization is
    used; otherwise the exact MIQP. With ``inference=True``, blank-period
    placebo inference is computed for every cluster and the aggregate.
    """
    solve = solve_design_relaxed if relaxed else solve_design
    solve_kw = dict(
        Y_full=Y_full, T0=T0, clusters=clusters, blank_periods=blank_periods,
        m_eq=m_eq, m_min=m_min, m_max=m_max, exclusive=exclusive, design=design,
        beta=beta, lambda1=lambda1, lambda2=lambda2, xi=xi,
        lambda1_unit=lambda1_unit, lambda2_unit=lambda2_unit,
        costs=costs, budget=budget, covariates=covariates,
        covariate_weight=covariate_weight, standardize=standardize,
        solver=solver, verbose=verbose,
    )
    # Restrictions are MIQP-only (the relaxed path's rounding can't guarantee
    # them); the config rejects the combination, so only the exact solver sees it.
    if not relaxed:
        solve_kw["restrictions"] = restrictions
        solve_kw["warm_start"] = warm_start
        solve_kw["time_limit"] = time_limit
    raw = solve(**solve_kw)

    df = raw["df"]
    Y_full_np = df.to_numpy() if hasattr(df, "to_numpy") else np.asarray(df)
    # Identity comes from the IndexSet (the single source of truth); fall back to
    # the frame's own index only when a caller does not supply one.
    if unit_index is not None:
        unit_labels = list(unit_index.labels)
    elif hasattr(df, "index"):
        unit_labels = list(df.index)
    else:
        unit_labels = list(range(Y_full_np.shape[0]))
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

    # Standardized post-fit diagnostics (ATE / total / lift / RMSEs / SMDs).
    # Constructed inline because MAREXResults is frozen; one source of truth
    # for these quantities lives in mlsynth.utils.post_fit.compute_post_fit.
    n_fit = T0_eff - blank_periods
    n_post = max(0, Y_full_np.shape[1] - T0_eff)
    cov_matrix = covariates if isinstance(covariates, np.ndarray) else None
    cov_names_seq = tuple(covariate_names) if covariate_names else None
    post_fit = compute_post_fit(
        treated_series=globres.synthetic_treated,
        control_series=globres.synthetic_control,
        n_fit=n_fit, n_blank=blank_periods, n_post=n_post,
        cov_matrix=cov_matrix, cov_names=cov_names_seq,
        treated_weights=adj_treated, control_weights=adj_control,
        inference=globres.inference,
        n_treated_units=int(np.sum(adj_treated > 1e-8)),
    )
    # Power analysis: analytical AR(1)-inflated MDE from the placebo/blank
    # gap residuals (or the pre-period gap when no blank window was carved
    # out). Skipped when no pre window or both windows are degenerate.
    try:
        power = compute_power_analysis(post_fit, alpha=alpha)
        post_fit = _dc_replace(post_fit, power=power)
    except Exception:                # never let power analysis break a fit
        pass

    # ----- standardized DesignResult surface -----
    # Treated / control assignment aggregated across clusters.
    treated_labels, control_labels = [], []
    for c in clusters_out.values():
        treated_labels.extend(c.unit_weight_map.get("Treated", {}).keys())
        control_labels.extend(c.unit_weight_map.get("Control", {}).keys())
    treated_w_map = {str(unit_labels[i]): float(adj_treated[i])
                     for i in range(len(unit_labels)) if adj_treated[i] > 1e-8}
    control_w_map = {str(unit_labels[i]): float(adj_control[i])
                     for i in range(len(unit_labels)) if adj_control[i] > 1e-8}
    design_weights = WeightsResults(
        donor_weights=treated_w_map,
        summary_stats={
            "weights_are": "marex_treated_weights_agg",
            "n_treated": len(treated_labels),
            "n_control": len(control_labels),
            "control_weights_agg": control_w_map,
        },
    )
    # The realized effect as a standardized EffectResult (the family adapter).
    time_periods = (np.asarray(time_index.labels) if time_index is not None
                    else (np.asarray(df.columns) if hasattr(df, "columns") else None))
    intervention = (time_periods[T0_eff]
                    if time_periods is not None and T0_eff < time_periods.shape[0]
                    else None)
    report = to_effect_result(
        post_fit, time_periods=time_periods, intervention_time=intervention,
        method_name="MAREX", donor_weights=treated_w_map,
    ) if post_fit is not None else None

    return MAREXResults(
        clusters=clusters_out, study=study, globres=globres, post_fit=post_fit,
        report=report,
        selected_units=list(treated_labels),
        assignment={"treated": list(treated_labels),
                    "control": list(control_labels)},
        design_weights=design_weights,
        power=(post_fit.power if post_fit is not None else None),
        metadata={
            "design": design, "T0": T0_eff, "blank_periods": blank_periods,
            "n_clusters": len(clusters_out),
        },
    )
