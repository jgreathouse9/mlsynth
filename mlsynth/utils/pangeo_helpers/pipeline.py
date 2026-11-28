"""Orchestration for the PANGEO design estimator.

For each treatment arm: enumerate admissible supergeo pairs over the arm's
geos (scored by pre-period DiD parallelism), solve the set-partitioning
MIP to choose the exact-cover design of minimum total non-parallelism,
and assemble the per-arm supergeo pairs with their treatment/control
halves.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np

from .fast_partition import fast_partition
from . import columns as _columns
from .columns import score_columns, solve_exact
from .mip import solve_partition
from .pairs import solve_pairs
from .parallelism import (
    _wavg,
    gap_variance,
    parallelism_r2,
)
from .power import compute_pangeo_power
from .setup import PangeoInputs
from .structures import (ArmDesign, MipSolveReport, PangeoExcludedUnitWarning, PangeoHeuristicFallbackWarning,
                         PangeoResults,
                         SupergeoPair)

# Upper bound on Q for automatic selection (bounds the candidate-pair
# enumeration when no Q is supplied; pass Q explicitly for large arms).
_AUTO_Q_CAP = 6


def _holdout_resid(yT, yC, e_idx, b_idx, augment, trend):
    """Held-out (DiD or augmented-DiD) residual on the blank window B.

    Fit the same counterfactual model the realised effect uses -- plain DiD
    (``yT - yC = d1 [+ g*t]``) or augmented DiD (``yT = d1 + d2*yC [+ g*t]``)
    -- on the estimation window E, then return the residual on B. Keeping the
    power model identical to the evaluation model makes the planning MDE and
    the realised SE use the same noise.
    """
    n = yT.size
    t = np.arange(n, dtype=float)
    cols = [np.ones(n)] + ([yC] if augment else []) + ([t] if trend else [])
    X = np.column_stack(cols)
    y = yT if augment else yT - yC
    XE = X[e_idx]
    # The same QR route the effects module uses, so a residual computed here
    # and an ATT reported there rest on the same fit.
    beta = np.linalg.lstsq(XE, y[e_idx], rcond=None)[0]
    return y[b_idx] - X[b_idx] @ beta


def _warn_excluded(arm, n_units: int, excluded) -> None:
    """Name each market left out of an arm's design, as a PangeoExcludedUnitWarning."""
    for unit in excluded:
        warnings.warn(
            f"PANGEO: arm {arm!r} has an odd number of units ({n_units}); {unit!r} is left out of "
            "the design so the rest pair at the least cost (results.excluded_units).",
            PangeoExcludedUnitWarning, stacklevel=3)


def _coin_flips(sizes: Sequence[int], seed: int):
    """One fair coin per supergeo pair, per arm: for arm sizes ``sizes`` (pairs
    per arm, in the panel's arm order), a list of boolean arrays, the k-th entry
    of the a-th array True when the k-th pair of arm a treats the half holding
    its first unit.

    The supergeo design randomizes within each pair, independently across
    pairs (Chen, Doudchenko, Jiang, Stein and Ying, "Supergeo design:
    generalized matching for geographic experiments", arXiv:2301.12044,
    section 3); that draw is the randomization the sign-flip test in
    ``effects.py`` inverts. One generator, seeded once, deals every arm's coins
    in turn, so two arms are not dealt the same sequence.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, size=int(k)).astype(bool) for k in sizes]


def _check_seed(seed) -> int:
    """The coins' seed, an integer (not a bool) from 0 to 2^63 - 1, or raise."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2 ** 63:
        from ...exceptions import MlsynthConfigError
        raise MlsynthConfigError(
            f"assignment_seed must be an integer from 0 to 2^63 - 1, not {seed!r}.")
    return int(seed)


def _orient(pair: dict, first_half_treated: bool) -> dict:
    """The pair with ``side_a`` the treated half: the half holding the pair's
    first unit (its lowest index) when ``first_half_treated``, the other half
    otherwise. The score is a function of the unordered split and is kept."""
    a, b = np.asarray(pair["side_a"]), np.asarray(pair["side_b"])
    first_in_a = a.min() < b.min()
    if first_in_a != bool(first_half_treated):
        a, b = b, a
    return {**pair, "side_a": a, "side_b": b}


def _build_pair(
    pair: dict, Y: np.ndarray, unit_names, T0: int,
    cov: Optional[np.ndarray] = None, cov_scales: Optional[np.ndarray] = None,
    cov_names=None, unit_weights: Optional[np.ndarray] = None,
    e_idx: Optional[np.ndarray] = None, b_idx: Optional[np.ndarray] = None,
    att_augment: bool = True, att_trend: bool = True,
    first_half_treated: bool = True,
) -> SupergeoPair:
    side_a, side_b = pair["side_a"], pair["side_b"]
    mean_a = _wavg(Y[:, :T0], side_a, unit_weights)   # full pre (for plotting)
    mean_b = _wavg(Y[:, :T0], side_b, unit_weights)
    gap = mean_a - mean_b
    # Design metrics are read off the estimation window E (what the split was
    # optimised on). The counterfactual gap level and the residual reservoir
    # come from the held-out blank window B -- it is excluded from the
    # optimisation (so the residuals are honest) and is the pre window closest
    # to the post period (so the level is not contaminated by gap drift).
    e_idx = np.arange(T0) if e_idx is None else e_idx
    b_idx = np.arange(T0) if b_idx is None else b_idx
    gap_level = float(gap[b_idx].mean())
    # Held-out reservoir for the power analysis: the residual of the *same*
    # counterfactual model the realised effect uses (plain DiD or augmented
    # DiD), fit on E and evaluated on the blank window B -- so the planning
    # MDE and the realised SE are built from the same noise.
    holdout_resid = _holdout_resid(mean_a, mean_b, e_idx, b_idx,
                                   att_augment, att_trend)
    covariate_smd: Dict[str, float] = {}
    if cov is not None:
        smd = (_wavg(cov, side_a, unit_weights)
               - _wavg(cov, side_b, unit_weights)) / cov_scales
        covariate_smd = {name: float(v) for name, v in zip(cov_names, smd)}
    return SupergeoPair(
        treatment=[unit_names[i] for i in side_a],
        control=[unit_names[i] for i in side_b],
        gap_variance=float(gap_variance(mean_a[e_idx], mean_b[e_idx])),
        parallelism_r2=float(parallelism_r2(mean_a[e_idx], mean_b[e_idx])),
        treatment_mean=mean_a,
        control_mean=mean_b,
        covariate_smd=covariate_smd,
        gap_level=gap_level,
        holdout_resid=holdout_resid,
        first_half_treated=bool(first_half_treated),
    )


def run_pangeo(
    inputs: PangeoInputs,
    *,
    max_supergeo_size: Optional[int] = None,
    min_pairs: int = 1,
    fast: Optional[bool] = None,
    fast_candidates: int = 5,
    objective: str = "ss_res",
    recency_decay: float = 0.97,
    frac_E: float = 0.7,
    covariate_weights: Optional[Dict[str, float]] = None,
    compute_power: bool = True,
    power_target: float = 0.80,
    power_alpha: float = 0.05,
    power_post_periods: Optional[Sequence[int]] = None,
    att_augment: bool = True,
    att_trend: bool = True,
    q_selection: str = "mde_min",
    q_min_pairs: int = 1,
    compute_q_sweep: bool = False,
    mip_backend: str = "auto",
    mip_rel_gap: float = 0.0,
    mip_abs_gap: float = 0.0,
    assignment_seed: int = 0,
    certify_node_limit: Optional[int] = None,
    popmusic_node_limit: Optional[int] = None,
    search_node_limit: Optional[int] = None,
) -> PangeoResults:
    """Design parallel supergeo pairs within each arm.

    Parameters
    ----------
    inputs : PangeoInputs
        Preprocessed pre-treatment panel.
    max_supergeo_size : int, optional
        Q -- the maximum size of either supergeo within a pair. If ``None``
        (the default), Q is **selected automatically**: every feasible Q in
        ``1..min(ceil(smallest_arm/2), 6)`` is designed and the one
        minimising the program-level MDE is returned (see
        :func:`_auto_select_q`). The sweep is recorded in
        ``results.metadata["q_sweep"]``.
    min_pairs : int
        Minimum number of supergeo pairs per arm.
    objective : {"ss_res", "r2", "weighted", "supergeo_total"}
        Per-pair cost minimised by the MIP (see
        :func:`mlsynth.utils.pangeo_helpers.parallelism.split_cost`, and
        :func:`mlsynth.utils.pangeo_helpers.parallelism.supergeo_total` for
        the paper's objective, available for pairs on the exact path).
    mip_backend : {"auto", "highspy", "cvxpy"}
        Backend of the exact-cover MIP (see
        :func:`mlsynth.utils.pangeo_helpers.mip.solve_partition`).
    mip_rel_gap, mip_abs_gap : float
        Relative and absolute gaps the exact-cover MIP must close, 0 by
        default; each arm's solve is reported in ``results.mip_reports``.
    certify_node_limit : int, optional
        Nodes the exact path may place after a proof stopped by its cap, in
        the iterative cutoff (:func:`mlsynth.utils.pangeo_helpers.columns.solve_exact`);
        ``cover.CERTIFY_NODE_LIMIT`` when None, 0 to leave a capped proof to
        the MIP loop.
    popmusic_node_limit : int, optional
        Nodes POPMUSIC (:mod:`mlsynth.utils.pangeo_helpers.popmusic`) may
        place improving the design after a proof stopped by its cap, before
        the iterative cutoff; ``cover.POPMUSIC_NODE_LIMIT`` when None, 0 to
        skip it.
    search_node_limit : int, optional
        Nodes the exact path's capped search may place (quick search, first
        window and proof); ``cover.SEARCH_NODE_LIMIT`` when None.
    fast : bool, optional
        Deprecated: passing it raises a DeprecationWarning and changes
        nothing. Every arm is designed by the exact path, except an arm with
        more candidate groups than ``columns.MAX_COLUMNS``, which falls back
        to the clustering heuristic with a ``PangeoHeuristicFallbackWarning``
        and is listed in ``metadata["heuristic_arms"]``.
    assignment_seed : int
        Seed of the fair coin drawn in each pair, after the pairs are chosen,
        to decide which half is treated (:func:`_coin_flips`). The pairs do
        not depend on it; the labels, and everything computed from them, do.
    recency_decay : float
        Geometric recency-weight decay for ``objective="weighted"``:
        period ``t`` gets weight ``recency_decay**(T0-1-t)`` (recent
        periods up-weighted), normalised to sum to ``T0``.
    frac_E : float
        Fraction of the pre-period used as the **estimation window** E that
        the split is optimised over; the remaining tail is the **blank**
        window B, held out so its gap residuals are an honest, out-of-sample
        estimate of the parallel-trends noise (powering the MDE and the
        conformal CIs). Mirrors LEXSCM / SPCD. Falls back to the full pre
        when the panel is too short to leave a usable B.
    covariate_weights : dict, optional
        ``{covariate_name: weight}`` on the standardized SMD^2 imbalance
        penalty (default 1.0 each). Only used when ``inputs.covariates`` is
        present.
    compute_power : bool
        Attach a program- and arm-level MDE / power analysis to the result
        (see :mod:`mlsynth.utils.pangeo_helpers.power`).
    power_target : float
        Target power for the stored MDE (default 0.80).
    power_alpha : float
        Two-sided significance level for the MDE (default 0.05).
    power_post_periods : sequence of int, optional
        Post-period horizons to evaluate (default ``range(2, 13)`` = 2..12).
    """
    assignment_seed = _check_seed(assignment_seed)
    if fast is not None:
        from .config import _FAST_DEPRECATED
        warnings.warn(_FAST_DEPRECATED, DeprecationWarning, stacklevel=2)
        fast = None                     # the exact path runs whatever its value; inner designs do not warn again
    if objective == "supergeo_total" and max_supergeo_size != 1:
        from ...exceptions import MlsynthConfigError
        raise MlsynthConfigError(
            "objective='supergeo_total' supports pairs (max_supergeo_size=1) only.")
    if max_supergeo_size is None:
        return _auto_select_q(
            inputs, min_pairs=min_pairs, fast=fast,
            fast_candidates=fast_candidates, objective=objective,
            recency_decay=recency_decay, frac_E=frac_E,
            covariate_weights=covariate_weights,
            compute_power=compute_power, power_target=power_target,
            power_alpha=power_alpha, power_post_periods=power_post_periods,
            att_augment=att_augment, att_trend=att_trend,
            q_selection=q_selection, q_min_pairs=q_min_pairs,
            mip_backend=mip_backend, mip_rel_gap=mip_rel_gap,
            mip_abs_gap=mip_abs_gap, assignment_seed=assignment_seed,
            certify_node_limit=certify_node_limit,
            popmusic_node_limit=popmusic_node_limit,
            search_node_limit=search_node_limit,
        )
    if max_supergeo_size < 1:
        from ...exceptions import MlsynthConfigError
        raise MlsynthConfigError("max_supergeo_size (Q) must be >= 1.")

    Y = inputs.Y
    T0 = Y.shape[1]                     # all observed periods are pre-period
    unit_names = inputs.unit_names

    # Estimation window E (optimise the split) vs held-out blank window B.
    n_E = int(round(frac_E * T0))
    if n_E < 2 or (T0 - n_E) < 2:       # too short to hold out: use full pre
        e_idx, b_idx = np.arange(T0), np.arange(T0)
        holdout = False
    else:
        e_idx, b_idx = np.arange(n_E), np.arange(n_E, T0)
        holdout = True

    weights = None
    if objective == "weighted":
        nE = e_idx.size
        raw = recency_decay ** (nE - 1 - np.arange(nE))
        weights = raw / raw.sum() * nE  # normalise to ~uniform scale

    cov = inputs.covariates
    cov_scales = inputs.covariate_scales
    cov_names = inputs.covariate_names
    uw = inputs.weights
    cov_w = None
    if cov is not None:
        cw = covariate_weights or {}
        cov_w = np.array([float(cw.get(name, 1.0)) for name in cov_names])

    # Pairs on the exact path are a minimum-weight perfect matching: solved in
    # closed form (sorted totals) for the paper's objective without covariates,
    # by the blossom matcher for every other cost; an explicit MIP backend
    # keeps the MIP (pairs.py; CLAUDE.md invariant 8, closed forms first).
    pairs_path = max_supergeo_size == 1 and mip_backend == "auto"
    closed_form = pairs_path and objective == "supergeo_total" and cov is None

    arm_designs = {}
    assignment = {}
    solver_diagnostics = {}
    mip_reports = {}
    chosen_by_arm = {}
    fell_back = []
    for arm, idx in inputs.arm_units.items():
        if closed_form:
            Ypre = Y[:, e_idx]
            totals = np.array([float(Ypre[[u]].sum()) for u in idx])
            chosen, solver_diagnostics[arm] = solve_pairs(
                None, idx, totals=totals, min_pairs=min_pairs)
        elif pairs_path:
            candidates = score_columns(
                idx, Y[:, e_idx], max_supergeo_size,
                objective=objective, weights=weights,
                cov=cov, cov_scales=cov_scales, cov_weights=cov_w,
                unit_weights=uw,
            ).candidates()
            chosen, solver_diagnostics[arm] = solve_pairs(
                candidates, idx, min_pairs=min_pairs)
        elif _columns.count_columns(int(idx.size), max_supergeo_size) > _columns.MAX_COLUMNS:
            # More groups than the exact path scores: the clustering heuristic, uncertified, until phase 4b's
            # strata replace it (benchmarks/studies/supergeo/PANGEO_ROADMAP.md, phase 4).
            warnings.warn(
                f"PANGEO: arm {arm!r} has {int(idx.size)} markets, which at max_supergeo_size="
                f"{max_supergeo_size} make {_columns.count_columns(int(idx.size), max_supergeo_size):,} candidate "
                f"groups, more than the {_columns.MAX_COLUMNS:,} (MAX_COLUMNS) the exact path scores; its design "
                f"comes from the clustering heuristic and carries no certified gap.",
                PangeoHeuristicFallbackWarning, stacklevel=2)
            fell_back.append(arm)
            chosen, solver_diagnostics[arm] = fast_partition(
                idx, Y[:, e_idx], max_supergeo_size,
                objective=objective, weights=weights,
                cov=cov, cov_scales=cov_scales, cov_weights=cov_w,
                unit_weights=uw, n_candidates=fast_candidates,
                min_pairs=min_pairs, seed=0, return_diagnostics=True,
            )
        else:
            # Every group scored in compiled code; the MIP sees only the
            # groups the root LP's Lagrangian bound cannot exclude (columns.py).
            chosen, solver_diagnostics[arm] = solve_exact(
                idx, Y[:, e_idx], max_supergeo_size,
                objective=objective, weights=weights,
                cov=cov, cov_scales=cov_scales, cov_weights=cov_w,
                unit_weights=uw, min_pairs=min_pairs, backend=mip_backend,
                mip_rel_gap=mip_rel_gap, mip_abs_gap=mip_abs_gap,
                solve=solve_partition, certify_node_limit=certify_node_limit,
                popmusic_node_limit=popmusic_node_limit, search_node_limit=search_node_limit)
            mip_reports[arm] = MipSolveReport.from_diagnostics(
                solver_diagnostics[arm])
        # Pairs in a canonical order (by first unit) so the k-th coin goes to
        # the same pair whatever order the solver returned them in.
        chosen_by_arm[arm] = sorted(chosen, key=lambda c: int(np.min(c["members"])))

    coins = dict(zip(chosen_by_arm, _coin_flips(
        [len(c) for c in chosen_by_arm.values()], assignment_seed)))
    for arm, idx in inputs.arm_units.items():
        pairs = [_build_pair(_orient(p, coin), Y, unit_names, T0, cov,
                             cov_scales, cov_names,
                             e_idx=e_idx, b_idx=b_idx, unit_weights=uw,
                             att_augment=att_augment, att_trend=att_trend,
                             first_half_treated=bool(coin))
                 for p, coin in zip(chosen_by_arm[arm], coins[arm])]
        # Sort pairs by quality (most parallel first) for readability.
        pairs.sort(key=lambda p: p.gap_variance)

        treatment_units, control_units = [], []
        for p in pairs:
            treatment_units.extend(p.treatment)
            control_units.extend(p.control)
            for u in p.treatment:
                assignment[u] = "treatment"
            for u in p.control:
                assignment[u] = "control"

        r2s = [p.parallelism_r2 for p in pairs if np.isfinite(p.parallelism_r2)]
        excluded = [unit_names[i] for i in
                    solver_diagnostics[arm].get("excluded_units", [])]
        _warn_excluded(arm, int(idx.size), excluded)
        arm_designs[arm] = ArmDesign(
            arm=arm,
            pairs=pairs,
            n_units=int(idx.size),
            total_gap_variance=float(sum(p.gap_variance for p in pairs)),
            mean_parallelism_r2=float(np.mean(r2s)) if r2s else np.nan,
            treatment_units=treatment_units,
            control_units=control_units,
            excluded_units=excluded,
        )

    metadata = {
        "n_arms": len(inputs.arm_units),
        "arm_sizes": {a: int(idx.size) for a, idx in inputs.arm_units.items()},
        "max_supergeo_size": max_supergeo_size,
        "T_pre": T0,
        "solver": ("clustering heuristic (more candidate groups than MAX_COLUMNS)"
                   if fell_back else "sorted pairing (closed form)" if closed_form
                   else "Edmonds blossom matching" if pairs_path
                   else "cvxpy/HiGHS set-partitioning MIP"),
        "fast_candidates": fast_candidates if fell_back else None,
        "heuristic_arms": list(fell_back),
        "solver_diagnostics": solver_diagnostics,
        "objective": objective,
        "mip_backend": mip_backend,
        "recency_decay": recency_decay if objective == "weighted" else None,
        "covariates": list(cov_names) if cov is not None else None,
        "frac_E": frac_E,
        "n_estimation": int(e_idx.size),
        "n_holdout": int(b_idx.size) if holdout else 0,
        "att_augment": att_augment,
        "att_trend": att_trend,
    }

    power = None
    if compute_power:
        power = compute_pangeo_power(
            arm_designs, post_periods=power_post_periods,
            alpha=power_alpha, power_target=power_target,
            unit_weights=(None if uw is None
                          else dict(zip(inputs.unit_names, map(float, uw)))),
        )

    if compute_q_sweep:
        # Diagnostic-only sweep in fixed-Q mode: run the full Q sweep so the
        # user can audit their chosen Q against the Pareto/1-SE alternatives,
        # without changing the honoured (fixed) Q. Inner designs never recurse.
        run_kwargs = dict(
            min_pairs=min_pairs, fast=fast, fast_candidates=fast_candidates,
            objective=objective, recency_decay=recency_decay, frac_E=frac_E,
            covariate_weights=covariate_weights, power_target=power_target,
            power_alpha=power_alpha, power_post_periods=power_post_periods,
            att_augment=att_augment, att_trend=att_trend,
            mip_backend=mip_backend, mip_rel_gap=mip_rel_gap,
            mip_abs_gap=mip_abs_gap, assignment_seed=assignment_seed,
            certify_node_limit=certify_node_limit,
            popmusic_node_limit=popmusic_node_limit,
            search_node_limit=search_node_limit,
        )
        metadata["q_sweep"], _ = _sweep_designs(
            inputs, q_upper=_q_upper(inputs), run_kwargs=run_kwargs)

    selected_units = [u for u, s in assignment.items() if s == "treatment"]
    return PangeoResults(
        arm_designs=arm_designs,
        max_supergeo_size=max_supergeo_size,
        assignment=assignment,
        selected_units=selected_units,
        time_labels=inputs.time_labels,
        metadata=metadata,
        power=power,
        assignment_seed=assignment_seed,
        excluded_units=[u for d in arm_designs.values() for u in d.excluded_units],
        mip_reports=mip_reports,
    )


def _mean_program_mde(result: PangeoResults) -> float:
    """Mean program-level MDE (% of baseline) across horizons; ``inf`` if
    unavailable. The selection score for automatic Q (lower is better).
    """
    if result.power is None:
        return float("inf")
    vals = [pt.mde_pct for pt in result.power.program.points
            if np.isfinite(pt.mde_pct)]
    return float(np.mean(vals)) if vals else float("inf")


def _select_q(sweep, *, method: str = "mde_min", q_min_pairs: int = 1):
    """Choose Q from a completed sweep of ``{q, feasible, n_program_pairs,
    mean_program_mde_pct, mde_se}`` rows. Returns the chosen ``q`` (int) or
    ``None`` if no feasible Q exists.

    * ``"mde_min"`` -- the feasible Q of smallest program MDE (historical).
    * ``"pareto_1se"`` -- treat Q as a two-objective choice (minimise MDE,
      maximise pair count ``K``). Keep the Pareto-efficient rows on
      ``(MDE down, K up)``, then a 1-SE tie-break: among the frontier (subject
      to ``K >= q_min_pairs`` when any such row qualifies) take the *largest K*
      whose MDE is within one SE of the frontier's best MDE. This spends pairs
      on power only when the power gain exceeds its own sampling noise.
    """
    feas = [r for r in sweep
            if r.get("feasible") and r.get("mean_program_mde_pct") is not None]
    if not feas:
        return None
    if method == "mde_min":
        return min(feas, key=lambda r: r["mean_program_mde_pct"])["q"]
    if method != "pareto_1se":
        from ...exceptions import MlsynthConfigError
        raise MlsynthConfigError(
            f"Unknown q_selection {method!r}; use 'mde_min' or 'pareto_1se'."
        )

    def _dominated(a) -> bool:
        # a is dominated if some other row is >= on K and <= on MDE (strictly
        # better on at least one axis).
        for o in feas:
            if o is a:
                continue
            if (o["n_program_pairs"] >= a["n_program_pairs"]
                    and o["mean_program_mde_pct"] <= a["mean_program_mde_pct"]
                    and (o["n_program_pairs"] > a["n_program_pairs"]
                         or o["mean_program_mde_pct"] < a["mean_program_mde_pct"])):
                return True
        return False

    frontier = [r for r in feas if not _dominated(r)]
    best = min(frontier, key=lambda r: r["mean_program_mde_pct"])
    se = best.get("mde_se")
    threshold = best["mean_program_mde_pct"] + (se if se else 0.0)
    within = [r for r in frontier if r["mean_program_mde_pct"] <= threshold]
    floored = [r for r in within if r["n_program_pairs"] >= q_min_pairs]
    eligible = floored or within
    return max(eligible, key=lambda r: r["n_program_pairs"])["q"]


def _q_upper(inputs: PangeoInputs) -> int:
    """Upper bound on Q for the sweep: the range over which Q meaningfully
    trades matching granularity against pair count for the smallest arm,
    stopped at the largest Q at which the largest arm's candidate groups fit
    within ``columns.MAX_COLUMNS``, so that every Q the sweep designs is
    designed by the exact path (a larger Q would fall back to the clustering
    heuristic, whose designs measured 6.4 to 26.8 times the exact path's cost and
    would be compared with exact designs). Q = 1 is always kept."""
    sizes = [int(idx.size) for idx in inputs.arm_units.values()]
    q = max(1, min((min(sizes) + 1) // 2, _AUTO_Q_CAP))
    biggest = max(sizes)
    while q > 1 and _columns.count_columns(biggest, q) > _columns.MAX_COLUMNS:
        q -= 1
    return q


def _sweep_designs(inputs: PangeoInputs, *, q_upper: int, run_kwargs: dict):
    """Design every feasible Q in ``1..q_upper``; return ``(sweep, results)``.

    ``sweep`` is the list of per-Q diagnostic rows (feasibility, pair count,
    mean program MDE, its small-sample SE, and the randomization-inference
    p-value floor). ``results`` maps each feasible Q to its ``PangeoResults``.
    Power is always computed here (the sweep needs the MDE); the inner designs
    never recurse into a sweep of their own.
    """
    from ...exceptions import MlsynthEstimationError

    sweep = []
    results = {}
    for q in range(1, q_upper + 1):
        try:
            # The sweep's designs are candidates; only the one returned names its
            # excluded markets (_auto_select_q re-warns for it).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PangeoExcludedUnitWarning)
                cand = run_pangeo(
                    inputs, max_supergeo_size=q, compute_power=True,
                    compute_q_sweep=False, **run_kwargs,
                )
        except MlsynthEstimationError:
            sweep.append({"q": q, "feasible": False, "n_program_pairs": 0,
                          "mean_program_mde_pct": None, "mde_se": None,
                          "rand_floor": None})
            continue
        results[q] = cand
        mde = _mean_program_mde(cand)
        n_pairs = cand.power.program.n_pairs if cand.power else 0
        b = int(cand.metadata.get("n_holdout", 0) or 0)
        # SD-of-SD small-sample SE of the MDE: relative SE ~ 1/sqrt(2(B-1)).
        mde_se = (mde / np.sqrt(2.0 * (b - 1))
                  if (b > 1 and np.isfinite(mde)) else None)
        sweep.append({
            "q": q, "feasible": True, "n_program_pairs": n_pairs,
            "mean_program_mde_pct": mde,
            "mde_se": float(mde_se) if mde_se is not None else None,
            "rand_floor": 2.0 / 2 ** n_pairs if n_pairs else None,
        })
    return sweep, results


def _auto_select_q(
    inputs: PangeoInputs,
    *,
    min_pairs: int,
    fast: Optional[bool] = None,
    fast_candidates: int = 5,
    objective: str,
    recency_decay: float,
    frac_E: float,
    covariate_weights: Optional[Dict[str, float]],
    compute_power: bool,
    power_target: float,
    power_alpha: float,
    power_post_periods: Optional[Sequence[int]],
    att_augment: bool = True,
    att_trend: bool = True,
    q_selection: str = "mde_min",
    q_min_pairs: int = 1,
    mip_backend: str = "auto",
    mip_rel_gap: float = 0.0,
    mip_abs_gap: float = 0.0,
    assignment_seed: int = 0,
    certify_node_limit: Optional[int] = None,
    popmusic_node_limit: Optional[int] = None,
    search_node_limit: Optional[int] = None,
) -> PangeoResults:
    """Choose Q automatically from a sweep over feasible sizes.

    Designs every feasible ``Q`` in ``1..min(ceil(smallest_arm/2), 6)`` --
    the range over which Q meaningfully trades off matching granularity (and
    hence absolute residual variance) against pair count -- and picks one via
    :func:`_select_q` (``q_selection``: ``"mde_min"`` or ``"pareto_1se"``).
    Infeasible Q (no exact cover, e.g. ``Q=1`` for an odd-sized arm) are
    skipped. The full sweep -- each Q's pair count, mean program MDE and its
    SE, and the ``2/2**pairs`` randomization p-value floor -- is stored in
    ``metadata["q_sweep"]`` so the choice is auditable.
    """
    from ...exceptions import MlsynthEstimationError

    run_kwargs = dict(
        min_pairs=min_pairs, fast=fast, fast_candidates=fast_candidates,
        objective=objective, recency_decay=recency_decay, frac_E=frac_E,
        covariate_weights=covariate_weights, power_target=power_target,
        power_alpha=power_alpha, power_post_periods=power_post_periods,
        att_augment=att_augment, att_trend=att_trend,
        mip_backend=mip_backend, mip_rel_gap=mip_rel_gap,
        mip_abs_gap=mip_abs_gap, assignment_seed=assignment_seed,
        certify_node_limit=certify_node_limit,
        popmusic_node_limit=popmusic_node_limit,
        search_node_limit=search_node_limit,
    )
    sweep, results = _sweep_designs(
        inputs, q_upper=_q_upper(inputs), run_kwargs=run_kwargs)

    q_selected = _select_q(sweep, method=q_selection, q_min_pairs=q_min_pairs)
    if q_selected is None:
        raise MlsynthEstimationError(
            "PANGEO: no feasible Q found for automatic selection; pass "
            "max_supergeo_size explicitly."
        )
    best = results[q_selected]
    for arm, design in best.arm_designs.items():
        _warn_excluded(arm, design.n_units, design.excluded_units)
    metadata = {**best.metadata, "q_auto_selected": True,
                "q_selected": q_selected, "q_selection": q_selection,
                "q_sweep": sweep}
    power = best.power if compute_power else None
    return best.model_copy(update={"metadata": metadata, "power": power})
