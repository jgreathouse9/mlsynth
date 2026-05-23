"""Orchestration for the PANGEO design estimator.

For each treatment arm: enumerate admissible supergeo pairs over the arm's
geos (scored by pre-period DiD parallelism), solve the set-partitioning
MIP to choose the exact-cover design of minimum total non-parallelism,
and assemble the per-arm supergeo pairs with their treatment/control
halves.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence

import numpy as np

from .mip import solve_partition
from .parallelism import (
    enumerate_candidate_pairs,
    gap_variance,
    parallelism_r2,
)
from .power import compute_pangeo_power
from .setup import PangeoInputs
from .structures import ArmDesign, PangeoResults, SupergeoPair

# Upper bound on Q for automatic selection (bounds the candidate-pair
# enumeration when no Q is supplied; pass Q explicitly for large arms).
_AUTO_Q_CAP = 6


def _build_pair(
    pair: dict, Y: np.ndarray, unit_names, T0: int,
    cov: Optional[np.ndarray] = None, cov_scales: Optional[np.ndarray] = None,
    cov_names=None,
) -> SupergeoPair:
    side_a, side_b = pair["side_a"], pair["side_b"]
    mean_a = Y[side_a, :T0].mean(axis=0)
    mean_b = Y[side_b, :T0].mean(axis=0)
    covariate_smd: Dict[str, float] = {}
    if cov is not None:
        smd = (cov[side_a].mean(axis=0) - cov[side_b].mean(axis=0)) / cov_scales
        covariate_smd = {name: float(v) for name, v in zip(cov_names, smd)}
    return SupergeoPair(
        treatment=[unit_names[i] for i in side_a],
        control=[unit_names[i] for i in side_b],
        gap_variance=float(gap_variance(mean_a, mean_b)),
        parallelism_r2=float(parallelism_r2(mean_a, mean_b)),
        treatment_mean=mean_a,
        control_mean=mean_b,
        covariate_smd=covariate_smd,
    )


def run_pangeo(
    inputs: PangeoInputs,
    *,
    max_supergeo_size: Optional[int] = None,
    min_pairs: int = 1,
    objective: str = "ss_res",
    recency_decay: float = 0.97,
    covariate_weights: Optional[Dict[str, float]] = None,
    compute_power: bool = True,
    power_target: float = 0.80,
    power_alpha: float = 0.05,
    power_post_periods: Optional[Sequence[int]] = None,
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
    objective : {"ss_res", "r2", "weighted"}
        Per-pair parallelism cost minimised by the MIP (see
        :func:`mlsynth.utils.pangeo_helpers.parallelism.split_cost`).
    recency_decay : float
        Geometric recency-weight decay for ``objective="weighted"``:
        period ``t`` gets weight ``recency_decay**(T0-1-t)`` (recent
        periods up-weighted), normalised to sum to ``T0``.
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
    if max_supergeo_size is None:
        return _auto_select_q(
            inputs, min_pairs=min_pairs, objective=objective,
            recency_decay=recency_decay, covariate_weights=covariate_weights,
            compute_power=compute_power, power_target=power_target,
            power_alpha=power_alpha, power_post_periods=power_post_periods,
        )
    if max_supergeo_size < 1:
        from ...exceptions import MlsynthConfigError
        raise MlsynthConfigError("max_supergeo_size (Q) must be >= 1.")

    Y = inputs.Y
    T0 = Y.shape[1]                     # all observed periods are pre-period
    unit_names = inputs.unit_names

    weights = None
    if objective == "weighted":
        raw = recency_decay ** (T0 - 1 - np.arange(T0))
        weights = raw / raw.sum() * T0  # normalise to ~uniform scale

    cov = inputs.covariates
    cov_scales = inputs.covariate_scales
    cov_names = inputs.covariate_names
    cov_w = None
    if cov is not None:
        cw = covariate_weights or {}
        cov_w = np.array([float(cw.get(name, 1.0)) for name in cov_names])

    arm_designs = {}
    assignment = {}
    for arm, idx in inputs.arm_units.items():
        candidates = enumerate_candidate_pairs(
            idx, Y[:, :T0], max_supergeo_size,
            objective=objective, weights=weights,
            cov=cov, cov_scales=cov_scales, cov_weights=cov_w,
        )
        chosen = solve_partition(candidates, idx, min_pairs=min_pairs)
        pairs = [_build_pair(p, Y, unit_names, T0, cov, cov_scales, cov_names)
                 for p in chosen]
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
        arm_designs[arm] = ArmDesign(
            arm=arm,
            pairs=pairs,
            n_units=int(idx.size),
            total_gap_variance=float(sum(p.gap_variance for p in pairs)),
            mean_parallelism_r2=float(np.mean(r2s)) if r2s else np.nan,
            treatment_units=treatment_units,
            control_units=control_units,
        )

    metadata = {
        "n_arms": len(inputs.arm_units),
        "arm_sizes": {a: int(idx.size) for a, idx in inputs.arm_units.items()},
        "max_supergeo_size": max_supergeo_size,
        "T_pre": T0,
        "solver": "cvxpy/HiGHS set-partitioning MIP",
        "objective": objective,
        "recency_decay": recency_decay if objective == "weighted" else None,
        "covariates": list(cov_names) if cov is not None else None,
    }

    power = None
    if compute_power:
        power = compute_pangeo_power(
            arm_designs, post_periods=power_post_periods,
            alpha=power_alpha, power_target=power_target,
        )

    return PangeoResults(
        arm_designs=arm_designs,
        max_supergeo_size=max_supergeo_size,
        assignment=assignment,
        time_labels=inputs.time_labels,
        metadata=metadata,
        power=power,
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


def _auto_select_q(
    inputs: PangeoInputs,
    *,
    min_pairs: int,
    objective: str,
    recency_decay: float,
    covariate_weights: Optional[Dict[str, float]],
    compute_power: bool,
    power_target: float,
    power_alpha: float,
    power_post_periods: Optional[Sequence[int]],
) -> PangeoResults:
    """Choose Q by minimising the program-level MDE.

    Designs every feasible ``Q`` in ``1..min(ceil(smallest_arm/2), 6)`` --
    the range over which Q meaningfully trades off matching granularity (and
    hence absolute residual variance) against pair count -- and returns the
    design with the smallest mean program MDE. Infeasible Q (no exact cover,
    e.g. ``Q=1`` for an odd-sized arm) are skipped. The full sweep is stored
    in ``metadata["q_sweep"]`` (with each Q's program-pair count and the
    ``2/2**pairs`` randomization-inference p-value floor) so the choice is
    auditable and a user who prizes design-based inference can override.
    """
    from ...exceptions import MlsynthEstimationError

    min_arm = min(int(idx.size) for idx in inputs.arm_units.values())
    q_upper = max(1, min((min_arm + 1) // 2, _AUTO_Q_CAP))

    best: Optional[PangeoResults] = None
    best_score = float("inf")
    sweep = []
    for q in range(1, q_upper + 1):
        try:
            cand = run_pangeo(
                inputs, max_supergeo_size=q, min_pairs=min_pairs,
                objective=objective, recency_decay=recency_decay,
                covariate_weights=covariate_weights, compute_power=True,
                power_target=power_target, power_alpha=power_alpha,
                power_post_periods=power_post_periods,
            )
        except MlsynthEstimationError:
            sweep.append({"q": q, "feasible": False, "n_program_pairs": 0,
                          "mean_program_mde_pct": None, "rand_floor": None})
            continue
        score = _mean_program_mde(cand)
        n_pairs = cand.power.program.n_pairs if cand.power else 0
        sweep.append({
            "q": q, "feasible": True, "n_program_pairs": n_pairs,
            "mean_program_mde_pct": score,
            "rand_floor": 2.0 / 2 ** n_pairs if n_pairs else None,
        })
        if score < best_score:
            best, best_score = cand, score

    if best is None:
        raise MlsynthEstimationError(
            "PANGEO: no feasible Q found for automatic selection; pass "
            "max_supergeo_size explicitly."
        )

    metadata = {**best.metadata, "q_auto_selected": True,
                "q_selected": best.max_supergeo_size, "q_sweep": sweep}
    power = best.power if compute_power else None
    return dataclasses.replace(best, metadata=metadata, power=power)
