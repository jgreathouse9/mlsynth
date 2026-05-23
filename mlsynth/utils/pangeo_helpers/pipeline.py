"""Orchestration for the PANGEO design estimator.

For each treatment arm: enumerate admissible supergeo pairs over the arm's
geos (scored by pre-period DiD parallelism), solve the set-partitioning
MIP to choose the exact-cover design of minimum total non-parallelism,
and assemble the per-arm supergeo pairs with their treatment/control
halves.
"""

from __future__ import annotations

import numpy as np

from .mip import solve_partition
from .parallelism import (
    enumerate_candidate_pairs,
    gap_variance,
    parallelism_r2,
)
from .setup import PangeoInputs
from .structures import ArmDesign, PangeoResults, SupergeoPair


def _build_pair(pair: dict, Y: np.ndarray, unit_names, T0: int) -> SupergeoPair:
    side_a, side_b = pair["side_a"], pair["side_b"]
    mean_a = Y[side_a, :T0].mean(axis=0)
    mean_b = Y[side_b, :T0].mean(axis=0)
    return SupergeoPair(
        treatment=[unit_names[i] for i in side_a],
        control=[unit_names[i] for i in side_b],
        gap_variance=float(gap_variance(mean_a, mean_b)),
        parallelism_r2=float(parallelism_r2(mean_a, mean_b)),
        treatment_mean=mean_a,
        control_mean=mean_b,
    )


def run_pangeo(
    inputs: PangeoInputs,
    *,
    max_supergeo_size: int = 3,
    min_pairs: int = 1,
) -> PangeoResults:
    """Design parallel supergeo pairs within each arm.

    Parameters
    ----------
    inputs : PangeoInputs
        Preprocessed pre-treatment panel.
    max_supergeo_size : int
        Q -- the maximum size of either supergeo within a pair.
    min_pairs : int
        Minimum number of supergeo pairs per arm.
    """
    if max_supergeo_size < 1:
        from ...exceptions import MlsynthConfigError
        raise MlsynthConfigError("max_supergeo_size (Q) must be >= 1.")

    Y = inputs.Y
    T0 = Y.shape[1]                     # all observed periods are pre-period
    unit_names = inputs.unit_names

    arm_designs = {}
    assignment = {}
    for arm, idx in inputs.arm_units.items():
        candidates = enumerate_candidate_pairs(idx, Y[:, :T0], max_supergeo_size)
        chosen = solve_partition(candidates, idx, min_pairs=min_pairs)
        pairs = [_build_pair(p, Y, unit_names, T0) for p in chosen]
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
        "objective": "pre-period DiD parallelism (level-removed gap variance)",
    }
    return PangeoResults(
        arm_designs=arm_designs,
        max_supergeo_size=max_supergeo_size,
        assignment=assignment,
        time_labels=inputs.time_labels,
        metadata=metadata,
    )
