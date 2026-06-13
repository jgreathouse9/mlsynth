"""Placebo-permutation inference for the MicroSynth panel method.

Port of the permutation test in the R ``microsynth`` package (the ``perm`` /
``test`` arguments). Inference is by **placebo permutation**: draw many random
sets of ``n_T`` control units, treat each as a placebo "treated area", refit the
panel QP from the remaining controls, and record the placebo treatment effect.
The observed effect is then compared to that null distribution of placebo
effects to form a one- or two-sided p-value (and a permutation-based CI).

This mirrors ``microsynth``'s construction, which samples permutation groups of
the same size as the treated area from the units **not** in the real treated
area (``microsynth/R/weights.r``), and reports p-values as the rank of the
observed effect among the placebos (``get.pval``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .panel_qp import solve_panel_qp
from ...exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class PanelPermutationResult:
    """Placebo-permutation inference for the panel method."""

    p_value: float                 # ATT-level p-value
    p_values_by_period: np.ndarray # per-post-period p-values
    placebo_atts: np.ndarray       # null distribution of placebo ATTs
    se: float                      # SD of the placebo distribution
    ci: np.ndarray                 # [low, high] permutation CI for the ATT
    n_perm: int                    # successful placebo groups
    test: str


def _tail_pvalue(obs: float, placebos: np.ndarray, test: str) -> float:
    """Rank-based p-value of ``obs`` against the placebo distribution."""
    n = placebos.size
    if test == "lower":
        count = int(np.sum(placebos <= obs))
    elif test == "upper":
        count = int(np.sum(placebos >= obs))
    else:  # twosided: compare magnitudes
        count = int(np.sum(np.abs(placebos) >= abs(obs)))
    return (1 + count) / (1 + n)


def panel_permutation_test(
    *,
    cov_C: np.ndarray,
    lag_C: Optional[np.ndarray],
    Y_C_post: np.ndarray,
    n_T: int,
    obs_gap_trajectory: np.ndarray,
    obs_att: float,
    ridge: float,
    n_perm: int,
    test: str = "twosided",
    seed: int = 1400,
    confidence: float = 0.95,
) -> PanelPermutationResult:
    """Placebo-permutation inference for the panel-method ATT.

    Parameters
    ----------
    cov_C : np.ndarray
        Raw control covariate matrix, shape ``(n_C, d_cov)``.
    lag_C : np.ndarray or None
        Raw control lagged-outcome matrix, shape ``(n_C, m)``; ``None`` (or a
        zero-column array) for covariates-only weighting.
    Y_C_post : np.ndarray
        Control post-period outcomes, shape ``(n_C, T_post)``.
    n_T : int
        Treated-area size (placebo groups are this many controls).
    obs_gap_trajectory : np.ndarray
        Observed per-post-period total effects, shape ``(T_post,)``.
    obs_att : float
        Observed ATT (mean of ``obs_gap_trajectory``).
    ridge : float
        Panel QP ridge (same as the main fit).
    n_perm : int
        Number of placebo groups to draw.
    test : str
        ``'lower'``, ``'upper'`` or ``'twosided'``.
    seed : int
        RNG seed for placebo-group sampling.
    confidence : float
        Confidence level for the permutation CI.

    Returns
    -------
    PanelPermutationResult
    """
    n_C = cov_C.shape[0]
    if n_perm <= 0:
        raise MlsynthEstimationError(
            "panel_permutation_test requires n_perm > 0."
        )
    if n_T >= n_C:
        raise MlsynthEstimationError(
            "Not enough control units to form placebo groups "
            f"(n_T={n_T}, n_C={n_C})."
        )
    if lag_C is not None and lag_C.shape[1] == 0:
        lag_C = None

    rng = np.random.default_rng(seed)
    T_post = Y_C_post.shape[1]
    placebo_traj = []
    all_idx = np.arange(n_C)
    for _ in range(n_perm):
        placebo = rng.choice(n_C, size=n_T, replace=False)
        donor = np.setdiff1d(all_idx, placebo, assume_unique=False)
        hard_C = np.column_stack([np.ones(donor.size), cov_C[donor]])
        hard_t = np.concatenate([[float(n_T)], cov_C[placebo].sum(axis=0)])
        if lag_C is not None:
            soft_C = lag_C[donor]
            soft_t = lag_C[placebo].sum(axis=0)
        else:
            soft_C = soft_t = None
        try:
            sol = solve_panel_qp(hard_C, hard_t, soft_C, soft_t, ridge=ridge)
        except MlsynthEstimationError:     # pragma: no cover - rare infeasible draw
            continue                       # infeasible placebo group: skip
        cf = sol.w @ Y_C_post[donor]       # (T_post,)
        placebo_traj.append(Y_C_post[placebo].sum(axis=0) - cf)

    if not placebo_traj:                    # pragma: no cover - degenerate pool
        raise MlsynthEstimationError(
            "All placebo permutation groups were infeasible; cannot compute "
            "panel-method inference."
        )
    placebo_traj = np.asarray(placebo_traj)            # (n_ok, T_post)
    placebo_atts = placebo_traj.mean(axis=1)           # (n_ok,)

    p_value = _tail_pvalue(obs_att, placebo_atts, test)
    p_by_period = np.array([
        _tail_pvalue(obs_gap_trajectory[t], placebo_traj[:, t], test)
        for t in range(T_post)
    ])
    se = float(placebo_atts.std(ddof=1)) if placebo_atts.size > 1 else float("nan")
    alpha = 1.0 - confidence
    lo_q, hi_q = np.quantile(placebo_atts, [alpha / 2, 1 - alpha / 2])
    ci = np.array([obs_att - hi_q, obs_att - lo_q])     # placebo-null CI
    return PanelPermutationResult(
        p_value=float(p_value),
        p_values_by_period=p_by_period,
        placebo_atts=placebo_atts,
        se=se,
        ci=ci,
        n_perm=int(placebo_atts.size),
        test=test,
    )
