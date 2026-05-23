"""Pre-treatment parallelism scoring for PANGEO supergeo pairs.

The design objective replaces Supergeo's scalar sum-matching with a
**difference-in-differences parallelism** score on the full pre-period
*vector*. For a supergeo pair split into halves ``A`` and ``B`` with mean
trajectories :math:`\\bar Y_A, \\bar Y_B`, define the DiD level shift
:math:`\\delta = \\overline{(\\bar Y_A - \\bar Y_B)}` and score the pair by
the variance of the *level-removed* pre-period gap

.. math::

   \\text{score}(A, B) = \\sum_{t} \\big[(\\bar Y_{A,t} - \\bar Y_{B,t}) - \\delta\\big]^2 .

This is exactly the pre-period residual sum of squares of a DiD fit (cf.
:func:`mlsynth.utils.selector_helpers._did_from_mean`): minimising it makes
the two halves run parallel, so the within-pair DiD comparison is clean
regardless of their *levels* (the level is absorbed by :math:`\\delta`).
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

_EPS = 1e-12


def _wavg(matrix: np.ndarray, idx, weights: Optional[np.ndarray]) -> np.ndarray:
    """Weighted column-mean of ``matrix[idx]`` (plain mean if ``weights`` is
    None). The supergeo aggregate -- weighting by e.g. population makes the
    design and the downstream ATT population-weighted and consistent.
    """
    if weights is None:
        return matrix[idx].mean(axis=0)
    return np.average(matrix[idx], axis=0, weights=weights[idx])


def gap_variance(mean_a: np.ndarray, mean_b: np.ndarray) -> float:
    """Variance of the level-removed gap between two trajectories (the DiD
    pre-period residual sum of squares).
    """
    gap = mean_a - mean_b
    resid = gap - gap.mean()
    return float(resid @ resid)


def parallelism_r2(mean_a: np.ndarray, mean_b: np.ndarray) -> float:
    """R^2 of the DiD parallel-trends fit (1 = perfectly parallel)."""
    gap = mean_a - mean_b
    resid = gap - gap.mean()
    ss_res = float(resid @ resid)
    # Total variation in A around its level (the scale R^2 is relative to).
    a_c = mean_a - mean_a.mean()
    ss_tot = float(a_c @ a_c)
    if ss_tot <= _EPS:
        return np.nan
    return 1.0 - ss_res / ss_tot


def split_cost(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    objective: str = "ss_res",
    weights: Optional[np.ndarray] = None,
) -> float:
    """Per-pair cost minimised by the MIP (lower = more parallel).

    All three objectives are precomputed scalars, so the outer selection
    problem stays a linear MILP.

    * ``"ss_res"`` -- absolute DiD residual sum of squares
      :math:`\\sum_t (g_t - \\bar g)^2` (scale-dependent; big-amplitude
      pairs weigh more).
    * ``"r2"`` -- ``1 - R^2`` = ``ss_res / ss_tot`` (scale-free; every pair
      counts equally, FDID's R^2 criterion but optimised exactly).
    * ``"weighted"`` -- weighted residual SS
      :math:`\\sum_t w_t (g_t - \\bar g_w)^2` with the level removed at the
      *weighted* mean :math:`\\bar g_w = \\sum_t w_t g_t / \\sum_t w_t`
      (e.g. recency weighting, so recent parallelism matters more).
    """
    gap = mean_a - mean_b
    if objective == "weighted":
        if weights is None:
            raise ValueError("objective='weighted' requires weights.")
        wsum = float(weights.sum())
        delta = float((weights * gap).sum()) / max(wsum, _EPS)
        resid = gap - delta
        return float((weights * resid ** 2).sum())
    resid = gap - gap.mean()
    ss_res = float(resid @ resid)
    if objective == "r2":
        a_c = mean_a - mean_a.mean()
        ss_tot = float(a_c @ a_c)
        if ss_tot <= _EPS:
            return ss_res
        return ss_res / ss_tot          # = 1 - R^2 (lower is better)
    return ss_res                        # "ss_res"


def covariate_imbalance(
    cov_a: np.ndarray, cov_b: np.ndarray, scales: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted standardized SMD^2 between two supergeos' covariate means.

    For supergeo means :math:`\\bar c_A, \\bar c_B` (averaged over each
    half's units) and per-covariate scales :math:`s_m`,

    .. math::

       \\sum_m w_m \\Big(\\frac{\\bar c_{A,m} - \\bar c_{B,m}}{s_m}\\Big)^2 .

    A precomputed scalar, so adding it to the trajectory cost keeps the
    outer set-partitioning problem a linear MILP.
    """
    smd = (cov_a - cov_b) / scales
    if weights is None:
        return float(smd @ smd)
    return float((weights * smd ** 2).sum())


def best_split(
    members: np.ndarray, Ypre: np.ndarray, max_size: int,
    objective: str = "ss_res", weights: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    cov_scales: Optional[np.ndarray] = None,
    cov_weights: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
) -> Tuple[float, List[int], List[int]]:
    """Best treatment/control split of a candidate supergeo pair.

    Parameters
    ----------
    members : np.ndarray
        Row indices (into ``Ypre``) of the units in this candidate pair.
    Ypre : np.ndarray
        Pre-period outcomes, shape ``(n_units, T0)``.
    max_size : int
        Maximum size of either supergeo (Q).
    objective : {"ss_res", "r2", "weighted"}
        Per-pair cost to minimise (see :func:`split_cost`).
    weights : np.ndarray, optional
        Length-``T0`` weights for ``objective="weighted"``.
    cov : np.ndarray, optional
        Baseline covariate matrix, shape ``(n_units, M)`` aligned with the
        rows of ``Ypre``. When given, a standardized SMD^2 imbalance term is
        added to each split's trajectory cost (see :func:`covariate_imbalance`).
    cov_scales : np.ndarray, optional
        Length-``M`` standardization scales for the covariates.
    cov_weights : np.ndarray, optional
        Length-``M`` per-covariate penalty weights (default 1 each).
    unit_weights : np.ndarray, optional
        Length-``n_units`` per-unit aggregation weights (e.g. population);
        the supergeo mean trajectory is the weighted average of its members.

    Returns
    -------
    score : float
        Minimum cost over admissible splits (``inf`` if none).
    side_a, side_b : list of int
        The treatment / control halves (unit indices) achieving it.
    """
    m = len(members)
    best = (np.inf, [], [])
    # Enumerate splits where side A has 1..min(max_size, m-1) members and
    # side B = remainder also respects max_size. Use combinations of A; the
    # complementary B is determined. Avoid double-counting by fixing the
    # first member to side A.
    first = members[0]
    rest = members[1:]
    for size_a in range(1, min(max_size, m - 1) + 1):
        size_b = m - size_a
        if size_b > max_size or size_b < 1:
            continue
        for combo in combinations(rest, size_a - 1):
            side_a = [first] + list(combo)
            set_a = set(side_a)
            side_b = [u for u in members if u not in set_a]
            mean_a = _wavg(Ypre, side_a, unit_weights)
            mean_b = _wavg(Ypre, side_b, unit_weights)
            s = split_cost(mean_a, mean_b, objective=objective, weights=weights)
            if cov is not None:
                s += covariate_imbalance(
                    _wavg(cov, side_a, unit_weights),
                    _wavg(cov, side_b, unit_weights),
                    cov_scales, cov_weights,
                )
            if s < best[0]:
                best = (s, side_a, side_b)
    return best


def enumerate_candidate_pairs(
    unit_indices: np.ndarray, Ypre: np.ndarray, max_size: int,
    objective: str = "ss_res", weights: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    cov_scales: Optional[np.ndarray] = None,
    cov_weights: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
) -> List[dict]:
    """All admissible supergeo pairs over ``unit_indices`` with their scores.

    A candidate pair is any subset of size ``2 .. 2*max_size`` that can be
    split into two halves each of size ``<= max_size``. Returns a list of
    ``{"members", "score", "side_a", "side_b"}`` dicts -- the inputs to the
    set-partitioning MIP. ``score`` is the chosen ``objective`` (plus the
    optional standardized covariate-imbalance penalty when ``cov`` is given).
    """
    n = len(unit_indices)
    pairs: List[dict] = []
    upper = min(2 * max_size, n)
    for size in range(2, upper + 1):
        for combo in combinations(range(n), size):
            members = np.array([unit_indices[c] for c in combo])
            score, side_a, side_b = best_split(
                members, Ypre, max_size, objective=objective, weights=weights,
                cov=cov, cov_scales=cov_scales, cov_weights=cov_weights,
                unit_weights=unit_weights)
            if np.isfinite(score):
                pairs.append({
                    "members": members,
                    "score": score,
                    "side_a": np.array(side_a, dtype=int),
                    "side_b": np.array(side_b, dtype=int),
                })
    return pairs
