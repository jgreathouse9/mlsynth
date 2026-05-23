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


def best_split(
    members: np.ndarray, Ypre: np.ndarray, max_size: int,
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

    Returns
    -------
    score : float
        Minimum gap variance over admissible splits (``inf`` if none).
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
            mean_a = Ypre[side_a].mean(axis=0)
            mean_b = Ypre[side_b].mean(axis=0)
            s = gap_variance(mean_a, mean_b)
            if s < best[0]:
                best = (s, side_a, side_b)
    return best


def enumerate_candidate_pairs(
    unit_indices: np.ndarray, Ypre: np.ndarray, max_size: int,
) -> List[dict]:
    """All admissible supergeo pairs over ``unit_indices`` with their scores.

    A candidate pair is any subset of size ``2 .. 2*max_size`` that can be
    split into two halves each of size ``<= max_size``. Returns a list of
    ``{"members", "score", "side_a", "side_b"}`` dicts -- the inputs to the
    set-partitioning MIP.
    """
    n = len(unit_indices)
    pairs: List[dict] = []
    upper = min(2 * max_size, n)
    for size in range(2, upper + 1):
        for combo in combinations(range(n), size):
            members = np.array([unit_indices[c] for c in combo])
            score, side_a, side_b = best_split(members, Ypre, max_size)
            if np.isfinite(score):
                pairs.append({
                    "members": members,
                    "score": score,
                    "side_a": np.array(side_a, dtype=int),
                    "side_b": np.array(side_b, dtype=int),
                })
    return pairs
