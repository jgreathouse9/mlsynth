"""Ibragimov-Muller inference for ISCM (paper Section 5, eq. 16).

ISCM yields one treatment-effect estimate :math:`\\widehat\\alpha_i` per
contributing unit, with relative weights :math:`v_i` (so
:math:`\\widehat\\alpha = \\sum_{i \\in C} v_i \\widehat\\alpha_i`). Under
the null :math:`H_0: \\alpha = \\alpha_0` the weighted deviations
:math:`v_i(\\widehat\\alpha_i - \\alpha_0)` are treated as approximately
symmetric, and a sign-flip (Rademacher) randomization test calibrates the
p-value -- conservative but valid even with a handful of contributing
units, where a permutation test cannot reach standard thresholds.
"""

from __future__ import annotations

import numpy as np

from .structures import ISCMInference

_EPS = 1e-12


def ibragimov_muller_inference(
    att: float,
    unit_att: np.ndarray,
    contribution: np.ndarray,
    *,
    N: int,
    null_value: float = 0.0,
    alpha_level: float = 0.05,
    n_draws: int = 10000,
    random_state: int = 0,
) -> ISCMInference:
    """Sign-flip randomization test over the per-unit estimates.

    Parameters
    ----------
    att : float
        Aggregate ATT.
    unit_att : np.ndarray
        Per-unit estimates (``NaN`` outside the contributing set).
    contribution : np.ndarray
        Per-unit weights :math:`v_i` (sum to one over ``C``).
    N : int
        Total number of units (for the finite-sample variance scaling).
    null_value : float
        Tested null :math:`\\alpha_0`.
    alpha_level : float
        Two-sided level for the reported CI.
    n_draws : int
        Number of Rademacher sign-flip draws.
    random_state : int
        RNG seed.
    """
    contributing = np.isfinite(unit_att) & (contribution > _EPS)
    a_i = unit_att[contributing]
    v_i = contribution[contributing]
    q = int(contributing.sum())

    # Weighted deviations of the per-unit estimates around the point ATT.
    var = (N / max(N - 1, 1)) * float(np.sum(v_i * (a_i - att) ** 2))
    se = float(np.sqrt(max(var, 0.0)))
    t_stat = (att - null_value) / se if se > _EPS else np.nan

    # Sign-flip distribution of the weighted-sum statistic under H0.
    c_i = v_i * (a_i - null_value)            # contributions; sum = att - null
    observed = float(np.abs(np.sum(c_i)))
    if q >= 1:
        rng = np.random.default_rng(random_state)
        signs = rng.choice([-1.0, 1.0], size=(n_draws, q))
        flipped = np.abs(signs @ c_i)
        p_value = float((flipped >= observed - _EPS).mean())
    else:
        p_value = np.nan

    z = 1.959963984540054 if abs(alpha_level - 0.05) < 1e-9 else _z(alpha_level)
    ci = (att - z * se, att + z * se)

    return ISCMInference(
        method="ibragimov_muller",
        null_value=float(null_value),
        t_stat=float(t_stat),
        p_value=p_value,
        se=se,
        ci=ci,
        alpha_level=float(alpha_level),
        n_contributing=q,
        n_draws=int(n_draws),
    )


def _z(alpha_level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha_level / 2.0))
