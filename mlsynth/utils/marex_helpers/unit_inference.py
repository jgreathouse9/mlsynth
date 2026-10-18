"""Inference for each treated unit of a MAREX design.

Abadie and Zhao's Unit-level design (their formulation 10) gives each treated
unit ``j`` its own synthetic control ``v_.j``, and their eq. (11) writes the
aggregate effect as the weighted average of the per-unit contrasts,

    u_jt = y_jt - sum_i v_ij y_it,        tau_hat_t = sum_j w_j u_jt.

Each ``u_j`` has a blank stretch and a post stretch, so MAREX's inference runs on
it as it runs on the aggregate. The argument of their Theorem 2 needs only
weights chosen on the fitting periods and factors exchangeable over the blank
and post periods; ``v_.j`` is such a weight vector.

``v_.j`` is fitted after the design, when the assignment is known, on the
fitting periods and from the untreated units of ``j``'s cluster: the units a
cluster's controls come from, and eq. (10)'s constraint that no treated unit is
another's donor. It is computed for every design, not only ``unit_penalized``.

On simulated MAREX designs the per-unit Theorem 2 test held its size under
exchangeable factors (0.047-0.060 at 5%) and the pointwise band its coverage
(0.962-0.968 at 95%). A unit's contrast does not average noise across treated
units, so its band is about twice the aggregate's; and ``m`` per-unit tests are
``m`` tests, so the family gets Bonferroni-adjusted p-values and a joint test.
On a ``standard`` design eq. (11)'s aggregate trailed the shared contrast,
which the design optimised, by 11-21% in RMSE; on a ``unit_penalized`` design,
whose penalty prices each unit's own fit, the two were within 3%. The study is
in ``benchmarks/studies/marex_unit_conformal/``.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from ...exceptions import MlsynthDataError
from ..post_fit import _ar1_rho, _effective_n, _variance_inflation
from ..solvers.active_set import solve_simplex_qp
from .inference import compute_inference
from .structures import (MAREXBlankCheck, MAREXInference, MAREXUnitInference,
                         MAREXUnitLevel)


def joint_p_value(U, T_post: int, *, max_combinations: int,
                  rng: np.random.Generator) -> float:
    """Permutation p-value for no effect on any of the units in ``U``.

    ``U`` is ``(m, n)``: each row a unit's contrast over the blank and post
    periods, the post block last. The statistic is
    ``max_j mean_post |u_j| / mean_window |u_j|``. The denominator is a mean over
    the whole window, which no arrangement changes, so the statistic ignores
    each unit's scale and the test stays a permutation test. Every unit gets the
    same arrangement of periods, which keeps their common factors aligned. A
    unit with no variation contributes zero.
    """
    U = np.abs(np.asarray(U, dtype=float))
    m, n = U.shape
    if not 1 <= T_post <= n:
        raise MlsynthDataError(
            f"T_post must be between 1 and the {n} periods of the window; got "
            f"{T_post}.")
    scale = U.mean(axis=1, keepdims=True)
    Z = np.divide(U, scale, out=np.zeros_like(U), where=scale > 0)
    t_obs = float(np.max(Z[:, n - T_post:].mean(axis=1)))
    tol = 1e-12 * (1.0 + t_obs)
    if math.comb(n, T_post) <= max_combinations:
        pi = np.array(list(itertools.combinations(range(n), T_post)))
        t = np.max(Z[:, pi].mean(axis=2), axis=0)
        return float(np.mean(t >= t_obs - tol))
    pi = np.array([rng.choice(n, size=T_post, replace=False)
                   for _ in range(max_combinations)])
    t = np.max(Z[:, pi].mean(axis=2), axis=0)
    return float((1 + np.sum(t >= t_obs - tol)) / (1 + max_combinations))


def blank_check(blank_contrast, blank_level) -> MAREXBlankCheck:
    """Mean of a unit's contrast over the blank periods, and its standard error.

    ``blank_contrast`` is the unit minus its synthetic control on the blank
    periods, ``blank_level`` the unit's outcome there. The standard error allows
    for AR(1) dependence in the contrast; see :class:`MAREXBlankCheck`.
    """
    b = np.asarray(blank_contrast, dtype=float)
    n = int(b.size)
    mean = float(b.mean())
    scale = float(np.mean(np.abs(np.asarray(blank_level, dtype=float))))
    relative = mean / scale if scale > 0 else math.nan
    if n < 2:
        return MAREXBlankCheck(mean=mean, se=math.nan, t=math.nan,
                               p_value=math.nan, rho=0.0, n=n,
                               relative=relative)
    rho = _ar1_rho(b)
    se = float(np.std(b, ddof=1) * math.sqrt(_variance_inflation(n, rho)))
    if se > 0:
        t = mean / se
        p = float(2.0 * stats.t.sf(abs(t), _effective_n(n, rho) - 1.0))
    elif mean == 0:
        t, p = 0.0, 1.0
    else:
        t, p = math.copysign(math.inf, mean), 0.0
    return MAREXBlankCheck(mean=mean, se=se, t=t, p_value=p, rho=rho, n=n,
                           relative=relative)


def unit_level_inference(
    Y_full: np.ndarray,
    clusters: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    treated_agg: np.ndarray,
    unit_labels: Sequence,
    *,
    T0: int,
    TcE: int,
    Tb: int,
    alpha: float,
    max_combinations: int,
    random_state: Optional[int],
    T_post: Optional[int],
    cumulative_band: bool = False,
    cumulative_method: str = "resample",
    cumulative_block: int = 0,
    cumulative_n_sim: int = 2000,
    cumulative_seed: int = 0,
) -> MAREXUnitLevel:
    """Inference for each treated unit and for eq. (11)'s aggregate.

    Parameters
    ----------
    Y_full : np.ndarray
        Outcomes, ``(N, T)``.
    clusters : sequence of (label, members, treated_weights)
        Each cluster's label, its member rows, and its design's treated
        weights over all ``N`` units.
    treated_agg : np.ndarray
        The aggregate treated weights, ``(N,)``, summing to one over the treated
        units; eq. (11) weights each unit's contrast by them.
    unit_labels : sequence
        Labels of the rows of ``Y_full``.
    T0, TcE, Tb : int
        Pre-periods, fitting periods (where the blank window starts), blank
        periods.
    alpha, max_combinations, random_state, T_post, cumulative_* :
        As for :func:`~mlsynth.utils.marex_helpers.inference.compute_inference`,
        applied to every unit.
    """
    common = dict(alpha=alpha, max_combinations=max_combinations,
                  random_state=random_state, T_post=T_post,
                  cumulative_band=cumulative_band,
                  cumulative_method=cumulative_method,
                  cumulative_block=cumulative_block,
                  cumulative_n_sim=cumulative_n_sim,
                  cumulative_seed=cumulative_seed)
    found = []
    for label, members, treated_w in clusters:
        members = np.asarray(members)
        on = np.asarray(treated_w)[members] > 1e-8
        pool = members[~on]
        for j in members[on]:
            v = solve_simplex_qp(Y_full[pool, :TcE].T, Y_full[j, :TcE])
            syn_c = v @ Y_full[pool]
            inf = compute_inference(Y_full[j], syn_c, T0, TcE, Tb,
                                    control_outcomes=Y_full[pool], **common)
            weights = {str(unit_labels[i]): float(x) for i, x in zip(pool, v)}
            found.append((int(j), str(label), weights, syn_c, inf))

    m = len(found)
    units: Dict[str, MAREXUnitInference] = {}
    for j, label, weights, syn_c, inf in found:
        units[str(unit_labels[j])] = MAREXUnitInference(
            unit=str(unit_labels[j]), cluster=label, control_weights=weights,
            synthetic_control=syn_c, inference=inf,
            p_bonferroni=float(min(1.0, m * inf.global_p_value)),
            blank_check=blank_check(inf.placebo_effects, Y_full[j, TcE:T0]))

    window = np.array([np.r_[f[4].placebo_effects, f[4].treated_effects]
                       for f in found])
    H = found[0][4].treated_effects.size
    joint = joint_p_value(window, H, max_combinations=max_combinations,
                          rng=np.random.default_rng(random_state))

    treated_agg = np.asarray(treated_agg, dtype=float)
    eq11_control = sum(treated_agg[j] * syn_c for j, _, _, syn_c, _ in found)
    untreated = treated_agg <= 1e-8
    eq11 = compute_inference(treated_agg @ Y_full, eq11_control, T0, TcE, Tb,
                             control_outcomes=Y_full[untreated], **common)
    return MAREXUnitLevel(units=units, joint_p_value=joint, eq11=eq11,
                          eq11_control=eq11_control)
