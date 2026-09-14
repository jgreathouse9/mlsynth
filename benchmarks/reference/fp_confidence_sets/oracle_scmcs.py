"""Port of ``SCM.CS`` -- Firpo & Possebom (2018) confidence sets.

Firpo, S. & Possebom, V. (2018), "Synthetic Control Method: Inference,
Sensitivity Analysis and Confidence Sets", Journal of Causal Inference 6(2),
20160026. Reference: the authors' supplement ``function_SCM-CS_v07.R``; line
numbers below cite that file.

The procedure inverts Abadie's placebo test over a one-parameter class of effect
paths. For a candidate parameter it builds the null path -- zero through the
pre-period, then constant ``phi`` or linear ``phi * (t - T0)`` -- and imposes it
across the WHOLE panel: for each placebo unit ``j`` the path is added to ``j``'s
own outcome and subtracted from the treated unit's column inside ``j``'s donor
pool (R:140-152). Only then is each unit's post/pre MSPE ratio recomputed and
ranked. That re-imposition is what separates this from taking quantiles of a
placebo distribution computed at zero: the reference distribution is not
invariant to the hypothesised effect when the statistic is a ratio.

The p-value is ``sum_j prob_j * 1{rank_j >= rank_treated}`` with
``prob = softmax(phi_sens * v)`` (R:159-163). With ``phi_sens = 0`` that is the
uniform Abadie p-value; raising it tilts assignment probability toward the units
flagged in ``v``, which is the paper's sensitivity mechanism.

Weights are taken as given, exactly as the reference takes them.
"""
from __future__ import annotations

import numpy as np


def _null_path(value: float, n_periods: int, T0: int, kind: str) -> np.ndarray:
    """The candidate effect path (R:126-133, and again at :198-205).

    Zero over the pre-period; afterwards constant, or linear in periods since
    treatment starting at one.
    """
    path = np.zeros(n_periods)
    if kind == "constant":
        path[T0:] = value
    elif kind == "linear":
        path[T0:] = value * np.arange(1, n_periods - T0 + 1)
    else:
        raise ValueError(f"type must be 'constant' or 'linear', got {kind!r}")
    return path


def _statistics(Y: np.ndarray, W: np.ndarray, treated0: int, T0: int,
                path: np.ndarray) -> np.ndarray:
    """Every unit's post/pre MSPE ratio under the null (R:135-157).

    ``treated0`` is 0-based. For the treated unit the panel is read as observed;
    for a placebo unit the null is added to its outcome and removed from the
    treated column of its donor pool, so the hypothesis is imposed everywhere it
    would act.
    """
    n_periods, n_units = Y.shape
    stats = np.empty(n_units)
    for j in range(n_units):
        donors = [k for k in range(n_units) if k != j]
        if j == treated0:
            y1 = Y[:, treated0]
            y0 = Y[:, donors]
        else:
            y1 = Y[:, j] + path
            y0 = Y[:, donors].copy()
            # the treated unit's position inside j's donor pool
            pos = treated0 - 1 if j < treated0 else treated0
            y0[:, pos] = y0[:, pos] - path
        gaps = y1 - y0 @ W[:, j] - path
        post = float(gaps[T0:] @ gaps[T0:]) / (n_periods - T0)
        pre = float(gaps[:T0] @ gaps[:T0]) / T0
        stats[j] = post / pre
    return stats


def _rank_ascending(x: np.ndarray) -> np.ndarray:
    """R's ``rank`` default: average ranks for ties, ascending, 1-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties, as R does
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def pvalue(Y, W, treated0, T0, path, phi_sens=0.0, v=None) -> float:
    """The weighted rank p-value (R:158-163)."""
    stats = _statistics(Y, W, treated0, T0, path)
    ranks = _rank_ascending(stats)
    n_units = Y.shape[1]
    v = np.zeros(n_units) if v is None else np.asarray(v, float).ravel()
    w = np.exp(phi_sens * v)
    prob = w / w.sum()
    return float(prob @ (ranks >= ranks[treated0]).astype(float))


def confidence_set(Y, W, treated, T0, *, kind="linear", significance=0.05,
                   precision=30, phi_sens=0.0, v=None):
    """Invert the placebo test into a confidence set for the effect parameter.

    ``treated`` is 1-based, matching the reference's argument.

    The search is the reference's (R:112-241): start at the point estimate,
    walk outward in steps of ``sign(param) * (1/2)**power``, and on each
    rejection step back in. ``power`` runs to ``precision``, so the bracket
    tightens by a factor of two per level.

    Returns ``(lower, upper)`` for the parameter, and the per-period paths.
    """
    Y = np.asarray(Y, float)
    W = np.asarray(W, float)
    n_periods, n_units = Y.shape
    t0 = treated - 1

    # the point estimate that seeds the search (R:99-108)
    gaps = Y[:, t0] - Y[:, [k for k in range(n_units) if k != t0]] @ W[:, t0]
    if kind == "constant":
        param = float(gaps[T0:].mean())
    else:
        param = float(gaps[-1]) / (n_periods - T0)
    s = float(np.sign(param))
    ub = lb = param

    def rejects(value):
        p = pvalue(Y, W, t0, T0, _null_path(value, n_periods, T0, kind),
                   phi_sens, v)
        return p <= significance

    first_u = first_l = True
    for power in range(0, precision + 1):
        step = 0.5 ** power
        while True:                                   # upper bound (R:117-176)
            if not rejects(ub):
                ub = param * (ub / param + s * step)
                first_u = False
            else:
                if first_u:
                    raise ValueError("confidence set is empty for this class")
                ub = param * (ub / param - s * step)
                break
            if abs(ub) > 100 * abs(param):
                raise ValueError("upper bound was not found")
        while True:                                   # lower bound (R:228-238)
            # the sign is mirrored against the upper branch: the lower bound
            # walks away from the point estimate on a non-rejection and steps
            # back in on a rejection (R:229 vs :235).
            if not rejects(lb):
                lb = param * (lb / param - s * step)
                first_l = False
            else:
                if first_l:
                    raise ValueError("confidence set is empty for this class")
                lb = param * (lb / param + s * step)
                break
            if abs(lb) > 100 * abs(param):
                raise ValueError("lower bound was not found")

    return (lb, ub, _null_path(lb, n_periods, T0, kind),
            _null_path(ub, n_periods, T0, kind))
