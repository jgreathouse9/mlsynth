"""Inference across a registry of designs (Ferman, Pinto and Possebom 2020).

A single placebo test asks whether the treated unit's post/pre ratio of mean
squared gaps is extreme among the ratios every unit gets when it is the one
treated. Ferman, Pinto and Possebom (2020) show that choosing among several
designs inflates that test, and combine the designs instead: the joint test
ranks each unit's mean statistic across designs, and the Firpo-Possebom (2018)
band collects the effect paths, shifted from the average effect across designs
by a constant, that the joint test does not reject.

Everything here is array-level. Gaps are ``(T, N)`` (periods by units); a stack
of designs is ``(D, T, N)``; a weight stack is ``(D, N, N)`` whose row ``j``
holds unit ``j``'s donor weights over every unit, zero on itself, and is all
missing when unit ``j``'s fit failed. A unit without a statistic (a failed fit,
or no pre-period variation to divide by) is left out of every ranking.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError


def _windows(pre, post, T: int) -> tuple[np.ndarray, np.ndarray]:
    pre, post = np.asarray(pre, bool), np.asarray(post, bool)
    if pre.shape != (T,) or post.shape != (T,):
        raise MlsynthConfigError(f"pre and post must be boolean masks of length {T}.")
    if not pre.any() or not post.any() or (pre & post).any():
        raise MlsynthConfigError("pre and post must be nonempty and disjoint.")
    return pre, post


def ratio_statistic(gaps, pre, post, root: bool = False) -> np.ndarray:
    """Post/pre mean squared gap per unit (FPP eq. 3); ``root`` takes its square root.

    A unit gets NaN when its gaps are not all finite or its pre-period mean
    squared gap is zero (the ratio is undefined).
    """
    gaps = np.asarray(gaps, float)
    if gaps.ndim != 2:
        raise MlsynthDataError("gaps must be a (periods, units) array.")
    pre, post = _windows(pre, post, gaps.shape[0])
    # The ratio is unchanged by rescaling a unit's gaps, so square them only
    # after dividing by their largest magnitude: neither tiny nor huge gaps
    # underflow or overflow on the way.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        scale = np.max(np.abs(gaps[pre | post]), axis=0)
        scaled = gaps / np.where(scale > 0, scale, 1.0)
        pre_ms = np.mean(scaled[pre] ** 2, axis=0)
        ratio = np.mean(scaled[post] ** 2, axis=0) / pre_ms
    ratio[~np.isfinite(gaps).all(axis=0) | ~(pre_ms > 0)] = np.nan
    ratio[~np.isfinite(ratio)] = np.nan
    return np.sqrt(ratio) if root else ratio


def permutation_pvalue(statistic, position: int) -> float:
    """Share of units whose statistic is at least the treated unit's.

    Units without a statistic are left out of the count and the denominator.
    """
    stat = np.asarray(statistic, float).ravel()
    if not 0 <= position < stat.size:
        raise MlsynthConfigError(f"position {position} is outside the {stat.size} units.")
    if not np.isfinite(stat[position]):
        raise MlsynthDataError("The treated unit has no statistic to rank.")
    finite = stat[np.isfinite(stat)]
    return float(np.mean(finite >= stat[position]))


def joint_pvalue(statistics, position: int) -> float:
    """Permutation p-value of each unit's mean statistic across designs.

    ``statistics`` is ``(units, designs)``; a unit's mean runs over the designs
    in which it has a statistic.
    """
    stats = np.asarray(statistics, float)
    if stats.ndim != 2:
        raise MlsynthDataError("statistics must be a (units, designs) array.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing rows -> NaN
        means = np.nanmean(stats, axis=1)
    return permutation_pvalue(means, position)


def _stack(gaps) -> np.ndarray:
    gaps = np.asarray(gaps, float)
    if gaps.ndim != 3:
        raise MlsynthDataError("gaps must be a (designs, periods, units) stack.")
    return gaps


def _window(mask, T: int) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if mask.shape != (T,) or not mask.any():
        raise MlsynthConfigError(f"The window must be a nonempty boolean mask of length {T}.")
    return mask


def dube_zipperer(gaps, post, position: int) -> np.ndarray:
    """Mean post-period squared gap of the placebo units, per design (choose the minimum).

    Dube and Zipperer (2015) judge a design by how well it predicts units that
    were not treated; the treated unit and placebos without a fit are left out.
    """
    gaps = _stack(gaps)
    post = _window(post, gaps.shape[1])
    out = np.empty(gaps.shape[0])
    for d, G in enumerate(gaps):
        placebo = [j for j in range(G.shape[1])
                   if j != position and np.isfinite(G[:, j]).all()]
        out[d] = np.mean(G[np.ix_(post, placebo)] ** 2) if placebo else np.nan
    return out


def donohue(gaps, window, position: int) -> np.ndarray:
    """The treated unit's mean squared gap over a validation window, per design.

    Donohue, Aneja and Weber (2019) choose the design that best predicts the
    treated unit over pre-treatment periods held out of its fit.
    """
    gaps = _stack(gaps)
    window = _window(window, gaps.shape[1])
    return np.mean(gaps[:, window, position] ** 2, axis=1)


def synthetic_gaps(outcomes, weights) -> np.ndarray:
    """``(T, N)`` gaps: each unit's outcome minus its weighted donors.

    Row ``j`` of ``weights`` is unit ``j``'s weight on every unit; a row that
    is all missing (a failed fit) gives that unit missing gaps.
    """
    Y, W = np.asarray(outcomes, float), np.asarray(weights, float)
    N = Y.shape[1]
    if W.shape != (N, N):
        raise MlsynthDataError(f"weights must be ({N}, {N}) for {N} units.")
    fitted = np.isfinite(W).all(axis=1)
    if np.any(np.abs(np.diag(W)[fitted]) > 0):
        raise MlsynthDataError("A unit cannot be its own donor.")
    G = np.full(Y.shape, np.nan)
    G[:, fitted] = Y[:, fitted] - Y @ W[fitted].T
    return G


@dataclass(frozen=True)
class ConfidenceBand:
    """Effect paths the joint test does not reject, around the average effect.

    ``average`` is the treated unit's gap averaged across the designs;
    ``lower`` and ``upper`` are ``average`` shifted by ``shift_lower`` and
    ``shift_upper`` over the post-treatment periods and missing before. An
    infinite shift means the search found no rejection on that side.
    """

    average: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    shift_lower: float
    shift_upper: float


def fp_band(outcomes, weights, position: int, n_pre: int, significance: float,
            precision: int = 30, initial_step: float = 1.0,
            max_steps: int = 100_000) -> ConfidenceBand:
    """Firpo-Possebom (2018) band from the joint test, as Ferman, Pinto and Possebom compute it.

    The nulls are the average post-treatment effect across designs plus a
    constant ``c``. Under each null the treated unit's outcome is netted of the
    hypothesised effect, both in its own gap and wherever it is a placebo's
    donor. Weights stay fixed: they depend on pre-treatment data only, which
    the nulls leave unchanged. The test statistic is the mean across designs of
    each unit's post/pre mean squared gap, rows ``[:n_pre]`` against
    ``[n_pre:]``, and a null is rejected when its p-value is at most
    ``significance``. Each bound is found by the authors' search: walk out
    from zero in steps of ``initial_step``, stop at the first rejection, step
    back, and repeat with the step halved, ``precision + 1`` passes in all. The
    walk runs on the outcome's scale, so ``initial_step`` should be of that
    order; a walk longer than ``max_steps`` stops with an infinite shift and a
    warning.
    """
    Y, W = np.asarray(outcomes, float), np.asarray(weights, float)
    T, N = Y.shape
    if W.ndim != 3 or W.shape[1:] != (N, N):
        raise MlsynthDataError(f"weights must be a (designs, {N}, {N}) stack.")
    if not 0 < significance < 1:
        raise MlsynthConfigError("significance must lie strictly between 0 and 1.")
    if precision < 0 or initial_step <= 0 or max_steps < 1:
        raise MlsynthConfigError("precision >= 0, initial_step > 0 and max_steps >= 1 are required.")
    if not 0 < n_pre < T:
        raise MlsynthConfigError("n_pre must leave at least one period on each side.")
    pre = np.arange(T) < n_pre
    post = ~pre
    own = np.stack([synthetic_gaps(Y, Wd)[:, position] for Wd in W])
    if not np.isfinite(own).all():
        raise MlsynthDataError("The treated unit needs a fit in every design.")
    average = own.mean(axis=0)
    base = np.where(post, average, 0.0)

    def pvalue(c: float) -> float:
        Yn = Y.copy()
        Yn[:, position] = Y[:, position] - (base + c * post)
        stats = np.column_stack(
            [ratio_statistic(synthetic_gaps(Yn, Wd), pre, post) for Wd in W])
        return joint_pvalue(stats, position)

    def walk(sign: float) -> float:
        bound = 0.0
        for power in range(precision + 1):
            step = initial_step * 0.5 ** power
            for _ in range(max_steps):
                if pvalue(bound) <= significance:
                    bound -= sign * step
                    break
                bound += sign * step
            else:
                warnings.warn(
                    f"No rejection within {max_steps} steps of {step:g}; the band "
                    "is unbounded on this side or initial_step is small for the "
                    "outcome's scale.", RuntimeWarning, stacklevel=3)
                return sign * np.inf
        return bound

    upper, lower = walk(1.0), walk(-1.0)
    nan = np.full(T, np.nan)
    return ConfidenceBand(average=average,
                          lower=np.where(post, average + lower, nan),
                          upper=np.where(post, average + upper, nan),
                          shift_lower=float(lower), shift_upper=float(upper))


def identified_effect_range(fit_matrix, weights, outcome_post, donors_post,
                            tol: float = 1e-9) -> tuple[float, float]:
    """Range of the average post-treatment gap over every equally well fitting weight vector.

    The donor weights solve ``min ||b - A w||`` on the simplex for the design's
    fit matrix ``A`` (V-weighted scaled predictors, or the outcomes of the fit
    periods). The loss depends on ``w`` only through ``A w``, so every simplex
    point with ``A w = A w*`` fits exactly as well as the solution ``w*``.
    Minimising and maximising the average gap ``mean(y_post - Y0_post w)`` over
    that set (two linear programs) gives the range the design leaves open; a
    range of width zero means the design pins the effect down. ``tol`` is the
    slack allowed on ``A w = A w*``, relative to the largest entry of ``A``.
    """
    A = np.atleast_2d(np.asarray(fit_matrix, float))
    w = np.asarray(weights, float).ravel()
    y = np.asarray(outcome_post, float).ravel()
    Y0 = np.atleast_2d(np.asarray(donors_post, float))
    J = w.size
    if A.shape[1] != J or Y0.shape != (y.size, J):
        raise MlsynthDataError("fit_matrix, weights and the post-period donors disagree in shape.")
    if not (np.isfinite(A).all() and np.isfinite(w).all() and np.isfinite(y).all()
            and np.isfinite(Y0).all()):
        raise MlsynthDataError("identified_effect_range needs finite inputs.")
    scale = max(1.0, float(np.abs(A).max()))
    As = A / scale
    target = As @ w
    A_ub = np.vstack([As, -As])
    b_ub = np.concatenate([target + tol, -(target - tol)])
    g = Y0.mean(axis=0)  # average gap = mean(y) - g @ w
    options = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
    ends = []
    for sign in (1.0, -1.0):
        res = linprog(sign * g, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, J)), b_eq=[1.0],
                      bounds=[(0, None)] * J, method="highs", options=options)
        if res.status != 0:
            raise MlsynthEstimationError(f"The identified-set program failed: {res.message}")
        ends.append(float(y.mean() - g @ res.x))
    lo, hi = sorted(ends)
    return lo, hi
