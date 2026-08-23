"""A band for the cumulative effect, obtained by inverting a joint conformal test.

:mod:`.cumulative` calibrates a total on out-of-sample errors: a withheld stretch
is split into horizon-length windows and their summed errors become the
conformity scores. A total over ``H`` periods is ``H`` times the mean error over
them, so the term that sizes it is a level observed once per window, and the
calibration set for that term has size ``m`` -- the number of windows -- not
``m * H``. Coverage is then bounded by ``m``: 0.86 at three windows and 0.94 at
twenty-six, even with every distributional assumption granted (see
``benchmarks/cases/conformal_window_count.py``). A design whose withheld stretch
supports three windows cannot reach nominal by any resampling of them.

Test inversion has no such ceiling. Under a sharp null the whole post block is
imputed, the control is refit on every period, and the post-block statistic is
compared against permutations of the residual path, so the reference set is the
``T0 + H`` periods themselves. Chernozhukov, Wuthrich and Zhu give a
nonasymptotic size bound whose leading term is ``(H / T0) ** 0.25 * log T`` with
``H`` fixed: the rate is in the length of the pre-period, and no window count
enters it.

What that costs is the shape of the null. A one-dimensional interval requires a
one-dimensional null, so the effect is taken to be constant across the post
block and the total is ``H`` times it. The band therefore covers the total under
a constant effect and says nothing about an arbitrary path -- CWZ's Remark 1
statistic (the signed sum) targets the same average deviation, and neither
recovers a path from one number. Where the effect plainly grows or decays,
``.cumulative`` or ``.resample`` make no such restriction and pay for it in
``m`` instead.

Two answers here are not numbers, and both are real answers, not failures to
compute. An empty set means no constant effect conforms, which a multiplicative
lift on a trending panel produces. An infinite endpoint means nothing is excluded
in that direction: the block scheme's reference set is the ``T`` cyclic shifts,
one of which is the observed path, so no p-value falls below ``1 / T`` and at
``alpha < 1 / T`` no candidate is ever rejected. Returning a finite bound in
either case would report a rejection that never happened.

The search, the refit rules and the permutation schemes are
:func:`~mlsynth.utils.bilevel.ridge_inference.conformal_att_interval`'s; this
module supplies the validation, the cumulative reading and the container, so an
estimator can ask for a band for the total without reaching into the ridge
machinery.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864. Remark 1 and Assumption 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .refit import CONFORMAL_REFIT_RULES

#: Permutation sets, CWZ's ``Pi_all`` and ``Pi_->``. ``block`` uses the ``T``
#: cyclic shifts and preserves serial dependence, which Assumption 2.2 requires
#: of a stationary strongly-mixing shock process; ``iid`` draws permutations and
#: is valid under Assumption 2.1, giving a finer p-value grid.
CONFORMAL_PERMUTATIONS = ("block", "iid")

#: How the accepted set is located. ``bisect`` brackets outward from the point
#: estimate and bisects each crossing, which finds a narrow set at any width but
#: assumes the survivors form one interval. ``grid`` evaluates the candidates it
#: is given and returns the range of everything that survived, which tolerates
#: gaps but only sees what the grid covers.
CONFORMAL_SEARCHES = ("bisect", "grid")


@dataclass(frozen=True)
class CumulativeInversionBand:
    """Interval for the total effect, from inverting a constant-effect null.

    Parameters
    ----------
    point : float
        ``horizon`` times the mean post-period gap of a fit on the pre-period
        alone -- the estimator's own effect, not the null refit's.
    lower, upper : float
        Endpoints for the total. ``nan`` on both when the accepted set is empty;
        ``-inf`` / ``+inf`` on a side nothing excludes. The two need not be
        equidistant from ``point``: the test's asymmetry is preserved.
    per_period_lower, per_period_upper : float
        The same interval before multiplying by the horizon, i.e. for the
        constant per-period effect that was actually tested.
    horizon : int
        Post periods accumulated.
    alpha : float
        Level; the band is the candidates with ``p > alpha``.
    refit : str
        Which control the null refit used, from
        :data:`~mlsynth.utils.conformal.refit.CONFORMAL_REFIT_RULES`.
    conformal_type : str
        Permutation set, from :data:`CONFORMAL_PERMUTATIONS`.
    n_reference : int
        Periods entering the test, ``pre + horizon``. Under the block scheme this
        is also the size of the reference set, so ``1 / n_reference`` is the
        smallest attainable p-value.
    search : str
        How the set was located, from :data:`CONFORMAL_SEARCHES`.
    has_interior_gaps : bool or None
        Whether accepted candidates were separated by rejected ones. ``None``
        under ``bisect``, which evaluates only one outward walk per side and so
        cannot know; reporting ``False`` there would claim more than it has
        seen.
    is_empty : bool
        No constant effect conformed.
    is_unbounded : bool
        At least one endpoint is infinite.
    """

    point: float
    lower: float
    upper: float
    per_period_lower: float
    per_period_upper: float
    horizon: int
    alpha: float
    refit: str
    conformal_type: str
    n_reference: int
    is_empty: bool
    is_unbounded: bool
    search: str
    has_interior_gaps: Optional[bool]


def _check_alpha(alpha) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating,
                                                         np.integer)):
        raise MlsynthConfigError(f"alpha must be a number in (0, 1); got {alpha!r}.")
    if not 0.0 < float(alpha) < 1.0:
        raise MlsynthConfigError(
            f"alpha must lie in the open interval (0, 1); got {alpha!r}.")
    return float(alpha)


def _check_series(y, Y0):
    y = np.asarray(y, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    if y.ndim != 1:
        raise MlsynthDataError(
            f"y must be one-dimensional; got shape {y.shape}.")
    if Y0.ndim != 2:
        raise MlsynthDataError(
            f"Y0 must be two-dimensional (n_periods, n_donors); got shape "
            f"{Y0.shape}.")
    if Y0.shape[0] != y.size:
        raise MlsynthDataError(
            f"y and Y0 must cover the same periods; y has {y.size} periods and "
            f"Y0 has {Y0.shape[0]}.")
    if not np.isfinite(y).all() or not np.isfinite(Y0).all():
        raise MlsynthDataError("y and Y0 must be finite; they contain nan or inf.")
    return y, Y0


def _check_counts(pre, horizon, n_periods):
    if isinstance(pre, bool) or not isinstance(pre, (int, np.integer)):
        raise MlsynthConfigError(f"pre must be a positive int; got {pre!r}.")
    if not 1 <= pre < n_periods:
        raise MlsynthConfigError(
            f"pre must be between 1 and {n_periods - 1}; got {pre}.")
    if horizon is None:
        horizon = n_periods - int(pre)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise MlsynthConfigError(
            f"horizon must be a positive int; got {horizon!r}.")
    if not 1 <= horizon <= n_periods - int(pre):
        raise MlsynthConfigError(
            f"horizon must be between 1 and the {n_periods - int(pre)} post "
            f"periods available; got {horizon}.")
    return int(pre), int(horizon)


def _check_grid(grid):
    grid = np.asarray(grid, dtype=float).ravel() if not isinstance(grid, str) \
        else None
    if grid is None:
        raise MlsynthConfigError(
            "grid must be an array of candidate per-period effects.")
    if grid.size < 2:
        raise MlsynthConfigError(
            f"grid must hold at least two candidates; got {grid.size}.")
    if not np.isfinite(grid).all():
        raise MlsynthDataError("grid must be finite; it contains nan or inf.")
    return np.sort(grid)


def _grid_bounds(pvalues, grid, alpha):
    """Range of the surviving candidates, and whether they had holes.

    Delegates the accept rule and the range to
    :func:`~mlsynth.utils.conformal.inversion.confidence_set_bounds`, which owns
    the strict ``p > alpha`` convention and the decision to take the range of the
    whole surviving set instead of its largest contiguous run.
    """
    from .inversion import confidence_set_bounds

    pvalues = np.asarray(pvalues, dtype=float)
    lo, hi = confidence_set_bounds(grid, pvalues, alpha)
    kept = pvalues > alpha
    if not kept.any():
        return float("nan"), float("nan"), False
    first, last = int(np.argmax(kept)), int(kept.size - 1 - np.argmax(kept[::-1]))
    gaps = bool((~kept[first:last + 1]).any())
    return float(lo), float(hi), gaps


def cumulative_conformal_by_inversion(
    y,
    Y0,
    pre: int,
    horizon: Optional[int] = None,
    *,
    alpha: float = 0.1,
    refit: str = "sc",
    conformal_type: str = "block",
    search: str = "bisect",
    grid=None,
    **kwargs: Any,
) -> CumulativeInversionBand:
    """Band for the total effect over ``horizon`` periods, by test inversion.

    Parameters
    ----------
    y : array-like, shape ``(T,)``
        Treated outcomes over the full sample.
    Y0 : array-like, shape ``(T, J)``
        Donor outcomes aligned to ``y``.
    pre : int
        Number of pre-treatment periods.
    horizon : int, optional
        Post periods to accumulate. Defaults to every post period. Only
        ``pre + horizon`` periods enter the test, so a longer series does not
        lend it extra permutations.
    alpha : float
        Level. The band is the candidates a level-``alpha`` test does not reject,
        i.e. ``p > alpha``.
    refit : {"sc", "ridge"}
        Which control the null refit uses; see
        :mod:`~mlsynth.utils.conformal.refit`. It should match the estimator
        whose effect is being tested. ``"sc"`` is the procedure as published.
    conformal_type : {"block", "iid"}
        Permutation set. See :data:`CONFORMAL_PERMUTATIONS`.
    search : {"bisect", "grid"}
        How to locate the accepted set. See :data:`CONFORMAL_SEARCHES`. A
        conformal p-value is not unimodal in the candidate, so the survivors can
        form more than one island; ``bisect`` walks outward from the point
        estimate and stops at the first crossing, which is fast and finds a
        narrow set, but can exclude an accepted candidate beyond a hole.
        ``grid`` sees whatever the grid covers and says whether there were holes,
        and chains its refits through
        :func:`~mlsynth.utils.bilevel.ridge_inference.conformal_pvalue_sweep`,
        so its cost is far below one cold solve per candidate.
    grid : array-like, optional
        Candidate per-period effects, required by ``search="grid"``. There is no
        default: a grid has to be scaled to the panel's own dispersion, and one
        guessed here would step over a narrow set and report it empty.
    **kwargs
        Forwarded to
        :func:`~mlsynth.utils.bilevel.ridge_inference.conformal_att_interval`
        (``q``, ``ns``, ``seed``, ``lambda_``, ``fixed_effects``, the search
        controls).

    Returns
    -------
    CumulativeInversionBand

    Raises
    ------
    MlsynthConfigError
        On a bad ``alpha``, ``pre``, ``horizon``, ``refit`` or
        ``conformal_type``.
    MlsynthDataError
        On mismatched, misshapen or non-finite series.
    """
    y, Y0 = _check_series(y, Y0)
    pre, horizon = _check_counts(pre, horizon, y.size)
    alpha = _check_alpha(alpha)
    if refit not in CONFORMAL_REFIT_RULES:
        raise MlsynthConfigError(
            f"refit must be one of {CONFORMAL_REFIT_RULES}; got {refit!r}.")
    if conformal_type not in CONFORMAL_PERMUTATIONS:
        raise MlsynthConfigError(
            f"conformal_type must be one of {CONFORMAL_PERMUTATIONS}; got "
            f"{conformal_type!r}.")
    if search not in CONFORMAL_SEARCHES:
        raise MlsynthConfigError(
            f"search must be one of {CONFORMAL_SEARCHES}; got {search!r}.")
    if search == "grid":
        if grid is None:
            raise MlsynthConfigError(
                'search="grid" needs a grid of candidate per-period effects.')
        grid = _check_grid(grid)

    from ..bilevel.ridge_inference import conformal_att_interval
    from ..bilevel.simplex import simplex_lstsq

    n_reference = pre + horizon
    y_w, Y0_w = y[:n_reference], Y0[:n_reference]

    if search == "bisect":
        lo, hi = conformal_att_interval(y_w, Y0_w, pre, alpha=alpha, refit=refit,
                                        conformal_type=conformal_type, **kwargs)
        gaps = None
    else:
        # One sweep, not one call per candidate. The sweep walks the grid sorted
        # so consecutive solves are neighbouring problems and each seeds the next
        # from its own solution; the p-values come back in the caller's order and
        # are identical to solving every candidate cold, since a seed moves how
        # many pivots the active set takes and never where it certifies.
        from ..bilevel.ridge_inference import conformal_pvalue_sweep

        pvalues = conformal_pvalue_sweep(y_w, Y0_w, pre, grid, refit=refit,
                                         conformal_type=conformal_type, **kwargs)
        lo, hi, gaps = _grid_bounds(pvalues, grid, alpha)

    # The point is the estimator's own effect: weights fitted on the pre-period
    # alone, so the post gaps are the ones the band is meant to describe.
    w = simplex_lstsq(Y0[:pre], y[:pre])
    point = horizon * float(np.mean(y[pre:n_reference]
                                    - Y0[pre:n_reference] @ w))

    is_empty = bool(np.isnan(lo) or np.isnan(hi))
    is_unbounded = bool(np.isinf(lo) or np.isinf(hi))
    return CumulativeInversionBand(
        point=point,
        lower=lo * horizon,
        upper=hi * horizon,
        per_period_lower=float(lo),
        per_period_upper=float(hi),
        horizon=horizon,
        alpha=alpha,
        refit=refit,
        conformal_type=conformal_type,
        n_reference=n_reference,
        is_empty=is_empty,
        is_unbounded=is_unbounded,
        search=search,
        has_interior_gaps=gaps,
    )
