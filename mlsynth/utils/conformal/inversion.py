"""Inverting a conformal test into a confidence set.

A test-inversion interval is defined by duality: sweep a grid of candidate
effects, test each at level ``alpha``, and keep the candidates the test does not
reject. The interval is the range of what survives, so the whole construction
reduces to one decision -- what counts as "not rejected" -- and that decision
lives here, not at each call site, because the library had it both ways.

The decision is ``p > alpha``. A level-``alpha`` test rejects when ``p <=
alpha``, so a candidate with ``p == alpha`` is rejected and belongs outside the
set. For a continuous statistic that boundary is a measure-zero quibble. A
conformal p-value is not continuous: its reference set is finite -- ``T0 + 1``
residuals for a single-period inversion, ``T`` cyclic blocks for the
moving-block scheme -- so ``p`` takes only the values ``k / n``, and whenever
``alpha`` is one of them the boundary carries a whole level of the reference
set. On the JASA gonorrhea panel (``T0 + 1 = 20``, ``alpha = 0.1``, both
attained exactly) the inclusive rule widens every per-period interval by 15 to
40 percent against the authors' own package.

The strict rule is what ``scinference::confidence_interval`` uses
(``ci_grid[ps_temp > alpha]``, v1.0.0 at ``567c688``) and what
:func:`~mlsynth.utils.bilevel.ridge_inference.conformal_att_interval` already
used for its bracketing search.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...exceptions import MlsynthDataError


def confidence_set_bounds(
    grid: np.ndarray, pvalues: np.ndarray, alpha: float
) -> Tuple[float, float]:
    r"""Endpoints of the nulls a level-``alpha`` test does not reject.

    Returns ``(lower, upper)``, the range of ``{tau in grid : p(tau) > alpha}``.

    The endpoints are the range of the surviving set, not the ends of its
    largest contiguous run: a conformal p-value path need not be unimodal in the
    candidate, so the accepted set can have interior gaps, and the object being
    reported is an interval. Summarising by the range is what the reference does
    (``min``/``max`` of the accepted grid) and it is the conservative reading --
    the alternative could report an interval that excludes candidates the test
    accepted.

    Parameters
    ----------
    grid : array-like, shape (m,)
        Candidate effects that were tested. Order is irrelevant; only the range
        of the surviving subset is used.
    pvalues : array-like, shape (m,)
        ``pvalues[i]`` is the p-value for the null that the effect is
        ``grid[i]``. Must be finite -- a NaN compares ``False`` against any
        threshold and would drop its candidate without saying so.
    alpha : float
        Level in ``(0, 1)``. The set targets ``1 - alpha`` coverage.

    Returns
    -------
    tuple of float
        ``(lower, upper)``, or ``(nan, nan)`` when no candidate survives.
        An empty set is a real answer: it says no effect on the grid conforms,
        and it is what a constant-effect null returns on a panel whose effect is
        not constant.

    Raises
    ------
    MlsynthDataError
        Empty grid, mismatched lengths, non-finite p-values, or ``alpha``
        outside ``(0, 1)``.
    """
    grid = np.asarray(grid, dtype=float).ravel()
    pvalues = np.asarray(pvalues, dtype=float).ravel()
    if grid.size == 0:
        raise MlsynthDataError(
            "inverting a conformal test needs at least one candidate effect; "
            "the grid is empty."
        )
    if grid.size != pvalues.size:
        raise MlsynthDataError(
            "the candidate grid and its p-values must have the same length; got "
            f"{grid.size} candidates and {pvalues.size} p-values."
        )
    if not np.isfinite(pvalues).all():
        raise MlsynthDataError(
            "every conformal p-value must be finite; a non-finite value compares "
            "False against the level and would drop its candidate silently."
        )
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise MlsynthDataError(f"alpha must lie strictly in (0, 1); got {alpha!r}.")

    # Strict: the test rejects at p <= alpha, so p == alpha is a rejection.
    kept = grid[pvalues > float(alpha)]
    if kept.size == 0:
        return float("nan"), float("nan")
    return float(kept.min()), float(kept.max())
