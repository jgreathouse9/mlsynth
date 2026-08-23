"""The moving-block permutation p-value, in one place.

A split-conformal band carves the pre-period into disjoint calibration windows
and asks where the post-treatment window falls among them. That is simple, and it
is wasteful when the pre-period is short relative to the horizon: an eight-period
horizon over a hundred pre-periods yields around ten windows, and a p-value built
from ten reference values takes only the multiples of one tenth. A band inverted
at ``alpha = 0.1`` from such a p-value has no resolution left at the level it is
trying to hit.

Chernozhukov, Wuthrich and Zhu calibrate against the ``T`` cyclic shifts of the
residual path instead. Every position in the series supplies a reference block,
so the same panel gives a hundred-odd blocks rather than ten, and the p-value
takes the multiples of ``1 / T``. The blocks overlap and are therefore dependent,
which is the price; the permutation argument runs on exchangeability of the
residual sequence under cyclic rotation rather than on independence between
blocks.

The statistic is a parameter because the estimand chooses it. For a per-period
band the natural score is mean absolute residual, which is what the existing call
sites use and what stays the default here. For a band on a running total it is
the absolute block sum: a block that is uniformly positive and one that
oscillates around zero share a mean absolute residual and have entirely different
totals, so the per-period statistic cannot see the thing a cumulative band is
about.

The precondition the two statistics do not share
------------------------------------------------

The permutation argument needs the trailing block to be exchangeable with its own
rotations *in whatever the statistic measures*, and a fitted synthetic-control
residual satisfies that for magnitude but not for sign.

Measured on a fitted PPSCM residual path, the per-period magnitude profile is
flat -- 0.99 to 1.04 of the path mean at every position -- while the standardized
absolute block sum ramps from 0.733 in the interior to 0.956 near the end of the
balanced window and 1.114 at the final position. The quadratic program leaves
end-of-window residuals positively correlated, so their signs stop cancelling and
their sums grow, without their magnitudes growing at all. Wrapping blocks, which
splice the end of the path to its start, show the same inflation (0.874 against
0.757 for contiguous ones) because they join two edge regions.

The trailing block always occupies the most inflated position, so a sum statistic
reads that coherence as evidence: on a null panel, ``abs_sum`` rejects at 0.225
against a nominal 0.10, while ``mean_abs`` on the identical paths sits at 0.092.
Placing the block at a random position instead restores calibration (0.108), which
is what identifies the position -- not the path, and not the statistic -- as the
fault.

So ``abs_sum`` is valid on an exchangeable path (0.091 to 0.096 on iid and AR
draws, constrained and unconstrained) and invalid on a fitted residual path
without first removing the position effect. A cumulative band cannot be obtained
merely by swapping the statistic underneath this function. ``test_conformal_
moving_block.py`` pins both halves of that.

This function is the single definition. The statistic is currently written out
again in :mod:`mlsynth.utils.cscipca_helpers.inference`,
:mod:`mlsynth.utils.bilevel.ridge_inference`,
:mod:`mlsynth.utils.syndes_helpers.inference`,
:mod:`mlsynth.utils.shc_helpers.inference` and
:mod:`mlsynth.utils.rrsc_helpers.inference`, two of them rolling in the opposite
direction. Those five enumerate the same block set today, so they agree; they
agree by coincidence rather than by construction, which is the same situation
:func:`~mlsynth.utils.conformal.inversion.confidence_set_bounds` was centralised
to end. Migrating them is deliberately not bundled here -- each is pinned by its
own golden benchmark and moves on its own PR.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError

#: Statistics a block may be scored by. ``mean_abs`` is the per-period score of
#: the reference implementation; ``abs_sum`` is its counterpart for a running
#: total, which needs the signed sum before the absolute value is taken.
BLOCK_STATISTICS = ("mean_abs", "abs_sum")


def _score(window: np.ndarray, statistic: str) -> float:
    if statistic == "mean_abs":
        return float(np.mean(np.abs(window)))
    return abs(float(np.sum(window)))


def moving_block_pvalue(residuals, *, block: int,
                        statistic: str = "mean_abs") -> float:
    r"""Permutation p-value of the trailing block against every cyclic shift.

    The post-treatment block is the last ``block`` entries of ``residuals``. The
    ``T`` cyclic rotations of the path each supply a block of the same length,
    every one is scored, and the p-value is the share scoring at least as extreme
    as the post block:

    .. math::

        p = \frac{1}{T} \sum_{s=0}^{T-1}
            \mathbb{1}\{ S(\text{roll}(u, s)) \ge S(u) \}

    The post block is itself one of the rotations, so ``p`` is never zero: the
    smallest attainable value is ``1 / T``, which is the most evidence a
    permutation test over ``T`` reference values can produce.

    Parameters
    ----------
    residuals : array-like, shape (T,)
        The residual path, pre-period first, with the block being tested last.
        Must be finite throughout -- dropping an entry would change ``T`` and so
        change the reference set the guarantee is stated over.
    block : int
        Length of the trailing block, in ``[1, T]``. Keyword-only.
    statistic : {"mean_abs", "abs_sum"}, default "mean_abs"
        How a block is scored. ``mean_abs`` matches the reference implementation
        and suits a per-period band; ``abs_sum`` suits a band on a running total.
        Keyword-only.

    Returns
    -------
    float
        A p-value in ``[1 / T, 1]``, always a multiple of ``1 / T``.

    Raises
    ------
    MlsynthConfigError
        If ``block`` is not an integer in ``[1, T]``, or ``statistic`` is not one
        of :data:`BLOCK_STATISTICS`.
    MlsynthDataError
        If ``residuals`` is empty or holds a non-finite entry.
    """
    if statistic not in BLOCK_STATISTICS:
        raise MlsynthConfigError(
            f"statistic must be one of {BLOCK_STATISTICS}; got {statistic!r}.")
    u = np.asarray(residuals, dtype=float).ravel()
    if u.size == 0:
        raise MlsynthDataError("residuals is empty; there is nothing to permute.")
    if not np.isfinite(u).all():
        raise MlsynthDataError(
            "residuals must be finite. A non-finite entry cannot be scored, and "
            "dropping it would shrink the cyclic reference set the p-value's "
            "denominator is defined over.")
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise MlsynthConfigError(
            f"block must be an integer; got {block!r}. A fractional block would "
            "be truncated, silently scoring a shorter window than requested.")
    block = int(block)
    if not 1 <= block <= u.size:
        raise MlsynthConfigError(
            f"block must lie in [1, {u.size}]; got {block}.")

    observed = _score(u[-block:], statistic)
    stats = np.array([_score(np.roll(u, s)[-block:], statistic)
                      for s in range(u.size)], dtype=float)
    return float(np.mean(stats >= observed))
