"""Turning a calibration error series into accumulable post-period paths.

:mod:`.scores` produces out-of-sample errors; :mod:`.cumulative` reduces them to a
split-conformal order statistic. This module takes the third route: it resamples
the errors into whole paths, which a caller accumulates and reads a band off. That
is what a band for a *running total* needs, because the total's spread depends on
how the period errors move together and a per-period order statistic has already
thrown that away.

The output is an ``(n_sim, H)`` matrix, one post-period error path per row, which
is the shape a simultaneous band already consumes:
:func:`mlsynth.utils.pda_helpers.inference.cumulative_supt_band` and PPSCM's
counterpart both accumulate such a matrix and take a standard error afterwards,
indifferent to where the rows came from. A bootstrap produces them at one refit
per replicate; this produces them at one refit per calibration origin, which on the
panel lengths these estimators see is two orders of magnitude fewer.

The construction is the one Andrew Wheeler uses in LassoSynth. He forms leave-one-out
conformity scores, mirrors them into a symmetric pool (``np.concatenate([cs, -cs])``),
draws one pool element per post period independently, and accumulates. Drawing
periods independently assumes the period errors are uncorrelated, which a panel's
rarely are: the variance of an ``H``-period total is ``H * gamma_0 + 2 * sum_k
(H - k) * gamma_k``, so positive autocorrelation makes the total more variable than
``H`` independent draws imply and an independent draw too narrow by a factor that
grows with the horizon.

So blocks are drawn instead of periods. Each path is assembled from
``ceil(H / block)`` circular blocks of the centred series, truncated to ``H``, and
each block's sign is flipped with probability one half. The sign flip is Wheeler's
symmetrisation applied to a block: the multiset ``{+e_i, -e_i}`` is the multiset
``{+|e_i|, -|e_i|}``, so at ``block = 1`` the two constructions coincide in
distribution, which ``tests/test_pda_resample_band.py`` pins against a transcription
of his code. Flipping periods individually would destroy the very correlation the
blocks were drawn to keep.

The series is centred first. A fit sitting a little high shifts every one of its
errors by the same amount, and an uncentred draw would charge the band for that
shift once per period it accumulates.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError

#: Blocks shorter than this cannot carry the autocorrelation they are drawn for,
#: but the choice is the caller's; this is only the default stand-in for "the
#: whole horizon", requested as ``block = 0``.
WHOLE_HORIZON = 0


def _check_horizon(horizon) -> int:
    """Return ``horizon`` as a positive int, or raise.

    Parameters
    ----------
    horizon : int
        Number of post-treatment periods the paths span.

    Returns
    -------
    int

    Raises
    ------
    MlsynthConfigError
        If ``horizon`` is not an integer, or is not at least one.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise MlsynthConfigError(
            f"horizon must be an integer of at least 1; got {horizon!r}."
        )
    if int(horizon) < 1:
        raise MlsynthConfigError(
            f"horizon must be at least 1; got {int(horizon)}."
        )
    return int(horizon)


def resolve_block(block: Union[int, np.integer], horizon: int) -> int:
    """Turn a configured block length into the one the draw will use.

    ``0`` asks for the whole horizon, which is the longest block that carries
    correlation the accumulated total is sensitive to. A block longer than the
    horizon is clamped to it, since the surplus periods are truncated away and a
    longer block would only reduce the number of distinct starting positions.

    Parameters
    ----------
    block : int
        Requested block length in periods. ``0`` means the horizon.
    horizon : int
        Number of post-treatment periods.

    Returns
    -------
    int
        A block length in ``[1, horizon]``.

    Raises
    ------
    MlsynthConfigError
        If ``block`` is not an integer, or is negative. Booleans and floats are
        refused outright: ``True`` is an ``int`` in Python and ``2.5`` has no
        unambiguous reading, so neither is reinterpreted.
    """
    h = _check_horizon(horizon)
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise MlsynthConfigError(
            f"block must be a non-negative integer (0 means the horizon); "
            f"got {block!r}."
        )
    b = int(block)
    if b < 0:
        raise MlsynthConfigError(
            f"block must be non-negative (0 means the horizon); got {b}."
        )
    if b == WHOLE_HORIZON:
        return h
    return min(b, h)


def block_error_paths(
    series,
    *,
    horizon: int,
    block: int = WHOLE_HORIZON,
    n_sim: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """``(n_sim, horizon)`` post-period error paths, drawn as sign-flipped blocks.

    Parameters
    ----------
    series : array-like
        Out-of-sample per-period errors, in time order. Non-finite entries are
        dropped, so an origin whose refit failed contributes nothing instead of a
        zero that would shrink the band.
    horizon : int
        Number of post-treatment periods each path spans.
    block : int, optional
        Block length in periods; ``0`` (default) means the horizon. See
        :func:`resolve_block`.
    n_sim : int, optional
        Number of paths to draw (default 2000).
    rng : numpy.random.Generator, optional
        Source of randomness. A fresh default generator is used when omitted, so
        a caller that wants reproducibility passes its own.

    Returns
    -------
    numpy.ndarray, shape ``(n_sim, horizon)``
        One error path per row, ready for
        :func:`mlsynth.utils.pda_helpers.inference.cumulative_supt_band` or for
        equal-tailed quantiles of the accumulated draws.

    Raises
    ------
    MlsynthConfigError
        If ``horizon``, ``block`` or ``n_sim`` is not a positive integer of the
        right kind.
    MlsynthDataError
        If ``series`` holds no finite value, since there is nothing to resample,
        or if the resolved block is not shorter than the series, which would make
        every drawn path sum to the same value and collapse the band to zero
        width.
    """
    h = _check_horizon(horizon)
    b = resolve_block(block, h)
    if isinstance(n_sim, bool) or not isinstance(n_sim, (int, np.integer)) \
            or int(n_sim) < 1:
        raise MlsynthConfigError(
            f"n_sim must be an integer of at least 1; got {n_sim!r}."
        )
    n_sim = int(n_sim)
    if rng is None:
        rng = np.random.default_rng()

    e = np.asarray(series, dtype=float).ravel()
    e = e[np.isfinite(e)]
    if e.size == 0:
        raise MlsynthDataError(
            "cannot resample from an empty error series: no finite out-of-sample "
            "error was supplied. A calibration pass that produced no usable "
            "window cannot support a band."
        )
    # Blocks are circular, so a block of length b >= n wraps through whole cycles
    # of the n-period series. A whole cycle of a centred series sums to exactly
    # zero, so the effective block is b % n and not the b that was asked for. At
    # b % n == 0 -- which b == n always is -- every block is a rotation of the
    # whole series, every path sums to zero, and the quantile of those totals is
    # zero: a band asserting perfect certainty about the accumulated effect, which
    # is a worse answer than the infinite half-width this construction was reached
    # for. Neither outcome is delivered under the requested block length.
    if b >= e.size:
        raise MlsynthDataError(
            f"block length {b} is not shorter than the {e.size}-period "
            f"calibration series. Blocks are circular, so such a block wraps "
            f"through whole cycles of the series; a whole cycle of the centred "
            f"series sums to zero, making the effective block {b % e.size} and not "
            f"{b}, and at {b} % {e.size} == {b % e.size} every drawn path sums to "
            f"the same value, so the band would have zero width. Use a shorter "
            f"block (block <= {e.size - 1}, or block=1 to draw periods "
            f"independently) or calibrate on a longer series."
        )
    e = e - e.mean()

    n_blocks = int(np.ceil(h / b))
    starts = rng.integers(0, e.size, size=(n_sim, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % e.size
    draws = e[idx]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_sim, n_blocks, 1))
    return (draws * signs).reshape(n_sim, n_blocks * b)[:, :h]


def resample_cumulative_paths(
    y,
    refit_at,
    pre_periods: int,
    horizon: int,
    *,
    block: int = WHOLE_HORIZON,
    n_sim: int = 2000,
    seed=0,
    min_train_frac: float = 0.3,
) -> np.ndarray:
    """From a fitted control to resampled cumulative error paths, in one call.

    The composition every estimator wanting a resampled band performs:
    :func:`~mlsynth.utils.conformal.rolling_origin_counterfactual_errors` for the
    calibration series, then :func:`block_error_paths` for the draw. Keeping it in
    one place means an estimator adopting the band inherits the construction that
    was cross-validated against Wheeler's LassoSynth, and a change to either half
    cannot reach only some callers.

    Parameters
    ----------
    y : array-like, shape ``(T,)``
        Treated-unit outcome over the full sample.
    refit_at : callable
        ``refit_at(origin) -> counterfactual``, the estimator refit on the periods
        strictly before ``origin`` and evaluated over the whole sample.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    horizon : int
        Post-period length ``L``; also the calibration window length and the
        origin stride.
    block : int, optional
        Block length for the draw; ``0`` (default) means the horizon. See
        :func:`resolve_block`.
    n_sim : int, optional
        Number of paths to draw (default 2000). These cost no refits.
    seed : optional
        Seed for the draw, so a band is reproducible. Passed to
        :func:`numpy.random.default_rng`.
    min_train_frac : float, optional
        Earliest calibration origin as a fraction of ``T0`` (default ``0.3``).

    Returns
    -------
    numpy.ndarray, shape ``(n_sim, horizon)``

    Raises
    ------
    MlsynthConfigError
        For a malformed ``horizon``, ``block``, ``n_sim`` or ``min_train_frac``.
    MlsynthDataError
        If the pre-period admits no calibration window, or a refit fails. Either
        way there is nothing to resample, and that is reported instead of an empty
        band being returned.
    """
    from .scores import rolling_origin_counterfactual_errors

    series = rolling_origin_counterfactual_errors(
        y, refit_at, pre_periods, horizon, min_train_frac=min_train_frac)
    return block_error_paths(series, horizon=horizon, block=block, n_sim=n_sim,
                             rng=np.random.default_rng(seed))


def resample_cumulative_paths_from_weights(
    y,
    Y0,
    pre_periods: int,
    horizon: int,
    weight_fn,
    *,
    block: int = WHOLE_HORIZON,
    n_sim: int = 2000,
    seed=0,
    min_train_frac: float = 0.3,
) -> np.ndarray:
    """The same composition for a fit that returns donor weights.

    :func:`resample_cumulative_paths` serves an estimator that hands back a
    counterfactual; this one serves the classical case, where the fit is a weight
    vector over donors and the error is ``y - Y0 @ w``. The two agree wherever
    both apply, which
    ``tests/test_conformal_resample_paths_weights.py`` asserts.

    Where a split-conformal band needs ``ceil(1/alpha) - 1`` calibration windows
    before its order statistic exists, this draws from the ``m * L`` per-period
    errors inside those windows, so it stays finite on pre-periods that leave a
    split band infinite.

    Parameters
    ----------
    y : array-like, shape ``(T,)``
        Treated-unit outcome over the full sample.
    Y0 : array-like, shape ``(T, J)``
        Donor outcomes aligned to ``y``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    horizon : int
        Post-period length ``L``; also the calibration window length and stride.
    weight_fn : callable
        ``weight_fn(keep_idx) -> w``, the estimator's refit on the periods indexed
        by ``keep_idx``, as :func:`rolling_origin_block_sums` takes it.
    block, n_sim, seed, min_train_frac
        As :func:`resample_cumulative_paths`.

    Returns
    -------
    numpy.ndarray, shape ``(n_sim, horizon)``

    Raises
    ------
    MlsynthConfigError
        For a malformed ``horizon``, ``block``, ``n_sim`` or ``min_train_frac``.
    MlsynthDataError
        If ``Y0`` does not align with ``y``, the pre-period admits no window, or
        the calibration pass yields nothing to resample.
    """
    from .scores import rolling_origin_period_errors

    series = rolling_origin_period_errors(
        y, Y0, pre_periods, horizon, weight_fn, min_train_frac=min_train_frac)
    return block_error_paths(series, horizon=horizon, block=block, n_sim=n_sim,
                             rng=np.random.default_rng(seed))
