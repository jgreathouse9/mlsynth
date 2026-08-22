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
    n_windows: int = 0,
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
    n_windows : int, optional
        Number of equal-length calibration windows the series is made of. ``0``,
        the default, treats it as one flat series. Any value of at least two
        splits the draw in two: each path takes one window's own level, and blocks
        of the within-window residuals on top of it.

        A horizon total is ``H`` times the mean error over that horizon, so it is
        driven by the level and not by fluctuation about it. Drawn flat, a block of
        length ``L`` from ``m`` concatenated windows usually straddles a boundary
        and averages two levels, so the very component that dominates the total
        enters at a fraction of its weight. Splitting the draw restores it.

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
        if the series holds a single value, which carries no spread, or if the
        resolved block is not shorter than the series, which would make every
        drawn path sum to the same value and collapse the band to zero width.
        Also if ``n_windows`` is one, since a single window's level deviation is
        zero by construction and the variance the total needs is unidentified, or
        if it exceeds the number of calibration periods.
    """
    h = _check_horizon(horizon)
    b = resolve_block(block, h)
    if isinstance(n_windows, bool) or not isinstance(n_windows, (int, np.integer)) \
            or int(n_windows) < 0:
        raise MlsynthConfigError(
            f"n_windows must be a non-negative integer (0 means one flat series); "
            f"got {n_windows!r}."
        )
    n_windows = int(n_windows)
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
    # The series is centred, so its circular sums carry an exact constraint: the
    # sum over a b-block is minus the sum over the complementary (n - b)-block,
    # since the two partition the series. Their spreads are therefore equal, and
    # the drawn spread is symmetric about b = n / 2. Past that midpoint the block
    # length stops meaning "how much serial correlation is carried" and reverses --
    # a longer block draws a NARROWER total, mirroring a shorter one -- until at
    # b = n every block is a rotation of the whole series, every path sums to zero,
    # and the band has no width at all. A parameter that inverts its own meaning is
    # not delivered silently, so the draw refuses past the midpoint.
    if e.size == 1:
        raise MlsynthDataError(
            "a single calibration error carries no spread: centring it leaves "
            "exactly zero, so every drawn path would sum to zero and the band "
            "would have no width. Calibrate on a series of at least two periods."
        )
    if 2 * b > e.size:
        raise MlsynthDataError(
            f"block length {b} is more than half the {e.size}-period calibration "
            f"series. The series is centred, so the circular sum of a {b}-block "
            f"is minus the sum of the complementary {e.size - b}-block: the two "
            f"have exactly equal spread, and past b = n / 2 a longer block draws "
            f"a narrower total, until at b = n every path sums to zero and the "
            f"band has no width. Use block <= {e.size // 2} (block=1 draws periods "
            f"independently) or calibrate on a longer series."
        )
    e = e - e.mean()

    if n_windows == 1:
        raise MlsynthDataError(
            "n_windows=1 cannot identify a level: a single window's deviation from "
            "its own mean is zero by construction, so the between-window variance a "
            "horizon total is made of is unidentified. Calibrate on at least two "
            "windows, or pass n_windows=0 to draw the series flat and accept that "
            "the total carries fluctuation only."
        )
    if n_windows > e.size:
        raise MlsynthDataError(
            f"n_windows={n_windows} exceeds the {e.size} calibration periods, so the "
            f"windows would be empty."
        )
    if n_windows >= 2 and e.size < 2 * h:
        raise MlsynthDataError(
            f"a level needs two windows and a window is the horizon, so splitting "
            f"asks for at least {2 * h} calibration periods; there are {e.size}. "
            f"Calibrate on a longer series, shorten the horizon, or pass "
            f"n_windows=0 to draw the series flat and accept that the total carries "
            f"fluctuation only."
        )
    if n_windows >= 2:
        mu, e, width = _level_split(e, n_windows, h)
        # one level per path, sign-flipped like everything else, held across the
        # whole horizon; the fluctuation is drawn from the within-window residuals
        # below and added on top.
        lvl = rng.choice(mu, size=(n_sim, 1))
        lvl = lvl * rng.choice(np.array([-1.0, 1.0]), size=(n_sim, 1))
        # Each window's residuals now sum to zero, so the zero-sum constraint binds
        # at the window length and not at the series length: a block as long as a
        # window is degenerate exactly as a block as long as the whole series was
        # before. Cap at half a window, and correct the spread against the window.
        b = min(b, max(1, width // 2))
        span = width
    else:
        lvl = 0.0
        span = e.size

    n_blocks = int(np.ceil(h / b))
    starts = rng.integers(0, e.size, size=(n_sim, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % e.size
    draws = e[idx]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_sim, n_blocks, 1))
    # Centring costs the block sums some of their spread, and the loss is exact
    # enough to undo. Because the centred series sums to zero, its circular
    # autocovariances sum to zero too, so for a b-block
    #     Var(S_b) = sum_{|k|<b} (b - |k|) chat(k) = b v (1 - (b-1)/(n-1)),
    # and the drawn totals come out narrow by sqrt((n-b)/(n-1)) -- 16 percent at a
    # 13-period block drawn from 47 calibration periods. Scaling by the inverse
    # restores it. The factor is exactly 1 at b = 1, so the single-period draw --
    # Wheeler's construction, and the parity pinned on it -- is unchanged.
    #
    # It is exact for a horizon made of whole blocks, which the default
    # block = horizon always is. A horizon shorter than one block is scaled by
    # this factor too and comes out slightly wide, since a partial block of length
    # h < b loses sqrt((n-h)/(n-1)) and not sqrt((n-b)/(n-1)). Erring wide is the
    # safe direction for a band, and the alternative -- a per-horizon factor --
    # cannot be applied to a path without changing its shape.
    scale = np.sqrt((span - 1) / (span - b)) if b > 1 else 1.0
    fluct = (draws * signs).reshape(n_sim, n_blocks * b)[:, :h] * scale
    return fluct + lvl


def _level_split(e, n_windows, horizon):
    """Window levels and within-window residuals, from a centred series.

    The window is the horizon. A total over ``H`` periods is ``H`` times the mean
    error over ``H`` periods, so the level the band needs is a level over exactly
    that span; deriving the width from the series length instead would measure a
    level over some other span whenever the series does not divide evenly, and
    would rescale the block cap and the zero-sum correction against that other span
    too.

    Where the series holds more periods than ``m`` windows need, the ones kept are
    the most recent, since they sit closest to the horizon being predicted. For a
    caller whose series is exactly ``m`` horizons long -- a rolling-origin pass laid
    end to end -- nothing is dropped and the choice does not arise.

    The series is already centred, so the window means are deviations summing to
    zero and their spread is what a horizon total is made of.
    """
    width = int(horizon)
    m = min(int(n_windows), e.size // width)
    used = m * width
    panel = e[e.size - used:].reshape(m, width)
    mu = panel.mean(axis=1)
    mu = mu - mu.mean()
    resid = (panel - panel.mean(axis=1, keepdims=True)).ravel()

    # The window means scatter even when every window shares a level, since each is
    # an average of `width` noisy periods. Their variance estimates
    #     sigma_level^2 + sigma_within^2 / width,
    # so drawing from them whole charges the band for a level that may not be there.
    # Subtract the noise floor -- the ANOVA estimator for a one-way random effect --
    # and floor it at zero, since sampling can put s2_between below the floor.
    # A strongly levelled series is left almost untouched; a flat one collapses to
    # nothing and the draw returns to what it would have been without the split.
    v0 = float(np.mean(mu ** 2))                       # variance of drawing from mu
    if m > 1 and v0 > 0.0:
        s2_between = v0 * m / (m - 1.0)                # unbiased, means sum to zero
        s2_within = float(np.sum(resid ** 2) / max(m * (width - 1), 1))
        s2_level = max(0.0, s2_between - s2_within / width)
        mu = mu * np.sqrt(s2_level / v0)
    else:
        mu = np.zeros_like(mu)
    return mu, resid, width


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
    from .scores import origin_schedule, rolling_origin_counterfactual_errors

    series = rolling_origin_counterfactual_errors(
        y, refit_at, pre_periods, horizon, min_train_frac=min_train_frac)
    # As in the weight-vector composer: one horizon-length window per origin, so
    # the draw is told the structure and takes the level apart from the fluctuation.
    m = len(list(origin_schedule(pre_periods, horizon, min_train_frac)))
    return block_error_paths(series, horizon=horizon, block=block, n_sim=n_sim,
                             rng=np.random.default_rng(seed),
                             n_windows=m if m >= 2 else 0)


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
    from .scores import origin_schedule, rolling_origin_period_errors

    series = rolling_origin_period_errors(
        y, Y0, pre_periods, horizon, weight_fn, min_train_frac=min_train_frac)
    # The series is one horizon-length window per rolling origin, laid end to end.
    # Telling the draw so lets it take each window's level separately from the
    # fluctuation about it, which is what a horizon total is made of.
    m = len(list(origin_schedule(pre_periods, horizon, min_train_frac)))
    return block_error_paths(series, horizon=horizon, block=block, n_sim=n_sim,
                             rng=np.random.default_rng(seed),
                             n_windows=m if m >= 2 else 0)
