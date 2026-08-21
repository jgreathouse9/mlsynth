"""Block-resampled post-period error paths for a PDA cumulative band.

:func:`mlsynth.utils.pda_helpers.inference.cumulative_supt_band` takes an
``(B, H)`` matrix of post-period prediction-error paths and builds the band from
it, indifferent to where the rows came from. Today they come from
:func:`mlsynth.utils.inferutils.pda_prediction_intervals`, whose dependent wild
bootstrap refits the estimator once per replicate -- 999 times at the default.
This module is a second producer of that matrix, costing one refit per
rolling-origin calibration window instead of one per replicate.

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
        unambiguous reading, so neither is quietly reinterpreted.
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
        If ``series`` holds no finite value, since there is nothing to resample.
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
    e = e - e.mean()

    n_blocks = int(np.ceil(h / b))
    starts = rng.integers(0, e.size, size=(n_sim, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % e.size
    draws = e[idx]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_sim, n_blocks, 1))
    return (draws * signs).reshape(n_sim, n_blocks * b)[:, :h]


def rolling_origin_counterfactual_errors(
    y,
    refit_at,
    pre_periods: int,
    horizon: int,
    *,
    min_train_frac: float = 0.3,
):
    """Per-period out-of-sample errors from a PDA fit's own rolling-origin refits.

    :func:`mlsynth.utils.conformal.rolling_origin_period_errors` reads the error as
    ``y[block] - Y0[block] @ w``, which needs the fit to be a bare linear
    combination of the donors. A PDA fit is not: LASSO carries an intercept, the
    modified-BIC path standardizes the donors as ``glmnet`` does, and forward
    selection and HCW return a counterfactual with no single weight vector to
    multiply. So this asks the estimator for the counterfactual and subtracts.

    The schedule is :func:`mlsynth.utils.conformal.origin_schedule`, shared with the
    conformal readers, so a resampled band and a split-conformal band calibrate on
    the same windows. Origins step by a whole horizon, so the windows do not
    overlap and the concatenation is a contiguous stretch of out-of-sample periods.

    Parameters
    ----------
    y : array-like, shape ``(T,)``
        Treated-unit outcome over the full sample.
    refit_at : callable
        ``refit_at(origin) -> counterfactual``, the estimator refit on the periods
        strictly before ``origin`` and evaluated over the whole sample.
        ``orchestration._build_refit`` builds one per PDA variant: it closes over
        the pre-period length, so passing an origin in place of ``T0`` gives
        exactly this.
    pre_periods : int
        Number of pre-treatment periods ``T0``; origins are drawn from within it.
    horizon : int
        Window length ``L`` -- both the block length and the origin stride.
    min_train_frac : float, optional
        Earliest origin as a fraction of ``T0`` (default ``0.3``), floored at
        :data:`mlsynth.utils.conformal.MIN_TRAIN_PERIODS` absolute periods.

    Returns
    -------
    numpy.ndarray, shape ``(m * L,)``
        The per-period errors in time order, ready for :func:`block_error_paths`.
        Empty when the pre-period admits no origin.

    Raises
    ------
    MlsynthConfigError
        If ``horizon`` or ``pre_periods`` is not a positive integer, or
        ``min_train_frac`` is not a number in ``[0, 1]``.
    MlsynthDataError
        If ``pre_periods`` exceeds the sample, or a refit returns a counterfactual
        that does not span it or is not finite -- a failed refit is refused rather
        than scored, since its error would be arbitrary.
    """
    from ..conformal import origin_schedule

    h = _check_horizon(horizon)
    if isinstance(min_train_frac, bool) \
            or not isinstance(min_train_frac, (int, float, np.floating)) \
            or not 0.0 <= float(min_train_frac) <= 1.0:
        raise MlsynthConfigError(
            f"min_train_frac must be a number in [0, 1]; got {min_train_frac!r}."
        )
    if isinstance(pre_periods, bool) or not isinstance(pre_periods, (int, np.integer)) \
            or int(pre_periods) < 1:
        raise MlsynthConfigError(
            f"pre_periods must be an integer of at least 1; got {pre_periods!r}."
        )

    y = np.asarray(y, dtype=float).ravel()
    if int(pre_periods) > y.size:
        raise MlsynthDataError(
            f"pre_periods ({int(pre_periods)}) exceeds the {y.size} periods "
            "supplied; there is no pre-period to calibrate on."
        )

    blocks = []
    for origin in origin_schedule(int(pre_periods), h, float(min_train_frac)):
        cf = np.asarray(refit_at(origin), dtype=float).ravel()
        if cf.size != y.size:
            raise MlsynthDataError(
                f"refit_at({origin}) returned a counterfactual of {cf.size} "
                f"periods; it must span all {y.size}, since the window scored "
                "lies after the periods it was trained on."
            )
        if not np.isfinite(cf).all():
            raise MlsynthDataError(
                f"refit_at({origin}) returned a counterfactual that is not "
                "finite; a refit that failed is refused rather than scored."
            )
        blocks.append((y - cf)[origin:origin + h])

    if not blocks:
        return np.asarray([], dtype=float)
    return np.concatenate(blocks)
