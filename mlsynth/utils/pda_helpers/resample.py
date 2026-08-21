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
