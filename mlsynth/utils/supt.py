"""Simultaneous (sup-t) confidence bands from resampling draws.

A pointwise band covers each horizon with probability ``1 - alpha`` one horizon
at a time. Nobody reads an event study one horizon at a time. Read across the
path -- "the effect is positive from week three onward", "it never leaves the
band" -- the pointwise band covers the whole path with much less than
``1 - alpha``, and the shortfall grows with the number of horizons. With nine
independent horizons a nominal 95% pointwise band covers the path about 63% of
the time.

A simultaneous band restores the level for the path as a whole. Every interval
is inflated by one shared critical value ``c``, chosen so that the *largest*
standardized deviation across horizons stays inside with probability
``1 - alpha``:

    P( max_h |theta_h - theta_h| / se_h  <=  c )  =  1 - alpha

Following Montiel Olea and Plagborg-Moller (2019), ``c`` comes from the
correlation across horizons: standardize the draws, estimate their correlation
matrix, draw ``z ~ N(0, R)``, and take the ``1 - alpha`` quantile of
``max_h |z_h|``. Simulating instead of reading the quantile off the draws
themselves is what makes this usable with a delete-one jackknife -- there are
only as many replicates as units, so an empirical quantile at 0.95 is coarse,
and jackknife deviations sit on a different scale from the estimator's own
sampling distribution. Only the correlation is taken from the draws; the scale
comes from ``se_h`` separately.

That last sentence is also the limit of the method, and worth knowing before
choosing it: taking only the correlation makes ``c`` a functional of ``R``
alone. Two ensembles with the same correlation matrix get the same multiplier
however different their joint laws, and the quantile being estimated is not such
a functional. Where the draws are close to normal the two agree to a few
thousandths. Where they are not -- a block bootstrap over a short calibration
series draws from ``m`` distinct blocks, so a large ``n_sim`` resamples a small
support instead of exploring a continuum -- they part company, by an amount that
runs either side of parity and shrinks as the support grows.
``supt_critical_value(..., method="empirical")`` reads the quantile off the
draws for callers whose ensemble makes that the better instrument.

The cumulative case is why this module exists at all. An interval for a running
total is not the running total of the period intervals: adding endpoints treats
the period errors as moving in lockstep, so the width grows like ``L``, while
independent errors grow it like ``sqrt(L)``. Neither assumption measures
anything. :func:`cumulative_from_paths` accumulates the draws themselves, so
whatever correlation the errors actually have is carried into the standard
error, and the band inherits it.

References
----------
Montiel Olea, J. L., & Plagborg-Moller, M. (2019). Simultaneous confidence bands:
Theory, implementation, and an application to SVARs. Journal of Applied
Econometrics, 34(1), 1-17.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..exceptions import MlsynthConfigError, MlsynthDataError

#: Draws used to tabulate the ``max_h |z_h|`` quantile. Large enough that the
#: critical value is stable to about three decimals, which is far finer than the
#: correlation matrix it is conditioned on is known.
DEFAULT_N_SIMS = 200_000


def _check_alpha(alpha) -> float:
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating))
            or not 0.0 < float(alpha) < 1.0):
        raise MlsynthConfigError(
            f"alpha must be a number in the open interval (0, 1); got {alpha!r}."
        )
    return float(alpha)


def cumulative_from_paths(paths: np.ndarray) -> np.ndarray:
    """Accumulate each draw's period path into its running total.

    Parameters
    ----------
    paths : np.ndarray, shape (n_draws, H)
        One per-period path per draw.

    Returns
    -------
    np.ndarray, shape (n_draws, H)
        ``out[i, L]`` is the total over horizons ``0 .. L`` of draw ``i``, so a
        standard error taken down its columns carries whatever correlation the
        period errors have.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise MlsynthDataError(
            f"paths must be 2-D (n_draws, H); got shape {paths.shape}."
        )
    return np.cumsum(paths, axis=1)


def jackknife_se(draws: np.ndarray, *, jackknife: bool = True) -> np.ndarray:
    """Column-wise standard error of a set of replicates.

    Parameters
    ----------
    draws : np.ndarray, shape (n_draws, H)
        Replicates, one row each. ``NaN`` marks a replicate that could not be
        computed -- a leave-one-out fit that failed to converge, say -- and is
        omitted from that column rather than counted as zero.
    jackknife : bool, default True
        Apply the delete-one inflation ``(m-1)/m * sum (x - xbar)^2``. Jackknife
        replicates differ from the full-sample estimate by ``O(1/m)``, so they
        need it to estimate a sampling standard error. Set ``False`` for draws
        that are already on the estimator's own scale (a bootstrap, or a direct
        simulation), where the ordinary sample standard deviation is wanted.

    Returns
    -------
    np.ndarray, shape (H,)
        The standard error per column; ``NaN`` where fewer than two replicates
        survived, since one point has no spread.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise MlsynthDataError(
            f"draws must be 2-D (n_draws, H); got shape {draws.shape}."
        )
    out = np.full(draws.shape[1], np.nan)
    for h in range(draws.shape[1]):
        x = draws[:, h]
        x = x[~np.isnan(x)]
        m = x.size
        if m < 2:
            continue
        ss = float(np.sum((x - x.mean()) ** 2))
        out[h] = np.sqrt((m - 1) / m * ss) if jackknife else np.sqrt(ss / (m - 1))
    return out


def supt_critical_value(
    draws: np.ndarray,
    *,
    alpha: float,
    n_sims: int = DEFAULT_N_SIMS,
    seed: Optional[int] = 0,
    method: str = "gaussian",
) -> float:
    """The shared multiplier that makes a band cover the whole path at once.

    Parameters
    ----------
    draws : np.ndarray, shape (n_draws, H)
        Replicates of the quantity being banded, one row each. Only their
        *correlation* across horizons is used, so the draws need not be on the
        estimator's scale and rescaling any horizon leaves the answer unchanged.
        ``NaN`` entries are dropped pairwise.
    alpha : float
        Level; the band is simultaneous at ``1 - alpha``.
    n_sims : int
        Draws used to tabulate the quantile of ``max_h |z_h|``. Ignored by
        ``method="empirical"``, which consults no RNG.
    seed : int, optional
        RNG seed, so a reported band is reproducible. Ignored by
        ``method="empirical"``.
    method : {"gaussian", "empirical"}
        Which estimator of the same quantile to use.

        ``"gaussian"`` (default) reduces the draws to their correlation across
        horizons and tabulates ``max_h |z_h|`` from simulated normals carrying
        that correlation. This is the right instrument for a delete-one
        jackknife: there are only as many draws as units, so an empirical
        quantile at ``1 - alpha`` is coarse, and jackknife deviations are on a
        different scale from the estimator's sampling distribution anyway.

        ``"empirical"`` reads the quantile off the standardized draws
        themselves. Reach for it when the draws are a large resampling ensemble
        -- a block bootstrap, say -- where neither objection applies: the
        quantile is not coarse, and the draws are already on the estimator's
        scale. It also keeps what the correlation matrix cannot carry. Reducing
        the draws to second moments discards *scale heterogeneity across draws*
        -- a bootstrap replicate that catches an unusual stretch is large at
        every horizon at once -- and that is precisely what drives ``max_h``. A
        Gaussian vector's own scale is tightly concentrated, so no correlation
        matrix can reproduce it, and the simulated multiplier comes back too
        small. The gap is not a constant: it runs either side of parity, ordered
        by how dispersed the per-draw scale is.

    Returns
    -------
    float
        ``c``, to be applied as ``theta_h +/- c * se_h``. Always at least the
        pointwise ``z_{1 - alpha/2}``, and equal to it when there is one horizon
        or when the horizons are perfectly correlated.

    Raises
    ------
    MlsynthConfigError
        ``alpha`` outside ``(0, 1)``, a non-positive ``n_sims``, or a ``method``
        that is neither ``"gaussian"`` nor ``"empirical"``.
    MlsynthDataError
        ``draws`` not 2-D, or fewer than two replicates.
    """
    alpha = _check_alpha(alpha)
    if method not in ("gaussian", "empirical"):
        raise MlsynthConfigError(
            f"method must be 'gaussian' or 'empirical'; got {method!r}."
        )
    if isinstance(n_sims, bool) or not isinstance(n_sims, (int, np.integer)) or n_sims < 1:
        raise MlsynthConfigError(f"n_sims must be a positive integer; got {n_sims!r}.")
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise MlsynthDataError(
            f"draws must be 2-D (n_draws, H); got shape {draws.shape}."
        )
    n, H = draws.shape
    if n < 2:
        raise MlsynthDataError(
            f"need at least 2 draws to estimate a correlation; got {n}."
        )

    if method == "empirical":
        # The same statistic the simulation targets, taken over the draws in
        # hand. Standardizing by each horizon's own spread is what makes it
        # scale-free, exactly as in the simulated route; a horizon with no
        # variation is given an infinite scale so it contributes zero rather
        # than a division by zero.
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(draws, axis=0)
            centred = draws - np.nanmean(draws, axis=0)
            z = centred / np.where(np.isfinite(sd) & (sd > 0.0), sd, np.inf)
        m = np.nanmax(np.abs(z), axis=1)
        m = m[np.isfinite(m)]
        if m.size < 2:
            raise MlsynthDataError(
                "no usable draws left after standardizing; every replicate was "
                "non-finite or every horizon was constant."
            )
        return float(np.quantile(m, 1.0 - alpha))

    # Correlation across horizons, pairwise-complete so one failed replicate does
    # not discard a whole horizon. A horizon with no variation carries no
    # information about co-movement; it is given a unit diagonal and zero
    # off-diagonals, which is the neutral choice and keeps the matrix usable.
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.eye(H)
        sd = np.nanstd(draws, axis=0)
        good = np.isfinite(sd) & (sd > 0.0)
        idx = np.flatnonzero(good)
        if idx.size > 1:
            sub = draws[:, idx]
            keep = ~np.isnan(sub).any(axis=1)
            if keep.sum() > 1:
                C = np.corrcoef(sub[keep], rowvar=False)
                C = np.nan_to_num(C, nan=0.0)
                R[np.ix_(idx, idx)] = C
    np.fill_diagonal(R, 1.0)

    # Nearest usable covariance: the estimated correlation can be indefinite when
    # there are fewer replicates than horizons, which a jackknife over a modest
    # panel reaches easily. Clipping the eigenvalues at zero is the standard
    # projection and leaves a matrix that is still a correlation on the diagonal.
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 0.0, None)
    L = V * np.sqrt(w)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(int(n_sims), H)) @ L.T
    scale = np.sqrt(np.clip(np.einsum("ij,ij->i", L, L), 1e-300, None))
    m = np.max(np.abs(z / scale), axis=1)
    return float(np.quantile(m, 1.0 - alpha))
