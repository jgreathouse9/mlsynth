"""The reference objective for many designs at once (#962).

:func:`~mlsynth.utils.tbr_helpers.design.objective.score_split` refits the TBR
regression and every gate from the two aggregates of one design. A search that
weighs a whole neighbourhood of designs pays that cost per design, though every
term it computes is a quadratic form in the membership indicators. With
``u_T`` and ``u_C`` the indicator rows of a design and ``M`` the panel centred
over the scoring window,

    S_yy = u_T' G u_T,   S_xy = u_T' G u_C,   S_xx = u_C' G u_C,   G = M'M,

and the correlation, the slope, the residual sum of squares and so the required
impact all follow from those three numbers. The Durbin-Watson statistic needs
the same three forms over the first-differenced series, the A/A refit the same
three over the fit window together with each geo's fit-window and holdout means,
and the Brownian-bridge gate the cumulative residual path, which is the
cumulative centred series contracted with the two rows. All of it is cached once
per window in :class:`DesignCache` and applied to a matrix of designs by
:func:`score_designs`.

The relative objective's key is the treated group's level over its required
impact. The impact is the reference's power term, from the same three forms, and
the level is the treated aggregate's mean over the window, ``u_T' m`` with ``m``
each geo's mean, so the screen computes it for the cost of one more product.

The Gram form loses digits where the direct fit does not
(``agents/agents_solvers.md``, rule 6): ``S_xx`` and the residual sum of squares
are differences of large numbers when the control aggregate barely moves or the
fit is nearly exact. A design whose quadratic forms are within their own
rounding of zero on either window is therefore scored by ``score_split`` itself,
and a search reading these keys confirms the move it takes with ``score_split``
before taking it. The keys are for screening; the direct fit decides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from ....exceptions import MlsynthConfigError, MlsynthDataError
from . import objective as obj


@dataclass(frozen=True)
class DesignCache:
    """Everything :func:`score_designs` needs from one scoring window."""

    y: np.ndarray          # (n, N) the window, kept for the direct fallback
    n_test: int
    G: np.ndarray          # Gram of the series centred over the window
    absG: np.ndarray       # |G|, the size of each form's terms
    Gd: np.ndarray         # Gram of the first differences
    G1: np.ndarray         # Gram of the series centred over the A/A fit window
    absG1: np.ndarray
    fit_mean: np.ndarray   # (N,) each geo's mean over the A/A fit window
    hold_mean: np.ndarray  # (N,) each geo's mean over the held-out window
    P: np.ndarray          # (n, N) cumulative sums of the centred series
    col_mean: np.ndarray   # (N,) each geo's mean over the whole window

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_fit(self) -> int:
        return self.n - self.n_test


def design_cache(y_matrix: np.ndarray, *, n_test: int) -> DesignCache:
    """The Grams and window sums of a scoring window of periods by geos."""
    y = np.asarray(y_matrix, dtype=float)
    if y.ndim != 2:
        raise MlsynthDataError(
            f"the scoring window has to be periods by geos; got shape {y.shape}.")
    n_fit = y.shape[0] - int(n_test)
    if n_fit < obj.MIN_HOLDOUT_PRETEST:
        raise MlsynthDataError(
            f"the A/A test fits on the window before the last {n_test} periods "
            f"and needs at least {obj.MIN_HOLDOUT_PRETEST} of them; a scoring "
            f"window of {y.shape[0]} leaves {n_fit}.")
    centred = y - y.mean(axis=0)
    diffs = np.diff(y, axis=0)
    fit = y[:n_fit]
    fit_centred = fit - fit.mean(axis=0)
    G = centred.T @ centred
    G1 = fit_centred.T @ fit_centred
    return DesignCache(
        y=y, n_test=int(n_test), G=G, absG=np.abs(centred).T @ np.abs(centred),
        Gd=diffs.T @ diffs, G1=G1,
        absG1=np.abs(fit_centred).T @ np.abs(fit_centred),
        fit_mean=fit.mean(axis=0), hold_mean=y[n_fit:].mean(axis=0),
        P=np.cumsum(centred, axis=0), col_mean=y.mean(axis=0))


def _forms(G: np.ndarray, UT: np.ndarray, UC: np.ndarray):
    """``u_T'G u_T``, ``u_T'G u_C`` and ``u_C'G u_C`` for every row."""
    GT = UT @ G
    return (np.einsum("mi,mi->m", GT, UT), np.einsum("mi,mi->m", GT, UC),
            np.einsum("mi,mi->m", UC @ G, UC))


def _check_rows(cache: DesignCache, UT: np.ndarray, UC: np.ndarray) -> None:
    N = cache.y.shape[1]
    if UT.ndim != 2 or UC.shape != UT.shape or UT.shape[1] != N:
        raise MlsynthDataError(
            f"designs are rows of indicators over the panel's {N} geos; got "
            f"treatment rows of shape {UT.shape} and control rows of shape "
            f"{UC.shape}.")
    if np.any(UC.sum(axis=1) == 0):
        raise MlsynthDataError(
            "a design has no control geo, so there is no control aggregate to "
            "regress on.")
    if np.any(UT.sum(axis=1) == 0):
        raise MlsynthDataError("a design has no treatment geo.")
    if np.any((UT > 0) & (UC > 0)):
        raise MlsynthDataError(
            "a geo sits in both groups of a design; it would enter both "
            "aggregates and share its variance between them.")


def score_designs(cache: DesignCache, UT: np.ndarray, UC: np.ndarray, *,
                  objective: str = "reference") -> np.ndarray:
    """The key of every design under ``objective``, one row each.

    ``UT`` and ``UC`` are (m, N) indicator rows of the treatment and control
    groups. Under ``"reference"`` the result is (m, 6): the four gates as 0/1,
    the correlation rounded to two places, and the inverse required impact.
    Under ``"relative"`` it is (m, 1): the treated level over the required
    impact. Each row is the tuple
    :func:`~mlsynth.utils.tbr_helpers.design.objective.score_split` returns
    under the same objective. ``"paper"`` is refused, since its p-values are not
    quadratic forms in the rows.
    """
    if objective not in ("reference", "relative"):
        raise MlsynthConfigError(
            f"the batched screen scores objective='reference' or 'relative'; "
            f"got {objective!r}, whose terms are not quadratic forms in the "
            f"group memberships.")
    UT = np.asarray(UT, dtype=float)
    UC = np.asarray(UC, dtype=float)
    _check_rows(cache, UT, UC)
    n, n_test, n_fit = cache.n, cache.n_test, cache.n_fit
    eps = float(np.finfo(float).eps)
    noise = (UT.shape[1] + 2) * eps
    const = obj.window_constants(n, n_test)

    s_yy, s_xy, s_xx = _forms(cache.G, UT, UC)
    a_yy, a_xy, a_xx = _forms(cache.absG, UT, UC)
    f_yy, f_xy, f_xx = _forms(cache.G1, UT, UC)
    b_yy, b_xy, b_xx = _forms(cache.absG1, UT, UC)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = s_xy / s_xx
        rss = s_yy - beta * s_xy
        b1 = f_xy / f_xx
        rss1 = f_yy - b1 * f_xy
    # A form within its own rounding of zero is a computed zero (rule 4); the
    # direct fit handles those designs exactly, including its rank-deficient
    # branch, which no closed form about S_xx describes.
    degenerate = ((s_xx <= noise * a_xx) | (s_yy <= noise * a_yy)
                  | (rss <= noise * (a_yy + np.abs(beta) * a_xy))
                  | (f_xx <= noise * b_xx)
                  | (rss1 <= noise * (b_yy + np.abs(b1) * b_xy)))
    ok = ~degenerate
    beta, rss, b1, rss1 = (np.where(ok, v, 1.0) for v in (beta, rss, b1, rss1))
    s_xx = np.where(ok, s_xx, 1.0)
    f_xx = np.where(ok, f_xx, 1.0)

    # The square root of each form, as score_split takes it (#990).
    corr = np.clip(s_xy / (np.sqrt(s_xx) * np.sqrt(np.where(ok, s_yy, 1.0))),
                   -0.999999, 0.999999)
    sigma = np.sqrt(rss / (n - 2))
    sq = np.sqrt(const.phi * (n + 1) / (n * n_test * (n - 1))
                 + 1.0 / n + 1.0 / n_test)
    inv_impact = 1.0 / ((const.tq_sig + const.tq_pow) * n_test * sq * sigma)
    if objective == "relative":
        level = UT @ cache.col_mean
        return _direct_fallback(cache, (level * inv_impact)[:, None], degenerate,
                                UT, UC, objective)

    d_yy, d_xy, d_xx = _forms(cache.Gd, UT, UC)
    dw = (d_yy - 2.0 * beta * d_xy + beta * beta * d_xx) / rss
    dw_ok = (dw > obj.DW_RANGE[0]) & (dw < obj.DW_RANGE[1])

    path = cache.P @ UT.T - beta[None, :] * (cache.P @ UC.T)
    z = np.abs(path / sigma[None, :])[:-1]
    envelope = obj.brownian_bridge_envelope(n)
    bb_ok = ~np.any(z > envelope[:, None], axis=0)

    # The A/A test: refit on the first n - n_test periods, estimate the rest.
    y_fit, x_fit = UT @ cache.fit_mean, UC @ cache.fit_mean
    y_hold, x_hold = UT @ cache.hold_mean, UC @ cache.hold_mean
    alpha1 = y_fit - b1 * x_fit
    sigma1 = np.sqrt(rss1 / (n_fit - 2))
    estimate = n_test * (y_hold - alpha1 - b1 * x_hold)
    scale = n_test * sigma1 * np.sqrt(1.0 / n_fit + (x_hold - x_fit) ** 2 / f_xx
                                      + 1.0 / n_test)
    half = const.tq_holdout * scale
    lower, upper = estimate - half, estimate + half
    true_mean = np.minimum(np.abs(lower), np.abs(upper))
    tq = half / sigma1
    post = sigma1 * np.sqrt(1.0 / n_fit + 1.0 / n_test)
    false_pos = (1.0 - special.stdtr(const.holdout_df, tq - true_mean / post)
                 + special.stdtr(const.holdout_df, -tq - true_mean / post))
    aa_ok = (lower * upper < 0.0) | (false_pos <= obj.AA_THRESHOLD_PROB)

    keys = np.column_stack([corr >= obj.MIN_CORR, aa_ok, bb_ok, dw_ok,
                            np.round(corr, 2), inv_impact]).astype(float)
    return _direct_fallback(cache, keys, degenerate, UT, UC, objective)


def _direct_fallback(cache: DesignCache, keys: np.ndarray,
                     degenerate: np.ndarray, UT: np.ndarray, UC: np.ndarray,
                     objective: str) -> np.ndarray:
    """Overwrite the degenerate rows with the direct fit's key."""
    for r in np.flatnonzero(degenerate):
        T = np.flatnonzero(UT[r] > 0)
        C = np.flatnonzero(UC[r] > 0)
        keys[r] = obj.score_split(cache.y[:, T].sum(axis=1),
                                  cache.y[:, C].sum(axis=1),
                                  objective=objective, n_test=cache.n_test).key
    return keys
