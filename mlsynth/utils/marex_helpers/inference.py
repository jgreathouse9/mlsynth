"""Placebo/permutation inference for a MAREX synthetic treated-vs-control pair.

The held-out *blank* pre-periods act as placebos: the synthetic treated minus
synthetic control there should be noise, so its distribution calibrates a
permutation p-value and a split-conformal confidence band for the post-period
effect (Abadie & Zhao 2026, OA; Chernozhukov-Wuthrich-Zhu 2021).

The band for the total effect has two constructions. ``"resample"`` draws paths
from the blank-period effects alone. ``"inversion"`` also reads the post
outcomes: it inverts CWZ's test of a constant effect with their Remark 1
statistic, the magnitude of the signed post sum of the residuals, against the
cyclic shifts of the residual path over the blank and post periods. The fitting
periods stay out of that reference because the design was chosen on them. On
simulated MAREX designs their residuals are about 30% smaller in mean square
than the blank and post residuals, which behave alike, and a reference that
includes them understates the spread of a 13-period post sum by a factor of
1.17 with independent factors and 1.59 with factors autocorrelated at 0.8. The
study is in ``benchmarks/studies/marex_cumulative_conformal/``.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import replace
from typing import Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..conformal import (CumulativeConformalBand, block_error_paths,
                         cumulative_conformal_by_inversion)
from ..conformal.quantile import split_conformal_quantile
from .structures import MAREXInference
from .windows import min_blank_periods

#: How the band for the total effect is built; see the module docstring.
CUMULATIVE_METHODS = ("resample", "inversion")


def compute_inference(
    Y_treated: np.ndarray,
    Y_control: np.ndarray,
    T0: int,
    TcE: int,
    Tb: int,
    alpha: float = 0.05,
    max_combinations: int = 1000,
    random_state: Optional[int] = None,
    cumulative_band: bool = False,
    cumulative_block: int = 0,
    cumulative_n_sim: int = 2000,
    cumulative_seed: int = 0,
    cumulative_method: str = "resample",
    control_outcomes: Optional[np.ndarray] = None,
    T_post: Optional[int] = None,
) -> MAREXInference:
    """Compute permutation inference for one synthetic contrast.

    Parameters
    ----------
    Y_treated, Y_control : np.ndarray
        Synthetic treated / control outcomes over the full timeline, shape ``(T,)``.
    T0 : int
        Number of pre-treatment periods.
    TcE : int
        Start index of the blank (held-out) window.
    Tb : int
        Number of blank periods.
    alpha : float
        Two-sided significance level.
    max_combinations : int
        Number of permutation draws for the global test.
    random_state : int, optional
        Seed for reproducibility.
    cumulative_band : bool
        Whether to add a band for the total post-period effect.
    cumulative_block : int
        ``"resample"`` only: block length for the draw; ``0`` means the whole
        horizon.
    cumulative_n_sim : int
        ``"resample"`` only: number of accumulated paths to draw.
    cumulative_seed : int
        ``"resample"`` only: seed for the draw, held separate from
        ``random_state`` so the permutation test and the band are reproducible
        independently.
    cumulative_method : {"resample", "inversion"}
        How the band for the total is built; see the module docstring.
    control_outcomes : np.ndarray, optional
        ``"inversion"`` only: outcomes of the units the refit may draw on, shape
        ``(J, T)``, aligned to ``Y_treated``. MAREX passes the units the design
        left untreated.

    Returns
    -------
    MAREXInference

    Raises
    ------
    MlsynthConfigError
        On an unknown ``cumulative_method``, or a band by ``"inversion"``
        without ``control_outcomes``.

    Warns
    -----
    UserWarning
        When the inversion's reference is too short for any p-value to reach
        ``alpha``, so the band is unbounded on both sides.
    """
    if cumulative_method not in CUMULATIVE_METHODS:
        raise MlsynthConfigError(
            f"cumulative_method must be one of {CUMULATIVE_METHODS}; got "
            f"{cumulative_method!r}.")
    if (cumulative_band and cumulative_method == "inversion"
            and control_outcomes is None):
        raise MlsynthConfigError(
            "cumulative_method='inversion' needs control_outcomes, the "
            "outcomes of the units the refit may draw on.")
    rng = np.random.default_rng(random_state)
    Y_treated = np.asarray(Y_treated, dtype=float)
    Y_control = np.asarray(Y_control, dtype=float)
    T_post = _check_t_post(T_post, len(Y_treated) - T0)

    blank_idx = np.arange(TcE, TcE + Tb)
    post_idx = np.arange(T0, T0 + T_post)

    placebo_effects = Y_treated[blank_idx] - Y_control[blank_idx]
    treated_effects = Y_treated[post_idx] - Y_control[post_idx]
    full_effects = Y_treated - Y_control

    all_effects = np.concatenate([placebo_effects, treated_effects])
    s_obs = float(np.mean(np.abs(treated_effects))) if T_post > 0 else 0.0
    if T_post > 0:
        global_p = _global_p_value(all_effects, T_post, s_obs,
                                   max_combinations, rng)
    else:
        global_p = float("nan")

    if placebo_effects.size and treated_effects.size:
        # Each post period joins the blank periods as one more score, so its
        # own rank counts: (1 + k) / (Tb + 1), never zero.
        exceed = np.sum(
            np.abs(placebo_effects)[:, None] >= np.abs(treated_effects)[None, :],
            axis=0)
        per_period = (1.0 + exceed) / (placebo_effects.size + 1.0)
        q = split_conformal_quantile(placebo_effects, alpha)
        if np.isinf(q):
            warnings.warn(
                f"The pointwise band needs at least "
                f"{min_blank_periods(alpha)} blank periods at alpha={alpha}: "
                f"with {placebo_effects.size}, the conformal quantile, order "
                f"statistic ceil((Tb + 1)(1 - alpha)), does not exist, and the "
                f"band is reported as (-inf, inf). The permutation p-values do "
                f"not depend on it.", UserWarning, stacklevel=2)
    else:
        per_period = np.full(T_post, np.nan)
        q = np.nan
    interval = np.column_stack([treated_effects - q, treated_effects + q])
    ci = np.full((len(Y_treated), 2), np.nan)
    ci[T0:T0 + T_post] = interval

    # The blank periods are held out of the fit, so their effects are an
    # out-of-sample error series. Block-resampling them into whole paths and
    # accumulating gives the spread of an L-period total, which the pointwise
    # quantile above has already averaged away: the variance of that total is
    # L * gamma_0 + 2 * sum_k (L - k) * gamma_k, and only the block length
    # carries the gamma_k terms.
    band = None
    if (cumulative_band and cumulative_method == "inversion"
            and placebo_effects.size and treated_effects.size):
        band = _inversion_band(Y_treated, control_outcomes, T0, TcE, T_post,
                               point=float(treated_effects.sum()), alpha=alpha)
    elif cumulative_band and placebo_effects.size and treated_effects.size:
        # The blank window is one contiguous stretch, so it is split into as many
        # horizon-length sub-windows as it holds and the level is drawn apart from
        # the fluctuation. A horizon total is H times the mean error over that
        # horizon, so the level is the term that dominates it; a draw that cannot
        # see the level vary between windows cannot size the total.
        n_windows = int(placebo_effects.size // max(int(T_post), 1))
        paths = block_error_paths(
            placebo_effects, horizon=int(T_post), block=int(cumulative_block),
            n_sim=int(cumulative_n_sim),
            rng=np.random.default_rng(cumulative_seed),
            n_windows=n_windows if n_windows >= 2 else 0)
        point = float(treated_effects.sum())
        # The draws are centred and sign-symmetric, so the quantile of their
        # magnitudes is the symmetric half-width the band reports.
        half = float(np.quantile(np.abs(paths.sum(axis=1)), 1.0 - alpha))
        band = CumulativeConformalBand(
            point=point, lower=point - half, upper=point + half,
            half_width=half, n_scores=int(placebo_effects.size),
            alpha=float(alpha), horizon=int(T_post))

    return MAREXInference(
        treated_effects=treated_effects,
        placebo_effects=placebo_effects,
        fulltreated_effects=full_effects,
        s_obs=s_obs,
        global_p_value=global_p,
        per_period_pvals=per_period,
        ci=ci,
        alpha=alpha,
        cumulative_band=band,
    )


def _check_t_post(T_post, available):
    """Post periods the inference reads; every one after ``T0`` by default."""
    if T_post is None:
        return available
    if isinstance(T_post, bool) or not isinstance(T_post, (int, np.integer)):
        raise MlsynthConfigError(f"T_post must be an integer; got {T_post!r}.")
    if not 1 <= T_post <= available:
        raise MlsynthDataError(
            f"T_post must be between 1 and the {available} periods after T0; "
            f"got {T_post}.")
    return int(T_post)


def _global_p_value(all_effects, T_post, s_obs, max_combinations, rng):
    """Abadie and Zhao's eq. (17), exactly when it is affordable.

    The arrangements are the ``T_post``-subsets of the blank and post periods.
    When there are at most ``max_combinations`` of them every one is used, the
    observed one included, and the p-value is the paper's. Otherwise
    ``max_combinations`` are drawn and the observed arrangement is counted as
    one more, ``(1 + k) / (1 + B)``, which keeps the test valid at any ``B``
    (Phipson and Smyth 2010) and never reports zero.
    """
    n_total = all_effects.size
    abs_effects = np.abs(all_effects)
    tol = 1e-12 * (1.0 + s_obs)
    if math.comb(n_total, T_post) <= max_combinations:
        pi = np.array(list(itertools.combinations(range(n_total), T_post)))
        s_perm = np.mean(abs_effects[pi], axis=1)
        return float(np.mean(s_perm >= s_obs - tol))
    pi = np.array([rng.choice(n_total, size=T_post, replace=False)
                   for _ in range(max_combinations)])
    s_perm = np.mean(abs_effects[pi], axis=1)
    return float((1 + np.sum(s_perm >= s_obs - tol)) / (1 + max_combinations))


def _inversion_band(y, controls, T0, TcE, T_post, *, point, alpha):
    """The inversion band for ``y``'s total, reported about MAREX's own total.

    The refit matches on every period and the reference starts at the blank
    window, ``TcE``. The library's own point is an SC fit of ``y`` on the
    pre-period; the band reports MAREX's total in its place, the number the
    resampled band reports, and the endpoints are the test's either way.
    """
    band = cumulative_conformal_by_inversion(
        y, np.asarray(controls, dtype=float).T, int(T0), alpha=float(alpha),
        horizon=int(T_post),
        refit="sc", conformal_type="block", statistic="signed",
        reference_start=int(TcE))
    if 1.0 / band.n_reference > alpha:
        warnings.warn(
            f"The cumulative band by inversion cannot reject any constant "
            f"effect: its reference holds {band.n_reference} blank and post "
            f"periods, so no p-value falls below 1/{band.n_reference}, above "
            f"alpha={alpha}. The band is unbounded on both sides; it needs at "
            f"least 1/alpha blank and post periods.",
            UserWarning, stacklevel=3)
    return replace(band, point=point)
