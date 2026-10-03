"""Post-inference update module for synthetic experimental design."""

import numpy as np
from typing import List, Optional

from .structure import SEDCandidate
from .inference import compute_moving_block_conformal_ci   # Adjust path if needed


def update_post_inference(
    candidate_results: List[SEDCandidate],
    Y_full: np.ndarray,
    post_idx: np.ndarray,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = 42
) -> List[SEDCandidate]:
    """
    Update all candidates with:
      - Full-timeline synthetic treated, synthetic control, and effects
      - Average Treatment Effect (ATE) over post periods
      - Permutation p-value
      - Block conformal confidence intervals

    Parameters
    ----------
    candidate_results : list of SEDCandidate
        List of evaluated candidates (must have .weights and .identification)
    Y_full : np.ndarray
        Full timeline matrix (pre + post), shape (T_total, N)
    post_idx : np.ndarray
        Indices corresponding to the post-treatment period in Y_full
    n_sims : int
        Number of simulations for permutation test
    alpha : float
        Significance level for conformal CI
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    candidate_results : list of SEDCandidate (updated in place)
    """
    if len(post_idx) == 0:
        return candidate_results

    for cand in candidate_results:
        # Get column indices of the treated units for this candidate
        treated_col_idx = np.asarray(cand.identification.treated_idx, dtype=int)

        # Extract weights
        treated_weights = cand.weights.treated      # shape (m,)
        control_weights = cand.weights.control      # shape (N,)

        # Compute predictions on full timeline
        synth_treated_full = Y_full[:, treated_col_idx] @ treated_weights
        synth_control_full = Y_full @ control_weights

        # Store results
        cand.predictions.synthetic_treated = synth_treated_full
        cand.predictions.synthetic_control = synth_control_full
        cand.predictions.effects = synth_treated_full - synth_control_full

        # Point estimate (ATE)
        post_gap = cand.predictions.effects[post_idx]
        cand.inference.ate = float(np.mean(post_gap))

        # Store metadata
        cand.inference.treated_col_idx = treated_col_idx.tolist()

        # === Inference ===
        # Permutation p-value
        inference_result = compute_post_inference(
            candidate=cand,
            post_idx=post_idx,
            n_perms=n_sims,
            seed=seed
        )
        cand.inference.p_value = inference_result.inference.p_value

        # Block conformal confidence intervals
        cand = compute_moving_block_conformal_ci(
            candidate=cand,
            post_idx=post_idx,
            alpha=alpha,
            seed=seed
        )

    return candidate_results


# ---------------------------------------------------------------------------
# Cumulative effect path for a simplex-weighted synthetic control.
#
# TBR reads its cumulative effect off a Student-t posterior because AdID's
# counterfactual is an unconstrained two-parameter fit, so it is affine in the
# treated series and a sum of affine functions stays affine. Simplex weights are
# not affine in y -- measured departure from affinity is 3e-16 for least
# squares, 6e-16 with the weights summed to one, and 3e-2 once they are also
# constrained non-negative -- so the closed form does not carry over and the
# interval is built by inverting a pivot instead.
#
# The pivot is the sum over a block of held-out residuals. Blocks preserve the
# serial dependence weekly panels carry, and summing them matches the statistic
# being reported, so nothing is assumed about the shape of the effect path. An
# alternative that inverts a constant-effect sharp null was measured and
# discarded: it conflates the size of the cumulative with the flatness of the
# path, and returns an empty set for 1.7 to 2.5 per cent of post windows.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _stats

from ...exceptions import MlsynthConfigError, MlsynthDataError

#: Blank-window bias beyond this many standard errors means the donors cannot
#: approximate the treated unit, where the interval is biased and not merely
#: wide. The threshold sits where coverage starts to fall away. Over 1,600
#: simulated designs the cumulative interval covers 0.901 where the statistic
#: is below 1, 0.844 between 2 and 3, 0.842 between 3 and 4, 0.719 between 4
#: and 5 and 0.513 between 5 and 6, against a nominal 0.90. At 3.0 the gate
#: admits 84 per cent of designs, whose coverage is 0.883, and the designs it
#: refuses cover 0.696. Moving it to 5.0 admits 93 per cent at 0.875, so the
#: trade is flat across 2 to 5 and the choice inside that range is not sharp.
APPROXIMABILITY_T = 3.0


@dataclass(frozen=True)
class CumulativePoint:
    """The cumulative effect through one horizon, with its interval."""

    horizon: int
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Approximability:
    """Whether the donor pool can reproduce the treated unit off-sample."""

    bias: float
    scale: float
    t_stat: float
    p_value: float
    ok: bool
    serial_correlation: float = 0.0
    effective_n: float = 0.0


def approximability(blank_gaps: Sequence[float], *,
                    threshold: float = APPROXIMABILITY_T) -> Approximability:
    """Test the held-out gap for a location offset.

    The cumulative interval is centred only when the fitted weights reproduce
    the treated unit on periods they were not fitted to. A miss on those periods
    runs through the post window as a constant, reads as a treatment effect, and
    no resampling of residuals removes it. Read this before the interval, not
    beside it.

    The statistic predicts interval failure better than the convex-hull
    condition it stands in for. Over 700 designs whose hull membership is
    verified by a linear program, the gate admits 81 per cent of them with
    coverage 0.875 and refuses the rest at 0.540; splitting the same designs on
    hull membership admits 47 per cent at 0.897 and leaves 0.732 in the refused
    half. Hull membership is the wrong cut because what decides the interval is
    the size of the realized offset against the noise: a design just outside the
    hull carries an offset too small to matter, and a design inside it can be
    fitted badly enough to carry a large one. The two overlap heavily -- the
    largest in-hull statistic is 8.2, the smallest out-of-hull one is 0.001, and
    82 per cent of out-of-hull designs fall below 5.5 -- so no threshold sorts
    designs by hull membership, and this one does not try to.

    Serially correlated periods carry less information than independent ones, so
    the offset is tested on the Bartlett effective sample size
    ``n (1 - rho) / (1 + rho)`` and not the raw period count. On zero-offset
    gaps of 20 periods the raw count refuses 0.007 of designs at ``rho = 0``,
    0.043 at 0.3, 0.173 at 0.6 and 0.385 at 0.8, against the 0.003 a correctly
    sized test would refuse at this threshold; the correction brings the two
    dependent cases to 0.060 and 0.125.

    The correction costs something where there is no dependence: at ``rho = 0``
    it refuses 0.019 against the raw count's 0.007. ``(1 - rho)/(1 + rho)`` is
    convex, so a lag-one estimate scattered about zero averages to an effective
    sample size above the period count -- 1.24 n over 20 periods -- and the test
    charges the offset against more information than the window holds. Clipping
    the effective count at the period count measures 0.005 at ``rho = 0`` and
    0.059 at 0.6, and is not applied here.
    """
    gaps = np.asarray(blank_gaps, dtype=float).ravel()
    if gaps.size < 2:
        raise MlsynthDataError(
            f"testing the held-out gap for an offset needs at least 2 periods; "
            f"got {gaps.size}.")
    if not np.all(np.isfinite(gaps)):
        raise MlsynthDataError("the held-out gap contains non-finite values.")

    scale = float(gaps.std(ddof=1))
    if scale <= 0.0:
        raise MlsynthDataError(
            "the held-out gap has no spread, so an offset cannot be tested "
            "against it.")
    bias = float(gaps.mean())
    centred = gaps - bias
    rho = 0.0
    if gaps.size > 2:
        lag = float(np.corrcoef(centred[1:], centred[:-1])[0, 1])
        rho = float(np.clip(lag, -0.99, 0.99)) if np.isfinite(lag) else 0.0
    effective_n = max(gaps.size * (1.0 - rho) / (1.0 + rho), 2.0)
    t_stat = bias / (scale / np.sqrt(effective_n))
    p_value = float(2.0 * (1.0 - _stats.t.cdf(abs(t_stat), effective_n - 1.0)))
    return Approximability(bias=bias, scale=scale, t_stat=float(t_stat),
                           p_value=p_value, ok=bool(abs(t_stat) <= threshold),
                           serial_correlation=rho, effective_n=float(effective_n))


def _blocks(pool: Sequence[np.ndarray], horizon: int) -> np.ndarray:
    """Every circular block of ``horizon`` periods from every usable series.

    Circular blocks keep each series contributing as many blocks as it has
    periods, so the tail quantiles do not thin out as the horizon grows. A
    series shorter than the horizon contributes none and is skipped; the pool
    is refused only when no series can form a block at all.
    """
    found = []
    for series in pool:
        if series.size < horizon:
            continue
        extended = np.concatenate([series, series[:horizon - 1]])
        found.append(np.lib.stride_tricks.sliding_window_view(extended, horizon))
    if not found:
        raise MlsynthDataError(
            f"no placebo series is long enough to form a block of {horizon} "
            f"period(s); the longest holds "
            f"{max((s.size for s in pool), default=0)}.")
    return np.vstack(found)


def _prepare_pool(pool: Sequence[Sequence[float]],
                  standardize: bool) -> List[np.ndarray]:
    """Validate the pool and put every series on the first one's scale.

    The first series is the treated unit's own held-out residual and sets the
    scale; the rest are placebos. Pooling raises the number of blocks the tail
    quantiles are read from, which a single blank window cannot supply -- its
    own residuals alone cover 0.83 at a nominal 0.90 and fall to 0.72 by the
    eighth horizon. Pooling without rescaling overshoots instead, to 0.93-0.95,
    because the placebos are noisier than the unit under test. Doing both
    covers 0.915, flat in horizon.
    """
    series = [np.asarray(s, dtype=float).ravel() for s in pool]
    series = [s for s in series if s.size > 0]
    if not series:
        raise MlsynthDataError(
            "the placebo pool is empty; the cumulative interval is read from "
            "held-out residuals and has nothing to read.")
    if not all(np.all(np.isfinite(s)) for s in series):
        raise MlsynthDataError("the placebo pool contains non-finite values.")

    spreads = np.array([s.std(ddof=1) if s.size > 1 else 0.0 for s in series])
    if not np.any(spreads > 0.0):
        raise MlsynthDataError(
            "every placebo series is degenerate, so the pool has no spread to "
            "build an interval from.")
    if not standardize:
        return series

    reference = spreads[0] if spreads[0] > 0.0 else float(spreads[spreads > 0].mean())
    scaled = [series[0]]
    for s, spread in zip(series[1:], spreads[1:]):
        scaled.append(s * (reference / spread) if spread > 0.0 else s)
    return scaled


def cumulative_path(post_gaps: Sequence[float],
                    placebo_pool: Sequence[Sequence[float]], *,
                    level: float = 0.90,
                    standardize: bool = True) -> List[CumulativePoint]:
    """The cumulative effect through each horizon, with a block-sum interval.

    ``post_gaps`` are the treated unit's post-period gaps. ``placebo_pool``
    holds held-out residual series, the first being the treated unit's own and
    setting the scale for the rest.

    The point estimate is arithmetic -- the running sum of the gaps. Only the
    interval is inferential, and it inverts the distribution of the sum over
    blocks of the same length drawn from the pool. Because the sum over a window
    is the window length times its mean, an interval for one rescales exactly
    into an interval for the other; the dependence between periods is already
    priced by the blocks.

    Coverage holds under moderate serial dependence and degrades under strong
    dependence: 0.915 at an AR(1) coefficient of 0, 0.881 at 0.3 and 0.863 at
    0.6, against a nominal 0.90. Blocks absorb dependence shorter than the
    horizon they span, and at 0.6 some outlives them.

    The estimand is the effect on the treated aggregate the gaps were formed
    from, Abadie and Zhao's ``tau^T``. It is not the population effect. Under an
    experimental design whose treated units are weighted to represent a wider
    population, the two separate as soon as the effect varies across units: at a
    cross-unit standard deviation of 0.8 around a mean effect of 1.0 they differ
    by 0.30 per period, and this interval covers the treated effect 0.880 of the
    time and the population effect 0.714. Reporting the result as a national
    number needs a second term for the representation error, which this does not
    carry.
    """
    if not 0.0 < float(level) < 1.0:
        raise MlsynthConfigError(
            f"the interval's level has to sit strictly inside (0, 1); got "
            f"{level}.")
    gaps = np.asarray(post_gaps, dtype=float).ravel()
    if gaps.size == 0:
        raise MlsynthDataError(
            "there are no post-period gaps to accumulate.")
    if not np.all(np.isfinite(gaps)):
        raise MlsynthDataError("the post-period gaps contain non-finite values.")

    pool = _prepare_pool(placebo_pool, standardize)
    alpha = 1.0 - float(level)
    running = np.cumsum(gaps)

    path: List[CumulativePoint] = []
    for horizon in range(1, gaps.size + 1):
        sums = _blocks(pool, horizon).sum(axis=1)
        high, low = np.quantile(sums, [1.0 - alpha / 2.0, alpha / 2.0])
        observed = float(running[horizon - 1])
        path.append(CumulativePoint(horizon=horizon, estimate=observed,
                                    lower=observed - float(high),
                                    upper=observed - float(low)))
    return path


@dataclass(frozen=True)
class UnitLevelCumulative:
    """A cumulative path per treated unit, and the aggregate they compose."""

    aggregate: List[CumulativePoint]
    per_unit: Tuple[List[CumulativePoint], ...]
    weights: Tuple[float, ...]
    estimand: str = "treated"
    centered: bool = False
    level_scale: Tuple[float, ...] = ()


def _level_scale(series: np.ndarray) -> float:
    """Standard error of a window's mean, charged on its effective length."""
    if series.size < 2:
        return 0.0
    rho = _lag_one(series)
    return float(np.sqrt(series.var(ddof=1) / _bartlett_n(series.size, rho)))


def _widen(sums: np.ndarray, scale: float) -> np.ndarray:
    """Convolve a block-sum null with an independent normal term."""
    if scale <= 0.0:
        return sums
    return (sums[:, None] + scale * _NORMAL_NODES[None, :]).ravel()


def _path(post: np.ndarray, pool: List[np.ndarray], alpha: float,
          level_scale: float) -> List[CumulativePoint]:
    """One cumulative path, widened for a subtracted level if there was one."""
    running = np.cumsum(post)
    out: List[CumulativePoint] = []
    for horizon in range(1, post.size + 1):
        sums = _widen(_blocks(pool, horizon).sum(axis=1), horizon * level_scale)
        high, low = np.quantile(sums, [1.0 - alpha / 2.0, alpha / 2.0])
        observed = float(running[horizon - 1])
        out.append(CumulativePoint(horizon=horizon, estimate=observed,
                                   lower=observed - float(high),
                                   upper=observed - float(low)))
    return out


def unit_level_cumulative(post_gaps: np.ndarray, blank_gaps: np.ndarray,
                          weights: Sequence[float], *,
                          level: float = 0.90,
                          center: bool = False,
                          extra_pool: Optional[Sequence[Sequence[float]]] = None
                          ) -> UnitLevelCumulative:
    """Cumulative paths under Abadie and Zhao's Unit-level design, equation (10).

    That design fits the treated aggregate to the population and each treated
    unit to its own synthetic control at once, so the estimate decomposes by
    their equation (11): the aggregate gap is the weighted mean of the per-unit
    gaps. Both are returned, and the aggregate is computed from the parts, so a
    breakdown and a headline cannot contradict each other.

    ``post_gaps`` and ``blank_gaps`` are period-by-unit. Each unit's own interval
    reads its null from the other treated units, which the design has already
    built proper synthetic controls for; ``extra_pool`` adds further held-out
    series, which a small treated group needs because a handful of short series
    leaves the tail quantiles thin.

    The aggregate and the per-unit paths do not respond to that alike. Measured
    on a four-unit design at a nominal 0.90, with the number of extra series
    running 0, 4, 10, 20, the aggregate covers 0.874, 0.901, 0.913, 0.932 and
    each unit covers its own effect 0.774, 0.805, 0.833, 0.852. Four extra
    series bring the aggregate to its nominal level and twenty carry it past,
    so a thicker pool is not uniformly better.

    A thicker pool leaves the per-unit paths short of their nominal level,
    because one treated unit's synthetic control sits at its own fitted level
    and no resampling shifts a location offset. ``center`` subtracts each unit's
    blank-window mean from both windows and charges the noise in that mean back
    as the path accumulates, which ``level_scale`` reports.

    It is off by default, because removing the offset costs more than it buys.
    Measured at a nominal 0.90 on a four-unit design: the aggregate covers 0.913
    without it and 0.858 with it, the population 0.895 and 0.849, and the
    per-unit paths 0.833 and 0.838. The offsets also sit in the residuals the
    null is built from, so taking them out contracts the null -- the aggregate
    interval narrows by a third -- and the bias removed is worth less than the
    width lost, except marginally for a single unit, whose offset is largest
    relative to its noise. Reach for it only for a per-unit reading.

    Centring also assumes the offset is the same in both windows, which
    :func:`approximability` tests on the blank gap.

    The estimand is ``tau^T``, the weighted effect on the treated. See
    :func:`cumulative_path` for why that is not the population effect.
    """
    post = np.asarray(post_gaps, dtype=float)
    blank = np.asarray(blank_gaps, dtype=float)
    if post.ndim == 1:
        post = post[:, None]
    if blank.ndim == 1:
        blank = blank[:, None]
    w = np.asarray(weights, dtype=float).ravel()

    if np.any(w < 0.0):
        raise MlsynthConfigError(
            f"the design weights have to be non-negative; the smallest is "
            f"{w.min():.6g}.")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise MlsynthConfigError(
            f"the design weights have to sum to one; they sum to {w.sum():.6g}.")
    if post.shape[1] != w.size or blank.shape[1] != w.size:
        raise MlsynthDataError(
            f"the gaps cover {post.shape[1]} treated unit(s) after treatment and "
            f"{blank.shape[1]} before it, against {w.size} weight(s).")

    extra = [np.asarray(s, dtype=float).ravel() for s in (extra_pool or [])]
    n_units = w.size
    alpha = 1.0 - float(level)

    # A unit's synthetic control sits at its own fitted level. That offset runs
    # through the post window unchanged and reads as a treatment effect, so it
    # comes off both windows, and the noise in the mean that removed it is
    # charged back as the path accumulates.
    levels = blank.mean(axis=0) if center else np.zeros(n_units)
    scales = ([_level_scale(blank[:, k] - levels[k]) for k in range(n_units)]
              if center else [0.0] * n_units)
    post_c, blank_c = post - levels, blank - levels
    extra_c = [s - s.mean() for s in extra] if center else extra

    per_unit: List[List[CumulativePoint]] = []
    for k in range(n_units):
        others = [blank_c[:, j] for j in range(n_units) if j != k]
        pool = _prepare_pool([blank_c[:, k], *others, *extra_c], True)
        per_unit.append(_path(post_c[:, k], pool, alpha, scales[k]))

    units = [blank_c[:, j] for j in range(n_units)] if n_units > 1 else []
    agg_pool = _prepare_pool([blank_c @ w, *units, *extra_c], True)
    agg_scale = _level_scale(blank_c @ w) if center else 0.0
    aggregate = _path(post_c @ w, agg_pool, alpha, agg_scale)

    return UnitLevelCumulative(aggregate=aggregate,
                               per_unit=tuple(per_unit),
                               weights=tuple(float(x) for x in w),
                               centered=bool(center),
                               level_scale=tuple(float(x) for x in scales))


#: Standard-normal nodes the representation error is convolved over. A fixed
#: grid instead of sampling keeps the interval deterministic.
_NORMAL_NODES = _stats.norm.ppf(np.linspace(0.5 / 41.0, 1.0 - 0.5 / 41.0, 41))


@dataclass(frozen=True)
class PopulationCumulative:
    """A cumulative path for the population effect, and what it cost to get it."""

    aggregate: List[CumulativePoint]
    effect_dispersion: float
    weight_distance: float
    estimand: str = "population"


def _bartlett_n(n: int, rho: float) -> float:
    """How many independent periods ``n`` correlated ones amount to."""
    return max(n * (1.0 - rho) / (1.0 + rho), 2.0)


def _lag_one(series: np.ndarray) -> float:
    if series.size < 3:
        return 0.0
    centred = series - series.mean()
    value = float(np.corrcoef(centred[1:], centred[:-1])[0, 1])
    return float(np.clip(value, -0.99, 0.99)) if np.isfinite(value) else 0.0


def _effect_dispersion(post: np.ndarray, blank: np.ndarray) -> float:
    """Spread of the per-unit effects, with two sources of residue removed.

    Each unit's effect is read as its post-window mean less its blank-window
    mean. The subtraction matters because a unit's synthetic control sits at its
    own fitted level, that offset runs through the post window unchanged, and
    nothing in the post window separates it from a treatment effect. Taking the
    spread of raw post-window means on a four-unit design reports 0.849 where
    the truth is zero; differencing the level out brings it to 0.297.

    What survives is the sampling noise in both means, which is larger than
    ``variance / periods`` because the periods are serially correlated, so it is
    charged on the Bartlett effective sample size of each window. Without that
    the remaining 0.297 is read as heterogeneity, since it exceeds the 0.150
    that independence would have subtracted.
    """
    n_post, n_units = post.shape
    n_blank = blank.shape[0]
    if n_units < 2 or n_post < 2 or n_blank < 2:
        return 0.0

    effects = post.mean(axis=0) - blank.mean(axis=0)
    between = float(effects.var(ddof=1))

    within = 0.0
    for k in range(n_units):
        rho = _lag_one(blank[:, k])
        scale = float(blank[:, k].var(ddof=1))
        within += scale / _bartlett_n(n_post, rho) + scale / _bartlett_n(n_blank, rho)
    return max(between - within / n_units, 0.0)


def population_cumulative(post_gaps: np.ndarray, blank_gaps: np.ndarray,
                          weights: Sequence[float],
                          population_weights: Sequence[float],
                          treated_index: Sequence[int], *,
                          level: float = 0.90,
                          center: bool = False,
                          extra_pool: Optional[Sequence[Sequence[float]]] = None
                          ) -> PopulationCumulative:
    """The cumulative effect on the population, not on the treated group.

    The gap estimates ``tau^T = w . tau``. The population effect is
    ``tau = f . tau``, and the two differ by ``sum_j (w_j - f_j) tau_j``. Both
    weight vectors sum to one, so when the per-unit effects are exchangeable
    with a common mean that difference has expectation zero: it needs a variance
    and not a bias correction. Its variance is ``var(tau_j) * || w - f ||^2``,
    and the Unit-level design supplies the first factor, since it estimates an
    effect for every treated unit.

    The point estimate is therefore the same as
    :func:`unit_level_cumulative`'s; only the interval widens. Over ``h``
    periods a constant per-period offset accumulates as ``h``, so the
    representation term enters the variance as ``h^2`` against the gap term's
    ``h`` and the widening grows with the horizon. That is the shape of the
    shortfall it exists to repair: carrying only the gap term, the interval
    covers ``tau^T`` 0.880 of the time and ``tau`` 0.714 at a nominal 0.90, on a
    design whose per-unit effects have standard deviation 0.8 about a mean of
    1.0.

    Measured against a known truth on a four-unit design, the term takes
    population coverage from 0.775 to 0.895 at a nominal 0.90, costing 40 per
    cent more width. Where the effects are in fact identical it costs 8 per cent
    more width for nothing and covers 0.931 against 0.913, because the spread
    estimate retains about 0.13 it cannot resolve: the correction errs wide.

    Two assumptions carry it, and neither is testable from the treated units
    alone. The untreated units' effects are drawn from the same distribution as
    the treated ones, which is what makes the offset mean zero. And
    ``var(tau_j)`` is read off however many units were treated, so a design with
    two or three of them estimates it poorly; the interval is then honest about
    the correction's form and vague about its size.
    """
    post = np.atleast_2d(np.asarray(post_gaps, dtype=float))
    if post.shape[0] == 1 and np.asarray(post_gaps).ndim == 1:
        post = post.T
    w = np.asarray(weights, dtype=float).ravel()
    f = np.asarray(population_weights, dtype=float).ravel()
    idx = np.asarray(treated_index, dtype=int).ravel()

    if not np.isclose(f.sum(), 1.0, atol=1e-8):
        raise MlsynthConfigError(
            f"the population weights have to sum to one; they sum to "
            f"{f.sum():.6g}.")
    if idx.size != w.size:
        raise MlsynthDataError(
            f"the treated index names {idx.size} unit(s) against {w.size} "
            f"design weight(s).")
    if idx.size and (idx.min() < 0 or idx.max() >= f.size):
        raise MlsynthDataError(
            f"the treated index runs outside the population of {f.size} "
            f"unit(s); it spans {idx.min()} to {idx.max()}.")

    blank = np.atleast_2d(np.asarray(blank_gaps, dtype=float))
    if blank.shape[0] == 1 and np.asarray(blank_gaps).ndim == 1:
        blank = blank.T

    embedded = np.zeros_like(f); embedded[idx] = w
    distance = float(np.sum((embedded - f) ** 2))
    dispersion = _effect_dispersion(post, blank)

    n_units = w.size
    levels = blank.mean(axis=0) if center else np.zeros(n_units)
    post_c, blank_c = post - levels, blank - levels
    extra = [np.asarray(x, dtype=float).ravel() for x in (extra_pool or [])]
    extra_c = [x - x.mean() for x in extra] if center else extra
    units = [blank_c[:, j] for j in range(n_units)] if n_units > 1 else []
    pool = _prepare_pool([blank_c @ w, *units, *extra_c], True)

    # Two independent terms, each accumulating as the horizon: the noise in the
    # level that centring removed, and the representation error.
    level_scale = _level_scale(blank_c @ w) if center else 0.0
    combined = float(np.sqrt(level_scale ** 2 + dispersion * distance))
    path = _path(post_c @ w, pool, 1.0 - float(level), combined)

    return PopulationCumulative(aggregate=path,
                                effect_dispersion=float(dispersion),
                                weight_distance=float(np.sqrt(distance)))
