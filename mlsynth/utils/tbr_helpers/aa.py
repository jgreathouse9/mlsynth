r"""The A/A test: how often a TBR design calls a null window significant.

An A/A test takes a window where the truth is zero and asks how often the
design reports an effect anyway. That is the design's false-positive rate. It
is not power, which is the smallest effect the design could detect, and the two
come apart: a design can have excellent power and an unacceptable
false-positive rate. A search ranks on the first; this module screens on the
second.

The primitive is three operations on top of a pretest fit. Fit on the earlier
pretest periods, project onto the last ``n_test`` held-out pretest periods, and
test the cumulative interval against zero. No treatment is applied anywhere and
nothing is taken from the real post window, so an interval that excludes zero
is a false positive by construction.

The gate rule
-------------

The rule is not "the interval covers zero". A narrow interval sitting just off
zero is a small error and not a broken design, so an interval that excludes
zero gets a second look. Take the true mean at the point of the interval
nearest zero, which is the most forgiving value consistent with it, and ask how
often a design like this one would call a null window significant. Writing that
mean as :math:`\mu_0`, the posterior scale as :math:`s`, and :math:`t` for the
:math:`0.5(1 + \text{level})` quantile on the fit's degrees of freedom:

.. math::

   p = \operatorname{sf}(t - |\mu_0| / s) + \operatorname{cdf}(-t - |\mu_0| / s)

Because :math:`\mu_0` is the most forgiving mean the interval admits, :math:`p`
is a lower bound on how often the design cries wolf, and the candidate fails
only if even that bound exceeds a threshold.

Writing :math:`\mu_0` as the point of the interval nearest zero, which is
``clip(0, lower, upper)``, makes the rule one expression with no branch on
whether zero is covered. Two consequences follow and both are pinned by tests.
The expression is bounded below by :math:`1 - \text{level}`, attained exactly
when the interval covers zero, because then the nearest point is zero itself.
And it is continuous at the boundary: an interval that just touches zero scores
:math:`1 - \text{level}`, the same as one centred there.

The floor is the reason ``threshold`` is validated against it. A threshold at
or below :math:`1 - \text{level}` rejects every candidate, including a design
whose interval sits dead on zero, so it is a configuration error and not a
strict gate.
"""
from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .posterior import cumulative_posterior, cumulative_posterior_hac, fit_pretest

#: Pretest periods a backdated fit needs before its interval means anything.
#: The fit spends two on the intercept and the slope, so eight leaves six
#: degrees of freedom; below that the t quantile is wide enough that the gate
#: cannot separate a sound design from a broken one.
MIN_FIT = 8


class AADraw(BaseModel):
    """One A/A draw: the estimate, the interval, and the gate's verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_fit: int = Field(description="Pretest periods the fit was given.")
    n_test: int = Field(description="Held-out pretest periods it predicted.")
    level: float = Field(description="Interval level the gate is read at.")
    variance: Literal["iid", "hac"] = Field(
        description="'iid' is equation 6 as published; 'hac' is Proposition "
                    "3.4's Newey-West scale.")
    bandwidth: Optional[int] = Field(
        default=None, description="Newey-West truncation lag under 'hac'.")
    estimate: float = Field(
        description="Cumulative effect over the held-out window. The truth is "
                    "zero, so this is the error the design made.")
    scale: float = Field(description="Posterior scale of that cumulative effect.")
    df: int = Field(description="Degrees of freedom of the t posterior.")
    lower: float = Field(description="Lower end of the interval.")
    upper: float = Field(description="Upper end of the interval.")
    covers_zero: bool = Field(
        description="Whether the interval covers zero. On its own this is not "
                    "the gate: an interval that excludes zero gets a second "
                    "look.")
    nearest_mean: float = Field(
        description="The point of the interval nearest zero, which is the most "
                    "forgiving true mean consistent with it.")
    false_positive_probability: float = Field(
        description="How often a design like this one would call a null window "
                    "significant, at that most forgiving mean. A lower bound, "
                    "never below 1 - level.")
    threshold: float = Field(
        description="The probability above which the candidate fails.")
    passes: bool = Field(
        description="Whether the design clears the gate.")


def false_positive_probability(mu0: float, scale: float, df: int,
                              level: float) -> float:
    """How often a design with this scale calls a null window significant.

    Evaluated at a true mean of ``mu0``. At ``mu0 = 0`` this is ``1 - level``
    exactly, for every scale and every degrees of freedom, which is the floor
    the threshold is checked against.
    """
    t_crit = float(stats.t.ppf(0.5 * (1.0 + level), df))
    z = abs(float(mu0)) / float(scale)
    return float(stats.t.sf(t_crit - z, df) + stats.t.cdf(-t_crit - z, df))


def _validate(scale: float, df: int, level: float, threshold: float) -> None:
    if not (0.0 < level < 1.0):
        raise MlsynthConfigError(
            f"level must sit strictly inside (0, 1); got {level}.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise MlsynthConfigError(
            f"the posterior scale must be finite and positive; got {scale}.")
    if int(df) <= 0:
        raise MlsynthConfigError(
            f"the t posterior needs at least one degree of freedom; got {df}.")
    # The floor is compared with a tolerance because 1 - level is not the
    # number it looks like: 1 - 0.90 is 0.09999999999999998, so a threshold of
    # 0.10 sits 2e-17 above the floor and a strict comparison admits it. The
    # direction flips by level -- at 0.99, 0.95 and 0.50 the rounded value is
    # at or below the floor and is caught -- so a strict test passes at some
    # levels and leaks at others.
    floor = 1.0 - level
    at_floor = math.isclose(threshold, floor, rel_tol=1e-9, abs_tol=0.0)
    if not np.isfinite(threshold) or threshold <= floor or at_floor:
        raise MlsynthConfigError(
            f"threshold {threshold} is at or below the gate's floor of "
            f"1 - level = {floor:.4g}, which every design attains when its "
            f"interval covers zero. A threshold there rejects every candidate, "
            f"including a design whose interval sits on zero. Raise the "
            f"threshold above {floor:.4g}, or read the gate at a lower level.")


def gate(*, estimate: float, scale: float, df: int, level: float,
         threshold: float, n_fit: int = 0, n_test: int = 0,
         variance: Literal["iid", "hac"] = "iid",
         bandwidth: Optional[int] = None) -> AADraw:
    """Apply the gate rule to one estimate and its scale.

    Separate from :func:`aa_draw` so the rule can be tested on scalars chosen
    to sit exactly on its boundaries, which no panel reaches by accident.
    """
    _validate(scale, df, level, threshold)
    half = float(stats.t.ppf(0.5 * (1.0 + level), df)) * float(scale)
    lower, upper = float(estimate) - half, float(estimate) + half
    nearest = min(max(0.0, lower), upper)
    p = false_positive_probability(nearest, scale, df, level)
    return AADraw(
        n_fit=int(n_fit), n_test=int(n_test), level=float(level),
        variance=variance, bandwidth=bandwidth, estimate=float(estimate),
        scale=float(scale), df=int(df), lower=lower, upper=upper,
        covers_zero=bool(lower <= 0.0 <= upper), nearest_mean=nearest,
        false_positive_probability=p, threshold=float(threshold),
        passes=bool(p <= threshold))


def aa_draw(y: np.ndarray, x: np.ndarray, n_test: int, *,
            level: float = 0.90,
            variance: Literal["iid", "hac"] = "iid",
            bandwidth: Optional[int] = None,
            threshold: Optional[float] = None) -> AADraw:
    """One A/A draw on a treated and a control aggregate.

    ``y`` and ``x`` are the whole pretest. The last ``n_test`` periods are held
    out, the fit is taken on what remains, and the cumulative effect over the
    held-out window is tested against zero.

    ``threshold`` defaults to twice the nominal rate, ``2 (1 - level)``, which
    is admissible at every level by construction. It is a starting point for the
    calibration harness to replace with a measured one, not a value from the
    reference.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n_test = int(n_test)
    if n_test < 1:
        raise MlsynthConfigError(
            f"n_test must hold at least one period; got {n_test}.")
    if y.size != x.size:
        raise MlsynthDataError(
            f"the two aggregates have different lengths: {y.size} and {x.size}.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise MlsynthDataError(
            "both aggregates must be finite; found a nan or an infinity.")
    n_fit = y.size - n_test
    if n_fit < MIN_FIT:
        raise MlsynthDataError(
            f"holding out {n_test} of {y.size} periods leaves {n_fit} to fit "
            f"on, against the {MIN_FIT} periods a backdated fit needs.")

    fit = fit_pretest(y[:n_fit], x[:n_fit])
    if variance == "hac":
        loc, scale = cumulative_posterior_hac(
            fit, y[:n_fit], x[:n_fit], y[n_fit:], x[n_fit:], bandwidth=bandwidth)
    else:
        loc, scale = cumulative_posterior(fit, y[n_fit:], x[n_fit:])

    if threshold is None:
        threshold = 2.0 * (1.0 - level)
    return gate(estimate=float(loc[-1]), scale=float(scale[-1]), df=int(fit.df),
                level=level, threshold=float(threshold), n_fit=n_fit,
                n_test=n_test, variance=variance, bandwidth=bandwidth)


# ---------------------------------------------------------------------------
# The calibration harness.
#
# One draw says whether one design cried wolf once. Running the primitive over
# many random treated/control splits of a panel with no treatment anywhere
# gives the rate, at each nominal level and window length, and that is the
# quantity the per-candidate gate cannot report because it sees one design at a
# time. A cell carries counts and an interval on the coverage; nothing on it
# says pass, because the decision the table feeds is a human one.
# ---------------------------------------------------------------------------

#: Kerman (2011)'s neutral prior for a binomial rate. Beta(1/3, 1/3) is the
#: prior whose posterior median is closest to unbiased, which is the criterion
#: a coverage count wants: the quantity being estimated is itself a coverage,
#: and a count of 0 or n has to come back with a usable interval instead of a
#: degenerate one.
NEUTRAL_PRIOR = 1.0 / 3.0


class CoverageCell(BaseModel):
    """One (level, window, variance) cell of the calibration table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: float = Field(description="Nominal interval level.")
    n_test: int = Field(description="Held-out window length, in periods.")
    variance: Literal["iid", "hac"] = Field(description="Which scale was read.")
    screened: bool = Field(
        description="Whether a design screen was applied before measuring.")
    n_draws: int = Field(description="Splits measured, after any screen.")
    n_covered: int = Field(description="Of those, how many covered zero.")
    n_rejected_by_screen: int = Field(
        default=0, description="Splits the screen refused, which are counted "
                               "and not measured.")
    coverage: float = Field(
        description="n_covered / n_draws, the realised coverage. Not finite "
                    "when the screen left nothing to measure.")
    ci_lower: float = Field(description="Lower end of the interval on coverage.")
    ci_upper: float = Field(description="Upper end of the interval on coverage.")
    mean_width: float = Field(
        description="Mean interval width over the measured splits, in outcome "
                    "units. The cost side of a variance correction.")


def random_split(rng: np.random.Generator, n_geos: int, n_treated: int,
                 eligible: Optional[Sequence[int]] = None):
    """A random treated/control partition of the geo indices."""
    n_geos = int(n_geos)
    n_treated = int(n_treated)
    if n_treated < 1 or n_treated >= n_geos:
        raise MlsynthConfigError(
            f"n_treated must leave both groups non-empty: got {n_treated} of "
            f"{n_geos} geos.")
    pool = np.arange(n_geos) if eligible is None else np.asarray(eligible, int)
    treated = np.sort(rng.choice(pool, size=n_treated, replace=False))
    control = np.setdiff1d(np.arange(n_geos), treated)
    return treated, control


def beta_interval(n_covered: int, n_draws: int, level: float = 0.90):
    """An interval on a coverage rate, under Kerman's neutral prior.

    Beta(1/3 + y, 1/3 + n - y). A normal approximation collapses at y = 0 and
    y = n, which are exactly the cells a coverage study produces when a design
    is badly calibrated, so those have to come back with real bounds.
    """
    n_draws = int(n_draws)
    n_covered = int(n_covered)
    if n_draws < 1:
        raise MlsynthConfigError(
            f"a coverage interval needs at least one draw; got {n_draws}.")
    if not (0 <= n_covered <= n_draws):
        raise MlsynthConfigError(
            f"{n_covered} covered out of {n_draws} draws is not a count.")
    if not (0.0 < level < 1.0):
        raise MlsynthConfigError(
            f"level must sit strictly inside (0, 1); got {level}.")
    a = NEUTRAL_PRIOR + n_covered
    b = NEUTRAL_PRIOR + n_draws - n_covered
    tail = 0.5 * (1.0 - level)
    return (float(stats.beta.ppf(tail, a, b)),
            float(stats.beta.ppf(1.0 - tail, a, b)))


def coverage_grid(panel: np.ndarray, *, n_treated: int,
                  windows: Sequence[int], levels: Sequence[float],
                  variances: Sequence[str] = ("iid",), reps: int = 200,
                  seed: int = 0, bandwidth: Optional[int] = None,
                  screen=None, ci_level: float = 0.90):
    """Coverage of the A/A interval over random splits of an untreated panel.

    ``panel`` is periods by geos, with no treatment anywhere. Each replication
    draws a treated/control split, sums each group, holds out the last
    ``n_test`` periods and asks whether the interval covered zero.

    The same split sequence is used at every level and variance, so the arms
    are compared on identical designs and a difference between them is the
    variance and not the draw. ``screen`` is an optional
    ``(y, x, n_fit) -> bool`` admitting a candidate, which is where a
    design-time gate goes; rejected splits are counted, never measured.

    Returns one :class:`CoverageCell` per (level, window, variance). No cell
    carries a verdict.
    """
    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2:
        raise MlsynthDataError(
            f"the panel must be periods by geos; got shape {panel.shape}.")
    n_periods, n_geos = panel.shape
    if int(reps) < 1:
        raise MlsynthConfigError(f"reps must be at least one; got {reps}.")

    cells = []
    for n_test in windows:
        n_test = int(n_test)
        n_fit = n_periods - n_test
        # One split sequence per window, shared by every level and variance.
        rng = np.random.default_rng([seed, n_test])
        splits = [random_split(rng, n_geos, n_treated) for _ in range(int(reps))]

        admitted, rejected = [], 0
        for treated, control in splits:
            y = panel[:, treated].sum(axis=1)
            x = panel[:, control].sum(axis=1)
            if screen is not None and not screen(y, x, n_fit):
                rejected += 1
                continue
            admitted.append((y, x))

        for variance in variances:
            for level in levels:
                covered, widths = 0, []
                for y, x in admitted:
                    d = aa_draw(y, x, n_test, level=level, variance=variance,
                                bandwidth=bandwidth,
                                threshold=1.0 - 0.5 * (1.0 - level))
                    covered += int(d.covers_zero)
                    widths.append(d.upper - d.lower)
                n = len(admitted)
                lo, hi = (beta_interval(covered, n, ci_level) if n
                          else (float("nan"), float("nan")))
                cells.append(CoverageCell(
                    level=float(level), n_test=n_test, variance=variance,
                    screened=screen is not None, n_draws=n, n_covered=covered,
                    n_rejected_by_screen=rejected,
                    coverage=(covered / n) if n else float("nan"),
                    ci_lower=lo, ci_upper=hi,
                    mean_width=float(np.mean(widths)) if widths
                               else float("nan")))
    return cells
