"""The two objectives a matched-markets search can climb.

Au (2018) section 3.1 states the objective as

    f = min(CUSUM p-value, Breusch-Godfrey p-value, R^2)

with the CUSUM an OLS-based structural break test (Ploberger and Kramer 1992) on
the fitted TBR model, the Breusch-Godfrey a test for autocorrelation in its
residuals, and R^2 the model's fit. The reference implementation scores

    (corr_test, aa_test, bb_test, dw_test, corr, 1 / required_impact)

sorted lexicographically: four assumption gates, then the correlation between
the group aggregates, then the inverse of the smallest impact the experiment
could detect.

They are closer than their names suggest, and the difference that remains
decides which is the default. Measured over 400 random splits of the GeoLift
markets:

* R^2 and the reference's ``corr`` are the same quantity. A simple regression
  with an intercept has R^2 = corr^2, so they induce identical rankings
  (Spearman 1.000000, max absolute difference 1.8e-15).
* the reference's Brownian-bridge test is an OLS-CUSUM test. It compares the
  cumulative standardised residual against the Brownian bridge's own pointwise
  envelope where the classical statistic compares the supremum against a uniform
  one, so the boundary shape differs and the test does not. They agree on 80.5%
  of splits, and one-sidedly: the reference's boundary is the stricter.
* Breusch-Godfrey and Durbin-Watson agree on 96.2%, Durbin-Watson being the
  AR(1) case of the same question.
* Au's R^2 term is inert. It binds none of the 400 splits, because a p-value
  would have to exceed a pretest R^2 running 0.79 to 0.99 for the minimum to
  land on it, and the largest p-value reached is 0.89.

What is left is power, and only the reference has it. Au's ``f`` is indifferent
to whether a design can detect anything: the reference's ten best designs are
2.5 times better on detectable impact than Au's, and Au's ten best score 0.63 on
``f`` against the reference's 0.29, each objective winning on its own axis. Au's
own section 3.1 asks for a power term -- the p-values "do not assess the amount
of statistical power that TBR provides" -- and then leaves it out of ``f``. So
``reference`` is the default as the paper's stated intent, and ``paper`` is
available for fidelity.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from scipy import special

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.posterior import cumulative_posterior, fit_pretest

BB_BOUND = 3.0                     # the reference's Brownian-bridge constant
DW_RANGE = (1.5, 2.5)              # the reference's acceptable Durbin-Watson band
MIN_CORR = 0.8
SIG_LEVEL = 0.9
POWER_LEVEL = 0.8
FLEVEL = 0.9
AA_THRESHOLD_PROB = 0.2                # the reference's tolerated false-positive rate
MIN_HOLDOUT_PRETEST = 3            # periods the A/A test's own fit needs

#: Terms of the Kolmogorov series, which does not depend on the data.
_KOLMOGOROV_TERMS = np.arange(1, 201)


@dataclass(frozen=True)
class WindowConstants:
    """The quantiles a scoring window fixes, independent of any candidate split.

    Every field is a function of the window length and the planned test length
    alone, so all of them are the same for every split scored on one panel.
    Recomputing them per candidate cost more than the regressions they wrap: one
    ``scipy.stats`` frozen distribution is about 464 microseconds against 12 for
    the least squares, and most of that is scipy constructing an object and its
    docstring. They are computed once per window here instead.
    """

    phi: float          # F(1, n-1) at FLEVEL, the confidence-region bound
    tq_sig: float       # t(n-2) at SIG_LEVEL
    tq_pow: float       # t(n-2) at POWER_LEVEL
    tq_holdout: float   # t(n-n_test-2) at SIG_LEVEL, for the A/A window
    holdout_df: int     # degrees of freedom of the A/A fit


@functools.lru_cache(maxsize=None)
def window_constants(n: int, n_test: int) -> WindowConstants:
    """The quantiles for a window of ``n`` periods and a test of ``n_test``.

    ``scipy.special`` and not ``scipy.stats``: the wrappers broadcast, validate
    and mask for array input, which this never has. The values are identical --
    ``stats.f.ppf`` is ``special.fdtri`` and ``stats.t.ppf`` is
    ``special.stdtrit`` -- at a fortieth of the cost.
    """
    holdout_df = n - n_test - 2
    return WindowConstants(
        phi=float(special.fdtri(1, n - 1, FLEVEL)),
        tq_sig=float(special.stdtrit(n - 2, SIG_LEVEL)),
        tq_pow=float(special.stdtrit(n - 2, POWER_LEVEL)),
        tq_holdout=(float(special.stdtrit(holdout_df, SIG_LEVEL))
                    if holdout_df > 0 else float("nan")),
        holdout_df=holdout_df,
    )


@functools.lru_cache(maxsize=None)
def brownian_bridge_envelope(n: int) -> np.ndarray:
    """The Brownian-bridge boundary for a window of ``n`` periods.

    Read only, because the cache hands the same array to every caller.
    """
    k = np.arange(1, n)
    envelope = BB_BOUND * np.sqrt(k * (1.0 - k / float(n)))
    envelope.setflags(write=False)
    return envelope


@dataclass(frozen=True)
class SplitScore:
    """One candidate split's score, comparable and self-describing."""

    key: Tuple                     # what the search maximises, compared as a tuple
    value: float                   # a scalar readout of the same score
    detail: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "SplitScore") -> bool:
        return self.key < other.key


def _pretest_fit(y: np.ndarray, x: np.ndarray):
    """Eqn 1's residuals, residual scale and fit quality for one candidate split.

    The fit is TBR's own :func:`~mlsynth.utils.tbr_helpers.posterior.fit_pretest`
    and not a second least squares. Every term in both objectives is a
    functional of this regression -- the CUSUM and Breusch-Godfrey gates read
    its residuals, the correlation term is its fit quality, and
    ``required_impact`` is eqn 6's half-width solved for the smallest detectable
    effect -- so the search ranks designs by the model the experiment is later
    analysed with. Sharing the function means a correction to eqn 1 reaches the
    design stage and the analysis stage together.

    Two pretest periods leave ``df = 0``, where no residual scale exists. The
    duplicated fit this replaced returned ``sigma = nan`` with ``r2 = 1.0``
    there, scoring an unfittable window as a perfect one, so the window is
    refused.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 3:
        raise MlsynthDataError(
            f"scoring a split needs at least 3 pretest periods to leave "
            f"degrees of freedom for a residual scale; got {y.size}.")
    fit = fit_pretest(y, x)
    sigma = float(np.sqrt(fit.sigma_sq))
    # ``RSS`` is ``sigma_sq * df`` and ``TSS`` is the fit's own ``S_yy``, so the
    # fit quality needs neither a residual pass nor a second centring of ``y``.
    rss = fit.sigma_sq * fit.df
    r2 = 1.0 - rss / fit.sums.s_yy if fit.sums.s_yy > 0 else 0.0
    return fit, fit.resid, sigma, r2


def _kolmogorov_sf(t: float) -> float:
    """``P(sup |Brownian bridge| > t)``, the OLS-CUSUM null tail."""
    if not np.isfinite(t) or t <= 0.0:
        return 1.0
    j = _KOLMOGOROV_TERMS
    cdf = 1.0 + 2.0 * np.sum((-1.0) ** j * np.exp(-2.0 * j ** 2 * t ** 2))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def cusum_pvalue(resid: np.ndarray, sigma: float) -> float:
    """Ploberger and Kramer's OLS-CUSUM, against the uniform boundary."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 0.0
    stat = float(np.max(np.abs(np.cumsum(resid))) / (sigma * np.sqrt(resid.size)))
    return _kolmogorov_sf(stat)


def breusch_godfrey_pvalue(resid: np.ndarray, x: np.ndarray, lags: int = 1) -> float:
    """LM test for autocorrelation: the auxiliary fit's ``n R^2`` is chi-squared."""
    n = resid.size
    if n <= lags + 3:
        return 1.0
    target = resid[lags:]
    cols = [np.ones(target.size), x[lags:]]
    for k in range(1, lags + 1):
        cols.append(resid[lags - k: n - k])
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    aux = target - design @ coef
    tss = float(np.sum((target - target.mean()) ** 2))
    r2 = 0.0 if tss <= 0 else 1.0 - float(aux @ aux) / tss
    return float(special.chdtrc(lags, target.size * max(r2, 0.0)))


def brownian_bridge_ok(resid: np.ndarray, sigma: float) -> bool:
    """The reference's ``bb_test``: an OLS-CUSUM with a proportional boundary."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        return False
    envelope = brownian_bridge_envelope(resid.size)
    return bool(not np.any(np.abs(np.cumsum(resid / sigma))[:-1] > envelope))


def durbin_watson_ok(resid: np.ndarray) -> Tuple[bool, float]:
    rss = float(resid @ resid)
    if rss <= 0.0:
        return False, np.nan
    d = np.diff(resid)
    stat = float(d @ d) / rss
    return bool(DW_RANGE[0] < stat < DW_RANGE[1]), stat


@dataclass(frozen=True)
class HoldoutFit:
    """TBR's posterior for a held-out window, and the pretest it was fitted on."""

    estimate: float            # Delta(T) over the held-out window
    half_width: float          # the interval's half-width at SIG_LEVEL
    sigma: float               # the pretest residual standard deviation
    n_pretest: int             # periods the fit used
    df: int                    # degrees of freedom of the posterior


def holdout_fit(y: np.ndarray, x: np.ndarray, n_test: int) -> HoldoutFit:
    """Fit on all but the last ``n_test`` periods, then estimate that window.

    This is TBR run on the pretest against itself: eqn 1 on the earlier periods
    and eqns 4 and 6 on the held out ones, through the estimator's own
    :func:`~mlsynth.utils.tbr_helpers.posterior.cumulative_posterior`. The
    window carries no intervention, so the honest answer is zero and anything
    else is the design's own false-positive rate showing.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n_pretest = y.size - n_test
    if n_pretest < MIN_HOLDOUT_PRETEST:
        raise MlsynthDataError(
            f"the A/A test fits on the window before the last {n_test} periods "
            f"and needs at least {MIN_HOLDOUT_PRETEST} of them; a scoring "
            f"window of {y.size} leaves {n_pretest}.")
    fit = fit_pretest(y[:n_pretest], x[:n_pretest])
    loc, scale = cumulative_posterior(fit, y[n_pretest:], x[n_pretest:])
    sigma = float(np.sqrt(fit.sigma_sq))
    half_width = float(window_constants(y.size, n_test).tq_holdout * scale[-1])
    return HoldoutFit(estimate=float(loc[-1]), half_width=half_width,
                      sigma=sigma, n_pretest=n_pretest, df=fit.df)


def false_positive_probability(fit: HoldoutFit, n_test: int) -> float:
    """How often a design like this one calls a null window significant.

    Reached only when the held-out interval excludes zero. The true mean is
    taken at the interval bound nearest zero, which is the most forgiving value
    consistent with the interval, so the probability is a lower bound on how
    often the design would cry wolf.
    """
    lower = fit.estimate - fit.half_width
    upper = fit.estimate + fit.half_width
    true_mean = min(abs(lower), abs(upper))
    tq_sig = fit.half_width / fit.sigma
    posterior_scale = fit.sigma * np.sqrt(1.0 / fit.n_pretest + 1.0 / n_test)
    upper_tail = 1.0 - special.stdtr(fit.df, tq_sig - true_mean / posterior_scale)
    lower_tail = float(special.stdtr(fit.df, -tq_sig - true_mean / posterior_scale))
    return float(upper_tail + lower_tail)


def aa_test_ok(y: np.ndarray, x: np.ndarray, n_test: int) -> bool:
    """The reference's A/A test on the last ``n_test`` pretest periods.

    An interval covering zero is a pass: the design did not find an effect where
    there is none. An interval excluding zero is not an automatic failure, since
    a narrow interval sitting just off zero is a small error; it fails when the
    probability of that happening exceeds ``AA_THRESHOLD_PROB``.
    """
    fit = holdout_fit(y, x, n_test)
    lower = fit.estimate - fit.half_width
    upper = fit.estimate + fit.half_width
    if lower * upper < 0.0:
        return True
    return bool(false_positive_probability(fit, n_test) <= AA_THRESHOLD_PROB)

def required_impact(sigma: float, n: int, n_test: int) -> float:
    """The smallest impact the experiment could detect, the reference's formula.

    The reference writes eqn 6's residual scale as
    ``std(y, ddof=2) * sqrt(1 - corr^2)``. For a simple regression that is
    ``sqrt(sigma_sq)``, since ``RSS = TSS (1 - R^2)`` and ``corr^2 = R^2``, and
    the pretest fit already holds it. Taking it from there computes neither the
    variance nor the correlation a second time.
    """
    const = window_constants(n, n_test)
    phi, tq_sig, tq_pow = const.phi, const.tq_sig, const.tq_pow
    sq = np.sqrt(phi * (n + 1) / (n * n_test * (n - 1)) + 1.0 / n + 1.0 / n_test)
    return float((tq_sig + tq_pow) * n_test * sq * float(sigma))


def score_split(y: np.ndarray, x: np.ndarray, *, objective: str,
                n_test: int) -> SplitScore:
    """Score one candidate split under the named objective."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    fit, resid, sigma, r2 = _pretest_fit(y, x)

    if objective == "paper":
        p_cusum = cusum_pvalue(resid, sigma)
        p_bg = breusch_godfrey_pvalue(resid, x)
        f = float(min(p_cusum, p_bg, r2))
        return SplitScore(key=(f,), value=f,
                          detail={"p_cusum": p_cusum, "p_bg": p_bg, "r2": r2,
                                  "binding": ("cusum" if p_cusum == f else
                                              "bg" if p_bg == f else "r2")})

    sums = fit.sums
    corr = (float(sums.s_xy / np.sqrt(sums.s_xx * sums.s_yy))
            if sums.s_xx > 0.0 and sums.s_yy > 0.0 else 0.0)
    corr = float(np.clip(corr, -0.999999, 0.999999))
    impact = required_impact(sigma, sums.n, n_test)
    dw_ok, dw_stat = durbin_watson_ok(resid)
    gates = (corr >= MIN_CORR, aa_test_ok(y, x, n_test),
             brownian_bridge_ok(resid, sigma), dw_ok)
    inv_impact = 1.0 / impact if impact > 0 else np.inf
    return SplitScore(
        key=(int(gates[0]), int(gates[1]), int(gates[2]), int(gates[3]),
             round(corr, 2), inv_impact),
        value=inv_impact,
        detail={"corr_test": gates[0], "aa_test": gates[1], "bb_test": gates[2],
                "dw_test": gates[3], "corr": corr, "r2": r2,
                "required_impact": impact, "dw_stat": dw_stat,
                "gates_passed": int(sum(gates))})
