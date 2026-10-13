r"""What TBR assumes, and whether the panel it was handed bears it out.

Assumption 1's testable half and assumptions 2 and 4 on :doc:`the estimator's
page <../../docs/tbr>` are checkable from the data the fit already has.

Assumption 1 is that the treated aggregate's untreated path is the fitted
affine function of the control aggregate. Li and Van den Bulte (2022) section
2.2 splits it into a part that holds in the pretest, which is testable, and a
part that continues into the test window, which is not, because the treated
group's untreated path stops existing once the treatment lands. Under the
one-factor model of Li (2024) web appendix A the correlation between the two
series does not depend on time, so the first part reduces to parallel
pre-trends, and web appendix A states it as the difference being stationary
with a constant mean. Two checks address that part. ``backdating`` splits the
pretest, fits on the front and scores the tail, which is Li (2024) section
3.2's exercise; ``stationary_residual`` runs Engle-Granger on the pair.

Assumption 2, that the pretest residuals are independent, identically
distributed and normal, has three parts and each fails differently. Serial
correlation makes the interval too narrow, because the effective sample size is
smaller than the period count. Non-normality breaks the t posterior, which
matters at the pretest lengths TBR is used at and not asymptotically.
Heteroskedasticity moves the scale without moving the point estimate.

Assumption 4, that aggregation is over a fixed set of geos, fails in two ways
that look different in the data. A filled cell enters the group total as a
zero, so fills landing unevenly across the treatment boundary are confounded
with the effect; and a geo present for only part of the panel changes what the
totals mean from one period to the next.

What these checks do not do
---------------------------

They do not license the estimate. Everything here is the testable part, and a
panel that passes every check can still carry an estimate that is wrong because
the relation stopped holding after the treatment landed. Nothing computed from
the pretest reaches that.

Each check sees one failure and not its neighbours. Breusch-Pagan tests whether
the residual spread depends on the control aggregate, so spread that drifts
with time while the control aggregate stays put passes it. Breusch-Godfrey is
run at one lag, so correlation at a seasonal lag passes it. Backdating scores a
tail of the pretest, so a relation that held through the whole pretest and
broke at the boundary passes it. None of this is a reason to distrust what they
do report, and all of it is a reason not to read a pass as "the assumptions
hold".

How often they fire, measured
-----------------------------

Five of the seven checks are hypothesis tests read at the 5% level. Over 200
simulated sound panels per cell, the share firing:

    pretest periods          20      30      40      60
    backdating            0.060   0.055   0.050   0.060
    stationary_residual      nv   0.105   0.100   0.065
    serial_correlation    0.055   0.095   0.075   0.060
    normality             0.055   0.025   0.060   0.045
    homoskedasticity      0.050   0.045   0.045   0.050
    at least one          0.185   0.285   0.285   0.240

The last row is 1 - 0.95^5 and not a defect. It is the reason ``flagged`` names
the checks instead of reducing to one verdict, and the reason the warning is
grouped by consequence: on a sound panel something fires about one run in four,
and which one is the whole information. Engle-Granger reports no verdict at 20
periods and stays oversized at 30 and 40.

Against panels that really do violate, the share that says so:

    pretest periods             20      40      80
    AR(1) errors, rho 0.7    0.580   0.960   1.000
    t(2) errors              0.335   0.455   0.775
    spread varying with x    0.145   0.290   0.525

So serial correlation is caught reliably from about 40 pretest periods, and
the other two are not. At the lengths TBR is used at a passing normality or
homoskedasticity check is weak evidence, and these numbers are the reason the
passing text says "not detected at this length" and never that the assumption
holds. A check reports ``holds=None`` where the window is too short for the
test to mean anything at all, which is silence and not a pass.

The identification checks are measured against the violation Li (2024)
describes operationally, a treated series trending away from every control with
a true effect of zero. At 20, 30 and 40 pretest periods ``backdating`` catches
0.930, 0.925 and 0.915, and ``stationary_residual`` 0.000, 0.960 and 0.980.

The statistics' second home
---------------------------

``design/objective.py`` carries its own Durbin-Watson and Breusch-Godfrey for
the hill climb's design-time gates, which score candidate splits before an
experiment runs. These are the analysis-time versions, reporting a statistic
and not a gate. The two are independent implementations of the same
textbook quantities, which is a duplication to consolidate once the two callers
can share a module.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import special, stats

from .posterior import cumulative_posterior, fit_pretest

#: Significance level every hypothesis test here is read at.
ALPHA = 0.05
#: The band Au's design-time gate screens candidate splits on. It is reported
#: for continuity with that gate and does not decide this check: read as a
#: hypothesis test it is heavily oversized at short pretests.
DW_BAND = (1.5, 2.5)
#: Held-out periods below which the backdating exercise reports no verdict. The
#: question is whether the relation predicts a window, and one or two periods
#: do not make one: the F reference stays honest but has almost no power.
MIN_HOLD = 3
#: Pretest periods below which the Engle-Granger check reports no verdict. At
#: 20 it fails to establish stationarity on every sound panel measured, so
#: below this it would flag everything. Over 200 sound panels per cell it fires
#: on 0.105 at 30 periods, 0.100 at 40 and 0.065 at 60, so it stays oversized
#: at the shorter lengths, which is a small-sample property of the
#: Engle-Granger critical values.
MIN_PERIODS_STATIONARITY = 30
#: Pretest periods below which a check reports no verdict. The t posterior is
#: defined on three, but a distributional test on a handful of residuals
#: detects nothing, and reporting that as a pass would be worse than silence.
MIN_PERIODS = 8


class AssumptionCheck(BaseModel):
    """One check, its number, and what the number means."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="The check's name, as used in `flagged`.")
    statistic: Optional[float] = Field(
        default=None, description="The test statistic, or the count the check "
                                  "counts. None when the check did not run.")
    pvalue: Optional[float] = Field(
        default=None, description="The test's p-value, where it has one.")
    threshold: Optional[float] = Field(
        default=None, description="What the statistic or p-value was read "
                                  "against.")
    holds: Optional[bool] = Field(
        default=None, description="True when the check passed, False when it "
                                  "fired, and None when the window was too "
                                  "short to say. None is silence, not a pass.")
    detail: str = Field(
        ..., description="What the check found, and what to do about it.")


class AssumptionChecks(BaseModel):
    """The checks TBR can make on the panel it was handed.

    The two identification checks come first because they speak to the relation
    the method rests on. The three residual checks follow: they are about the
    conditions the posterior needs, and a panel can pass all three while the
    relation itself does not hold.
    """

    model_config = ConfigDict(frozen=True)

    backdating: AssumptionCheck
    stationary_residual: AssumptionCheck
    serial_correlation: AssumptionCheck
    normality: AssumptionCheck
    homoskedasticity: AssumptionCheck
    balanced_panel: AssumptionCheck
    stable_membership: AssumptionCheck
    flagged: List[str] = Field(
        default_factory=list,
        description="Names of the checks that fired, sorted. A check that "
                    "could not run is absent from this list and is not a pass.")

    def all_checks(self) -> List[AssumptionCheck]:
        return [self.backdating, self.stationary_residual,
                self.serial_correlation, self.normality, self.homoskedasticity,
                self.balanced_panel, self.stable_membership]


# --------------------------------------------------------------------- statistics
def durbin_watson(resid: np.ndarray) -> float:
    """``sum (e_t - e_{t-1})^2 / sum e_t^2``. Two is no first-order correlation."""
    rss = float(resid @ resid)
    if rss <= 0.0:
        return float("nan")
    d = np.diff(resid)
    return float(d @ d) / rss


def breusch_godfrey_pvalue(resid: np.ndarray, x: np.ndarray,
                           lags: int = 1) -> float:
    """LM test for autocorrelation: the auxiliary fit's ``n R^2`` is chi-squared."""
    n = resid.size
    if n <= lags + 3:
        return float("nan")
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


def breusch_pagan_pvalue(resid: np.ndarray, x: np.ndarray) -> float:
    """LM test for heteroskedasticity: squared residuals regressed on ``x``."""
    n = resid.size
    if n <= 4:
        return float("nan")
    sq = resid ** 2
    mean_sq = float(sq.mean())
    if mean_sq <= 0.0:
        return float("nan")
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, sq, rcond=None)
    fitted = design @ coef
    tss = float(np.sum((sq - mean_sq) ** 2))
    if tss <= 0.0:
        return float("nan")
    ess = float(np.sum((fitted - mean_sq) ** 2))
    # The LM statistic is n R^2 of the auxiliary regression of the squared
    # residuals on x, which is chi-squared on one degree of freedom here
    # because x is the single control aggregate.
    return float(special.chdtrc(1, n * ess / tss))


def shapiro_pvalue(resid: np.ndarray) -> float:
    """Shapiro-Wilk on the residuals. Needs a handful of points to say anything."""
    if resid.size < 3:
        return float("nan")
    spread = float(np.max(resid) - np.min(resid))
    if not np.isfinite(spread) or spread <= 0.0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.shapiro(resid).pvalue)


def _pretest_residuals(report) -> np.ndarray:
    """The fit's pretest residuals, read off a result."""
    ts = report.time_series
    observed = np.asarray(ts.observed_outcome, dtype=float).ravel()
    counterfactual = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    n = int(report.tbr_fit.n_pretest)
    return observed[:n] - counterfactual[:n]


def _filled_total(check: AssumptionCheck) -> int:
    """The filled-cell count carried by the ``balanced_panel`` check."""
    return 0 if check.statistic is None else int(round(check.statistic))


# ------------------------------------------------------------- assumption 2 checks
def _too_short(name: str, n: int) -> AssumptionCheck:
    return AssumptionCheck(
        name=name, holds=None,
        detail=f"the pretest window is too short to judge this: {n} period(s) "
               f"against the {MIN_PERIODS} this check needs. No verdict, which "
               f"is not a pass.")


def _no_variation(name: str) -> AssumptionCheck:
    return AssumptionCheck(
        name=name, holds=None,
        detail="the pretest residuals have no variation, so there is nothing "
               "for this check to test. That happens when the pretest design "
               "is rank deficient, which `tbr_fit.rank_deficient` reports. No "
               "verdict, which is not a pass.")


def _residual_checks(resid: np.ndarray, x_pre: np.ndarray
                     ) -> Tuple[AssumptionCheck, AssumptionCheck, AssumptionCheck]:
    n = resid.size
    if n < MIN_PERIODS or not np.isfinite(resid).all():
        return (_too_short("serial_correlation", n),
                _too_short("normality", n),
                _too_short("homoskedasticity", n))
    # A pretest that fits exactly leaves nothing to test, and every statistic
    # below would read it as clean: Breusch-Godfrey returns p = 1 on residuals
    # that are identically zero, which is a perfect score for a fit that said
    # nothing. Silence is the honest answer.
    if float(resid @ resid) <= 0.0:
        return (_no_variation("serial_correlation"),
                _no_variation("normality"),
                _no_variation("homoskedasticity"))

    dw = durbin_watson(resid)
    bg = breusch_godfrey_pvalue(resid, x_pre)
    # Breusch-Godfrey decides and Durbin-Watson is reported beside it. The band
    # DW_BAND is Au's screening rule for ranking candidate designs, not a
    # size-alpha test, and read as one it is badly oversized at the pretest
    # lengths TBR is used at: on sound panels it fires on 38% of them at 12
    # pretest periods and 23% at 20, against the 5% a test should hold to.
    correlated = bool(np.isfinite(bg) and bg < ALPHA)
    serial = AssumptionCheck(
        name="serial_correlation", statistic=dw,
        pvalue=None if not np.isfinite(bg) else bg, threshold=ALPHA,
        holds=not correlated,
        detail=(f"Breusch-Godfrey p {bg:.3g}, Durbin-Watson {dw:.3f} "
                f"(2 is no first-order correlation). "
                + ("Correlated pretest residuals make the posterior interval "
                   "too narrow, because the effective sample size is smaller "
                   "than the period count. Refit with variance='hac' to price "
                   "the interval on a Newey-West scale."
                   if correlated else
                   "No first-order correlation detected at this length.")))

    sh = shapiro_pvalue(resid)
    normal = not (np.isfinite(sh) and sh < ALPHA)
    normality = AssumptionCheck(
        name="normality", statistic=None,
        pvalue=None if not np.isfinite(sh) else sh, threshold=ALPHA,
        holds=normal,
        detail=(f"Shapiro-Wilk p {sh:.3g}. "
                + ("The posterior is a t on n-2 degrees of freedom, which "
                   "rests on normal pretest errors; at this pretest length "
                   "that is not an asymptotic argument. Check the pretest for "
                   "a spike or a level shift."
                   if not normal else
                   "No departure from normality detected at this length.")))

    bp = breusch_pagan_pvalue(resid, x_pre)
    homosk = not (np.isfinite(bp) and bp < ALPHA)
    hetero = AssumptionCheck(
        name="homoskedasticity", statistic=None,
        pvalue=None if not np.isfinite(bp) else bp, threshold=ALPHA,
        holds=homosk,
        detail=(f"Breusch-Pagan p {bp:.3g}. "
                + ("The residual spread moves with the control aggregate, so "
                   "equation 6's single s misprices the interval. The point "
                   "estimate is unaffected; variance='hac' is robust to this "
                   "as well as to correlation."
                   if not homosk else
                   "No dependence of the residual spread on the control "
                   "aggregate detected, which at this pretest length is weak "
                   "evidence: the check finds a sixfold spread gradient about "
                   "a third of the time at 40 periods.")))
    return serial, normality, hetero


# --------------------------------------------- assumption 1: the testable half
def backdating(y: np.ndarray, x: np.ndarray, n_pre: int, hold: int):
    """Li (2024) section 3.2's exercise, on the one control aggregate.

    Split the pretest, fit on the front, predict the tail, and look at what the
    prediction cost. Nothing is held out of the real test window, so this is a
    statement about the pretest and not about what follows it.

    Returns ``(statistic, in_sample_rmse, held_out_rmse, did_ratio)`` where the
    statistic is the mean squared standardised prediction error. Each held-out
    error is divided by its own prediction scale, equation 6 at a horizon of
    one, so a period the fit was always going to find hard is not counted as
    evidence against the model. Under a model that holds, each standardised
    error is a t on the backdated fit's degrees of freedom and the mean square
    sits near one.

    ``did_ratio`` is the same exercise with the slope forced to one, which is
    difference-in-differences and which TBR nests, divided by TBR's. Above one
    the slope adjustment paid for itself out of sample; near one on a large
    statistic, neither model traces the treated series and the problem is not
    which of the two to use.
    """
    T0 = n_pre - hold
    fit = fit_pretest(y[:T0], x[:T0])
    in_sample = y[:T0] - (fit.alpha + fit.beta * x[:T0])
    if float(in_sample @ in_sample) <= 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    z, err = [], []
    for t in range(T0, n_pre):
        loc, scale = cumulative_posterior(fit, y[t:t + 1], x[t:t + 1])
        if not np.isfinite(scale[-1]) or scale[-1] <= 0.0:   # pragma: no cover
            # Defensive. A front window the fit reproduces exactly is caught
            # above, and a rank-deficient fit still returns a finite positive
            # scale (measured: constant x over 32 periods gives 4.19), so
            # nothing known reaches this.
            return float("nan"), float("nan"), float("nan"), float("nan")
        z.append(float(loc[-1]) / float(scale[-1]))
        err.append(float(loc[-1]))
    alpha_did = float(np.mean(y[:T0] - x[:T0]))
    did_err = y[T0:n_pre] - (alpha_did + x[T0:n_pre])
    held = float(np.sqrt(np.mean(np.asarray(err) ** 2)))
    did = float(np.sqrt(np.mean(did_err ** 2)))
    in_rmse = float(np.sqrt(np.mean(in_sample ** 2)))
    # Both are RMSEs in the outcome's units, so the denominator is judged
    # against the in-sample one. A held-out error eight orders below it means
    # TBR reproduced the window to floating point and the comparison is
    # vacuous: testing held > 0 alone passes a denominator of 1e-16 and reports
    # a ratio of 2e15.
    defined = held > 1e-8 * in_rmse
    return (float(np.mean(np.asarray(z) ** 2)), in_rmse, held,
            did / held if defined else float("nan"))


def stationary_residual_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    """Engle-Granger on the pretest pair: can the residual be called stationary?

    Definition 1 of Li (2024) web appendix A states the identifying assumption
    as the treated series minus the control average being a zero-mean,
    finite-variance stationary process. The levels may trend; the difference
    may not, which is cointegration.

    The test is run through ``coint`` and not by putting an augmented
    Dickey-Fuller on the fitted residuals, because the residuals come from an
    estimated relation and the Dickey-Fuller critical values do not hold for
    them. Small p rejects the unit root, which is the outcome the assumption
    wants.
    """
    if y.size < 12:
        return float("nan")
    try:
        from statsmodels.tsa.stattools import coint
    except Exception:                 # pragma: no cover - statsmodels is a dependency
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(coint(y, x, trend="c")[1])
        except Exception:             # pragma: no cover - degenerate input
            return float("nan")


def _identification_checks(y: np.ndarray, x: np.ndarray, n_pre: int,
                           n_test: int) -> Tuple[AssumptionCheck, AssumptionCheck]:
    """The two checks that speak to the relation itself and not to its errors."""
    # Hold out as many pretest periods as the test window is long, so the
    # backdated prediction is the one the estimate actually makes.
    hold = int(min(max(n_test, 1), max(n_pre - MIN_PERIODS, 0)))
    T0 = n_pre - hold
    if hold < MIN_HOLD or T0 < MIN_PERIODS:
        back = AssumptionCheck(
            name="backdating", holds=None,
            detail=f"the pretest cannot be split into a window long enough "
                   f"to score: {n_pre} period(s) leaves {T0} to fit on "
                   f"after holding out {hold}, against the {MIN_PERIODS} a fit "
                   f"needs and the {MIN_HOLD} a held-out window needs. No "
                   f"verdict, which is not a pass.")
    else:
        stat, in_rmse, held_rmse, did_ratio = backdating(y, x, n_pre, hold)
        if not np.isfinite(stat):
            back = AssumptionCheck(
                name="backdating", holds=None,
                detail="the backdated fit is degenerate, so the held-out "
                       "errors have no scale to be judged against. No verdict.")
        else:
            # Each standardised error is a t on the backdated fit's degrees of
            # freedom, so their mean square is an F. Measured against sound
            # panels the match is close: the 95th percentile came out 2.46
            # against F's 2.45 at one configuration and 2.32 against 2.27 at
            # another, which is why the threshold is a distribution and not a
            # number chosen by hand.
            pval = float(stats.f.sf(stat, hold, max(T0 - 2, 1)))
            predicts = pval >= ALPHA
            back = AssumptionCheck(
                name="backdating", statistic=stat, pvalue=pval,
                threshold=ALPHA, holds=predicts,
                detail=(f"held out the last {hold} of {n_pre} pretest periods: "
                        f"RMSE {held_rmse:.4g} against {in_rmse:.4g} in "
                        f"sample, mean squared standardised error {stat:.3g} "
                        f"(p {pval:.3g})"
                        + (f"; difference-in-differences on the same window is "
                           f"{did_ratio:.3g} times TBR's error. "
                           if np.isfinite(did_ratio) else ". ")
                        + ("The fitted relation did not survive being asked to "
                           "predict a window it was not fitted on, so the "
                           "counterfactual over the real test window is not to "
                           "be relied on. This is the assumption the method "
                           "rests on and no variance correction addresses it; "
                           "the group split is the thing to revisit."
                           if not predicts else
                           "The fitted relation predicted a window it was not "
                           "fitted on.")))

    if n_pre < MIN_PERIODS_STATIONARITY:
        stat_check = AssumptionCheck(
            name="stationary_residual", holds=None,
            detail=f"{n_pre} pretest period(s) is below the "
                   f"{MIN_PERIODS_STATIONARITY} this test needs to say "
                   f"anything: at 20 it fails to establish stationarity on "
                   f"every sound panel measured. No verdict, which is not a "
                   f"pass.")
    else:
        pval = stationary_residual_pvalue(y[:n_pre], x[:n_pre])
        established = bool(np.isfinite(pval) and pval < ALPHA)
        stat_check = AssumptionCheck(
            name="stationary_residual", pvalue=None if not np.isfinite(pval) else pval,
            threshold=ALPHA, holds=established,
            detail=(f"Engle-Granger p {pval:.3g}. "
                    + ("Stationarity of the treated-minus-control relation "
                       "could not be established, which is what Li's "
                       "Definition 1 asks for. The levels may trend; their "
                       "fitted difference may not. This is a failure to "
                       "reject and so is weaker evidence than the other "
                       "checks: on sound panels it lands here 7% of the time "
                       "at 40 pretest periods and 12% at 30."
                       if not established else
                       "The fitted difference rejects a unit root, so the "
                       "relation is stable in the pretest in the sense "
                       "Definition 1 asks for.")))
    return back, stat_check


# ------------------------------------------------------------- assumption 4 checks
def _panel_checks(df: pd.DataFrame, unit: str, time: str, n_pre: int,
                  periods: Sequence) -> Tuple[AssumptionCheck, AssumptionCheck]:
    """Absent cells, and geos that are present for only part of the panel.

    The absent set is recomputed here instead of being carried out of ingestion. The
    count has to match what ``filled_cells`` reports, and a test holds the two
    in step so the second source of truth cannot drift unnoticed.
    """
    units = sorted(df[unit].unique())
    all_periods = sorted(df[time].unique())
    present = set(zip(df[unit], df[time]))
    pre_set = set(all_periods[:n_pre])

    missing = [(u, t) for u in units for t in all_periods
               if (u, t) not in present]
    n_missing = len(missing)
    n_pre_cells = len(units) * n_pre
    n_post_cells = len(units) * (len(all_periods) - n_pre)
    miss_pre = sum(1 for _, t in missing if t in pre_set)
    miss_post = n_missing - miss_pre

    if n_missing == 0:
        balanced = AssumptionCheck(
            name="balanced_panel", statistic=0.0, holds=True,
            detail="every geo-period cell is present; nothing was filled.")
    else:
        table = [[miss_pre, max(n_pre_cells - miss_pre, 0)],
                 [miss_post, max(n_post_cells - miss_post, 0)]]
        p = float(stats.fisher_exact(table).pvalue) if n_post_cells else float("nan")
        even = not (np.isfinite(p) and p < ALPHA)
        rate_pre = miss_pre / n_pre_cells if n_pre_cells else float("nan")
        rate_post = miss_post / n_post_cells if n_post_cells else float("nan")
        balanced = AssumptionCheck(
            name="balanced_panel", statistic=float(n_missing),
            pvalue=None if not np.isfinite(p) else p, threshold=ALPHA,
            holds=even,
            detail=(f"{n_missing} absent cell(s) filled with zero: "
                    f"{miss_pre} in the pretest ({rate_pre:.2%} of its cells) "
                    f"and {miss_post} in the test window ({rate_post:.2%}). "
                    + ("The fills fall unevenly across the treatment boundary, "
                       "and a filled cell enters the group total as a zero, so "
                       "the imbalance is confounded with the effect. Check "
                       "whether the absent cells are zero-valued or missing."
                       if not even else
                       "The fills are spread evenly across the boundary.")))

    span = {u: (df.loc[df[unit] == u, time].min(),
                df.loc[df[unit] == u, time].max()) for u in units}
    first, last = all_periods[0], all_periods[-1]
    partial = sorted(u for u in units
                     if span[u][0] != first or span[u][1] != last)
    if not partial:
        stable = AssumptionCheck(
            name="stable_membership", statistic=0.0, holds=True,
            detail="every geo spans the whole panel.")
    else:
        named = ", ".join(str(u) for u in partial[:5])
        stable = AssumptionCheck(
            name="stable_membership", statistic=float(len(partial)), holds=False,
            detail=(f"{len(partial)} geo(s) are present for only part of the "
                    f"panel: {named}"
                    + (", ..." if len(partial) > 5 else "")
                    + ". Aggregation is over a fixed set, so a geo that "
                      "arrives or leaves changes what the totals mean between "
                      "one period and the next. Drop it from the panel or "
                      "assign it to neither group."))
    return balanced, stable


# -------------------------------------------------------------------- the entry point
#: What a fired check costs, grouped by consequence. A check on the fitted
#: relation bears on the counterfactual itself; a check on the residual
#: distribution bears on the posterior built from it, since equation 6 scales a
#: flat-prior t on n-2 degrees of freedom and assumes nothing else; a check on
#: the panel bears on both. Grouping the warning by consequence keeps a nuisance
#: check from reading like a failure of identification. Five checks sized at 5
#: per cent fire on about a quarter of sound panels between them, so one warning
#: per check would train the reader to ignore all of them.
_CONSEQUENCE = (
    (("backdating", "stationary_residual"),
     "The fitted relation does not describe the pretest, so the "
     "counterfactual it extrapolates may be biased and the effect with it."),
    (("serial_correlation", "normality", "homoskedasticity"),
     "The point estimate is unaffected; the posterior interval around it is "
     "built from a single scale on a flat-prior t and may be miscalibrated."),
    (("balanced_panel", "stable_membership"),
     "The aggregates are taken over a changing set of units, which moves the "
     "estimate and its interval together."),
)


def run_checks(config, inputs, fit, n_pre: int, *, warn: bool = True
               ) -> AssumptionChecks:
    """Every check TBR can make, and a warning for each one that fires."""
    y, x = np.asarray(inputs.y, dtype=float), np.asarray(inputs.x, dtype=float)
    resid = y[:n_pre] - (fit.alpha + fit.beta * x[:n_pre])
    back, stationary = _identification_checks(y, x, n_pre, int(inputs.n_test))
    serial, normality, hetero = _residual_checks(resid, x[:n_pre])
    balanced, stable = _panel_checks(
        config.df, config.unitid, config.time, n_pre, inputs.time_labels)

    checks = AssumptionChecks(
        backdating=back, stationary_residual=stationary,
        serial_correlation=serial, normality=normality,
        homoskedasticity=hetero, balanced_panel=balanced,
        stable_membership=stable,
        flagged=sorted(c.name for c in (back, stationary, serial, normality,
                                        hetero, balanced, stable)
                       if c.holds is False))
    if warn:
        for group, consequence in _CONSEQUENCE:
            fired = [c for c in checks.all_checks()
                     if c.holds is False and c.name in group]
            if not fired:
                continue
            warnings.warn(
                "TBR: " + ", ".join(c.name.replace("_", " ") for c in fired)
                + (" fired. " if len(fired) == 1 else " fired. ")
                + consequence + " "
                + " ".join(c.detail for c in fired)
                + " These are checks on the pretest only; they say nothing "
                "about whether the fitted relation continues through the "
                "test window.",
                UserWarning, stacklevel=3)
    return checks
