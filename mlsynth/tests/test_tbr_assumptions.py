r"""TBR says what it assumes, and whether the panel bears it out.

Three of TBR's four assumptions are checkable from the data it was handed, one
of them only in half.

Assumption 1 is that the two group totals are linearly related and that the
relation is stable. Its pretest half is testable two ways. Li (2024) section
3.2 backdates: split the pretest, fit on the front, predict the tail, and see
what the prediction cost. Her web appendix Definition 1 states the assumption
as the fitted difference being stationary, which Engle-Granger tests. Neither
reaches the half that continues into the test window.

Assumption 2 is that the pretest residuals are independent, identically
distributed and normal. All three parts are testable and each fails
differently: serial correlation narrows the interval, non-normality breaks the
t posterior at the pretest lengths TBR is used at, and heteroskedasticity moves
the scale.

Assumption 4 is that aggregation is over a fixed set of geos. A filled cell
enters the group total as a zero, which is right when the cell is absent
because nothing happened and wrong when it is absent because the datum is. Two
things make that dangerous: fills landing unevenly across the treatment
boundary, which is confounded with the effect, and a geo that is present for
part of the panel, which changes what the totals mean.

What the checks do not do is license the estimate. Li and Van den Bulte split
parallel trends into a part that holds in the pretest, which is testable, and a
part that continues into the test window, which is not. Everything here is the
first part. A panel that passes every check can still carry an estimate that is
wrong for the second reason, and the warnings are worded so as not to suggest
otherwise.

Levels: smoke, unit invariants, a property over the domain, edge cases,
failure.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.utils.tbr_helpers import diagnostics as dg
from mlsynth.utils.tbr_helpers import posterior as pst

N_PRE, N_POST, N_GEOS = 40, 8, 8
TREATED = {0, 1, 2}


def _panel(n_pre=N_PRE, n_post=N_POST, n_geos=N_GEOS, seed=0, rho=0.0,
           drop=None, effect=0.0):
    """A clean panel, optionally with AR(1) noise or absent cells."""
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) + 80.0
    level = rng.uniform(10.0, 20.0, n_geos)
    load = rng.uniform(0.8, 1.2, n_geos)
    noise = rng.normal(0, 1.0, (T, n_geos))
    if rho:                                     # serially correlated errors
        for t in range(1, T):
            noise[t] += rho * noise[t - 1]
    Y = level[None, :] + load[None, :] * factor[:, None] + noise
    for j in TREATED:
        Y[n_pre:, j] += effect
    rows = [{"geo": f"g{j:02d}", "t": t, "sales": float(Y[t, j]),
             "post": int(t >= n_pre), "is_treat": int(j in TREATED),
             "is_ctrl": int(j not in TREATED)}
            for j in range(n_geos) for t in range(T)]
    df = pd.DataFrame(rows)
    if drop is not None:
        df = df[~df.apply(lambda r: (r["geo"], r["t"]) in drop, axis=1)]
    return df.reset_index(drop=True)


def _aggregates(df):
    """The treated and control totals the fit is built on."""
    wide = df.pivot_table(index="t", columns="geo", values="sales")
    tr = sorted(df[df.is_treat == 1].geo.unique())
    ct = sorted(df[df.is_ctrl == 1].geo.unique())
    return wide[tr].sum(axis=1).to_numpy(), wide[ct].sum(axis=1).to_numpy()


def _fit(df, **kw):
    base = dict(df=df, outcome="sales", unitid="geo", time="t",
                treatment_col="is_treat", control_col="is_ctrl",
                post_col="post")
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBR(TBRConfig(**base)).fit().report


# --------------------------------------------------------------------------- smoke
def test_a_fit_reports_every_check():
    a = _fit(_panel()).assumptions
    assert a is not None
    for name in ("backdating", "stationary_residual", "serial_correlation",
                 "normality", "homoskedasticity", "balanced_panel",
                 "stable_membership"):
        check = getattr(a, name)
        assert check.name == name
        assert isinstance(check.detail, str) and check.detail


def test_a_clean_panel_flags_nothing():
    a = _fit(_panel()).assumptions
    assert a.flagged == [], a.flagged


# ----------------------------------------------- assumption 2: the residual checks
def test_serial_correlation_is_found_when_it_is_there():
    clean = _fit(_panel(rho=0.0)).assumptions.serial_correlation
    corr = _fit(_panel(rho=0.8)).assumptions.serial_correlation
    assert clean.holds is True
    assert corr.holds is False
    assert corr.statistic < clean.statistic      # Durbin-Watson falls

def test_the_serial_correlation_warning_names_the_remedy():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TBR(TBRConfig(df=_panel(rho=0.8), outcome="sales", unitid="geo",
                      time="t", treatment_col="is_treat",
                      control_col="is_ctrl", post_col="post")).fit()
    text = " ".join(str(w.message) for w in caught)
    assert "serial" in text.lower()
    assert "hac" in text.lower()                 # the remedy the library has


def test_the_durbin_watson_statistic_is_the_textbook_one():
    df = _panel()
    rep = _fit(df)
    resid = dg._pretest_residuals(rep)
    d = np.diff(resid)
    assert rep.assumptions.serial_correlation.statistic == pytest.approx(
        float(d @ d) / float(resid @ resid), rel=1e-12)


def test_non_normal_residuals_are_found():
    """A heavy contaminant in the pretest is what Shapiro-Wilk is for."""
    df = _panel()
    pre = df[(df["t"] < 5) & (df["is_treat"] == 1)].index
    df.loc[pre, "sales"] *= 3.0
    assert _fit(df).assumptions.normality.holds is False


def test_heteroskedastic_residuals_are_found():
    """Residual spread growing with the control aggregate, which is what
    Breusch-Pagan tests. Spread that drifts with time while the control
    aggregate stays put is a different failure and this check does not see it;
    the module docstring says so."""
    rng = np.random.default_rng(3)
    df = _panel()
    control = (df[df["is_ctrl"] == 1].groupby("t")["sales"].sum())
    control = (control - control.min()) / (control.max() - control.min())
    for t in range(N_PRE):
        m = (df["t"] == t) & (df["is_treat"] == 1)
        df.loc[m, "sales"] += rng.normal(0, 0.3 + 8.0 * control[t], int(m.sum()))
    assert _fit(df).assumptions.homoskedasticity.holds is False


# -------------------------------------------------- assumption 4: the panel checks
def test_fills_concentrated_after_the_boundary_are_flagged():
    """A fill is a zero, so fills in the test window look like the effect."""
    drop = {(f"g{j:02d}", t) for j in range(4) for t in range(N_PRE, N_PRE + 6)}
    a = _fit(_panel(drop=drop)).assumptions
    assert a.balanced_panel.holds is False
    assert "balanced_panel" in a.flagged


def test_a_few_fills_spread_evenly_are_not_flagged():
    drop = {("g00", 3), ("g04", 7), ("g02", N_PRE + 1), ("g06", N_PRE + 4)}
    assert _fit(_panel(drop=drop)).assumptions.balanced_panel.holds is True


def test_a_geo_that_arrives_late_is_named():
    drop = {("g05", t) for t in range(0, 12)}
    check = _fit(_panel(drop=drop)).assumptions.stable_membership
    assert check.holds is False
    assert "g05" in check.detail


def test_a_geo_that_leaves_early_is_named():
    drop = {("g06", t) for t in range(N_PRE + 2, N_PRE + N_POST)}
    check = _fit(_panel(drop=drop)).assumptions.stable_membership
    assert check.holds is False
    assert "g06" in check.detail


def test_the_fill_count_agrees_with_what_ingestion_counted():
    """Two code paths count the absent cells; a test keeps them in step."""
    drop = {("g00", 3), ("g01", 9), ("g03", N_PRE + 2)}
    rep = _fit(_panel(drop=drop))
    assert rep.assumptions.balanced_panel.statistic is not None
    assert rep.filled_cells == len(drop)
    assert dg._filled_total(rep.assumptions.balanced_panel) == rep.filled_cells


# ------------------------------------------------------------------------- property
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n_pre=st.integers(min_value=10, max_value=45),
       n_post=st.integers(min_value=2, max_value=10),
       seed=st.integers(min_value=0, max_value=2 ** 31 - 1))
def test_every_check_is_reported_and_self_consistent(n_pre, n_post, seed):
    """Over the panel domain: flagged is exactly the checks that did not hold."""
    rep = _fit(_panel(n_pre=n_pre, n_post=n_post, seed=seed))
    assume(not rep.tbr_fit.rank_deficient)
    a = rep.assumptions
    names = [c.name for c in a.all_checks()]
    assert len(set(names)) == len(names) == 7
    assert a.flagged == sorted(c.name for c in a.all_checks() if c.holds is False)
    for c in a.all_checks():
        if c.pvalue is not None:
            assert 0.0 <= c.pvalue <= 1.0


# ------------------------------------------------------------------------ edge cases
def test_a_pretest_too_short_to_judge_reports_no_verdict_not_a_pass():
    """Silence about an untestable thing, not a clean bill of health."""
    a = _fit(_panel(n_pre=5, n_post=2)).assumptions
    assert a.normality.holds is None
    assert "too short" in a.normality.detail.lower()
    assert "normality" not in a.flagged


def test_a_perfectly_balanced_panel_has_nothing_to_report():
    check = _fit(_panel()).assumptions.balanced_panel
    assert check.holds is True
    assert check.statistic == 0.0


def test_a_degenerate_fit_still_reports_the_panel_checks():
    """The residual checks need a fit; the panel checks do not."""
    df = _panel()
    df.loc[(df["t"] < N_PRE) & (df["is_ctrl"] == 1), "sales"] = 100.0
    rep = _fit(df)
    assert rep.tbr_fit.rank_deficient is True
    assert rep.assumptions.stable_membership.holds is True


# --------------------------------------------------------------------------- failure
def test_the_container_is_frozen():
    a = _fit(_panel()).assumptions
    with pytest.raises(Exception):
        a.flagged = []


def test_a_check_that_did_not_run_is_not_counted_as_holding():
    """None is not True, anywhere the flag list is built."""
    from mlsynth.utils.tbr_helpers.diagnostics import AssumptionCheck
    c = AssumptionCheck(name="x", statistic=None, pvalue=None, threshold=None,
                        holds=None, detail="not run")
    assert c.holds is not True


# --------------------------------------- the statistics on input they cannot judge
# Each returns NaN, not a number, and the checks read NaN as "no verdict",
# so a degenerate pretest never arrives as a pass.

def test_durbin_watson_on_residuals_with_no_variation():
    assert np.isnan(dg.durbin_watson(np.zeros(10)))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_breusch_godfrey_on_a_window_too_short_for_its_lag(n):
    assert np.isnan(dg.breusch_godfrey_pvalue(np.arange(n, dtype=float),
                                              np.arange(n, dtype=float)))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_breusch_pagan_on_a_window_too_short(n):
    assert np.isnan(dg.breusch_pagan_pvalue(np.ones(n), np.arange(n, dtype=float)))


def test_breusch_pagan_on_residuals_that_are_exactly_zero():
    """Mean squared residual of zero: there is no spread to explain."""
    assert np.isnan(dg.breusch_pagan_pvalue(np.zeros(20), np.arange(20.0)))


def test_breusch_pagan_when_the_squared_residuals_do_not_vary():
    """Constant squared residuals: the auxiliary regression has no variance."""
    resid = np.tile([1.0, -1.0], 10)          # every square is exactly 1
    assert np.isnan(dg.breusch_pagan_pvalue(resid, np.arange(20.0)))


@pytest.mark.parametrize("n", [0, 1, 2])
def test_shapiro_on_too_few_points(n):
    assert np.isnan(dg.shapiro_pvalue(np.zeros(n)))


def test_shapiro_on_residuals_with_no_spread():
    assert np.isnan(dg.shapiro_pvalue(np.full(12, 3.0)))


def test_a_nan_statistic_never_becomes_a_pass():
    """The guard that keeps a degenerate pretest from reading as clean."""
    serial, normality, hetero = dg._residual_checks(np.zeros(20),
                                                    np.arange(20.0))
    for check in (serial, normality, hetero):
        assert check.holds is None             # no verdict, and not a pass
        assert check.pvalue is None
        assert "no variation" in check.detail


# ----------------------------------------------------------------- the size of it
def test_the_suite_does_not_cry_wolf_on_sound_panels():
    """A check read at the 5% level has to fire at about 5%.

    The first version of this suite judged serial correlation on Durbin-Watson
    falling outside Au's band (1.5, 2.5) as well as on Breusch-Godfrey. That
    band screens candidate designs; read as a hypothesis test it is heavily
    oversized at short pretests, and the check fired on 38% of sound panels at
    12 pretest periods and 23% at 20. Breusch-Godfrey alone decides now.

    The bound here is loose enough to be stable at this sample size and tight
    enough that the oversized version fails it every time.
    """
    n = 120
    fired = 0
    for seed in range(n):
        a = _fit(_panel(n_pre=20, n_post=6, seed=seed)).assumptions
        fired += "serial_correlation" in a.flagged
    assert fired / n <= 0.15, f"serial correlation fired on {fired}/{n} sound panels"


# ------------------------------- assumption 1's testable half: the two new checks
def _trending_away(n_pre=40, n_post=8, n_geos=8, seed=0, slope=0.30):
    """Li (2024) Assumption 2.1's own description of a violation: the treated
    carries a trend no control can trace. The true effect is zero."""
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    f = np.cumsum(rng.normal(size=T)) + 100.0
    lev = rng.uniform(10, 20, n_geos)
    load = rng.uniform(0.8, 1.2, n_geos)
    Y = lev[None, :] + load[None, :] * 8 * f[:, None] + rng.normal(0, 2.0, (T, n_geos))
    Y[:, :3] += (slope * np.arange(T) ** 1.6)[:, None]
    return pd.DataFrame([
        {"geo": f"g{j:02d}", "t": t, "sales": float(Y[t, j]),
         "post": int(t >= n_pre), "is_treat": int(j < 3), "is_ctrl": int(j >= 3)}
        for j in range(n_geos) for t in range(T)])


def test_backdating_passes_when_the_relation_predicts():
    c = _fit(_panel(n_pre=40)).assumptions.backdating
    assert c.holds is True
    assert c.statistic > 0.0 and 0.0 <= c.pvalue <= 1.0
    assert "held out the last" in c.detail


def test_backdating_fires_when_the_treated_trends_away():
    c = _fit(_trending_away()).assumptions.backdating
    assert c.holds is False
    assert "not to be relied on" in c.detail


def test_the_identification_failure_is_not_prescribed_a_variance_fix():
    """The reason this check exists.

    A treated series trending away from every control also reads as
    autocorrelation, so the residual checks fire and tell the user to refit
    with a HAC variance, which does nothing for it. The identification checks
    have to fire too, and say something else.
    """
    a = _fit(_trending_away()).assumptions
    assert "backdating" in a.flagged
    assert "hac" not in a.backdating.detail.lower()
    assert "group split" in a.backdating.detail


def test_backdating_reports_the_cost_against_difference_in_differences():
    """TBR nests DID at a slope of one, so the comparison is free."""
    assert "times TBR's error" in _fit(_panel(n_pre=40)).assumptions.backdating.detail


def test_backdating_has_no_verdict_when_the_pretest_cannot_be_split():
    """One or two held-out periods are not a window long enough to score."""
    c = _fit(_panel(n_pre=9, n_post=4)).assumptions.backdating
    assert c.holds is None
    assert "long enough to score" in c.detail
    assert str(dg.MIN_HOLD) in c.detail


def test_stationarity_has_no_verdict_on_a_pretest_too_short_for_it():
    """At 20 periods it fails to establish stationarity on every sound panel,
    so reporting a verdict there would flag all of them."""
    c = _fit(_panel(n_pre=20, n_post=6)).assumptions.stationary_residual
    assert c.holds is None
    assert str(dg.MIN_PERIODS_STATIONARITY) in c.detail


def test_stationarity_holds_on_a_panel_whose_difference_is_stationary():
    c = _fit(_panel(n_pre=40)).assumptions.stationary_residual
    assert c.holds is True
    assert c.pvalue < dg.ALPHA


def test_stationarity_fires_when_the_difference_carries_a_trend():
    c = _fit(_trending_away(n_pre=40)).assumptions.stationary_residual
    assert c.holds is False
    assert "failure to reject" in c.detail


def test_the_backdating_check_is_correctly_sized():
    """Its reference distribution is F(hold, T0 - 2), not a chosen number.

    Measured over sound panels the 95th percentile of the statistic tracked F
    closely -- 2.46 against 2.45 at one configuration, 2.32 against 2.27 at
    another -- which is what makes a p-value legitimate here.
    """
    n = 120
    fired = sum("backdating" in _fit(_panel(n_pre=40, n_post=8, seed=s))
                .assumptions.flagged for s in range(n))
    assert fired / n <= 0.15, f"backdating fired on {fired}/{n} sound panels"


# --- how a fired check is reported -------------------------------------------
#
# Five of the seven checks are hypothesis tests sized at 5 per cent, so between
# them they fire on about a quarter of sound panels. One warning per check would
# report that quarter as seven separate alarms, and a reader who sees an alarm
# on every fourth sound panel stops reading all of them. The warning is grouped
# by what a fired check costs instead, so a nuisance check cannot read as a
# failure of identification.

def _warnings_from(df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TBR(TBRConfig(df=df, outcome="sales", unitid="geo", time="t",
                      treatment_col="is_treat", control_col="is_ctrl",
                      post_col="post")).fit()
    return [str(w.message) for w in caught
            if str(w.message).startswith("TBR: ")]


def test_a_nuisance_check_does_not_report_itself_as_a_failure_of_identification():
    """Serially correlated residuals cost calibration, not identification."""
    texts = _warnings_from(_panel(rho=0.8))
    assert len(texts) == 1
    text, = texts
    assert "point estimate is unaffected" in text
    assert "interval" in text
    assert "biased" not in text


def test_an_identification_failure_is_reported_as_one():
    text = " ".join(_warnings_from(_trending_away()))
    assert "counterfactual it extrapolates may be biased" in text


def test_the_warning_is_grouped_by_consequence_not_emitted_per_check():
    """At most one warning per consequence, however many checks fire."""
    texts = _warnings_from(_trending_away(slope=0.45))
    rep = _fit(_trending_away(slope=0.45))
    assert len(rep.assumptions.flagged) > len(texts)
    assert len(texts) <= len(dg._CONSEQUENCE)


def test_every_check_belongs_to_exactly_one_consequence_group():
    """A check absent from the table would fire and never warn."""
    grouped = [n for group, _ in dg._CONSEQUENCE for n in group]
    reported = [c.name for c in _fit(_panel()).assumptions.all_checks()]
    assert sorted(grouped) == sorted(reported)
    assert len(grouped) == len(set(grouped))


def test_the_grouped_warning_still_carries_each_fired_checks_detail():
    rep = _fit(_trending_away())
    text = " ".join(_warnings_from(_trending_away()))
    for check in rep.assumptions.all_checks():
        if check.holds is False:
            assert check.detail in text


def test_the_pretest_only_caveat_survives_the_grouping():
    for df in (_panel(rho=0.8), _trending_away()):
        for text in _warnings_from(df):
            assert "checks on the pretest only" in text


def test_the_did_comparison_is_fitted_on_the_same_window_tbr_was():
    """Both models predict the held-out tail; neither is shown it first.

    The ratio is only a comparison if the difference-in-differences intercept
    comes from the front window too. Fitted on the tail it is scoring an
    in-sample fit against an out-of-sample one, and on a panel whose treated
    series trends away from every control that reads as 0.17 instead of 1.08 --
    the suite recommending difference-in-differences exactly where TBR traces
    the series better.
    """
    rng = np.random.default_rng(3)
    n_pre, hold = 40, 8
    x = np.cumsum(rng.normal(size=n_pre + 4)) + 200.0
    y = 5.0 + 1.6 * x + rng.normal(0, 1.0, x.size)

    _, _, held, ratio = dg.backdating(y, x, n_pre, hold)
    T0 = n_pre - hold

    front = float(np.mean(y[:T0] - x[:T0]))
    did = float(np.sqrt(np.mean((y[T0:n_pre] - (front + x[T0:n_pre])) ** 2)))
    assert ratio == pytest.approx(did / held, rel=1e-12)

    tail = float(np.mean(y[T0:n_pre] - x[T0:n_pre]))
    cheated = float(np.sqrt(np.mean((y[T0:n_pre] - (tail + x[T0:n_pre])) ** 2)))
    assert abs(ratio - cheated / held) > 0.1


def test_the_did_ratio_reaches_the_detail_as_a_number():
    stat, _, _, ratio = dg.backdating(
        *_aggregates(_panel(n_pre=40)), 40, 8)
    detail = _fit(_panel(n_pre=40)).assumptions.backdating.detail
    assert f"{ratio:.3g}" in detail or f"{ratio:.2f}" in detail


# --- the degenerate paths, which are reachable and so are tested -------------

def _exactly_affine(n=48):
    """A pretest the fit reproduces with no residual at all.

    The backdated fit then has zero scale, so equation 6 at a horizon of one
    divides by nothing and the standardised error is undefined.
    """
    x = np.linspace(100.0, 160.0, n)
    return 5.0 + 1.6 * x, x


def test_backdating_on_a_pretest_that_fits_exactly_returns_no_number():
    y, x = _exactly_affine()
    stat, ins, held, ratio = dg.backdating(y, x, 40, 8)
    assert not np.isfinite(stat)
    assert not np.isfinite(ins) and not np.isfinite(held)
    assert not np.isfinite(ratio)


def test_a_degenerate_backdated_fit_reports_no_verdict_not_a_pass():
    y, x = _exactly_affine()
    back, _ = dg._identification_checks(y, x, 40, 8)
    assert back.holds is None
    assert "degenerate" in back.detail


@pytest.mark.parametrize("n", [0, 1, 5, 11])
def test_engle_granger_on_a_series_too_short_to_run(n):
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    assert not np.isfinite(dg.stationary_residual_pvalue(y, rng.normal(size=n)))


def test_a_nan_did_ratio_never_reaches_the_reader_as_a_word():
    """An exactly-predicted window gives a 0/0 ratio; the clause drops out."""
    for df in (_panel(n_pre=40), _trending_away()):
        for c in _fit(df).assumptions.all_checks():
            assert "nan" not in (c.detail or "").lower()


def test_a_rank_deficient_backdated_fit_still_returns_a_finite_scale():
    """Measured, and the reason the scale guard in backdating is defensive."""
    rng = np.random.default_rng(0)
    x = np.full(48, 120.0)
    y = 100.0 + rng.normal(0, 5.0, 48)
    fit = pst.fit_pretest(y[:32], x[:32])
    assert fit.rank_deficient
    _, scale = pst.cumulative_posterior(fit, y[32:33], x[32:33])
    assert np.isfinite(scale[-1]) and scale[-1] > 0


def test_a_tail_predicted_exactly_drops_the_did_clause_instead_of_printing_nan():
    """The one way the ratio is undefined with a non-degenerate front window.

    TBR's held-out RMSE is the ratio's denominator. A front window that carries
    residual variation gets past the degenerate guard, so if the tail then
    happens to sit exactly on the fitted line the ratio is 0/0. Constructed
    here by fitting the front and placing the tail on that line.
    """
    rng = np.random.default_rng(11)
    n_pre, hold = 40, 8
    T0 = n_pre - hold
    x = np.linspace(100.0, 180.0, n_pre + 4)
    y = np.empty_like(x)
    y[:T0] = 5.0 + 1.6 * x[:T0] + rng.normal(0, 2.0, T0)

    fit = pst.fit_pretest(y[:T0], x[:T0])
    y[T0:] = fit.alpha + fit.beta * x[T0:]           # predicted exactly

    _, in_rmse, held_rmse, ratio = dg.backdating(y, x, n_pre, hold)
    assert in_rmse > 0                                # past the guard
    assert held_rmse == pytest.approx(0.0, abs=1e-9)  # nothing to divide by
    assert not np.isfinite(ratio)

    back, _ = dg._identification_checks(y, x, n_pre, 4)
    assert "nan" not in back.detail.lower()
    assert "difference-in-differences" not in back.detail
