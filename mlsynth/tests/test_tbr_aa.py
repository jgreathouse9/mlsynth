r"""The A/A test: how often a TBR design calls a null window significant.

An A/A test takes a window where the truth is zero and asks how often the
design reports an effect anyway. That is the false-positive rate, and it is not
power: a design can have excellent power and an unacceptable false-positive
rate, so the two are measured separately and screened separately.

The primitive is three operations on top of a TBR fit. Fit on the earlier
pretest periods, project onto the last ``n_test`` held-out pretest periods, and
test the resulting cumulative interval against zero. Nothing is held out of the
real post window, and no treatment is applied anywhere, so any interval that
excludes zero is a false positive by construction.

The gate rule is not "the interval covers zero". A narrow interval sitting just
off zero is a small error and not a broken design, so an interval that excludes
zero gets a second look: take the true mean at the point of the interval
nearest zero, which is the most forgiving value consistent with it, and ask how
often a design like this one would call a null window significant. That number
is a lower bound on how often the design cries wolf, and the candidate fails
only if it exceeds a threshold.

Writing the true mean as ``mu0`` and the posterior scale as ``s``, with
``t`` the ``0.5 (1 + level)`` quantile on the fit's degrees of freedom:

    p = sf(t - |mu0|/s) + cdf(-t - |mu0|/s)

Two properties of that expression decide the tests below. It is bounded below
by ``1 - level``, attained exactly whenever the interval covers zero, because
then the nearest point of the interval to zero is zero itself. And it is
continuous at the boundary: a design whose interval just touches zero scores
``1 - level``, the same as one centred on zero. The floor matters because a
threshold at or below ``1 - level`` rejects every candidate, including a design
whose interval sits dead on zero, so that is a configuration error and not a
strict gate.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tbr_helpers import aa
from mlsynth.utils.tbr_helpers import posterior as pst

N_PRE, N_GEOS = 48, 8


def _series(seed=0, n=N_PRE, rho=0.0, slope=0.0, scale=2.0):
    """A treated and a control aggregate with no treatment anywhere.

    One common factor with heterogeneous loadings, so the two series co-move
    the way real market aggregates do. ``rho`` puts AR(1) dependence in the
    treated series' idiosyncratic part; ``slope`` makes the treated series
    drift away from the control, which is the one thing an A/A window should
    report.
    """
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n)) + 100.0
    e = rng.normal(0, scale, n)
    if rho:
        for i in range(1, n):
            e[i] += rho * e[i - 1]
    x = 40.0 + 3.0 * f + rng.normal(0, scale, n)
    y = 12.0 + 1.1 * f + e + slope * np.arange(n)
    return y, x


# ----------------------------------------------------------------- smoke
def test_a_draw_runs_and_reports_the_three_things_the_ticket_asks_for():
    y, x = _series()
    d = aa.aa_draw(y, x, n_test=8)
    assert np.isfinite(d.estimate)
    assert np.isfinite(d.lower) and np.isfinite(d.upper)
    assert d.lower < d.upper
    assert 0.0 <= d.false_positive_probability <= 1.0
    assert isinstance(d.passes, bool)
    assert d.n_fit == N_PRE - 8 and d.n_test == 8


def test_the_fit_never_sees_the_held_out_window():
    """The whole reason the A/A estimate is honest."""
    y, x = _series()
    base = aa.aa_draw(y, x, n_test=8)
    moved = y.copy()
    moved[-8:] += 500.0                      # only the held-out tail changes
    after = aa.aa_draw(moved, x, n_test=8)
    assert after.scale == pytest.approx(base.scale, rel=1e-12)
    assert after.df == base.df
    assert after.estimate != pytest.approx(base.estimate)


# --------------------------------------------------- the gate rule's invariants
@pytest.mark.parametrize("level", [0.99, 0.95, 0.90, 0.80, 0.50])
@pytest.mark.parametrize("df", [5, 38, 200])
def test_the_false_positive_floor_is_one_minus_the_level(level, df):
    """At a true mean of zero the rule returns the nominal rate exactly."""
    p = aa.false_positive_probability(0.0, 1.0, df, level)
    assert p == pytest.approx(1.0 - level, abs=1e-12)


@pytest.mark.parametrize("level", [0.95, 0.90, 0.50])
def test_an_interval_that_just_touches_zero_scores_the_floor(level):
    """Continuity at the boundary: no jump between covering and excluding."""
    df, s = 38, 1.0
    t = stats.t.ppf(0.5 * (1 + level), df)
    d = aa.gate(estimate=t * s, scale=s, df=df, level=level, threshold=0.999)
    assert d.lower == pytest.approx(0.0, abs=1e-9)
    assert d.false_positive_probability == pytest.approx(1.0 - level, abs=1e-9)


def test_the_probability_rises_as_the_interval_moves_off_zero():
    df, level, s = 38, 0.90, 1.0
    t = stats.t.ppf(0.5 * (1 + level), df)
    ps = [aa.gate(estimate=m * s, scale=s, df=df, level=level,
                  threshold=0.999).false_positive_probability
          for m in (t, 2.0, 2.5, 3.0, 4.0, 6.0)]
    assert ps == sorted(ps)
    assert ps[0] == pytest.approx(1.0 - level, abs=1e-9)
    assert ps[-1] > 0.99


def test_the_rule_does_not_care_which_way_the_interval_missed():
    df, level = 38, 0.90
    up = aa.gate(estimate=3.0, scale=1.0, df=df, level=level, threshold=0.999)
    dn = aa.gate(estimate=-3.0, scale=1.0, df=df, level=level, threshold=0.999)
    assert up.false_positive_probability == pytest.approx(
        dn.false_positive_probability, rel=1e-12)
    assert up.passes == dn.passes


def test_a_narrow_interval_just_off_zero_is_not_a_broken_design():
    """The ticket's reason for the second look, as a test."""
    df, level = 38, 0.90
    t = stats.t.ppf(0.5 * (1 + level), df)
    d = aa.gate(estimate=1.02 * t * 0.01, scale=0.01, df=df, level=level,
                threshold=0.20)
    assert not d.covers_zero                 # it really does exclude zero
    assert d.passes                          # and it is still a sound design


def test_an_interval_far_from_zero_fails_however_narrow_it_is():
    d = aa.gate(estimate=6.0, scale=1.0, df=38, level=0.90, threshold=0.20)
    assert not d.covers_zero and not d.passes
    assert d.false_positive_probability > 0.90


def test_covering_zero_passes_at_any_admissible_threshold():
    for level in (0.99, 0.95, 0.90, 0.80, 0.50):
        d = aa.gate(estimate=0.0, scale=1.0, df=38, level=level,
                    threshold=(1 - level) + 1e-9)
        assert d.covers_zero and d.passes


# ------------------------------------------------------------- the config edges
@pytest.mark.parametrize("level,threshold", [
    (0.90, 0.10), (0.80, 0.20), (0.95, 0.05), (0.99, 0.01), (0.50, 0.50)])
def test_a_threshold_at_the_floor_would_reject_everything_and_so_is_refused(
        level, threshold):
    """A gate that refuses a design centred on zero is a misconfiguration.

    Each pair here is a threshold a reader would write as "the nominal rate".
    They are not all equal to ``1 - level`` in binary: ``1 - 0.90`` is
    ``0.09999999999999998``, so ``0.10`` sits above it and a strict comparison
    admits it, while at 0.95 and 0.99 the rounded value falls below and is
    caught. The guard is a tolerance for that reason, and this test fails at
    0.90 and 0.80 without it.
    """
    with pytest.raises(MlsynthConfigError, match="1 - level"):
        aa.gate(estimate=0.0, scale=1.0, df=38, level=level,
                threshold=threshold)


def test_a_threshold_below_the_floor_is_refused_too():
    with pytest.raises(MlsynthConfigError, match="1 - level"):
        aa.gate(estimate=0.0, scale=1.0, df=38, level=0.90, threshold=0.05)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_a_level_outside_the_open_unit_interval_is_refused(level):
    with pytest.raises(MlsynthConfigError):
        aa.gate(estimate=0.0, scale=1.0, df=38, level=level, threshold=0.999)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_a_non_positive_scale_is_refused(scale):
    with pytest.raises(MlsynthConfigError):
        aa.gate(estimate=1.0, scale=scale, df=38, level=0.90, threshold=0.999)


@pytest.mark.parametrize("df", [0, -3])
def test_a_non_positive_degrees_of_freedom_is_refused(df):
    with pytest.raises(MlsynthConfigError):
        aa.gate(estimate=1.0, scale=1.0, df=df, level=0.90, threshold=0.999)


# ------------------------------------------------------- the draw's data edges
@pytest.mark.parametrize("n_test", [0, -1])
def test_a_window_with_nothing_in_it_is_refused(n_test):
    y, x = _series()
    with pytest.raises(MlsynthConfigError, match="n_test"):
        aa.aa_draw(y, x, n_test=n_test)


@pytest.mark.parametrize("n,n_test", [(12, 10), (13, 8), (14, 8), (15, 8)])
def test_a_window_that_leaves_too_little_to_fit_on_is_refused(n, n_test):
    """Refused by this module's own guard, named in the message.

    The first case leaves two periods, which ``fit_pretest`` refuses on its own
    with a message that also says "periods", so matching on that word cannot
    tell the two guards apart. The later cases leave five, six and eight, which
    ``fit_pretest`` accepts: only MIN_FIT rejects them, and the message has to
    say so.
    """
    y, x = _series(n=n)
    with pytest.raises(MlsynthDataError, match="a backdated fit needs"):
        aa.aa_draw(y, x, n_test=n_test)


def test_a_fit_of_exactly_the_minimum_length_is_admitted():
    """The boundary is inclusive, so the refusal above starts one below it."""
    y, x = _series(n=aa.MIN_FIT + 8)
    d = aa.aa_draw(y, x, n_test=8)
    assert d.n_fit == aa.MIN_FIT
    assert np.isfinite(d.scale) and d.scale > 0



def test_mismatched_series_are_refused():
    y, x = _series()
    with pytest.raises(MlsynthDataError):
        aa.aa_draw(y[:-3], x, n_test=8)


def test_a_series_carrying_a_nan_is_refused():
    y, x = _series()
    y = y.copy(); y[5] = np.nan
    with pytest.raises(MlsynthDataError, match="finite"):
        aa.aa_draw(y, x, n_test=8)


# ------------------------------------------------------------------- the HAC arm
def test_hac_prices_dependence_the_published_scale_assumes_away():
    """Question 2 of the ticket, as an assertion about one draw."""
    y, x = _series(rho=0.8)
    iid = aa.aa_draw(y, x, n_test=8, variance="iid")
    hac = aa.aa_draw(y, x, n_test=8, variance="hac")
    assert hac.estimate == pytest.approx(iid.estimate, rel=1e-12)
    assert hac.scale > iid.scale
    assert hac.variance == "hac" and iid.variance == "iid"


def test_a_zero_bandwidth_matches_equation_six_in_its_error_term_exactly():
    """At lag zero the long-run variance is eqn 6's sigma^2, to the last bit.

    The Bartlett sum keeps only the diagonal and ``_newey_west`` carries the
    same ``n / (n - 2)`` correction eqn 6's sigma^2 does, so the test-window
    error term is identical. Asserted on the helper because the public scale
    returns the two terms already summed.
    """
    y, x = _series(rho=0.8)
    n_fit = len(y) - 8
    fit = pst.fit_pretest(y[:n_fit], x[:n_fit])
    design = np.column_stack([np.ones(n_fit), x[:n_fit]])
    resid = y[:n_fit] - design @ np.linalg.lstsq(
        design, y[:n_fit], rcond=None)[0]
    lrv, _ = pst._newey_west(resid, design, 0)
    assert lrv == pytest.approx(fit.sigma_sq, rel=1e-12)


def test_a_zero_bandwidth_does_not_reproduce_the_whole_published_scale():
    """It is not eqn 6: the coefficient term becomes HC1.

    Measured on this panel the error term agrees exactly while the coefficient
    term comes back 16.81 against eqn 6's 21.92, and that term is 29% of the
    variance, so the scale lands 3.4% below. The two are on one footing and
    coincide in expectation under homoskedasticity; they are not equal on a
    sample, and a reading of ``bandwidth=0`` as "reproduces the iid scale"
    overstates it.
    """
    y, x = _series(rho=0.8)
    iid = aa.aa_draw(y, x, n_test=8, variance="iid")
    hac0 = aa.aa_draw(y, x, n_test=8, variance="hac", bandwidth=0)
    assert hac0.scale != pytest.approx(iid.scale, rel=1e-6)
    assert hac0.scale == pytest.approx(iid.scale, rel=0.10)
    assert hac0.scale < aa.aa_draw(y, x, n_test=8, variance="hac").scale


# ------------------------------------------------------------------- properties
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(mu0=st.floats(-50, 50), scale=st.floats(0.01, 20),
       level=st.sampled_from([0.99, 0.95, 0.90, 0.80, 0.50]),
       df=st.integers(3, 300))
def test_the_probability_is_a_probability_and_never_below_the_floor(
        mu0, scale, level, df):
    p = aa.false_positive_probability(mu0, scale, df, level)
    assert 1.0 - level - 1e-12 <= p <= 1.0


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(a=st.floats(0, 30), step=st.floats(0.01, 10),
       scale=st.floats(0.1, 5), df=st.integers(3, 300))
def test_the_probability_never_falls_as_the_true_mean_moves_away(
        a, step, scale, df):
    lo = aa.false_positive_probability(a, scale, df, 0.90)
    hi = aa.false_positive_probability(a + step, scale, df, 0.90)
    assert hi >= lo - 1e-12


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**32 - 1), n_test=st.integers(2, 12))
def test_a_draw_is_self_consistent_on_any_panel(seed, n_test):
    y, x = _series(seed=seed)
    d = aa.aa_draw(y, x, n_test=n_test)
    assert d.lower <= d.estimate <= d.upper
    assert d.covers_zero == (d.lower <= 0.0 <= d.upper)
    assert d.n_fit + d.n_test == len(y)
    assert d.passes == (d.false_positive_probability <= d.threshold)


# ============================================================================
# The calibration harness: the same primitive over many random splits.
#
# One draw says whether one design cried wolf once. The harness runs the
# primitive over many random treated/control splits of a panel with no
# treatment anywhere and counts how often the interval covered zero, at each
# nominal level and each window length. That is the quantity the per-candidate
# gate cannot report, because it sees one design at a time.
#
# The output is a table and not a verdict. A cell carries its counts, its
# coverage, and an interval on that coverage; nothing on it says pass.
# ============================================================================

def _wide(seed=0, n_geos=12, n=N_PRE, rho=0.0):
    """A panel of geo series with a common factor and no treatment."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n)) + 100.0
    lev = rng.uniform(10, 25, n_geos)
    load = rng.uniform(0.8, 1.2, n_geos)
    e = rng.normal(0, 2.0, (n, n_geos))
    if rho:
        for i in range(1, n):
            e[i] += rho * e[i - 1]
    return lev[None, :] + load[None, :] * 3.0 * f[:, None] + e


def test_a_split_partitions_the_geos():
    rng = np.random.default_rng(0)
    treated, control = aa.random_split(rng, 12, n_treated=3)
    assert len(treated) == 3 and len(control) == 9
    assert not (set(treated) & set(control))
    assert set(treated) | set(control) == set(range(12))


def test_a_split_is_not_always_the_same_split():
    rng = np.random.default_rng(0)
    seen = {tuple(sorted(aa.random_split(rng, 12, 3)[0])) for _ in range(40)}
    assert len(seen) > 5


@pytest.mark.parametrize("n_treated", [0, 12, 13, -1])
def test_a_split_that_leaves_no_group_is_refused(n_treated):
    rng = np.random.default_rng(0)
    with pytest.raises(MlsynthConfigError):
        aa.random_split(rng, 12, n_treated=n_treated)


# ----------------------------------------------------- the coverage interval
def test_the_coverage_interval_brackets_the_count_it_came_from():
    for y, n in [(1, 50), (25, 50), (49, 50)]:
        lo, hi = aa.beta_interval(y, n, 0.90)
        assert 0.0 < lo <= y / n <= hi < 1.0


@pytest.mark.parametrize("y,n", [(0, 50), (50, 50)])
def test_a_unanimous_count_gets_bounds_strictly_inside_zero_and_one(y, n):
    """The prior's job. A normal approximation returns a point here.

    The posterior excludes both ends, so the raw proportion falls outside the
    interval at a unanimous count -- measured 1.78e-06 against a proportion of
    0 at 0 of 50. That is the interval being usable, not a bracketing failure.
    """
    lo, hi = aa.beta_interval(y, n, 0.90)
    assert 0.0 < lo < hi < 1.0
    assert (lo > y / n) if y == 0 else (hi < y / n)


def test_the_coverage_interval_tightens_as_the_draws_pile_up():
    widths = [aa.beta_interval(n // 2, n, 0.90)[1]
              - aa.beta_interval(n // 2, n, 0.90)[0]
              for n in (20, 100, 1000)]
    assert widths == sorted(widths, reverse=True)


def test_the_coverage_interval_survives_a_unanimous_count():
    """Zero and n are the cells that break a normal approximation."""
    for y, n in [(0, 200), (200, 200)]:
        lo, hi = aa.beta_interval(y, n, 0.95)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


@pytest.mark.parametrize("n", [0, -1])
def test_the_coverage_interval_refuses_an_empty_count(n):
    with pytest.raises(MlsynthConfigError):
        aa.beta_interval(0, n, 0.90)


def test_the_coverage_interval_refuses_more_successes_than_draws():
    with pytest.raises(MlsynthConfigError):
        aa.beta_interval(11, 10, 0.90)


# ------------------------------------------------------------------- the grid
def test_the_grid_has_one_cell_per_level_window_and_variance():
    cells = aa.coverage_grid(_wide(), n_treated=3, windows=(4, 8),
                             levels=(0.90, 0.50), variances=("iid", "hac"),
                             reps=12, seed=1)
    assert len(cells) == 2 * 2 * 2
    keys = {(c.level, c.n_test, c.variance) for c in cells}
    assert len(keys) == 8


def test_the_grid_is_reproducible_from_its_seed():
    kw = dict(n_treated=3, windows=(6,), levels=(0.90,), reps=16, seed=7)
    a = aa.coverage_grid(_wide(), **kw)
    b = aa.coverage_grid(_wide(), **kw)
    assert [c.coverage for c in a] == [c.coverage for c in b]
    assert [c.n_covered for c in a] == [c.n_covered for c in b]


def test_a_different_seed_gives_a_different_draw_sequence():
    kw = dict(n_treated=3, windows=(6,), levels=(0.90,), reps=40)
    a = aa.coverage_grid(_wide(), seed=1, **kw)
    b = aa.coverage_grid(_wide(), seed=2, **kw)
    assert a[0].n_covered != b[0].n_covered or a[0].mean_width != b[0].mean_width


def test_a_cell_is_internally_consistent():
    for c in aa.coverage_grid(_wide(), n_treated=3, windows=(4, 8),
                              levels=(0.99, 0.90, 0.50), reps=20, seed=3):
        assert c.n_covered <= c.n_draws
        assert c.coverage == pytest.approx(c.n_covered / c.n_draws)
        assert 0.0 < c.ci_lower <= c.ci_upper < 1.0
        if 0 < c.n_covered < c.n_draws:
            # The posterior excludes both ends, so a unanimous cell's own
            # proportion falls outside its interval. Only the interior cells
            # are bracketed.
            assert c.ci_lower <= c.coverage <= c.ci_upper
        assert c.mean_width > 0.0


def test_a_cell_carries_no_verdict():
    """Criterion 3: the output is a table, not a pass or a fail."""
    c = aa.coverage_grid(_wide(), n_treated=3, windows=(6,), levels=(0.90,),
                         reps=8, seed=0)[0]
    fields = set(type(c).model_fields)
    assert not fields & {"passes", "ok", "verdict", "usable", "recommended"}


def test_coverage_falls_as_the_level_falls():
    """A 50% interval has to cover less often than a 99% one."""
    cells = {c.level: c.coverage for c in
             aa.coverage_grid(_wide(), n_treated=3, windows=(6,),
                              levels=(0.99, 0.90, 0.50), reps=200, seed=5)}
    assert cells[0.99] >= cells[0.90] >= cells[0.50]


@pytest.mark.parametrize("level", [0.90, 0.50])
def test_the_estimator_is_calibrated_when_the_panel_is_redrawn(level):
    """One split, fresh panels: eqn 6 hits its nominal rate.

    This is the cell where the answer is known in advance, so it tests the
    whole path -- fit, project, interval -- and not just the arithmetic. The
    split is held fixed and the panel redrawn, which is the sampling
    distribution eqn 6 is a statement about.
    """
    rng = np.random.default_rng(0)
    treated, control = aa.random_split(rng, 12, 3)
    covered = 0
    reps = 400
    for r in range(reps):
        p = _wide(seed=5000 + r, rho=0.0)
        d = aa.aa_draw(p[:, treated].sum(axis=1), p[:, control].sum(axis=1),
                       n_test=6, level=level)
        covered += d.covers_zero
    lo, hi = aa.beta_interval(covered, reps, 0.99)
    assert lo <= level <= hi, (
        f"nominal {level} outside [{lo:.3f}, {hi:.3f}] at "
        f"{covered / reps:.3f}")


def test_resplitting_one_panel_reads_higher_than_the_nominal_rate():
    """What the harness's own number is, and what it is not.

    An A/A test re-splits one fixed panel, so every draw reuses the same noise
    and the draws are dependent. The spread across splits therefore understates
    the estimator's sampling spread, and on a panel that genuinely satisfies
    the model the intervals read too wide. Measured over 2000 splits of one
    independent panel: coverage 0.98 at a nominal 0.90, with the mean posterior
    scale 1.36 times the spread of the estimate across splits, against a ratio
    of 0.98 when the split is held and the panel redrawn.

    So the harness's coverage is a diagnostic of a design on a panel, not an
    estimate of the estimator's coverage. The reference's real-data numbers run
    the other way -- 0.777 at a nominal 0.90 -- because a real panel carries
    the dependence equation 6 assumes away, and that effect is the larger one.
    """
    c, = aa.coverage_grid(_wide(rho=0.0), n_treated=3, windows=(6,),
                          levels=(0.90,), reps=400, seed=9)
    assert c.coverage > 0.90
    assert c.ci_lower > 0.90


def test_hac_is_wider_and_covers_more_when_the_panel_is_dependent():
    """Question 2 of the ticket, as a population statement."""
    cells = {c.variance: c for c in
             aa.coverage_grid(_wide(rho=0.8), n_treated=3, windows=(8,),
                              levels=(0.90,), variances=("iid", "hac"),
                              reps=300, seed=13)}
    assert cells["hac"].mean_width > cells["iid"].mean_width
    assert cells["hac"].coverage > cells["iid"].coverage


def test_a_screen_is_applied_and_counted():
    """The gated arm is a parameter, so A3's gates can be passed in later."""
    seen = []
    def screen(y, x, n_fit):
        seen.append(n_fit)
        return len(seen) % 2 == 0          # admit every other candidate
    c, = aa.coverage_grid(_wide(), n_treated=3, windows=(6,), levels=(0.90,),
                          reps=20, seed=0, screen=screen)
    assert c.screened is True
    assert c.n_rejected_by_screen > 0
    assert c.n_draws + c.n_rejected_by_screen == 20
    assert all(n == N_PRE - 6 for n in seen)


def test_without_a_screen_nothing_is_rejected():
    c, = aa.coverage_grid(_wide(), n_treated=3, windows=(6,), levels=(0.90,),
                          reps=20, seed=0)
    assert c.screened is False
    assert c.n_rejected_by_screen == 0 and c.n_draws == 20


def test_a_screen_that_rejects_everything_is_reported_not_divided_by():
    c, = aa.coverage_grid(_wide(), n_treated=3, windows=(6,), levels=(0.90,),
                          reps=10, seed=0, screen=lambda y, x, n: False)
    assert c.n_draws == 0 and c.n_rejected_by_screen == 10
    assert not np.isfinite(c.coverage)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 2.0])
def test_the_coverage_interval_refuses_a_level_outside_the_unit_interval(level):
    with pytest.raises(MlsynthConfigError, match="level"):
        aa.beta_interval(5, 10, level)


@pytest.mark.parametrize("shape", [(48,), (2, 3, 4)])
def test_the_grid_refuses_a_panel_that_is_not_periods_by_geos(shape):
    with pytest.raises(MlsynthDataError, match="periods by geos"):
        aa.coverage_grid(np.zeros(shape), n_treated=2, windows=(4,),
                         levels=(0.90,), reps=2)


@pytest.mark.parametrize("reps", [0, -3])
def test_the_grid_refuses_a_run_with_no_replications(reps):
    with pytest.raises(MlsynthConfigError, match="reps"):
        aa.coverage_grid(_wide(), n_treated=3, windows=(4,), levels=(0.90,),
                         reps=reps)


def test_the_arms_are_compared_on_identical_designs():
    """The split sequence is keyed by (seed, window) and nothing else.

    If the generator were advanced inside the variance or level loop, the iid
    and hac arms would see different splits and a difference between them would
    be the draw and not the variance -- which is the whole comparison question
    2 of the ticket rests on.
    """
    kw = dict(n_treated=3, windows=(6,), levels=(0.90, 0.50), reps=30, seed=4)
    alone = {(c.level, c.variance): c
             for c in aa.coverage_grid(_wide(), variances=("iid",), **kw)}
    together = {(c.level, c.variance): c
                for c in aa.coverage_grid(_wide(),
                                          variances=("hac", "iid"), **kw)}
    for key, cell in alone.items():
        assert together[key].n_covered == cell.n_covered
        assert together[key].mean_width == pytest.approx(cell.mean_width,
                                                         rel=1e-12)


def test_the_coverage_interval_is_the_neutral_prior_and_not_a_uniform_one():
    """Beta(1/3 + y, 1/3 + n - y), asserted against the quantile directly."""
    from scipy import stats as _st
    y, n, level = 7, 20, 0.90
    lo, hi = aa.beta_interval(y, n, level)
    a, b = aa.NEUTRAL_PRIOR + y, aa.NEUTRAL_PRIOR + n - y
    tail = 0.5 * (1 - level)
    assert lo == pytest.approx(_st.beta.ppf(tail, a, b), rel=1e-12)
    assert hi == pytest.approx(_st.beta.ppf(1 - tail, a, b), rel=1e-12)
    assert aa.NEUTRAL_PRIOR == pytest.approx(1.0 / 3.0)
    # a uniform prior would move both ends
    assert lo != pytest.approx(_st.beta.ppf(tail, 1 + y, 1 + n - y), rel=1e-6)


def test_the_grid_counts_coverage_and_not_the_gate_verdict():
    """Two different questions, and the harness is asked the first one.

    A design whose interval sits just off zero passes the gate -- that is the
    whole point of the second look -- while its interval does not cover zero.
    So the gate verdict is the more generous count, and reading it as coverage
    overstates calibration. Derived here from the draws themselves.
    """
    panel = _wide()
    n_test, reps, level, seed = 6, 60, 0.90, 21
    c, = aa.coverage_grid(panel, n_treated=3, windows=(n_test,),
                          levels=(level,), reps=reps, seed=seed)

    rng = np.random.default_rng([seed, n_test])
    splits = [aa.random_split(rng, panel.shape[1], 3) for _ in range(reps)]
    covers = passes = 0
    for treated, control in splits:
        d = aa.aa_draw(panel[:, treated].sum(axis=1),
                       panel[:, control].sum(axis=1), n_test, level=level,
                       threshold=1.0 - 0.5 * (1.0 - level))
        covers += d.covers_zero
        passes += d.passes
    assert c.n_covered == covers
    assert passes > covers, "no draw separates the two counts on this panel"
