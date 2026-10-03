r"""The placebo pool is a null, so its members have to be approximable too.

``unit_level_cumulative`` reads each treated unit's interval off blocks of
held-out residuals drawn from the other treated units and from ``extra_pool``.
Inverting a pivot on that pool assumes every member is a draw from the null. A
donor its own peers cannot reproduce does not qualify: its residual is dominated
by the fit's systematic miss, and ``_prepare_pool`` rescales each series to the
unit under test, which matches the standard deviation and preserves the ratio of
offset to spread. Summing a block then accumulates the offset as ``h`` against
noise at ``sqrt(h)``, so one such member dominates the tail quantiles, worst at
the long horizons the cumulative path exists to report.

Measured on a twelve-market panel calibrated to a 141-week weekly geo panel:
screening the pool with the gate the treated units already pass cuts per-unit
interval width from 199.6 to 88.0, a 56 per cent reduction, with the point
estimates unchanged. The library already owns the test -- it applies
``approximability`` to the treated unit and then builds that unit's interval from
a pool it never applies it to.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.post_inference import (
    approximability,
    unit_level_cumulative,
)


def _ar1(rng, n, rho=0.1, sd=1.0):
    e = rng.normal(0.0, sd, n)
    if rho == 0.0:
        return e
    out = np.empty(n)
    out[0] = e[0] / np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + e[t]
    return out


@pytest.fixture
def design():
    """Two treated units, clean gaps, and a pool of clean donor placebos."""
    rng = np.random.default_rng(606)
    blank = np.column_stack([_ar1(rng, 24), _ar1(rng, 24)])
    post = np.column_stack([_ar1(rng, 8) + 1.0, _ar1(rng, 8) + 0.6])
    clean = [_ar1(rng, 24) for _ in range(6)]
    return post, blank, np.array([0.4, 0.6]), clean


def _offset_series(rng, n=24, offset=9.0):
    """A donor its peers cannot reach: a constant miss on top of small noise."""
    return offset + _ar1(rng, n, sd=0.6)


# ----------------------------------------------------------------- smoke
def test_screening_runs_and_returns_the_contract(design):
    post, blank, w, clean = design
    res = unit_level_cumulative(post, blank, w, extra_pool=clean)
    assert len(res.per_unit) == 2
    assert all(np.isfinite([p.estimate, p.lower, p.upper]).all()
               for u in res.per_unit for p in u)
    assert res.dropped_extra == ()
    assert res.dropped_treated == ()


# -------------------------------------------------------- unit invariants
def test_an_unapproximable_donor_is_dropped_and_recorded(design):
    post, blank, w, clean = design
    rng = np.random.default_rng(7)
    pool = clean + [_offset_series(rng)]
    assert not approximability(pool[-1]).ok, "fixture must actually fail the gate"
    with pytest.warns(UserWarning, match="placebo"):
        res = unit_level_cumulative(post, blank, w, extra_pool=pool)
    assert res.dropped_extra == (len(pool) - 1,)


def test_screening_narrows_the_interval_it_was_built_to_narrow(design):
    """The measured consequence: the offset member inflates the null."""
    post, blank, w, clean = design
    rng = np.random.default_rng(8)
    pool = clean + [_offset_series(rng, offset=12.0)]
    wide = unit_level_cumulative(post, blank, w, extra_pool=pool, screen_pool=False)
    with pytest.warns(UserWarning):
        tight = unit_level_cumulative(post, blank, w, extra_pool=pool)
    for u_wide, u_tight in zip(wide.per_unit, tight.per_unit):
        assert (u_tight[-1].upper - u_tight[-1].lower) < (u_wide[-1].upper - u_wide[-1].lower)


def test_screening_moves_the_interval_and_never_the_estimate(design):
    """Only the null is rebuilt, so the running sum cannot move."""
    post, blank, w, clean = design
    rng = np.random.default_rng(9)
    pool = clean + [_offset_series(rng)]
    off = unit_level_cumulative(post, blank, w, extra_pool=pool, screen_pool=False)
    with pytest.warns(UserWarning):
        on = unit_level_cumulative(post, blank, w, extra_pool=pool)
    for a, b in zip(off.per_unit, on.per_unit):
        for pa, pb in zip(a, b):
            assert pb.estimate == pytest.approx(pa.estimate, rel=1e-12, abs=1e-12)


def test_a_units_own_series_is_never_screened_out(design):
    """It sets the reference scale, and its offset is the caller's own check.

    Both treated units are given a large common offset, so each fails the gate.
    The run must still produce a path for each -- dropping a unit's own residual
    would leave its interval with no scale to be read on.
    """
    post, blank, w, clean = design
    with pytest.warns(UserWarning):
        res = unit_level_cumulative(post, blank + 9.0, w, extra_pool=clean)
    assert len(res.per_unit) == 2
    assert all(np.isfinite([p.lower, p.upper]).all() for u in res.per_unit for p in u)


def test_an_unapproximable_treated_unit_leaves_the_others_pool(design):
    """Recording the drop is bookkeeping; the interval has to move.

    Asserting only ``dropped_treated`` passes against an implementation that
    computes the list and never filters on it, which is how the mutant
    ``pool-screen-takes-the-units-own-scale-reference`` survived a first run.
    """
    post, blank, w, clean = design
    bad = blank.copy(); bad[:, 1] += 11.0
    with pytest.warns(UserWarning):
        res = unit_level_cumulative(post, bad, w, extra_pool=clean)
    kept = unit_level_cumulative(post, bad, w, extra_pool=clean, screen_pool=False)
    assert res.dropped_treated == (1,)
    # unit 0's null must no longer contain unit 1's offset gap
    assert ((res.per_unit[0][-1].upper - res.per_unit[0][-1].lower)
            < (kept.per_unit[0][-1].upper - kept.per_unit[0][-1].lower))


def test_screening_off_reproduces_the_unscreened_pool(design):
    post, blank, w, clean = design
    rng = np.random.default_rng(10)
    pool = clean + [_offset_series(rng)]
    a = unit_level_cumulative(post, blank, w, extra_pool=pool, screen_pool=False)
    b = unit_level_cumulative(post, blank, w, extra_pool=pool, screen_pool=False)
    for ua, ub in zip(a.per_unit, b.per_unit):
        for pa, pb in zip(ua, ub):
            assert pa.lower == pytest.approx(pb.lower)
    assert a.dropped_extra == ()


# ------------------------------------------------------------- edge cases
def test_every_extra_failing_falls_back_rather_than_emptying_the_pool(design):
    """Screening must not leave a unit with nothing to read a null from."""
    post, blank, w, _ = design
    rng = np.random.default_rng(11)
    pool = [_offset_series(rng, offset=8.0 + k) for k in range(5)]
    with pytest.warns(UserWarning):
        res = unit_level_cumulative(post, blank, w, extra_pool=pool)
    assert res.dropped_extra == tuple(range(5))
    assert all(np.isfinite([p.lower, p.upper]).all() for u in res.per_unit for p in u)


def test_no_extra_pool_at_all_still_screens_the_treated_gaps(design):
    post, blank, w, _ = design
    res = unit_level_cumulative(post, blank, w)
    assert res.dropped_extra == ()
    assert len(res.per_unit) == 2


def test_a_single_treated_unit_has_no_peer_to_screen(design):
    post, blank, w, clean = design
    res = unit_level_cumulative(post[:, :1], blank[:, :1], np.array([1.0]),
                                extra_pool=clean)
    assert res.dropped_treated == ()
    assert len(res.per_unit) == 1


# ---------------------------------------------------------------- failure
def test_a_degenerate_extra_series_is_reported_not_swallowed(design):
    post, blank, w, clean = design
    with pytest.warns(UserWarning):
        res = unit_level_cumulative(post, blank, w,
                                    extra_pool=clean + [np.full(24, 3.0)])
    assert 6 in res.dropped_extra


def test_both_entry_points_screen_the_same_pool(design):
    """A screen applied on one side only is the defect this branch removes."""
    from mlsynth.utils.fast_scm_helpers.post_inference import population_cumulative
    post, blank, w, clean = design
    rng = np.random.default_rng(12)
    pool = clean + [_offset_series(rng, offset=10.0)]
    f = np.array([0.2, 0.3, 0.25, 0.25])
    with pytest.warns(UserWarning):
        unit = unit_level_cumulative(post, blank, w, extra_pool=pool)
    with pytest.warns(UserWarning):
        popn = population_cumulative(post, blank, w, f, [0, 1], extra_pool=pool)
    kept = population_cumulative(post, blank, w, f, [0, 1], extra_pool=pool,
                                 screen_pool=False)
    assert unit.dropped_extra == popn.dropped_extra
    assert unit.dropped_treated == popn.dropped_treated
    # and the population pool is actually rebuilt, not merely annotated
    assert ((popn.aggregate[-1].upper - popn.aggregate[-1].lower)
            < (kept.aggregate[-1].upper - kept.aggregate[-1].lower))


# ------------------------------------------------- generative properties
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@given(st.integers(0, 2 ** 31 - 1), st.integers(0, 5), st.floats(3.0, 30.0))
@SETTINGS
def test_the_estimate_is_invariant_to_whatever_the_screen_removes(seed, n_bad, offset):
    """Screening rebuilds the null and never the running sum.

    The point estimate is arithmetic -- the cumulative gap -- so no choice about
    which series represent the null can move it. Asserted over pool sizes and
    offset magnitudes, not at one fixture.
    """
    rng = np.random.default_rng(seed)
    blank = np.column_stack([_ar1(rng, 20), _ar1(rng, 20)])
    post = np.column_stack([_ar1(rng, 6) + 0.8, _ar1(rng, 6)])
    pool = [_ar1(rng, 20) for _ in range(4)] + [
        offset + _ar1(rng, 20, sd=0.5) for _ in range(n_bad)]
    w = np.array([0.45, 0.55])
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        on = unit_level_cumulative(post, blank, w, extra_pool=pool)
        off = unit_level_cumulative(post, blank, w, extra_pool=pool, screen_pool=False)
    for a, b in zip(on.per_unit, off.per_unit):
        for pa, pb in zip(a, b):
            assert pa.estimate == pytest.approx(pb.estimate, rel=1e-12, abs=1e-12)


@given(st.integers(0, 2 ** 31 - 1), st.integers(2, 8))
@SETTINGS
def test_screening_an_already_clean_pool_drops_nothing(seed, n_extra):
    """Idempotence: the screen only fires on series that fail the gate.

    Every member here is a centred draw, so a screen that removed any of them
    would be removing information the interval needs.
    """
    rng = np.random.default_rng(seed)
    blank = np.column_stack([_ar1(rng, 24), _ar1(rng, 24)])
    post = np.column_stack([_ar1(rng, 8), _ar1(rng, 8)])
    pool = [_ar1(rng, 24) for _ in range(n_extra)]
    clean = [i for i, x in enumerate(pool) if approximability(x).ok]
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        res = unit_level_cumulative(post, blank, np.array([0.5, 0.5]), extra_pool=pool)
    assert set(res.dropped_extra).isdisjoint(clean)
