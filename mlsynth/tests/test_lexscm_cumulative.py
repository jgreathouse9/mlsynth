"""Tests for the cumulative-effect path on a simplex-weighted synthetic control.

TBR's cumulative posterior is closed form because AdID's counterfactual is affine
in the treated series, so summing the gaps keeps the Student-t pivot. Simplex
weights break that affinity, so the cumulative interval here is built by
inverting a block-sum pivot against held-out placebo residuals, and its
properties have to be asserted, not inherited.

The design behind these assertions, with the measured coverage, is in
``agents`` and the pull request; the numbers that matter:

* pooling placebo series and rescaling each to the treated unit's residual
  scale covers 0.915 at nominal 0.90, flat in horizon. Pooling without
  rescaling covers 0.93--0.95 because the pool is noisier than the unit under
  test; using the unit's own residuals alone covers 0.83 and falls to 0.72 by
  horizon 8, because a blank window yields too few distinct blocks.
* the interval is only centred when the donors can approximate the treated
  unit. Outside their convex hull coverage is 0.037 with intervals six times
  wider, so ``approximability`` is a precondition and not a diagnostic.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.post_inference import (
    approximability,
    cumulative_path,
)


@pytest.fixture
def pool() -> list[np.ndarray]:
    """Eleven held-out residual series, generous enough to form blocks from."""
    rng = np.random.default_rng(0)
    return [rng.normal(0.0, 1.0, 32) for _ in range(11)]


@pytest.fixture
def gaps() -> np.ndarray:
    """Eight post-period gaps carrying a constant effect of 2."""
    rng = np.random.default_rng(1)
    return 2.0 + rng.normal(0.0, 1.0, 8)


# --------------------------------------------------------------------- smoke

def test_path_has_one_row_per_horizon(gaps, pool):
    path = cumulative_path(gaps, pool)
    assert [p.horizon for p in path] == list(range(1, gaps.size + 1))
    assert all(np.isfinite([p.estimate, p.lower, p.upper]).all() for p in path)


# ----------------------------------------------------------------- invariants

def test_point_estimate_is_the_running_sum_of_gaps(gaps, pool):
    """The estimate is arithmetic; only the interval is inferential."""
    path = cumulative_path(gaps, pool)
    assert np.allclose([p.estimate for p in path], np.cumsum(gaps))


def test_interval_brackets_its_own_point_estimate(gaps, pool):
    for p in cumulative_path(gaps, pool):
        assert p.lower <= p.estimate <= p.upper


def test_higher_confidence_gives_a_wider_interval(gaps, pool):
    narrow = cumulative_path(gaps, pool, level=0.80)
    wide = cumulative_path(gaps, pool, level=0.99)
    for a, b in zip(narrow, wide):
        assert b.upper - b.lower >= a.upper - a.lower


def test_shifting_every_gap_shifts_the_interval_by_the_same_amount(gaps, pool):
    """The pivot is a location statistic, so a constant shift must translate it."""
    base = cumulative_path(gaps, pool)
    moved = cumulative_path(gaps + 5.0, pool)
    for h, (a, b) in enumerate(zip(base, moved), start=1):
        assert b.lower == pytest.approx(a.lower + 5.0 * h)
        assert b.upper == pytest.approx(a.upper + 5.0 * h)


def test_rescaling_shrinks_the_influence_of_a_noisier_pool(gaps):
    """Standardising is what stops a noisy pool inflating the interval."""
    rng = np.random.default_rng(2)
    noisy = [rng.normal(0.0, 1.0, 32)] + [rng.normal(0.0, 4.0, 32) for _ in range(8)]
    on = cumulative_path(gaps, noisy, standardize=True)
    off = cumulative_path(gaps, noisy, standardize=False)
    assert all(a.upper - a.lower < b.upper - b.lower for a, b in zip(on, off))


def test_a_single_post_period_still_returns_a_path(pool):
    path = cumulative_path(np.array([1.5]), pool)
    assert len(path) == 1 and path[0].estimate == pytest.approx(1.5)


# ---------------------------------------------------------------- edge cases

def test_series_shorter_than_the_horizon_are_skipped_not_fatal(gaps):
    """A short placebo cannot form a long block; the rest of the pool carries it."""
    rng = np.random.default_rng(3)
    ragged = [rng.normal(0, 1, 32), rng.normal(0, 1, 3), rng.normal(0, 1, 32)]
    path = cumulative_path(gaps, ragged)
    assert all(np.isfinite([p.lower, p.upper]).all() for p in path)


def test_pool_too_short_for_any_horizon_is_refused(gaps):
    """Short but non-degenerate, so this is the block length and not the spread."""
    rng = np.random.default_rng(6)
    short = [rng.normal(0.0, 1.0, 2), rng.normal(0.0, 1.0, 2)]
    with pytest.raises(MlsynthDataError, match="block"):
        cumulative_path(gaps, short)


def test_constant_pool_has_no_spread_and_is_refused(gaps):
    with pytest.raises(MlsynthDataError, match="degenerate|spread"):
        cumulative_path(gaps, [np.zeros(32), np.zeros(32)])


# ------------------------------------------------------------------ failures

def test_empty_pool_is_refused(gaps):
    with pytest.raises(MlsynthDataError, match="placebo"):
        cumulative_path(gaps, [])


def test_no_post_gaps_is_refused(pool):
    with pytest.raises(MlsynthDataError, match="post"):
        cumulative_path(np.array([]), pool)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_level_outside_the_unit_interval_is_refused(gaps, pool, bad):
    with pytest.raises(MlsynthConfigError, match="level"):
        cumulative_path(gaps, pool, level=bad)


def test_non_finite_gaps_are_refused(pool):
    with pytest.raises(MlsynthDataError, match="finite"):
        cumulative_path(np.array([1.0, np.nan, 3.0]), pool)


# ------------------------------------------------------------ approximability

def test_a_centred_blank_window_passes(gaps):
    rng = np.random.default_rng(4)
    check = approximability(rng.normal(0.0, 1.0, 32))
    assert check.ok and abs(check.t_stat) < 3.0


def test_an_offset_blank_window_fails_and_says_so(gaps):
    """Out of the donors' hull the gap is not centred, and the interval is biased."""
    rng = np.random.default_rng(5)
    check = approximability(12.0 + rng.normal(0.0, 1.0, 32))
    assert not check.ok
    assert abs(check.t_stat) > 3.0 and check.p_value < 0.01


def test_approximability_needs_more_than_one_period():
    with pytest.raises(MlsynthDataError, match="period"):
        approximability(np.array([1.0]))


def test_non_finite_blank_window_is_refused():
    with pytest.raises(MlsynthDataError, match="finite"):
        approximability(np.array([1.0, np.inf, 2.0, 3.0]))


def test_flat_blank_window_cannot_be_tested_for_an_offset():
    with pytest.raises(MlsynthDataError, match="spread"):
        approximability(np.full(16, 4.0))


def test_non_finite_pool_is_refused(gaps):
    rng = np.random.default_rng(7)
    bad = [rng.normal(0.0, 1.0, 32), np.concatenate([rng.normal(0, 1, 31), [np.nan]])]
    with pytest.raises(MlsynthDataError, match="finite"):
        cumulative_path(gaps, bad)


def _ar1(rng, n, rho, sd=1.0):
    e = np.empty(n)
    e[0] = rng.normal(0.0, sd / np.sqrt(max(1.0 - rho ** 2, 1e-9)))
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0.0, sd)
    return e


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6])
def test_the_gate_does_not_punish_a_centred_window_for_being_dependent(rho):
    """Serial dependence shrinks the information in a window, not its centring.

    Dividing by ``sd / sqrt(n)`` treats dependent periods as independent, which
    understates the standard error and fires the gate on designs the donors
    reproduce perfectly well. Refusals must stay near the nominal level as the
    dependence rises.
    """
    rng = np.random.default_rng(100 + int(rho * 10))
    refused = sum(not approximability(_ar1(rng, 32, rho)).ok for _ in range(300))
    assert refused / 300 <= 0.10, f"gate refused {refused / 300:.0%} of centred windows"


def test_the_gate_still_catches_an_offset_under_dependence():
    """Widening the standard error must not blind the gate to a real offset."""
    rng = np.random.default_rng(200)
    caught = sum(not approximability(12.0 + _ar1(rng, 32, 0.6)).ok for _ in range(100))
    assert caught == 100


# ------------------------------------------------- the Unit-level design (10)

from mlsynth.utils.fast_scm_helpers.post_inference import unit_level_cumulative


@pytest.fixture
def unit_panel():
    """Three treated units, their blank and post gaps, and the design weights."""
    rng = np.random.default_rng(11)
    blank = rng.normal(0.0, 1.0, (32, 3))
    post = 2.0 + rng.normal(0.0, 1.0, (8, 3))
    return post, blank, np.array([0.5, 0.3, 0.2])


def test_unit_level_returns_an_aggregate_and_one_path_per_unit(unit_panel):
    post, blank, w = unit_panel
    out = unit_level_cumulative(post, blank, w)
    assert len(out.aggregate) == post.shape[0]
    assert len(out.per_unit) == w.size
    assert all(len(p) == post.shape[0] for p in out.per_unit)


def test_the_aggregate_is_the_weighted_mean_of_its_parts(unit_panel):
    """Equation (11). A headline that is the weighted mean of the breakdown is
    what lets the two be reported together without contradicting each other."""
    post, blank, w = unit_panel
    out = unit_level_cumulative(post, blank, w)
    for h in range(post.shape[0]):
        parts = np.array([out.per_unit[k][h].estimate for k in range(w.size)])
        assert out.aggregate[h].estimate == pytest.approx(float(w @ parts), rel=1e-12)


def test_a_single_treated_unit_reproduces_the_plain_path(unit_panel):
    post, blank, _ = unit_panel
    out = unit_level_cumulative(post[:, :1], blank[:, :1], np.array([1.0]))
    plain = cumulative_path(post[:, 0], [blank[:, 0]])
    assert np.allclose([p.estimate for p in out.aggregate], [p.estimate for p in plain])


def test_every_path_brackets_its_own_estimate(unit_panel):
    post, blank, w = unit_panel
    out = unit_level_cumulative(post, blank, w)
    for path in [out.aggregate, *out.per_unit]:
        for p in path:
            assert p.lower <= p.estimate <= p.upper


def test_extra_placebos_tighten_nothing_they_should_not(unit_panel):
    """Thickening the pool changes the tails; it must not move the estimates."""
    post, blank, w = unit_panel
    rng = np.random.default_rng(12)
    extra = [rng.normal(0.0, 1.0, 32) for _ in range(6)]
    a = unit_level_cumulative(post, blank, w)
    b = unit_level_cumulative(post, blank, w, extra_pool=extra)
    assert np.allclose([p.estimate for p in a.aggregate],
                       [p.estimate for p in b.aggregate])


def test_weights_that_do_not_sum_to_one_are_refused(unit_panel):
    post, blank, _ = unit_panel
    with pytest.raises(MlsynthConfigError, match="sum"):
        unit_level_cumulative(post, blank, np.array([0.5, 0.3, 0.1]))


def test_negative_weights_are_refused(unit_panel):
    post, blank, _ = unit_panel
    with pytest.raises(MlsynthConfigError, match="negative"):
        unit_level_cumulative(post, blank, np.array([1.2, -0.1, -0.1]))


def test_mismatched_unit_counts_are_refused(unit_panel):
    post, blank, w = unit_panel
    with pytest.raises(MlsynthDataError, match="unit"):
        unit_level_cumulative(post, blank[:, :2], w)


def test_the_estimand_is_named_on_the_result(unit_panel):
    """The interval covers the w-weighted effect on the treated, not the
    population effect, which differ once effects are heterogeneous."""
    post, blank, w = unit_panel
    assert unit_level_cumulative(post, blank, w).estimand == "treated"


def test_one_dimensional_gaps_are_read_as_a_single_treated_unit(unit_panel):
    """A lone treated unit may be handed in flat, without a length-one axis."""
    post, blank, _ = unit_panel
    flat = unit_level_cumulative(post[:, 0], blank[:, 0], np.array([1.0]))
    shaped = unit_level_cumulative(post[:, :1], blank[:, :1], np.array([1.0]))
    assert [p.estimate for p in flat.aggregate] == [p.estimate for p in shaped.aggregate]
    assert len(flat.per_unit) == 1


# ----------------------------------------------------- the population estimand

from mlsynth.utils.fast_scm_helpers.post_inference import population_cumulative


@pytest.fixture
def pop_panel():
    """Three of ten markets treated, with the population's own weights."""
    rng = np.random.default_rng(21)
    blank = rng.normal(0.0, 1.0, (32, 3))
    # genuinely different per-unit effects, so sigma_tau does not clip to zero
    post = np.array([1.0, 2.0, 4.0]) + rng.normal(0.0, 1.0, (8, 3))
    w = np.array([0.5, 0.3, 0.2])
    f = np.full(10, 0.1)
    return post, blank, w, f, np.array([0, 1, 2])


def test_population_path_has_one_row_per_horizon(pop_panel):
    post, blank, w, f, idx = pop_panel
    out = population_cumulative(post, blank, w, f, idx)
    assert len(out.aggregate) == post.shape[0]
    assert out.estimand == "population"


def test_the_point_estimate_is_unchanged_from_the_treated_path(pop_panel):
    """The representation error is mean zero, so it widens without moving."""
    post, blank, w, f, idx = pop_panel
    pop = population_cumulative(post, blank, w, f, idx)
    treated = unit_level_cumulative(post, blank, w)
    assert np.allclose([p.estimate for p in pop.aggregate],
                       [p.estimate for p in treated.aggregate])


def test_the_population_interval_is_never_narrower(pop_panel):
    post, blank, w, f, idx = pop_panel
    pop = population_cumulative(post, blank, w, f, idx)
    treated = unit_level_cumulative(post, blank, w)
    for a, b in zip(treated.aggregate, pop.aggregate):
        assert b.upper - b.lower >= a.upper - a.lower - 1e-9


def test_homogeneous_effects_leave_the_interval_alone(pop_panel):
    """With no spread across units there is nothing for w minus f to multiply."""
    _, blank, w, f, idx = pop_panel
    flat = np.tile(np.array([3.0, 3.0, 3.0]), (8, 1))      # identical every unit
    pop = population_cumulative(flat, blank, w, f, idx)
    treated = unit_level_cumulative(flat, blank, w)
    for a, b in zip(treated.aggregate, pop.aggregate):
        assert b.upper - b.lower == pytest.approx(a.upper - a.lower, rel=1e-6)


def test_the_widening_grows_faster_than_the_gap_term(pop_panel):
    """The representation term scales as h^2 against the gap term's h."""
    post, blank, w, f, idx = pop_panel
    pop = population_cumulative(post, blank, w, f, idx)
    treated = unit_level_cumulative(post, blank, w)
    extra = [(b.upper - b.lower) - (a.upper - a.lower)
             for a, b in zip(treated.aggregate, pop.aggregate)]
    assert extra[-1] > extra[0]


def test_a_treated_set_matching_the_population_adds_nothing(pop_panel):
    """When w equals f on every market there is no representation error."""
    post, blank, _, _, _ = pop_panel
    w = np.full(3, 1 / 3)
    pop = population_cumulative(post, blank, w, w, np.array([0, 1, 2]))
    treated = unit_level_cumulative(post, blank, w)
    for a, b in zip(treated.aggregate, pop.aggregate):
        assert b.upper - b.lower == pytest.approx(a.upper - a.lower, rel=1e-6)


def test_population_weights_that_do_not_sum_to_one_are_refused(pop_panel):
    post, blank, w, f, idx = pop_panel
    with pytest.raises(MlsynthConfigError, match="population"):
        population_cumulative(post, blank, w, np.full(10, 0.2), idx)


def test_treated_index_out_of_range_is_refused(pop_panel):
    post, blank, w, f, _ = pop_panel
    with pytest.raises(MlsynthDataError, match="index"):
        population_cumulative(post, blank, w, f, np.array([0, 1, 99]))


def test_treated_index_length_must_match_the_weights(pop_panel):
    post, blank, w, f, _ = pop_panel
    with pytest.raises(MlsynthDataError, match="index"):
        population_cumulative(post, blank, w, f, np.array([0, 1]))


def test_one_treated_unit_leaves_the_dispersion_unmeasurable(pop_panel):
    """A single treated unit gives no spread to read, so nothing is added."""
    post, blank, _, f, _ = pop_panel
    pop = population_cumulative(post[:, 0], blank[:, 0], np.array([1.0]),
                                f, np.array([0]))
    treated = unit_level_cumulative(post[:, 0], blank[:, 0], np.array([1.0]))
    assert pop.effect_dispersion == 0.0
    for a, b in zip(treated.aggregate, pop.aggregate):
        assert b.upper - b.lower == pytest.approx(a.upper - a.lower, rel=1e-9)


def test_the_result_reports_the_two_factors_it_multiplied(pop_panel):
    """Both are design facts a reader should be able to check independently."""
    post, blank, w, f, idx = pop_panel
    pop = population_cumulative(post, blank, w, f, idx)
    embedded = np.zeros_like(f); embedded[idx] = w
    assert pop.weight_distance == pytest.approx(float(np.linalg.norm(embedded - f)))
    assert pop.effect_dispersion > 0.0
