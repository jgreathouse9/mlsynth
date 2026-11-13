"""Groups drawn inside ``population_tolerance``: placebos and local restarts.

Two consumers of ``DesignRules.admits`` draw random treatment groups: the
calibration's placebo groups, above 20,000 candidate groups, and the local
search's restarts. Both used to draw uniformly over every candidate and keep
what the rules admitted, on a fixed budget (4,800 draws for 24 placebos, 50 a
restart). The budget assumes the rules admit a sizable share of groups, and the
tolerance admits a tiny one: on simulated 80-market panels at a tolerance of
1.5 the placebos delivered had a median of 3 of the 24 requested, and the
calibrated band's coverage fell with them.

The tolerance now produces its groups itself, from the closed form of the
anchor distance: an exact enumeration up to its own limit, a batched uniform
sampler above it. What these tests pin:

* the defect, reproduced: on a 40-geo panel with 91,390 candidate groups of
  four, of which the tolerance admits 57 (1 in 1,603), the calibration gets
  all 24 placebos it asks for and no local restart is abandoned;
* the enumeration is the set of groups ``admits`` accepts, each confirmed by
  ``admits`` itself, so a group the closed form passes and ``admits`` refuses
  stays out;
* every drawn group is admitted, by the tolerance and the business rules,
  distinct, of the requested size, holds every forced geo and is not the real
  group;
* below the tolerance's limit a draw delivers ``min(n, admissible)``: the
  sampler's first draws when they fill the request, the enumeration with its
  exact count when they do not; above it the sampler stops at the count or at
  its draw cap;
* the same seed gives the same groups;
* without a tolerance the calibration's rejection draws are unchanged.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers import calibration as cal
from mlsynth.utils.tbr_helpers.design import local as loc
from mlsynth.utils.tbr_helpers.design import tolerance as tol
from mlsynth.utils.tbr_helpers.design.rules import DesignRules
from mlsynth.utils.tbr_helpers.design.search import (CONTROL, ROLES, TREATMENT,
                                                     UNASSIGNED, _Scorer,
                                                     greedy_search)

ALL = frozenset(ROLES)
FORCED = frozenset({TREATMENT})

#: The tight panel: 40 geos, 60 periods, the first 42 the design window.
N_GEOS, N_PERIODS, DESIGN, BLANK, FACTOR, K = 40, 60, 42, 18, 1.3, 4

#: Draws enough to reach every group of a nine-geo panel many times over.
SPARE = 100_000


def market_panel(n_geos, n_periods, seed, noise=0.04):
    """Periods by geos: geos of differing size whose response to one
    market-wide trend differs."""
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    beta = rng.uniform(0.2, 1.8, n_geos)
    t = np.arange(n_periods)
    trend = 0.25 * np.sin(2.0 * np.pi * t / 17.0) + 0.3 * t / n_periods
    e = noise * rng.normal(size=(n_periods, n_geos))
    return size * (1.0 + beta * trend[:, None] + e)


def groups_of(roles, k):
    """Every candidate group of size k: the forced geos and k - |forced| more."""
    forced = [j for j, r in enumerate(roles) if r == FORCED]
    free = [j for j, r in enumerate(roles) if TREATMENT in r and j not in forced]
    if k < len(forced):
        return []
    return [sorted(forced + list(c)) for c in combinations(free, k - len(forced))]


def brute(t, roles, k, rules=None):
    """The groups ``admits`` accepts, by asking it about every candidate."""
    return [g for g in groups_of(roles, k)
            if t.admits(g) and (rules is None or rules.admits_rules(g))]


@pytest.fixture(scope="module")
def tight():
    """A panel where the tolerance admits under 1 in 1,000 groups of four."""
    Y = market_panel(N_GEOS, N_PERIODS, seed=3)
    W = Y[:DESIGN]
    roles = [ALL] * N_GEOS
    t = tol.PopulationTolerance(W, roles, None, FACTOR)
    rules = DesignRules.build(roles).with_tolerance(t)
    real = sorted(t.anchor(K).witness)
    return SimpleNamespace(Y=Y, W=W, roles=roles, t=t, rules=rules, real=real,
                           admitted=brute(t, roles, K))


def small(seed=0, n_geos=9, factor=1.6, roles=None, rules=None, **kw):
    Y = market_panel(n_geos, 30, seed)
    roles = [ALL] * n_geos if roles is None else roles
    t = tol.PopulationTolerance(Y, roles, rules, factor, **kw)
    base = DesignRules.build(roles) if rules is None else rules
    return Y, roles, t, base.with_tolerance(t)


def assert_valid(groups, t, roles, k, *, rules=None, real=None):
    """Admitted, distinct, of size k, every forced geo, never the real group."""
    forced = {j for j, r in enumerate(roles) if r == FORCED}
    assert len({tuple(g) for g in groups}) == len(groups), "duplicates"
    for g in groups:
        assert g == sorted(g) and len(set(g)) == k, g
        assert forced <= set(g), g
        assert t.admits(g), g
        assert rules is None or rules.admits_rules(g), g
        assert real is None or g != sorted(real), g


# --------------------------------------------------------------------------- #
# the defect, reproduced
# --------------------------------------------------------------------------- #
def test_the_tight_panel_is_the_regime_that_failed(tight):
    total = math.comb(N_GEOS, K)
    assert total > cal.ENUMERATION_LIMIT          # the rejection branch
    assert total <= tight.t.limit                 # enumerable by the tolerance
    assert 25 <= len(tight.admitted) < total / 1000
    assert tight.real in tight.admitted


def test_placebo_groups_deliver_every_requested_group(tight):
    """Main delivered a handful here: 4,800 uniform draws at an acceptance of
    57 in 91,390 expect three admitted groups."""
    groups, n_admissible = cal.placebo_groups(tight.real, tight.roles,
                                              tight.rules, 24, seed=0)
    assert len(groups) == 24, f"{len(groups)} placebos of 24"
    assert n_admissible == len(tight.admitted) - 1
    assert_valid(groups, tight.t, tight.roles, K, real=tight.real)


def test_the_calibration_pools_every_requested_placebo(tight):
    scorer = _Scorer(tight.W, "relative", 6)
    control = cal.match_controls(scorer, tight.real, tight.roles, tight.rules)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blank = cal.calibrate(tight.Y, tight.real, control, blank_periods=BLANK,
                              horizon=6, eligibility=tight.roles,
                              rules=tight.rules, objective="relative",
                              n_test=6, n_placebos=24, seed=0)
    assert len(blank.placebos) == 24
    assert blank.n_admissible == len(tight.admitted) - 1
    assert not [w for w in caught if "placebo design" in str(w.message)]


def test_local_restarts_are_never_abandoned(tight):
    """Main abandoned about 97% of these: 50 draws a restart at 57 in 91,390."""
    out = greedy_search(tight.W, tight.roles, max_treatment_size=K, n_test=6,
                        objective="relative", rules=tight.rules)
    start = next(o for o in out if o.k == K)
    res = loc.local_search(tight.W, tight.roles, start, n_test=6, restarts=10,
                           seed=0, rules=tight.rules, objective="relative")
    assert res.abandoned_restarts == 0
    assert tight.rules.admits_design(res.treatment, res.control)


def test_the_restart_starts_are_admitted_designs(tight):
    rng = np.random.default_rng(0)
    starts = loc._tolerance_starts(rng, tight.roles, K, tight.rules, 12)
    assert len(starts) == 12 and None not in starts
    for T, C in starts:
        assert tight.rules.admits_design(T, C)
        assert tuple(T) in {tuple(g) for g in tight.admitted}


# --------------------------------------------------------------------------- #
# unit: the enumeration
# --------------------------------------------------------------------------- #
def test_the_enumeration_is_the_admitted_set(tight):
    got = tight.t.admitted(K)
    assert got.shape == (len(tight.admitted), K)
    assert sorted(map(tuple, got.tolist())) == sorted(map(tuple, tight.admitted))


def test_the_enumeration_is_kept_per_size(tight):
    assert tight.t.admitted(K) is tight.t.admitted(K)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_enumeration_matches_brute_force_on_a_small_panel(k):
    _, roles, t, _ = small()
    assert sorted(map(tuple, t.admitted(k).tolist())) == \
        sorted(map(tuple, brute(t, roles, k)))


def test_the_enumeration_runs_in_blocks():
    _, roles, t, _ = small(factor=3.0)
    whole = t.admitted(3)
    _, _, t2, _ = small(factor=3.0)
    t2._admitted[3] = t2._enumerate_admitted(3, block=7)
    assert len(whole) > 7
    assert t2.admitted(3).tolist() == whole.tolist()


def test_a_group_the_closed_form_passes_and_admits_refuses_stays_out():
    """The screen carries a margin above the bound, so a group just beyond it
    passes the closed form; ``admits`` refuses it and so must the set."""
    Y, roles, t, _ = small()
    d_min = t.anchor(3).d_min
    g = sorted(max(groups_of(roles, 3), key=lambda g: (t.distance(g) < 1.4 * d_min,
                                                       t.distance(g))))
    factor = t.distance(g) / (d_min * (1.0 + tol.TIE_RTOL)) / (1.0 + 1e-8)
    t2 = tol.PopulationTolerance(Y, roles, None, factor)
    P = tol._Problem(Y, roles, None)
    bound = t2.threshold(3)
    assert P.d2(np.asarray([g]))[0] <= tol._screen(bound)   # passes the screen
    assert not t2.admits(g)
    assert g not in t2.admitted(3).tolist()
    rng = np.random.default_rng(0)
    assert g not in t2.sample(3, 10_000, rng, cap=200_000)


def test_a_tie_with_the_witness_is_in_the_set_at_factor_one():
    Y = market_panel(8, 30, seed=1)
    p = Y.sum(axis=1)
    j = int(np.argmin([tol.scaled_gap(Y[:, i], p) for i in range(8)]))
    Y = np.column_stack([Y] + [c * Y[:, j] for c in (0.37, 2.9, 11.3)])
    t = tol.PopulationTolerance(Y, [ALL] * 11, None, 1.0)
    ties = [j, 8, 9, 10]
    assert max(t.distance([g]) for g in ties) > t.anchor(1).d_min
    assert sorted(t.admitted(1)[:, 0].tolist()) == sorted(ties)


def test_a_size_without_an_anchor_admits_nothing():
    Y = market_panel(6, 20, seed=2)
    roles = [ALL] * 3 + [frozenset({CONTROL})] * 3
    t = tol.PopulationTolerance(Y, roles, None, 2.0)
    assert t.admitted(4).shape == (0, 4)
    assert t.sample(4, 5, np.random.default_rng(0)) == []
    assert t.candidates(4) == 0
    assert t.draw(4, 5, np.random.default_rng(0), eligibility=roles) == ([], 0)


def test_the_forced_group_alone():
    Y = market_panel(6, 20, seed=2)
    roles = [FORCED, FORCED] + [ALL] * 4
    t = tol.PopulationTolerance(Y, roles, None, 1.0, limit=0)
    assert t.candidates(2) == 1
    assert t.admitted(2).tolist() == [[0, 1]]
    assert t.sample(2, 3, np.random.default_rng(0), cap=1000) == [[0, 1]]
    assert t.sample(2, 3, np.random.default_rng(0), distinct=False) == \
        [[0, 1]] * 3


@pytest.mark.parametrize("k,expected", [(1, 0), (2, 1), (3, 4), (4, 6), (6, 1),
                                        (7, 0)])
def test_the_candidate_count(k, expected):
    Y = market_panel(7, 20, seed=2)
    roles = [ALL] * 4 + [FORCED, FORCED, frozenset({CONTROL})]
    t = tol.PopulationTolerance(Y, roles, None, 2.0)
    assert t.candidates(k) == expected


# --------------------------------------------------------------------------- #
# unit: drawing
# --------------------------------------------------------------------------- #
def test_draw_below_the_limit_takes_from_the_enumeration(monkeypatch):
    monkeypatch.setattr(tol, "QUICK_DRAWS", 0)
    _, roles, t, rules = small()
    pool = brute(t, roles, 3)
    real = pool[0]
    groups, n = t.draw(3, 5, np.random.default_rng(1), eligibility=roles,
                       exclude=[real])
    assert n == len(pool) - 1
    assert len(groups) == 5
    assert_valid(groups, t, roles, 3, real=real)


def test_draw_takes_every_group_when_fewer_are_admissible():
    _, roles, t, _ = small()
    pool = brute(t, roles, 3)
    groups, n = t.draw(3, 10 * len(pool), np.random.default_rng(1),
                       eligibility=roles, exclude=[pool[0]])
    assert sorted(groups) == sorted(pool[1:]) and n == len(pool) - 1


def test_a_request_the_sampler_fills_is_not_enumerated():
    """A loose tolerance: the first draws find every group asked for, so the
    enumeration, which would confirm most candidates one by one, never runs."""
    _, roles, t, _ = small(factor=50.0)
    t._enumerate_admitted = None                   # never called
    groups, n = t.draw(3, 6, np.random.default_rng(1), eligibility=roles)
    assert n is None and len(groups) == 6
    assert_valid(groups, t, roles, 3)
    assert 3 not in t._admitted


def test_a_request_the_sampler_cannot_fill_is_enumerated(tight):
    """A tight tolerance: 200 draws per group asked for find a handful, and
    the enumeration supplies all 24 with the exact count."""
    quick = tight.t.sample(K, 24, np.random.default_rng(0), exclude=[tight.real],
                           cap=tol.QUICK_DRAWS * 24)
    assert len(quick) < 8
    groups, n = tight.t.draw(K, 24, np.random.default_rng(0),
                             eligibility=tight.roles, exclude=[tight.real])
    assert len(groups) == 24 and n == len(tight.admitted) - 1
    assert_valid(groups, tight.t, tight.roles, K, real=tight.real)


def test_draw_above_the_limit_samples():
    _, roles, t, _ = small(limit=10)
    pool = brute(t, roles, 3)
    assert t.candidates(3) > t.limit
    groups, n = t.draw(3, 4, np.random.default_rng(1), eligibility=roles,
                       exclude=[pool[0]])
    assert n is None and len(groups) == 4
    assert_valid(groups, t, roles, 3, real=pool[0])


def test_the_limit_is_inclusive(monkeypatch):
    monkeypatch.setattr(tol, "QUICK_DRAWS", 0)
    _, roles, t, _ = small(limit=math.comb(9, 3))
    assert t.draw(3, 2, np.random.default_rng(0), eligibility=roles)[1] \
        == len(brute(t, roles, 3))


def test_the_sampler_reaches_every_admitted_group():
    _, roles, t, _ = small(limit=0)
    pool = brute(t, roles, 3)
    got = t.sample(3, len(pool) + 5, np.random.default_rng(3), cap=SPARE)
    assert sorted(got) == sorted(pool)


def test_the_sampler_never_repeats_a_group():
    _, roles, t, _ = small(limit=0)
    pool = brute(t, roles, 3)
    got = t.sample(3, len(pool), np.random.default_rng(3))
    assert len(got) == len(set(map(tuple, got))) == len(pool)


def test_draws_with_replacement_repeat_groups():
    _, roles, t, _ = small()
    pool = brute(t, roles, 3)
    n = 4 * len(pool)
    for limit in (t.limit, 0):
        t.limit = limit
        groups, _ = t.draw(3, n, np.random.default_rng(2), eligibility=roles,
                           distinct=False)
        assert len(groups) == n
        assert len(set(map(tuple, groups))) < n
        assert all(g in pool for g in groups)


def test_the_sampler_is_uniform_over_the_admitted_groups():
    """A frequency check with a fixed seed: 30,000 draws with replacement over
    the admitted groups, a chi-square statistic against equal frequencies well
    inside its 0.999 quantile."""
    _, roles, t, _ = small(limit=0, factor=2.0)
    pool = [tuple(g) for g in brute(t, roles, 3)]
    n = 30_000
    got = Counter(map(tuple, t.sample(3, n, np.random.default_rng(11),
                                      distinct=False)))
    assert set(got) == set(pool)
    e = n / len(pool)
    chi2 = sum((got[g] - e) ** 2 / e for g in pool)
    df = len(pool) - 1
    assert chi2 < df + 4.0 * math.sqrt(2.0 * df) + 10.0, (chi2, df)


def test_the_sampler_honours_the_business_rules_and_the_exclusion():
    Y = market_panel(9, 30, seed=0)
    roles = [ALL] * 9
    t0 = tol.PopulationTolerance(Y, roles, None, 1.6)
    pool = brute(t0, roles, 3)
    witness = sorted(t0.anchor(3).witness)
    other = next(g for g in pool if g != witness
                 and not {g[0], g[1]} <= set(witness))
    rules = DesignRules.build(roles, conflict_pairs=[(other[0], other[1])])
    t = tol.PopulationTolerance(Y, roles, rules, 1.6, limit=0)
    kept = brute(t, roles, 3, rules)
    assert other not in kept and len(kept) >= 3
    got = t.sample(3, 1000, np.random.default_rng(0), keep=rules.admits_rules,
                   exclude=[kept[0]], cap=SPARE)
    assert sorted(got) == sorted(kept[1:])


def test_the_sampler_stops_at_its_cap():
    _, roles, t, _ = small(limit=0)
    assert t.sample(3, 5, np.random.default_rng(0), cap=0) == []
    few = t.sample(3, 1000, np.random.default_rng(0), cap=40)
    assert len(few) < len(brute(t, roles, 3))
    assert_valid(few, t, roles, 3)


def test_the_sampler_draws_in_batches():
    _, roles, t, _ = small(limit=0)
    a = t.sample(3, 6, np.random.default_rng(5), batch=7)
    assert len(a) == 6
    assert_valid(a, t, roles, 3)


def test_nothing_is_drawn_for_a_count_of_zero():
    _, roles, t, _ = small(limit=0)
    assert t.sample(3, 0, np.random.default_rng(0)) == []
    assert t.draw(3, 0, np.random.default_rng(0), eligibility=roles)[0] == []


def test_forced_geos_are_in_every_drawn_group():
    roles = [FORCED] + [ALL] * 8
    _, roles, t, _ = small(roles=roles, limit=0)
    got = t.sample(3, 20, np.random.default_rng(0), cap=SPARE)
    assert got and all(0 in g for g in got)
    assert_valid(got, t, roles, 3)


@pytest.mark.parametrize("limit", [None, 0])
def test_the_same_seed_gives_the_same_groups(limit):
    kw = {} if limit is None else {"limit": limit}
    _, roles, t, _ = small(**kw)
    a = t.draw(3, 5, np.random.default_rng(7), eligibility=roles)
    b = t.draw(3, 5, np.random.default_rng(7), eligibility=roles)
    assert a == b


def test_a_draw_against_other_roles_is_refused():
    _, roles, t, _ = small()
    other = [FORCED] + roles[1:]
    with pytest.raises(MlsynthDataError, match="eligibility"):
        t.draw(3, 5, np.random.default_rng(0), eligibility=other)


# --------------------------------------------------------------------------- #
# the consumers
# --------------------------------------------------------------------------- #
def main_rejection(treatment, eligibility, rules, n_placebos, seed):
    """The calibration's rejection draws as they were on main (848dfb08)."""
    real = sorted(int(j) for j in treatment)
    forced = [j for j, r in enumerate(eligibility) if r == frozenset({TREATMENT})]
    free = [j for j, r in enumerate(eligibility)
            if TREATMENT in r and j not in forced]
    m = len(real) - len(forced)
    rng = np.random.default_rng(seed)
    found, seen = [], set()
    for _ in range(cal.DRAWS_PER_PLACEBO * n_placebos):
        g = sorted(forced + [free[i] for i in
                             rng.choice(len(free), size=m, replace=False)])
        if tuple(g) not in seen and g != real and \
                (rules is None or rules.admits(g)):
            seen.add(tuple(g))
            found.append(g)
            if len(found) == n_placebos:
                break
    return found


@pytest.mark.parametrize("with_rules", [False, True])
def test_without_a_tolerance_the_rejection_draws_are_unchanged(with_rules):
    roles = [ALL] * 30
    rules = (DesignRules.build(roles, conflict_pairs=[(0, 1), (2, 3)])
             if with_rules else None)
    real = [4, 7, 9, 12]
    got = cal.placebo_groups(real, roles, rules, 24, seed=3)
    assert math.comb(30, 4) > cal.ENUMERATION_LIMIT
    assert got == (main_rejection(real, roles, rules, 24, 3), None)


def test_placebos_below_the_calibration_limit_keep_its_enumeration():
    """Below 20,000 candidates the calibration enumerates on its own, with
    or without the tolerance, and the tolerance's draw is not consulted."""
    _, roles, t, rules = small()
    real = sorted(t.anchor(3).witness)
    t.draw = None                                  # never called
    groups, n = cal.placebo_groups(real, roles, rules, 500, seed=0)
    assert n == len(brute(t, roles, 3)) - 1
    assert sorted(groups) == sorted(g for g in brute(t, roles, 3) if g != real)


def test_placebos_above_the_tolerance_limit_are_sampled(tight):
    t = tol.PopulationTolerance(tight.W, tight.roles, None, FACTOR, limit=1000)
    rules = DesignRules.build(tight.roles).with_tolerance(t)
    groups, n = cal.placebo_groups(tight.real, tight.roles, rules, 24, seed=0)
    assert n is None and len(groups) == 24
    assert_valid(groups, t, tight.roles, K, real=tight.real)


def test_placebos_keep_the_business_rules(tight):
    """A conflict between two geos of an admitted group: the tolerance admits
    the group on distance, and the rules refuse it."""
    other = next(g for g in tight.admitted
                 if not {g[0], g[1]} <= set(tight.real))
    base = DesignRules.build(tight.roles, conflict_pairs=[(other[0], other[1])])
    rules = base.with_tolerance(tight.t)
    kept = [g for g in tight.admitted if rules.admits_rules(g)]
    groups, n = cal.placebo_groups(tight.real, tight.roles, rules, 500, seed=0)
    assert other not in groups
    assert n == len(kept) - 1
    assert sorted(groups) == sorted(g for g in kept if g != tight.real)


def test_placebo_draws_are_reproducible(tight):
    a = cal.placebo_groups(tight.real, tight.roles, tight.rules, 24, seed=4)
    b = cal.placebo_groups(tight.real, tight.roles, tight.rules, 24, seed=4)
    c = cal.placebo_groups(tight.real, tight.roles, tight.rules, 24, seed=5)
    assert a == b and a != c


def test_restarts_without_a_tolerance_draw_as_before():
    """The rules alone: the restart draws by rejection, untouched."""
    roles = [ALL] * 8
    rules = DesignRules.build(roles, conflict_pairs=[(0, 1)])
    rng = np.random.default_rng(0)
    draws = [loc._random_design(rng, roles, 3, rules) for _ in range(5)]
    rng = np.random.default_rng(0)
    again = [loc._random_design(rng, roles, 3, rules) for _ in range(5)]
    assert draws == again and all(d is not None for d in draws)


def test_the_restart_controls_respect_the_band_and_the_bans():
    Y = market_panel(10, 30, seed=0)
    roles = [ALL] * 8 + [frozenset({CONTROL})] * 2
    base = DesignRules.build(roles, conflict_pairs=[(0, 8), (1, 2)],
                             exclude_bordering_donors=True, min_controls=2,
                             max_controls=4)
    t = tol.PopulationTolerance(Y, roles, base, 2.0)
    rules = base.with_tolerance(t)
    starts = loc._tolerance_starts(np.random.default_rng(0), roles, 2, rules, 20)
    assert None not in starts
    for T, C in starts:
        assert rules.admits_design(T, C), (T, C)
        assert {8, 9} - set(rules.banned_controls(T)) <= C


def test_restart_groups_whose_controls_cannot_fit_are_not_drawn():
    """Geo 0 borders a geo that has to be a control; a group holding it leaves no
    control group, so no restart starts from it."""
    Y = market_panel(8, 30, seed=0)
    roles = [ALL] * 6 + [frozenset({CONTROL})] * 2
    base = DesignRules.build(roles, conflict_pairs=[(0, 6)],
                             exclude_bordering_donors=True)
    t = tol.PopulationTolerance(Y, roles, base, 50.0)
    rules = base.with_tolerance(t)
    starts = loc._tolerance_starts(np.random.default_rng(0), roles, 2, rules, 40)
    assert None not in starts
    assert not any(0 in T for T, _ in starts)


def test_restarts_are_abandoned_only_when_the_sampler_finds_nothing(tight):
    t = tol.PopulationTolerance(tight.W, tight.roles, None, FACTOR, limit=0)
    rules = DesignRules.build(tight.roles).with_tolerance(t)
    starts = loc._tolerance_starts(np.random.default_rng(0), tight.roles, K,
                                   rules, 3)
    assert None not in starts
    t.sample = lambda *a, **kw: []
    assert loc._tolerance_starts(np.random.default_rng(0), tight.roles, K,
                                 rules, 3) == [None] * 3


def test_a_local_search_with_the_tolerance_is_reproducible():
    Y, roles, t, rules = small(n_geos=9, factor=1.4)
    out = greedy_search(Y, roles, max_treatment_size=3, n_test=6,
                        objective="relative", rules=rules)
    a = loc.local_search(Y, roles, out[-1], n_test=6, restarts=3, seed=2,
                         rules=rules, objective="relative")
    b = loc.local_search(Y, roles, out[-1], n_test=6, restarts=3, seed=2,
                         rules=rules, objective="relative")
    assert (a.treatment, a.control, a.score.key, a.evaluations) == \
        (b.treatment, b.control, b.score.key, b.evaluations)
    assert a.abandoned_restarts == 0
    assert rules.admits_design(a.treatment, a.control)


def test_a_group_that_leaves_too_few_controls_has_no_room():
    roles = [ALL] * 6
    rules = DesignRules.build(roles, min_controls=3)
    assert loc._control_room(roles, [0, 1, 2], rules) is not None
    assert loc._control_room(roles, [0, 1, 2, 3], rules) is None
