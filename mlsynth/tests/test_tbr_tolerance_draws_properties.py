"""Properties of the groups ``population_tolerance`` enumerates and draws.

Asserted by ``hypothesis`` over random panels, eligibilities, conflicts and
factors, not only at the fixtures in ``test_tbr_tolerance_draws.py``:

* the enumeration is the brute-force set of groups ``admits`` accepts;
* every group drawn, enumerated or sampled, is admitted by the tolerance and
  the business rules, distinct, of the requested size, holds every forced geo
  and is not the excluded group;
* below the limit a draw delivers ``min(n, admissible)``, and reports the
  count whenever it enumerated;
* a sampler with draws to spare reaches every admitted group, which with the
  sampler's uniform proposal is uniformity's support;
* the calibration's placebos under the tolerance are ``min(n, admissible)``
  whenever the candidates are within the tolerance's limit.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.tbr_helpers import calibration as cal
from mlsynth.utils.tbr_helpers.design import tolerance as tol
from mlsynth.utils.tbr_helpers.design.rules import DesignRules
from mlsynth.utils.tbr_helpers.design.search import (CONTROL, ROLES, TREATMENT,
                                                     UNASSIGNED)

ALL = frozenset(ROLES)
FORCED = frozenset({TREATMENT})
_SETTINGS = dict(max_examples=40, deadline=None, derandomize=True,
                 suppress_health_check=[HealthCheck.too_slow])

#: Draws enough to reach every group of a nine-geo panel many times over.
SPARE = 100_000


def _panel(seed, n_geos, n_periods):
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    beta = rng.uniform(0.2, 1.8, n_geos)
    t = np.arange(n_periods)
    trend = 0.25 * np.sin(2.0 * np.pi * t / 11.0) + 0.3 * t / n_periods
    noise = 0.05 * rng.normal(size=(n_periods, n_geos))
    return size * (1.0 + beta * trend[:, None] + noise)


def _groups(roles, k):
    forced = [j for j, r in enumerate(roles) if r == FORCED]
    free = [j for j, r in enumerate(roles) if TREATMENT in r and j not in forced]
    if k < len(forced):
        return []
    return [sorted(forced + list(c)) for c in combinations(free, k - len(forced))]


def _brute(t, roles, k, rules):
    return [g for g in _groups(roles, k)
            if t.admits(g) and (rules is None or rules.admits_rules(g))]


@st.composite
def problems(draw, min_geos=5, max_geos=9):
    """A panel, each geo's roles, conflicts, a tolerance and the rules."""
    N = draw(st.integers(min_geos, max_geos))
    Y = _panel(draw(st.integers(0, 10_000)), N, draw(st.integers(12, 40)))
    n_forced = draw(st.integers(0, 1))
    roles = [FORCED] * n_forced + [ALL] * (N - n_forced - 1)
    roles.append(draw(st.sampled_from([ALL, frozenset({CONTROL}),
                                       frozenset({CONTROL, UNASSIGNED})])))
    pairs = draw(st.lists(st.tuples(st.integers(n_forced, N - 1),
                                    st.integers(n_forced, N - 1))
                          .filter(lambda p: p[0] != p[1]), max_size=3))
    rules = DesignRules.build(roles, conflict_pairs=pairs) if pairs else None
    factor = draw(st.floats(1.0, 3.0))
    t = tol.PopulationTolerance(Y, roles, rules, factor)
    full = (DesignRules.build(roles) if rules is None else rules).with_tolerance(t)
    return Y, roles, rules, t, full


def _check(groups, t, roles, k, rules, real):
    forced = {j for j, r in enumerate(roles) if r == FORCED}
    assert len({tuple(g) for g in groups}) == len(groups)
    for g in groups:
        assert g == sorted(g) and len(set(g)) == k
        assert forced <= set(g)
        assert t.admits(g)
        assert rules is None or rules.admits_rules(g)
        assert real is None or g != real


@settings(**_SETTINGS)
@given(problems(), st.integers(1, 4))
def test_the_enumeration_is_the_brute_force_admitted_set(problem, k):
    _, roles, _, t, _ = problem
    got = sorted(map(tuple, t.admitted(k).tolist()))
    assert got == sorted(map(tuple, _brute(t, roles, k, None)))


@settings(**_SETTINGS)
@given(problems(), st.integers(1, 4), st.integers(0, 30), st.integers(0, 99),
       st.booleans())
def test_drawn_groups_are_admitted_distinct_and_exclude_the_real(
        problem, k, n, seed, sample):
    _, roles, rules, t, _ = problem
    pool = _brute(t, roles, k, rules)
    real = pool[0] if pool else None
    if sample:
        t.limit = 0
    cap, tol.SAMPLE_DRAWS = tol.SAMPLE_DRAWS, SPARE
    try:
        groups, count = t.draw(k, n, np.random.default_rng(seed),
                               eligibility=roles,
                               keep=None if rules is None else rules.admits_rules,
                               exclude=[] if real is None else [real])
    finally:
        tol.SAMPLE_DRAWS = cap
    _check(groups, t, roles, k, rules, real)
    others = len(pool) - (real is not None)
    if sample and t.candidates(k) > 0:
        assert count is None
        assert len(groups) <= min(n, others)
    else:
        # The quick draws fill the request (count None) or the enumeration
        # does (count exact); either way min(n, others) groups come back.
        assert count is None or count == others
        assert count is not None or len(groups) == n
        assert len(groups) == min(n, others)


@settings(**_SETTINGS)
@given(problems(), st.integers(1, 4), st.integers(0, 99))
def test_the_sampler_reaches_every_admitted_group(problem, k, seed):
    _, roles, rules, t, _ = problem
    pool = _brute(t, roles, k, rules)
    got = t.sample(k, len(pool) + 1, np.random.default_rng(seed),
                   keep=None if rules is None else rules.admits_rules,
                   cap=SPARE)
    assert sorted(got) == sorted(pool)


@settings(**_SETTINGS)
@given(problems(), st.integers(1, 4), st.integers(1, 40), st.integers(0, 99))
def test_draws_with_replacement_stay_in_the_admitted_set(problem, k, n, seed):
    _, roles, rules, t, _ = problem
    pool = {tuple(g) for g in _brute(t, roles, k, rules)}
    keep = None if rules is None else rules.admits_rules
    for limit in (t.limit, 0):
        t.limit = limit
        groups, _ = t.draw(k, n, np.random.default_rng(seed), eligibility=roles,
                           keep=keep, distinct=False)
        assert len(groups) == (n if pool else 0)
        assert {tuple(g) for g in groups} <= pool


@settings(**_SETTINGS)
@given(problems(min_geos=7, max_geos=9), st.integers(1, 30), st.integers(0, 99))
def test_the_calibration_delivers_min_n_admissible(problem, n, seed):
    """With the calibration's own limit at zero every size goes to the
    tolerance, and the tolerance enumerates."""
    _, roles, rules, t, full = problem
    real = sorted(t.anchor(3).witness) if t.anchor(3) else None
    if real is None:
        return
    pool = [g for g in _brute(t, roles, 3, rules) if g != real]
    old = cal.ENUMERATION_LIMIT
    cal.ENUMERATION_LIMIT = 0
    try:
        groups, count = cal.placebo_groups(real, roles, full, n, seed)
    finally:
        cal.ENUMERATION_LIMIT = old
    assert count is None or count == len(pool)
    assert len(groups) == min(n, len(pool))
    _check(groups, t, roles, 3, rules, real)
