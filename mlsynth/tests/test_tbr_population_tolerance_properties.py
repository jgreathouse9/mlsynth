"""Properties of ``population_tolerance``, over the input domain.

Each invariant the tolerance promises, asserted by ``hypothesis`` over random
panels, eligibilities and rules, not only at the fixtures in
``test_tbr_population_tolerance.py``:

* the anchor distance is unit-free: scaling every geo by one positive constant
  leaves it unchanged, and so does scaling the treated aggregate or the
  population path alone;
* ``d_min(k)`` is the distance of its witness, the witness is a usable group
  of size ``k`` and is admitted at every factor of at least one;
* the enumerated ``d_min`` is the brute-force minimum, under forced geos and
  conflicts;
* the searched ``d_min`` is at least the enumerated one and is a one-swap
  local minimum among the usable groups;
* every design the greedy records is within the tolerance of its own size, and
  every size with a usable group returns one;
* ``extendable`` does not read the tolerance.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design import tolerance as tol
from mlsynth.utils.tbr_helpers.design.rules import DesignRules
from mlsynth.utils.tbr_helpers.design.search import (CONTROL, ROLES, TREATMENT,
                                                     UNASSIGNED, greedy_search)

ALL = frozenset(ROLES)
FORCED = frozenset({TREATMENT})
_SETTINGS = dict(max_examples=40, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow])


def _panel(seed, n_geos, n_periods):
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    beta = rng.uniform(0.2, 1.8, n_geos)
    t = np.arange(n_periods)
    trend = 0.25 * np.sin(2.0 * np.pi * t / 11.0) + 0.3 * t / n_periods
    noise = 0.05 * rng.normal(size=(n_periods, n_geos))
    return size * (1.0 + beta * trend[:, None] + noise)


def _literal(Y, T):
    y = Y[:, sorted(T)].sum(axis=1)
    p = Y.sum(axis=1)
    return math.sqrt(float(np.mean((y / y.mean() - p / p.mean()) ** 2)))


def _usable(T, roles, rules):
    control = {j for j, r in enumerate(roles) if CONTROL in r}
    if not control - set(T):
        return False
    return rules is None or (rules.admits(T) and rules.extendable(T, len(T)))


def _groups(roles, k):
    forced = [j for j, r in enumerate(roles) if r == FORCED]
    free = [j for j, r in enumerate(roles) if TREATMENT in r and j not in forced]
    for c in combinations(free, k - len(forced)):
        yield sorted(forced + list(c))


@st.composite
def problems(draw, min_geos=5, max_geos=9):
    """A panel, each geo's roles and group rules, with every size feasible."""
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
    return Y, roles, rules


panels = st.tuples(st.integers(0, 10_000), st.integers(3, 9),
                   st.integers(8, 40))


@settings(**_SETTINGS)
@given(panels, st.floats(1e-3, 1e3), st.data())
def test_the_distance_is_invariant_to_the_panels_scale(panel, c, data):
    seed, N, n = panel
    Y = _panel(seed, N, n)
    T = data.draw(st.lists(st.integers(0, N - 1), min_size=1, max_size=N - 1,
                           unique=True))
    assert tol.anchor_distance(c * Y, T) == pytest.approx(
        tol.anchor_distance(Y, T), rel=1e-9, abs=1e-12)


@settings(**_SETTINGS)
@given(panels, st.floats(1e-3, 1e3), st.floats(1e-3, 1e3), st.data())
def test_the_gap_is_invariant_to_scaling_either_path(panel, a, b, data):
    """Scaling the treated geos jointly scales their aggregate, and the gap
    reads the aggregate only through its own mean; the population likewise."""
    seed, N, n = panel
    Y = _panel(seed, N, n)
    T = data.draw(st.lists(st.integers(0, N - 1), min_size=1, max_size=N,
                           unique=True))
    y, p = Y[:, T].sum(axis=1), Y.sum(axis=1)
    base = tol.scaled_gap(y, p)
    assert base == pytest.approx(_literal(Y, T), rel=1e-9, abs=1e-12)
    assert tol.scaled_gap(a * y, b * p) == pytest.approx(base, rel=1e-9,
                                                          abs=1e-12)


@settings(**_SETTINGS)
@given(problems(), st.integers(1, 4), st.floats(1.0, 3.0))
def test_the_witness_attains_d_min_and_is_admitted(problem, k, factor):
    Y, roles, rules = problem
    t = tol.PopulationTolerance(Y, roles, rules, factor)
    a = t.anchor(k)
    if a is None:
        assert not any(_usable(T, roles, rules) for T in _groups(roles, k))
        return
    assert len(a.witness) == k and _usable(list(a.witness), roles, rules)
    assert a.d_min == t.distance(a.witness)
    assert t.admits(a.witness)
    assert t.threshold(k) == factor * a.d_min


@settings(**_SETTINGS)
@given(problems(), st.integers(1, 4))
def test_the_enumerated_d_min_is_the_brute_force_minimum(problem, k):
    Y, roles, rules = problem
    a = tol.closest_group(Y, roles, rules, k)
    ds = [_literal(Y, T) for T in _groups(roles, k) if _usable(T, roles, rules)]
    if not ds:
        assert a is None
        return
    assert a.method == "enumerated"
    assert a.d_min == pytest.approx(min(ds), rel=1e-9, abs=1e-12)


@settings(**_SETTINGS)
@given(problems(min_geos=6, max_geos=10), st.integers(1, 4),
       st.integers(0, 3))
def test_the_searched_d_min_is_a_local_minimum_above_the_exact_one(
        problem, k, starts):
    Y, roles, rules = problem
    e = tol.closest_group(Y, roles, rules, k)
    s = tol.closest_group(Y, roles, rules, k, limit=0, starts=starts)
    if s is None:
        return
    assert s.method == "searched"
    assert e is not None and s.d_min >= e.d_min * (1.0 - 1e-12)
    assert _usable(list(s.witness), roles, rules)
    forced = {j for j, r in enumerate(roles) if r == FORCED}
    free = {j for j, r in enumerate(roles) if TREATMENT in r} - forced
    W = set(s.witness)
    for out in W - forced:
        for g in free - W:
            T = sorted(W - {out} | {g})
            if _usable(T, roles, rules):
                assert _literal(Y, T) >= s.d_min * (1.0 - 1e-9)


@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(problems(min_geos=6, max_geos=9), st.integers(2, 4),
       st.floats(1.0, 2.5), st.sampled_from(["reference", "relative"]))
def test_every_recorded_design_is_within_the_tolerance(problem, K, factor,
                                                       objective):
    Y, roles, rules = problem
    t = tol.PopulationTolerance(Y, roles, rules, factor)
    base = DesignRules.build(roles) if rules is None else rules
    start = max(sum(r == FORCED for r in roles), 1)
    try:
        out = greedy_search(Y, roles, max_treatment_size=K, n_test=4,
                            objective=objective, rules=base.with_tolerance(t))
    except MlsynthDataError as exc:
        # Only a K no partition can reach is refused; a size with a usable
        # group always returns a design.
        assert "not attainable" in str(exc)
        return
    recorded = {o.k for o in out}
    for o in out:
        assert t.distance(o.treatment) <= t.threshold(o.k) * (1.0 + tol.TIE_RTOL)
        assert base.admits(o.treatment)
    for k in range(start, K + 1):
        if t.anchor(k) is not None:
            assert k in recorded, k


@settings(**_SETTINGS)
@given(problems(), st.floats(1.0, 3.0), st.data())
def test_extendable_does_not_read_the_tolerance(problem, factor, data):
    Y, roles, rules = problem
    base = DesignRules.build(roles) if rules is None else rules
    with_tol = base.with_tolerance(tol.PopulationTolerance(Y, roles, rules,
                                                           factor))
    N = Y.shape[1]
    T = data.draw(st.lists(st.integers(0, N - 1), max_size=N - 1, unique=True))
    K = data.draw(st.integers(1, N))
    assert with_tol.extendable(T, K) == base.extendable(T, K)
