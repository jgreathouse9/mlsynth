"""``population_tolerance``: treatment groups kept close to the population.

The anchor distance of a treatment group ``T`` over the scoring window is

    d(T) = sqrt(mean_t (y_T[t] / mean(y_T) - p[t] / mean(p))^2),

``y_T`` the treated aggregate and ``p`` the sum of every geo in the panel: the
RMS gap between the group's path and the population's, each scaled to its own
mean. With ``population_tolerance=C`` a group of size ``k`` is admissible only
when ``d(T) <= C d_min(k)``, ``d_min(k)`` the smallest distance over the groups
of that size the other rules admit. What these tests pin:

* the distance is that formula, over every geo in the panel, and ``d_min`` is
  the brute-force minimum when the groups are enumerated, and an upper bound
  on it when they are searched, with the method recorded;
* every recorded design, greedy or local, under every objective, is within the
  tolerance of its own size, and every size returns a design, the walk
  continuing from the size's witness group when it cannot reach an admitted one;
* the witness, and a group tied with it in exact arithmetic, are admitted at
  ``C = 1``;
* ``extendable`` is unchanged, so a size with no admitted group does not stop
  the walk;
* calibrated inference draws its placebos from tolerance-admitted groups;
* the fit refuses a non-positive level among the treatment-eligible geos or in
  the population, and refuses the option in a named split;
* the default ``population_tolerance=None`` leaves every existing output as it
  was on ``origin/main`` (a40f1c0b), and adds ``population_distance`` alone.
"""
from __future__ import annotations

import math
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tbr_helpers import calibration as cal
from mlsynth.utils.tbr_helpers.design import local as loc
from mlsynth.utils.tbr_helpers.design import tolerance as tol
from mlsynth.utils.tbr_helpers.design.rules import DesignRules
from mlsynth.utils.tbr_helpers.design.search import (CONTROL, ROLES, TREATMENT,
                                                     UNASSIGNED, greedy_search)
from mlsynth.utils.tbr_helpers.design.structures import (TBRPopulationAnchor,
                                                         TBRPopulationTolerance)

ALL = frozenset(ROLES)
FORCED = frozenset({TREATMENT})
_GEOLIFT = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                        "geolift_test_data.csv")

# The goldens were computed on origin/main (a40f1c0b). numpy's summation order
# differs across CI Python versions, so they are compared at 1e-9 relative.
GOLDEN_RTOL = 1e-9


# --------------------------------------------------------------------------- #
# panels and references
# --------------------------------------------------------------------------- #
def sensitivity_panel(n_geos=10, n_periods=40, seed=0):
    """Periods by geos: geos of differing size whose response to one
    market-wide trend differs, so their paths differ in shape once each is
    scaled to its own mean."""
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    beta = rng.uniform(0.2, 1.8, n_geos)
    t = np.arange(n_periods)
    trend = 0.25 * np.sin(2.0 * np.pi * t / 17.0) + 0.3 * t / n_periods
    noise = 0.04 * rng.normal(size=(n_periods, n_geos))
    return size * (1.0 + beta * trend[:, None] + noise)


def long(Y):
    n, N = Y.shape
    return pd.DataFrame([dict(geo=f"g{j:02d}", date=t, Y=float(Y[t, j]))
                         for j in range(N) for t in range(n)])


def cfg(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=3, n_test=6, population_tolerance=1.5)
    kw.update(over)
    return TBRConfig(**kw)


def fit(df, **over):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBR(cfg(df, **over)).fit()


def literal_distance(Y, T):
    """The anchor distance, written out from its definition."""
    y = Y[:, sorted(T)].sum(axis=1)
    p = Y.sum(axis=1)
    return math.sqrt(float(np.mean((y / y.mean() - p / p.mean()) ** 2)))


def usable(T, roles, rules):
    """A group some design can use: the rules admit it and a control is left."""
    control = {j for j, r in enumerate(roles) if CONTROL in r}
    if not control - set(T):
        return False
    return rules is None or (rules.admits(T) and rules.extendable(T, len(T)))


def brute_min(Y, roles, rules, k):
    """The closest usable group of size k, by scoring every one."""
    forced = [j for j, r in enumerate(roles) if r == FORCED]
    free = [j for j, r in enumerate(roles) if TREATMENT in r and j not in forced]
    best = None
    for c in combinations(free, k - len(forced)):
        T = sorted(forced + list(c))
        if not usable(T, roles, rules):
            continue
        d = literal_distance(Y, T)
        if best is None or d < best[0]:
            best = (d, T)
    return best


def with_tolerance(Y, factor, roles=None, rules=None, **kw):
    """Group rules carrying a tolerance over Y, and the tolerance itself."""
    roles = [ALL] * Y.shape[1] if roles is None else roles
    t = tol.PopulationTolerance(Y, roles, rules, factor, **kw)
    base = DesignRules.build(roles) if rules is None else rules
    return base.with_tolerance(t), t


def anchors_of(res):
    return {a.k: a for a in res.population_tolerance.anchors}


def window_of(res, df):
    """The scoring window the search used, periods by geos in label order."""
    wide = df.pivot(index="date", columns="geo", values="Y").sort_index()
    wide = wide[sorted(wide.columns)]
    return wide.to_numpy(float)[:res.n_periods_scored], list(wide.columns)


def assert_within(res, df):
    """Every design is within the tolerance of its own size."""
    Y, units = window_of(res, df)
    anchors = anchors_of(res)
    for d in res.designs:
        a = anchors[d.k]
        T = [units.index(u) for u in d.treatment_units]
        assert d.population_distance == pytest.approx(literal_distance(Y, T),
                                                      rel=1e-12)
        assert a.threshold == res.population_tolerance.factor * a.d_min
        assert d.population_distance <= a.threshold * (1.0 + tol.TIE_RTOL), d.k


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_a_fit_with_the_tolerance_records_it():
    df = long(sensitivity_panel())
    res = fit(df)
    rec = res.population_tolerance
    assert isinstance(rec, TBRPopulationTolerance)
    assert rec.factor == 1.5
    assert rec.enumeration_limit == tol.ENUMERATION_LIMIT
    assert [a.k for a in rec.anchors] == [1, 2, 3]
    for a in rec.anchors:
        assert isinstance(a, TBRPopulationAnchor)
        assert a.method == "enumerated"
        assert a.n_candidate_groups == math.comb(10, a.k)
        assert len(a.witness_units) == a.k
        assert math.isfinite(a.d_min) and a.d_min > 0.0
    for d in res.designs:
        assert math.isfinite(d.population_distance)
    with pytest.raises(ValidationError):
        rec.factor = 2.0
    with pytest.raises(ValidationError):
        rec.anchors[0].d_min = 0.0


def test_the_default_reports_population_distance_and_no_record():
    df = long(sensitivity_panel())
    res = fit(df, population_tolerance=None)
    assert res.population_tolerance is None
    Y, units = window_of(res, df)
    for d in res.designs:
        T = [units.index(u) for u in d.treatment_units]
        assert d.population_distance == pytest.approx(literal_distance(Y, T),
                                                      rel=1e-12)


# --------------------------------------------------------------------------- #
# unit: the distance and d_min
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [[0], [3], [1, 4], [0, 2, 7], [5, 6, 8, 9]])
def test_the_anchor_distance_is_the_rms_gap_of_the_scaled_paths(T):
    Y = sensitivity_panel()
    assert tol.anchor_distance(Y, T) == pytest.approx(literal_distance(Y, T),
                                                      rel=1e-12)


def test_the_population_is_every_geo_in_the_panel():
    """A geo that cannot be treated is still part of the population."""
    Y = sensitivity_panel()
    roles = [ALL] * 9 + [frozenset({CONTROL})]
    Y[:, 9] *= 1.0 + 0.5 * np.sin(np.arange(Y.shape[0]))   # an odd shape
    t = tol.PopulationTolerance(Y, roles, None, 1.5)
    for T in ([0], [2, 5], [1, 3, 8]):
        assert t.distance(T) == pytest.approx(literal_distance(Y, T), rel=1e-12)
    a = t.anchor(2)
    assert a.d_min == pytest.approx(brute_min(Y, roles, None, 2)[0], rel=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_enumerated_d_min_is_the_brute_force_minimum(k):
    Y = sensitivity_panel(n_geos=8, seed=4)
    roles = [ALL] * 8
    a = tol.closest_group(Y, roles, None, k)
    d, T = brute_min(Y, roles, None, k)
    assert a.method == "enumerated"
    assert a.d_min == pytest.approx(d, rel=1e-12)
    assert list(a.witness) == T
    assert a.d_min == tol.anchor_distance(Y, a.witness)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_d_min_with_forced_geos_and_conflicts_is_the_brute_force_minimum(k):
    Y = sensitivity_panel(n_geos=8, seed=5)
    roles = [FORCED] + [ALL] * 6 + [frozenset({CONTROL, UNASSIGNED})]
    rules = DesignRules.build(roles, conflict_pairs=[(1, 2), (3, 4), (0, 5)])
    a = tol.closest_group(Y, roles, rules, k)
    d, T = brute_min(Y, roles, rules, k)
    assert 0 in a.witness and 5 not in a.witness and 7 not in a.witness
    assert a.d_min == pytest.approx(d, rel=1e-12)
    assert list(a.witness) == T


def test_the_forced_group_is_its_own_witness():
    Y = sensitivity_panel(n_geos=8)
    roles = [FORCED, FORCED] + [ALL] * 6
    a = tol.closest_group(Y, roles, None, 2)
    assert a.witness == (0, 1)
    assert a.n_candidate_groups == 1
    assert a.d_min == tol.anchor_distance(Y, [0, 1])


def test_the_searched_path_bounds_the_enumerated_minimum_from_above():
    Y = sensitivity_panel(n_geos=12, seed=2)
    roles = [ALL] * 12
    for k in (2, 3, 4):
        e = tol.closest_group(Y, roles, None, k)
        s = tol.closest_group(Y, roles, None, k, limit=0)
        assert (e.method, s.method) == ("enumerated", "searched")
        assert s.d_min >= e.d_min * (1.0 - 1e-12)
        assert s.d_min == tol.anchor_distance(Y, s.witness)
        assert len(set(s.witness)) == k


def test_the_searched_minimum_is_a_one_swap_local_minimum():
    Y = sensitivity_panel(n_geos=12, seed=3)
    roles = [ALL] * 12
    s = tol.closest_group(Y, roles, None, 3, limit=0)
    W = set(s.witness)
    for out in W:
        for g in set(range(12)) - W:
            assert tol.anchor_distance(Y, sorted(W - {out} | {g})) >= \
                s.d_min * (1.0 - 1e-9)


def test_the_searched_path_reaches_the_minimum_on_a_small_panel():
    """Measured, not promised: from the forward-greedy start and the random
    starts the descent reaches the enumerated minimum here."""
    Y = sensitivity_panel(n_geos=12, seed=2)
    roles = [ALL] * 12
    for k in (2, 3, 4):
        e = tol.closest_group(Y, roles, None, k)
        s = tol.closest_group(Y, roles, None, k, limit=0)
        assert s.d_min == pytest.approx(e.d_min, rel=1e-12)


def test_the_forward_greedy_start_alone_is_descended():
    """With no random start the search still returns a local minimum from the
    forward-greedy group, and a repeat call returns the same group."""
    Y = sensitivity_panel(n_geos=12, seed=3)
    roles = [ALL] * 12
    a = tol.closest_group(Y, roles, None, 3, limit=0, starts=0)
    b = tol.closest_group(Y, roles, None, 3, limit=0, starts=0)
    assert a.witness == b.witness and a.method == "searched"
    W = set(a.witness)
    for out in W:
        for g in set(range(12)) - W:
            assert tol.anchor_distance(Y, sorted(W - {out} | {g})) >= \
                a.d_min * (1.0 - 1e-9)


def test_the_searched_path_honours_the_rules():
    Y = sensitivity_panel(n_geos=12, seed=6)
    roles = [FORCED] + [ALL] * 11
    rules = DesignRules.build(roles, conflict_pairs=[(0, j) for j in (1, 2, 3)])
    s = tol.closest_group(Y, roles, rules, 3, limit=0)
    e = tol.closest_group(Y, roles, rules, 3)
    assert usable(list(s.witness), roles, rules)
    assert s.d_min >= e.d_min * (1.0 - 1e-12)


def test_the_enumeration_limit_is_inclusive():
    Y = sensitivity_panel(n_geos=8)
    roles = [ALL] * 8
    assert tol.closest_group(Y, roles, None, 2, limit=28).method == "enumerated"
    assert tol.closest_group(Y, roles, None, 2, limit=27).method == "searched"


def test_the_enumeration_runs_in_blocks():
    """A block smaller than the candidate count gives the same minimum."""
    Y = sensitivity_panel(n_geos=10, seed=7)
    roles = [ALL] * 10
    whole = tol.closest_group(Y, roles, None, 3)
    blocked = tol.closest_group(Y, roles, None, 3, block=7)
    assert blocked.witness == whole.witness
    assert blocked.d_min == whole.d_min


# --------------------------------------------------------------------------- #
# unit: admission
# --------------------------------------------------------------------------- #
def test_the_witness_is_admitted_at_factor_one():
    Y = sensitivity_panel()
    t = tol.PopulationTolerance(Y, [ALL] * 10, None, 1.0)
    for k in (1, 2, 3):
        a = t.anchor(k)
        assert t.distance(a.witness) == a.d_min
        assert t.threshold(k) == a.d_min
        assert t.admits(a.witness)


def test_a_tie_with_the_witness_is_admitted_at_factor_one():
    """A geo and three scaled copies of it have the same distance in exact
    arithmetic and distances a rounding step apart in floating point. All four
    are the closest single geo, and all four are admitted at C = 1, including
    the copies whose computed distance lands above the witness's."""
    Y = sensitivity_panel(n_geos=8, seed=1)
    j = int(np.argmin([literal_distance(Y, [i]) for i in range(8)]))
    Y = np.column_stack([Y] + [c * Y[:, j] for c in (0.37, 2.9, 11.3)])
    t = tol.PopulationTolerance(Y, [ALL] * 11, None, 1.0)
    ties = [j, 8, 9, 10]
    d = [t.distance([g]) for g in ties]
    a = t.anchor(1)
    assert a.witness[0] in ties
    assert max(d) > a.d_min                      # a rounding tie, above d_min
    assert max(d) == pytest.approx(a.d_min, rel=1e-12)
    assert all(t.admits([g]) for g in ties)


def test_a_group_beyond_the_tolerance_is_refused():
    Y = sensitivity_panel()
    t = tol.PopulationTolerance(Y, [ALL] * 10, None, 1.5)
    d = [t.distance([j]) for j in range(10)]
    far = int(np.argmax(d))
    assert d[far] > 1.5 * min(d)
    assert not t.admits([far])
    assert all(t.admits([j]) == (d[j] <= 1.5 * min(d)) for j in range(10))


def test_the_threshold_reads_the_groups_own_size():
    Y = sensitivity_panel()
    t = tol.PopulationTolerance(Y, [ALL] * 10, None, 1.0)
    for k in (1, 2, 3):
        W = t.anchor(k).witness
        assert t.admits(W)
        others = [list(c) for c in combinations(range(10), k)
                  if tuple(c) != tuple(W)]
        assert not any(t.admits(T) for T in others[:60])


def test_a_size_with_no_usable_group_has_no_anchor():
    Y = sensitivity_panel(n_geos=4)
    roles = [ALL] * 4
    rules = DesignRules.build(roles, conflict_pairs=list(combinations(range(4), 2)))
    assert tol.closest_group(Y, roles, rules, 2) is None
    assert tol.closest_group(Y, roles, rules, 2, limit=0) is None
    rt, t = with_tolerance(Y, 1.5, roles, rules)
    assert t.anchor(2) is None and t.threshold(2) is None
    assert not t.admits([0, 1])
    assert t.anchor(1) is not None


def test_a_size_the_eligibility_cannot_fill_has_no_anchor():
    """Below the forced geos' count, or above the treatment-eligible pool."""
    Y = sensitivity_panel(n_geos=6)
    roles = [FORCED, FORCED, ALL, ALL] + [frozenset({CONTROL})] * 2
    assert tol.closest_group(Y, roles, None, 1) is None
    assert tol.closest_group(Y, roles, None, 5) is None
    assert tol.closest_group(Y, roles, None, 4).witness == (0, 1, 2, 3)


def test_the_witness_takes_the_whole_pool_when_it_holds_every_carried_control():
    """The carried control group lies inside the witness, so the controls are
    re-derived from the pool the witness leaves, minus its banned neighbours."""
    from mlsynth.utils.tbr_helpers.design.search import _read_eligibility, _reseat
    roles = [ALL] * 6
    rules = DesignRules.build(roles, conflict_pairs=[(0, 5)],
                              exclude_bordering_donors=True)
    pools = _read_eligibility(roles)
    T, C = _reseat([0, 1], {1}, pools, rules)
    assert T == [0, 1] and C == {2, 3, 4}
    T, C = _reseat([0, 1], {1, 3}, pools, rules)
    assert C == {3}


def test_a_group_that_leaves_no_control_is_not_a_witness():
    """Geo 0 has the population's shape exactly and is the only geo that may
    be a control, so the one-geo group {0} has no design and is passed over."""
    rng = np.random.default_rng(0)
    p = 100.0 * (1.0 + 0.3 * np.sin(np.arange(30) / 3.0))
    e = 0.1 * rng.normal(size=30)
    Y = np.column_stack([p, p * (1.0 + e), p * (1.0 - e)])
    roles = [ALL, frozenset({TREATMENT, UNASSIGNED}),
             frozenset({TREATMENT, UNASSIGNED})]
    assert tol.anchor_distance(Y, [0]) < 1e-12
    a = tol.closest_group(Y, roles, None, 1)
    assert a.witness != (0,)
    assert a.d_min == pytest.approx(brute_min(Y, roles, None, 1)[0], rel=1e-12)


def test_the_tolerance_leaves_extendable_unchanged():
    Y = sensitivity_panel(n_geos=9)
    roles = [ALL] * 9
    base = DesignRules.build(roles, conflict_pairs=[(0, 1)],
                             strata=[([0, 1, 2], 1, None), ([3, 4, 5], 1, 2)])
    rt, _ = with_tolerance(Y, 1.0, roles, base)
    groups = [list(c) for m in (1, 2, 3) for c in combinations(range(9), m)]
    for T in groups:
        for K in (2, 3, 4):
            assert rt.extendable(T, K) == base.extendable(T, K)


def test_a_tolerance_the_rules_already_carry_is_set_aside_for_d_min():
    Y = sensitivity_panel()
    roles = [ALL] * 10
    base = DesignRules.build(roles, conflict_pairs=[(0, 1)])
    rt, t = with_tolerance(Y, 1.0, roles, base)
    again = tol.PopulationTolerance(Y, roles, rt, 1.0)
    assert again.rules == base
    assert again.anchor(2) == t.anchor(2)


def test_the_rules_name_the_tolerance():
    Y = sensitivity_panel()
    rt, _ = with_tolerance(Y, 1.5)
    assert "population_tolerance" in rt.active
    assert "population_tolerance" not in DesignRules.build([ALL] * 10).active


def test_admits_rules_ignores_the_tolerance_and_admits_reads_it():
    Y = sensitivity_panel()
    rt, t = with_tolerance(Y, 1.0)
    far = int(np.argmax([t.distance([j]) for j in range(10)]))
    assert rt.admits_rules([far])
    assert not rt.admits([far])
    assert not rt.admits_design([far], [j for j in range(10) if j != far])


# --------------------------------------------------------------------------- #
# unit: the search
# --------------------------------------------------------------------------- #
def test_every_design_is_within_the_tolerance_of_its_own_size():
    df = long(sensitivity_panel())
    res = fit(df, max_treatment_size=4)
    assert [d.k for d in res.designs] == [1, 2, 3, 4]
    assert_within(res, df)


def test_at_factor_one_every_size_returns_its_witness():
    df = long(sensitivity_panel())
    res = fit(df, population_tolerance=1.0, max_treatment_size=4)
    anchors = anchors_of(res)
    assert [d.k for d in res.designs] == [1, 2, 3, 4]
    for d in res.designs:
        assert sorted(d.treatment_units) == sorted(anchors[d.k].witness_units)
    # Size 1 is reached by augmentation; some larger size is not, and the walk
    # carries on from that size's witness.
    assert not anchors[1].greedy_from_witness
    assert any(a.greedy_from_witness for a in anchors.values())
    assert_within(res, df)


def test_the_greedy_walk_continues_from_the_witness():
    """When no admitted geo can join, the walk takes the witness group and the
    next size grows from it."""
    Y = sensitivity_panel()
    rules, t = with_tolerance(Y, 1.0)
    out = greedy_search(Y, [ALL] * 10, max_treatment_size=4, n_test=6,
                        objective="reference", rules=rules)
    assert [o.k for o in out] == [1, 2, 3, 4]
    for o in out:
        assert sorted(o.treatment) == sorted(t.anchor(o.k).witness)
        assert o.from_witness == (o.k > 1 and not set(out[o.k - 2].treatment)
                                  <= set(t.anchor(o.k).witness))


@pytest.mark.parametrize("start", ["carried", "pool", "best"])
def test_every_control_start_respects_the_tolerance(start):
    df = long(sensitivity_panel(seed=3))
    res = fit(df, control_start=start, max_treatment_size=4)
    assert [d.k for d in res.designs] == [1, 2, 3, 4]
    assert_within(res, df)


@pytest.mark.parametrize("objective", ["reference", "paper", "relative"])
def test_every_objective_respects_the_tolerance(objective):
    df = long(sensitivity_panel(seed=1))
    res = fit(df, objective=objective, population_tolerance=1.2)
    assert [d.k for d in res.designs] == [1, 2, 3]
    assert_within(res, df)


@pytest.mark.parametrize("objective", ["reference", "relative"])
def test_the_local_search_respects_the_tolerance(objective):
    df = long(sensitivity_panel(seed=2))
    res = fit(df, objective=objective, search="local", local_restarts=4,
              population_tolerance=1.0)
    anchors = anchors_of(res)
    for d in res.designs:
        assert sorted(d.treatment_units) == sorted(anchors[d.k].witness_units)
    assert_within(res, df)


def test_the_neighbourhood_keeps_only_admitted_treatment_groups():
    Y = sensitivity_panel()
    roles = [ALL] * 10
    rules, t = with_tolerance(Y, 1.2)
    T = list(t.anchor(2).witness)
    C = set(range(10)) - set(T)
    base = loc.neighbourhood(T, C, roles, DesignRules.build(roles))
    kept = loc.neighbourhood(T, C, roles, rules)
    assert all(t.admits(T2) for T2, _ in kept)
    assert len(kept) < len(base)


def test_the_restarts_draw_only_admitted_groups():
    Y = sensitivity_panel()
    roles = [ALL] * 10
    rules, t = with_tolerance(Y, 1.3)
    rng = np.random.default_rng(0)
    drawn = [loc._random_design(rng, roles, 2, rules) for _ in range(40)]
    found = [d for d in drawn if d is not None]
    assert found
    for T, C in found:
        assert t.admits(T)
        assert C and not set(C) & set(T)


def test_a_huge_factor_leaves_the_greedy_designs_unchanged():
    df = long(sensitivity_panel(seed=4))
    off = fit(df, population_tolerance=None, max_treatment_size=4)
    on = fit(df, population_tolerance=1e9, max_treatment_size=4)
    for a, b in zip(off.designs, on.designs):
        assert a.treatment_units == b.treatment_units
        assert a.control_units == b.control_units
        assert a.matching_trace == b.matching_trace
        assert a.population_distance == b.population_distance
    assert not any(a.greedy_from_witness for a in on.population_tolerance.anchors)


def test_a_size_with_no_admitted_group_does_not_stop_the_walk():
    """Two strata asking for one treated geo each leave size 1 without an
    admitted group. The walk passes through it and records sizes 2 and 3."""
    df = long(sensitivity_panel())
    df["region"] = "r" + (df.geo.str[1:].astype(int) % 2).astype(str)
    res = fit(df, stratum_col="region", min_per_stratum=1,
              population_tolerance=1.0)
    assert [d.k for d in res.designs] == [2, 3]
    assert [a.k for a in res.population_tolerance.anchors] == [2, 3]
    assert_within(res, df)


def test_a_reached_group_the_rules_refuse_is_replaced_by_the_witness():
    """Two strata asking for one treated geo each. On this panel the walk
    reaches, at two geos, a pair within the tolerance whose geos share a
    stratum, which the quota refuses; it takes the witness for size two
    instead of losing the size (measured: skipping the size records [3])."""
    Y = sensitivity_panel(n_geos=10, seed=45)
    roles = [ALL] * 10
    base = DesignRules.build(roles, strata=[(list(range(0, 10, 2)), 1, None),
                                            (list(range(1, 10, 2)), 1, None)])
    rules, t = with_tolerance(Y, 1.0, roles, base)
    out = greedy_search(Y, roles, max_treatment_size=3, n_test=6,
                        objective="reference", rules=rules)
    assert [o.k for o in out] == [2, 3]
    assert out[0].from_witness
    assert sorted(out[0].treatment) == sorted(t.anchor(2).witness)
    for o in out:
        assert rules.admits(o.treatment)


def test_forced_geos_and_conflicts_with_the_tolerance():
    df = long(sensitivity_panel())
    df["dma"] = "d" + (df.geo.str[1:].astype(int) // 2).astype(str)
    res = fit(df, to_be_treated=["g04"], cluster_col="dma",
              population_tolerance=1.0, max_treatment_size=3)
    assert [d.k for d in res.designs] == [1, 2, 3]
    for d in res.designs:
        assert "g04" in d.treatment_units and "g05" not in d.treatment_units
    assert_within(res, df)
    assert res.design_constraints.rules == ["cluster_col/adjacency"]


# --------------------------------------------------------------------------- #
# calibrated inference
# --------------------------------------------------------------------------- #
def test_placebo_groups_are_drawn_from_tolerance_admitted_groups():
    Y = sensitivity_panel()
    roles = [ALL] * 10
    rules, t = with_tolerance(Y, 1.3)
    real = list(t.anchor(3).witness)
    groups, n_admissible = cal.placebo_groups(real, roles, rules, 500, seed=0)
    admitted = [list(c) for c in combinations(range(10), 3)
                if t.admits(list(c)) and list(c) != sorted(real)]
    assert n_admissible == len(admitted) < math.comb(10, 3) - 1
    assert sorted(map(tuple, groups)) == sorted(map(tuple, admitted))


def test_a_calibrated_fit_draws_selection_aware_placebos():
    df = long(sensitivity_panel(n_periods=70, seed=5))
    res = fit(df, inference="calibrated", n_placebos=8, population_tolerance=1.3)
    Y, units = window_of(res, df)
    assert res.n_periods_scored == 70 - 18
    t = tol.PopulationTolerance(Y, [ALL] * 10, None, 1.3)
    for group in res.calibration.placebo_treatment_units:
        assert t.admits([units.index(u) for u in group])
    for a in res.population_tolerance.anchors:     # d_min on the search window
        assert a.d_min == t.anchor(a.k).d_min
    for d in res.designs:
        T = [units.index(u) for u in d.treatment_units]
        assert d.population_distance == pytest.approx(literal_distance(Y, T),
                                                      rel=1e-12)
    assert_within(res, df)


def test_population_distance_is_on_the_search_window_under_calibration():
    """With the tolerance off the distance is still on the window the search
    scored, which under calibrated inference excludes the blank periods."""
    df = long(sensitivity_panel(n_periods=70, seed=5))
    res = fit(df, inference="calibrated", n_placebos=8,
              population_tolerance=None)
    Y, units = window_of(res, df)
    full = df.pivot(index="date", columns="geo", values="Y").sort_index()
    for d in res.designs:
        T = [units.index(u) for u in d.treatment_units]
        assert d.population_distance == pytest.approx(literal_distance(Y, T),
                                                      rel=1e-12)
        assert d.population_distance != pytest.approx(
            literal_distance(full.to_numpy(float), T), rel=1e-6)


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.5, 0.999, 0.0, -1.0, float("nan"),
                                 float("inf")])
def test_a_factor_below_one_or_not_finite_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="population_tolerance"):
        cfg(long(sensitivity_panel()), population_tolerance=bad)


def test_the_tolerance_is_refused_with_a_named_split():
    df = long(sensitivity_panel())
    df["post"] = (df.date >= 30).astype(int)
    df["tr"] = (df.geo == "g00").astype(int)
    df["ct"] = (df.geo != "g00").astype(int)
    with pytest.raises(MlsynthConfigError, match="named split"):
        TBRConfig(df=df, unitid="geo", time="date", outcome="Y", post_col="post",
                  treatment_col="tr", control_col="ct", population_tolerance=1.5)


def test_factor_one_is_accepted():
    assert cfg(long(sensitivity_panel()), population_tolerance=1.0) \
        .population_tolerance == 1.0


def test_a_treatment_eligible_geo_without_a_positive_level_is_named():
    Y = sensitivity_panel()
    Y[:, 3] -= 2.0 * Y[:, 3].mean()
    with pytest.raises(MlsynthDataError, match="g03"):
        fit(long(Y))


def test_a_geo_that_cannot_be_treated_may_have_a_negative_level():
    Y = sensitivity_panel()
    Y[:, 3] = -0.01 * Y[:, 3]
    df = long(Y)
    df["can_treat"] = (df.geo != "g03").astype(int)
    res = fit(df, treatment_eligible_col="can_treat")
    assert all("g03" not in d.treatment_units for d in res.designs)


def test_a_population_without_a_positive_level_is_refused():
    Y = sensitivity_panel()
    Y[:, 9] = -5.0 * Y[:, :9].sum(axis=1)
    df = long(Y)
    df["can_treat"] = (df.geo != "g09").astype(int)
    with pytest.raises(MlsynthDataError, match="population.*g09"):
        fit(df, treatment_eligible_col="can_treat")


def test_the_levels_are_checked_only_when_the_tolerance_is_on():
    Y = sensitivity_panel()
    Y[:, 3] -= 2.0 * Y[:, 3].mean()
    res = fit(long(Y), population_tolerance=None)
    Yw, units = window_of(res, long(Y))
    for d in res.designs:
        T = [units.index(u) for u in d.treatment_units]
        level = Yw[:, T].sum(axis=1).mean()
        if level > 0:
            assert math.isfinite(d.population_distance)
        else:
            assert d.population_distance is None


def test_population_distance_is_none_without_positive_levels():
    Y = sensitivity_panel()
    assert tol.population_distance(Y, [0]) == tol.anchor_distance(Y, [0])
    negative_group = Y.copy()
    negative_group[:, 0] *= -1.0
    assert tol.population_distance(negative_group, [0]) is None
    negative_population = Y.copy()
    negative_population[:, 1:] *= -1.0
    assert negative_population[:, 0].mean() > 0
    assert tol.population_distance(negative_population, [0]) is None


def test_the_tolerance_checks_its_inputs():
    Y = sensitivity_panel()
    with pytest.raises(MlsynthConfigError, match="at least 1"):
        tol.PopulationTolerance(Y, [ALL] * 10, None, 0.9)
    with pytest.raises(MlsynthDataError, match="10 geo"):
        tol.PopulationTolerance(Y, [ALL] * 9, None, 1.5)
    with pytest.raises(MlsynthDataError, match="periods by geos"):
        tol.PopulationTolerance(Y[:, 0], [ALL], None, 1.5)
    Y[:, 3] = -Y[:, 3]
    with pytest.raises(MlsynthDataError, match="g3"):
        tol.check_population_levels(Y, [ALL] * 10, [f"g{j}" for j in range(10)])


# --------------------------------------------------------------------------- #
# the default is unchanged
# --------------------------------------------------------------------------- #
# Computed on origin/main (a40f1c0b): the GeoLift panel's last 90 days,
# max_treatment_size=4, n_test=14. Each entry is the treatment group in the
# order the climb added it, the control group, the required impact, the key and
# the calibrated required impact.
GOLDEN_DEFAULT = {
    "recommended": 1,
    "designs": [
        (["dallas"],
         ["baltimore", "boston", "chicago", "cleveland", "denver", "houston",
          "kansas city", "los angeles", "memphis", "orlando", "philadelphia",
          "reno", "salt lake city", "san francisco", "washington"],
         1001.2738162137614,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.0009987278043297105),
         None),
        (["dallas", "denver"],
         ["baltimore", "baton rouge", "boston", "chicago", "cleveland",
          "houston", "kansas city", "los angeles", "memphis", "orlando",
          "reno", "san francisco", "washington"],
         2220.958753431646,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.0004502559979805482),
         None),
        (["dallas", "denver", "cincinnati"],
         ["atlanta", "baltimore", "baton rouge", "chicago", "cleveland",
          "houston", "jacksonville", "kansas city", "los angeles", "memphis",
          "minneapolis", "nashville", "new york", "orlando", "philadelphia",
          "reno", "san diego", "san francisco", "washington"],
         2310.1969704288385,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.00043286352324077863),
         None),
        (["dallas", "denver", "cincinnati", "chicago"],
         ["atlanta", "baltimore", "baton rouge", "cleveland", "houston",
          "kansas city", "memphis", "minneapolis", "nashville", "new york",
          "portland", "reno", "san diego", "washington"],
         2712.7196500232535,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.00036863374362751714),
         None),
    ],
}

GOLDEN_RELATIVE = {
    "recommended": 4,
    "designs": [
        (["denver"],
         ["baton rouge", "boston", "chicago", "dallas", "kansas city",
          "memphis", "nashville", "orlando", "portland", "reno", "san diego",
          "san francisco", "tucson", "washington"],
         1339.7136748708554,
         (2.6300478805299705,),
         None),
        (["denver", "cleveland"],
         ["baton rouge", "boston", "chicago", "dallas", "honolulu",
          "indianapolis", "kansas city", "memphis", "nashville", "orlando",
          "reno", "san diego", "san francisco", "tucson", "washington"],
         1992.1325919674723,
         (3.493832592188936,),
         None),
        (["denver", "cleveland", "austin"],
         ["atlanta", "baton rouge", "boston", "chicago", "cincinnati",
          "dallas", "indianapolis", "jacksonville", "kansas city", "memphis",
          "nashville", "oklahoma city", "orlando", "san diego",
          "san francisco", "tucson", "washington"],
         2222.976188372672,
         (4.320204220514616,),
         None),
        (["denver", "cleveland", "austin", "minneapolis"],
         ["baton rouge", "boston", "chicago", "cincinnati", "dallas",
          "jacksonville", "kansas city", "memphis", "nashville",
          "oklahoma city", "orlando", "reno", "san diego", "tucson",
          "washington"],
         2374.044990949892,
         (5.190531604300014,),
         None),
    ],
}

GOLDEN_RELATIVE_LOCAL = {
    "recommended": 4,
    "designs": [
        (["austin"],
         ["cincinnati", "los angeles", "miami", "milwaukee", "oakland",
          "oklahoma city", "portland"],
         950.5079679694952,
         (2.781179561261892,),
         None),
        (["austin", "denver"],
         ["boston", "chicago", "cincinnati", "dallas", "memphis", "nashville",
          "oklahoma city", "portland", "san diego", "tucson", "washington"],
         1519.410135462618,
         (4.058841191398761,),
         None),
        (["denver", "milwaukee", "new york"],
         ["baton rouge", "boston", "chicago", "cincinnati", "columbus",
          "dallas", "jacksonville", "kansas city", "memphis", "minneapolis",
          "nashville", "new orleans", "orlando", "portland", "salt lake city",
          "san antonio", "san diego", "tucson", "washington"],
         2295.9730851157515,
         (5.029472430749842,),
         None),
        (["atlanta", "minneapolis", "oklahoma city", "orlando"],
         ["austin", "baltimore", "cincinnati", "honolulu", "indianapolis",
          "miami", "nashville", "oakland", "phoenix", "reno", "salt lake city",
          "san diego", "tucson"],
         3466.4473773996497,
         (5.475535465879153,),
         None),
    ],
}

GOLDEN_RELATIVE_CALIBRATED = {
    "recommended": 4,
    "designs": [
        (["austin"],
         ["atlanta", "baton rouge", "chicago", "cincinnati", "indianapolis",
          "oakland", "salt lake city", "san diego"],
         881.8096905351728,
         (2.9419376930324064,),
         2447674.1017375514),
        (["austin", "orlando"],
         ["atlanta", "baltimore", "chicago", "detroit", "indianapolis",
          "kansas city", "los angeles", "nashville", "salt lake city",
          "san antonio", "san diego"],
         1552.011230031755,
         (3.894771538826422,),
         4307979.073181941),
        (["austin", "orlando", "saint paul"],
         ["atlanta", "chicago", "dallas", "detroit", "indianapolis",
          "las vegas", "los angeles", "nashville", "salt lake city",
          "san diego"],
         2027.4646005328168,
         (4.601590823113852,),
         5627713.83460534),
        (["austin", "orlando", "saint paul", "minneapolis"],
         ["atlanta", "chicago", "cincinnati", "dallas", "detroit",
          "las vegas", "los angeles", "nashville", "new orleans",
          "salt lake city", "san diego"],
         2269.9135913330156,
         (5.334509228119567,),
         6300689.1059633745),
    ],
}


def _geolift_df(n_last=90):
    df = pd.read_csv(_GEOLIFT).rename(columns={"location": "geo"})
    dates = sorted(df["date"].unique())[-n_last:]
    return df[df["date"].isin(dates)]


@pytest.mark.parametrize("over,golden", [
    ({}, GOLDEN_DEFAULT),
    ({"objective": "relative"}, GOLDEN_RELATIVE),
    ({"objective": "relative", "search": "local", "local_restarts": 2},
     GOLDEN_RELATIVE_LOCAL),
    ({"objective": "relative", "inference": "calibrated", "n_placebos": 8},
     GOLDEN_RELATIVE_CALIBRATED),
], ids=["reference", "relative", "relative-local", "relative-calibrated"])
def test_the_default_matches_main_on_geolift(over, golden):
    df = _geolift_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TBR(TBRConfig(df=df, unitid="geo", time="date", outcome="Y",
                            max_treatment_size=4, n_test=14, **over)).fit()
    assert res.population_tolerance is None
    assert res.recommended.k == golden["recommended"]
    assert len(res.designs) == len(golden["designs"])
    for d, (treated, controls, impact, key, calibrated) in zip(
            res.designs, golden["designs"]):
        assert d.treatment_units == treated
        assert d.control_units == controls
        got = tuple(float(v) for v in d.matching_trace[-1])
        assert got == pytest.approx(key, rel=GOLDEN_RTOL)
        assert d.objective_value == pytest.approx(key[-1], rel=GOLDEN_RTOL)
        assert d.detail["required_impact"] == pytest.approx(impact,
                                                            rel=GOLDEN_RTOL)
        if calibrated is None:
            assert d.calibrated_required_impact is None
        else:
            assert d.calibrated_required_impact == pytest.approx(
                calibrated, rel=GOLDEN_RTOL)
        assert math.isfinite(d.population_distance)
    assert list(res.power.columns) == (
        ["k", "objective_value"] + sorted(res.designs[0].detail)
        + (["calibrated_required_impact"] if "inference" in over else []))
