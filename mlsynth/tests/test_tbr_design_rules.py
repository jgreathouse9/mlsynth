"""The business rules on a TBR design, at the level of geo indices (#1046).

:mod:`mlsynth.utils.tbr_helpers.design.rules` holds the rules SYNDES, MAREX and
LEXSCM offer that the role sets cannot express: no two treated geos may
conflict, the treatment group meets per-stratum quotas, the treated geos' market
costs fit a budget, a treated geo's neighbours stay out of the control group,
and the control group's size sits in a band. Forced and forbidden geos and the
size band are folded into the roles upstream and are not rules here.

Four methods carry the search:

* ``admits(T)`` -- every treatment-side rule holds; the final check before a
  size is recorded;
* ``extendable(T, K)`` -- the rules that only tighten as the group grows hold,
  and the stratum minimums can still be met by geos that can still be added.
  It is a necessary condition for some admissible group of at most ``K`` geos
  to contain ``T``, so the augmentation can discard a geo that fails it without
  losing a design. The property test checks exactly that against enumeration;
* ``banned_controls(T)`` -- the conflict neighbours of ``T`` when
  ``exclude_bordering_donors`` is on;
* ``audit(forced, K)`` -- refuse, before any climbing, a problem no size admits,
  naming the rule that binds.

The oracle in this file reads the raw rule specification, not the module under
test, so a weakened check inside ``rules.py`` is caught by disagreement.
"""
from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.rules import DesignRules
from mlsynth.utils.tbr_helpers.design.search import (
    CONTROL, ROLES, TREATMENT, UNASSIGNED)

ALL = frozenset(ROLES)
T_ONLY = frozenset({TREATMENT})
C_ONLY = frozenset({CONTROL})
T_C = frozenset({TREATMENT, CONTROL})
C_U = frozenset({CONTROL, UNASSIGNED})


# --------------------------------------------------------------------------- #
# an oracle that reads the specification, not rules.py
# --------------------------------------------------------------------------- #
def _treatable(roles):
    return {j for j, r in enumerate(roles) if TREATMENT in r}


def oracle_treatment_ok(T, roles, spec):
    """Every treatment-side rule, read off the raw specification."""
    T = set(T)
    for i, j in spec.get("pairs", ()):
        if i in T and j in T:
            return False
    treatable = _treatable(roles)
    for members, lo, hi in spec.get("strata", ()):
        n = len(T & set(members))
        if hi is not None and n > hi:
            return False
        if lo is not None and set(members) & treatable and n < lo:
            return False
    costs, budget = spec.get("costs"), spec.get("budget")
    if budget is not None and sum(costs[j] for j in T) > budget + 1e-9:
        return False
    return True


def oracle_neighbours(T, spec):
    out = set()
    for i, j in spec.get("pairs", ()):
        if i in T:
            out.add(j)
        if j in T:
            out.add(i)
    return out - set(T)


def oracle_design_ok(T, C, roles, spec):
    """A whole design: roles, every rule, the control band."""
    T, C = set(T), set(C)
    n = len(roles)
    if not C or T & C:
        return False
    for j in range(n):
        r = roles[j]
        if j in T and TREATMENT not in r:
            return False
        if j in C and CONTROL not in r:
            return False
        if j not in T and j not in C and UNASSIGNED not in r:
            return False
    if not oracle_treatment_ok(T, roles, spec):
        return False
    if spec.get("bordering") and C & oracle_neighbours(T, spec):
        return False
    lo = spec.get("min_controls") or 1
    hi = spec.get("max_controls")
    if len(C) < lo or (hi is not None and len(C) > hi):
        return False
    return True


def oracle_has_controls(T, roles, spec):
    """Some control group completes ``T`` into a valid design."""
    T = set(T)
    n = len(roles)
    banned = oracle_neighbours(T, spec) if spec.get("bordering") else set()
    must = {j for j in range(n) if j not in T and UNASSIGNED not in roles[j]}
    avail = {j for j in range(n) if j not in T and CONTROL in roles[j]} - banned
    if not must <= avail:
        return False
    lo = max(1, spec.get("min_controls") or 1)
    hi = spec.get("max_controls")
    return len(avail) >= max(lo, len(must)) and (hi is None or len(must) <= hi)


def _rules(roles, spec):
    return DesignRules.build(
        roles, conflict_pairs=spec.get("pairs", ()), strata=spec.get("strata", ()),
        costs=spec.get("costs"), budget=spec.get("budget"),
        exclude_bordering_donors=spec.get("bordering", False),
        min_controls=spec.get("min_controls"), max_controls=spec.get("max_controls"))


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_an_empty_rule_set_admits_everything():
    rules = DesignRules.build([ALL] * 4)
    assert rules.active == ()
    assert rules.n_units == 4
    assert rules.admits([0, 1, 2])
    assert rules.extendable([0, 1, 2], 3)
    assert rules.banned_controls([0]) == frozenset()
    assert rules.control_band == (1, 4)
    assert rules.admits_design([0], {1, 2, 3})
    rules.audit([], 3)


def test_the_active_rules_are_named_by_their_config_fields():
    rules = DesignRules.build(
        [ALL] * 4, conflict_pairs=[(0, 1)], strata=[((0, 1), 1, 2)],
        costs=[1, 1, 1, 1], budget=3, exclude_bordering_donors=True,
        min_controls=1, max_controls=2)
    assert set(rules.active) == {
        "cluster_col/adjacency", "min_per_stratum", "max_per_stratum",
        "max_total_cost", "exclude_bordering_donors", "min_controls",
        "max_controls"}


# --------------------------------------------------------------------------- #
# unit: each rule in admits
# --------------------------------------------------------------------------- #
def test_two_conflicting_geos_are_not_both_treated():
    rules = DesignRules.build([ALL] * 4, conflict_pairs=[(0, 2)])
    assert not rules.admits([0, 2])
    assert not rules.admits([2, 1, 0])
    assert rules.admits([0, 1])
    assert rules.admits([2, 3])


def test_a_stratum_minimum_binds_the_final_group():
    rules = DesignRules.build([ALL] * 4, strata=[((0, 1), 1, None),
                                                 ((2, 3), 1, None)])
    assert not rules.admits([0])
    assert not rules.admits([0, 1])
    assert rules.admits([0, 2])


def test_a_stratum_maximum_binds_the_final_group():
    rules = DesignRules.build([ALL] * 5, strata=[((0, 1, 2), None, 1)])
    assert rules.admits([0, 3])
    assert not rules.admits([0, 1])


def test_a_stratum_with_no_treatable_member_carries_no_minimum():
    """SYNDES's reading: the minimum covers strata that can be covered."""
    roles = [ALL, ALL, C_U, C_U]
    rules = DesignRules.build(roles, strata=[((0, 1), 1, None), ((2, 3), 1, None)])
    assert rules.admits([0])
    assert "min_per_stratum" in rules.active


def test_the_budget_caps_the_treated_geos_total_cost():
    rules = DesignRules.build([ALL] * 4, costs=[1.0, 2.0, 3.0, 4.0], budget=4.0)
    assert rules.admits([0, 2])                    # exactly the budget
    assert not rules.admits([1, 2])
    assert rules.admits([3])


def test_a_budget_met_in_floating_point_is_met():
    """0.1 + 0.2 is 0.30000000000000004; a budget of 0.3 admits it."""
    rules = DesignRules.build([ALL] * 3, costs=[0.1, 0.2, 5.0], budget=0.3)
    assert rules.admits([0, 1])


# --------------------------------------------------------------------------- #
# unit: the control side
# --------------------------------------------------------------------------- #
def test_bordering_controls_are_banned_only_when_asked():
    pairs = [(0, 1), (0, 2), (3, 4)]
    off = DesignRules.build([ALL] * 5, conflict_pairs=pairs)
    on = DesignRules.build([ALL] * 5, conflict_pairs=pairs,
                           exclude_bordering_donors=True)
    assert off.banned_controls([0]) == frozenset()
    assert on.banned_controls([0]) == {1, 2}
    assert on.banned_controls([0, 3]) == {1, 2, 4}
    assert not on.admits_design([0], {1, 3})
    assert on.admits_design([0], {3, 4})
    assert off.admits_design([0], {1, 3})


def test_the_control_band_bounds_the_control_group():
    rules = DesignRules.build([ALL] * 6, min_controls=2, max_controls=3)
    assert rules.control_band == (2, 3)
    assert not rules.admits_design([0], {1})
    assert rules.admits_design([0], {1, 2})
    assert rules.admits_design([0], {1, 2, 3})
    assert not rules.admits_design([0], {1, 2, 3, 4})


def test_admits_design_reads_the_treatment_rules_too():
    rules = DesignRules.build([ALL] * 4, conflict_pairs=[(0, 1)])
    assert not rules.admits_design([0, 1], {2})


# --------------------------------------------------------------------------- #
# unit: extendable
# --------------------------------------------------------------------------- #
def test_extendable_refuses_a_group_already_past_k():
    assert not DesignRules.build([ALL] * 4).extendable([0, 1, 2], 2)


def test_extendable_reads_the_rules_that_only_tighten():
    rules = DesignRules.build(
        [ALL] * 6, conflict_pairs=[(0, 1)], strata=[((2, 3, 4), None, 1)],
        costs=[1, 1, 1, 1, 1, 9], budget=3)
    assert not rules.extendable([0, 1], 4)          # conflict
    assert not rules.extendable([2, 3], 4)          # stratum maximum
    assert not rules.extendable([5], 4)             # budget
    assert rules.extendable([0, 2], 4)


def test_extendable_counts_the_stratum_deficit_against_the_room_left():
    rules = DesignRules.build([ALL] * 6, strata=[((0, 1), 1, None),
                                                 ((2, 3), 1, None),
                                                 ((4, 5), 1, None)])
    assert rules.extendable([0], 3)                 # two deficits, room two
    assert not rules.extendable([0], 2)             # two deficits, room one
    assert not rules.extendable([0, 1], 3)          # two deficits, room one


def test_a_member_that_conflicts_with_the_group_cannot_fill_a_deficit():
    rules = DesignRules.build([ALL] * 5, conflict_pairs=[(0, 2), (0, 3)],
                              strata=[((2, 3), 1, None)])
    assert not rules.extendable([0], 3)
    assert rules.extendable([1], 3)


def test_a_member_over_the_remaining_budget_cannot_fill_a_deficit():
    rules = DesignRules.build([ALL] * 4, strata=[((2, 3), 1, None)],
                              costs=[2, 1, 5, 5], budget=4)
    assert not rules.extendable([0], 3)


def test_a_member_outside_the_treatment_pool_cannot_fill_a_deficit():
    roles = [ALL, ALL, ALL, C_U]
    rules = DesignRules.build(roles, strata=[((2, 3), 2, None)])
    # stratum {2, 3} has one treatable member and asks for two
    assert not rules.extendable([], 3)


def test_the_cheapest_deficits_have_to_fit_the_budget():
    """Each stratum alone fits; both together do not."""
    rules = DesignRules.build([ALL] * 5, strata=[((0, 1), 1, None),
                                                 ((2, 3), 1, None)],
                              costs=[3, 3, 3, 3, 0], budget=5)
    assert not rules.extendable([4], 4)
    loose = DesignRules.build([ALL] * 5, strata=[((0, 1), 1, None),
                                                 ((2, 3), 1, None)],
                              costs=[3, 3, 2, 3, 0], budget=5)
    assert loose.extendable([4], 4)


def test_a_geo_that_must_be_a_control_cannot_border_the_group():
    roles = [ALL, C_ONLY, ALL, ALL]
    rules = DesignRules.build(roles, conflict_pairs=[(0, 1)],
                              exclude_bordering_donors=True)
    assert not rules.extendable([0], 2)
    assert rules.extendable([2], 2)


def test_the_group_has_to_leave_enough_controls_for_the_band():
    rules = DesignRules.build([ALL] * 5, min_controls=3)
    assert rules.extendable([0, 1], 2)
    assert not rules.extendable([0, 1, 2], 3)


def test_the_group_has_to_be_able_to_shed_must_controls_by_k():
    """Three geos must be controls unless treated; max_controls is 1."""
    rules = DesignRules.build([ALL, T_C, T_C, T_C, ALL], max_controls=1)
    assert not rules.extendable([0], 2)            # three left, one more slot
    assert rules.extendable([1], 2)                # two left, one more slot
    assert not rules.extendable([1], 1)
    assert rules.extendable([1, 2], 2)


def test_bordering_shrinks_the_controls_the_group_leaves():
    rules = DesignRules.build([ALL] * 4, conflict_pairs=[(0, 1), (0, 2), (0, 3)],
                              exclude_bordering_donors=True)
    assert not rules.extendable([0], 1)


# --------------------------------------------------------------------------- #
# failure: the audit names the rule that binds
# --------------------------------------------------------------------------- #
def _audit_message(rules, forced, K):
    with pytest.raises(MlsynthDataError) as err:
        rules.audit(forced, K)
    return str(err.value)


def test_audit_refuses_forced_geos_that_conflict():
    roles = [T_ONLY, T_ONLY, ALL, ALL]
    msg = _audit_message(DesignRules.build(roles, conflict_pairs=[(0, 1)]), [0, 1], 3)
    assert "conflict under cluster_col/adjacency" in msg


def test_audit_refuses_forced_geos_over_a_stratum_maximum():
    roles = [T_ONLY, T_ONLY, ALL, ALL]
    msg = _audit_message(DesignRules.build(roles, strata=[((0, 1), None, 1)]),
                         [0, 1], 3)
    assert "fall in one stratum and max_per_stratum" in msg


def test_audit_refuses_a_stratum_short_of_treatable_members():
    roles = [ALL, C_U, ALL, ALL]
    msg = _audit_message(DesignRules.build(roles, strata=[((0, 1), 2, None)]), [], 3)
    assert "1 treatment-eligible geo(s)" in msg


def test_audit_refuses_quotas_that_need_more_geos_than_k():
    rules = DesignRules.build([ALL] * 6, strata=[((0, 1), 1, None),
                                                 ((2, 3), 1, None),
                                                 ((4, 5), 1, None)])
    msg = _audit_message(rules, [], 2)
    assert "quotas need at least 3 treated geos" in msg


def test_audit_refuses_a_budget_below_the_forced_geos_cost():
    roles = [T_ONLY, ALL, ALL]
    msg = _audit_message(DesignRules.build(roles, costs=[5, 1, 1], budget=4), [0], 2)
    assert "the forced geos cost 5" in msg


def test_audit_refuses_a_budget_below_every_treatable_geo():
    msg = _audit_message(DesignRules.build([ALL] * 3, costs=[5, 6, 7], budget=4), [], 2)
    assert "every treatment-eligible geo costs more" in msg


def test_audit_refuses_a_budget_the_quotas_cannot_fit():
    rules = DesignRules.build([ALL] * 4, strata=[((0, 1), 1, None),
                                                 ((2, 3), 1, None)],
                              costs=[3, 3, 3, 3], budget=5)
    msg = _audit_message(rules, [], 3)
    assert "meeting min_per_stratum costs at least 6" in msg


def test_audit_refuses_a_must_control_bordering_a_forced_geo():
    roles = [T_ONLY, C_ONLY, ALL, ALL]
    rules = DesignRules.build(roles, conflict_pairs=[(0, 1)],
                              exclude_bordering_donors=True)
    msg = _audit_message(rules, [0], 2)
    assert "has to be in the control group but borders" in msg


def test_audit_refuses_a_bordering_rule_that_leaves_no_control():
    roles = [T_ONLY, ALL, ALL]
    rules = DesignRules.build(roles, conflict_pairs=[(0, 1), (0, 2)],
                              exclude_bordering_donors=True)
    msg = _audit_message(rules, [0], 1)
    assert "exclude_bordering_donors leaves 0 control" in msg


def test_audit_refuses_a_control_minimum_no_pool_meets():
    msg = _audit_message(DesignRules.build([ALL] * 4, min_controls=4), [], 1)
    assert "4 control geo(s) (min_controls)" in msg


def test_audit_refuses_a_control_maximum_the_pinned_controls_exceed():
    roles = [ALL, C_ONLY, C_ONLY, C_ONLY]
    msg = _audit_message(DesignRules.build(roles, max_controls=2), [], 1)
    assert "max_controls is 2 but at least 3" in msg


def test_audit_counts_must_controls_that_can_still_be_treated():
    """Three geos must be assigned, two of them may be treated: K=1 leaves two."""
    roles = [T_C, T_C, C_ONLY, ALL]
    DesignRules.build(roles, max_controls=2).audit([], 1)
    msg = _audit_message(DesignRules.build(roles, max_controls=1), [], 1)
    assert "max_controls is 1 but at least 2" in msg


def test_audit_refuses_a_problem_the_combined_rules_close():
    """Every member of the stratum conflicts with the forced geo."""
    roles = [T_ONLY, ALL, ALL, ALL]
    rules = DesignRules.build(roles, conflict_pairs=[(0, 1), (0, 2)],
                              strata=[((1, 2), 1, None)])
    msg = _audit_message(rules, [0], 3)
    assert "no completion" in msg


# --------------------------------------------------------------------------- #
# failure: malformed index-level rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs, needle", [
    (dict(costs=[1, 1], budget=3), "entries"),
    (dict(costs=[1, 1, 1]), "budget"),
    (dict(budget=3), "costs"),
    (dict(costs=[1, -1, 1], budget=3), "non-negative"),
    (dict(costs=[1, math.nan, 1], budget=3), "non-negative"),
    (dict(conflict_pairs=[(0, 3)]), "conflict"),
    (dict(conflict_pairs=[(1, 1)]), "conflict"),
    (dict(strata=[((0, 5), 1, None)]), "stratum"),
    (dict(strata=[((0, 1), 1, None), ((1, 2), 1, None)]), "stratum"),
    (dict(strata=[((0, 1), 2, 1)]), "stratum"),
    (dict(min_controls=3, max_controls=2), "min_controls"),
    (dict(min_controls=0), "min_controls"),
])
def test_malformed_rules_are_refused(kwargs, needle):
    with pytest.raises(MlsynthDataError, match=needle):
        DesignRules.build([ALL] * 3, **kwargs)


def test_an_empty_role_list_is_refused():
    with pytest.raises(MlsynthDataError, match="geo"):
        DesignRules.build([])


# --------------------------------------------------------------------------- #
# properties against enumeration
# --------------------------------------------------------------------------- #
ROLE_CHOICES = [ALL, ALL, ALL, T_C, C_U, C_ONLY, T_ONLY,
                frozenset({TREATMENT, UNASSIGNED})]


@st.composite
def problems(draw):
    """A small role assignment and a random rule set over it."""
    n = draw(st.integers(3, 7))
    roles = [draw(st.sampled_from(ROLE_CHOICES)) for _ in range(n)]
    spec = {}
    if draw(st.booleans()):
        clusters = [draw(st.integers(0, n)) for _ in range(n)]
        spec["pairs"] = [(i, j) for i in range(n) for j in range(i + 1, n)
                         if clusters[i] == clusters[j]]
        spec["bordering"] = draw(st.booleans())
    if draw(st.booleans()):
        strata = [draw(st.integers(0, 2)) for _ in range(n)]
        lo = draw(st.sampled_from([None, 1, 2]))
        hi = draw(st.sampled_from([None, 1, 2, 3]))
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        spec["strata"] = [(tuple(j for j in range(n) if strata[j] == s), lo, hi)
                          for s in sorted(set(strata))]
    if draw(st.booleans()):
        spec["costs"] = [draw(st.integers(0, 4)) for _ in range(n)]
        spec["budget"] = draw(st.integers(1, 10))
    if draw(st.booleans()):
        spec["min_controls"] = draw(st.integers(1, 3))
    if draw(st.booleans()):
        spec["max_controls"] = draw(st.integers(spec.get("min_controls", 1), n))
    K = draw(st.integers(1, n - 1))
    return roles, spec, K


def _feasible_groups(roles, spec, K):
    """Every treatment group of at most K geos that some design completes."""
    forced = {j for j, r in enumerate(roles) if r == T_ONLY}
    treatable = sorted(_treatable(roles))
    out = []
    for size in range(max(len(forced), 1), K + 1):
        for T in itertools.combinations(treatable, size):
            if forced <= set(T) and oracle_treatment_ok(T, roles, spec) \
                    and oracle_has_controls(T, roles, spec):
                out.append(frozenset(T))
    return out


@settings(max_examples=150, deadline=None)
@given(problems(), st.data())
def test_extendable_never_discards_a_completable_group(problem, data):
    """Necessity: a group inside some feasible group of at most K passes."""
    roles, spec, K = problem
    rules = _rules(roles, spec)
    feasible = _feasible_groups(roles, spec, K)
    if not feasible:
        return
    target = data.draw(st.sampled_from(feasible))
    sub = data.draw(st.sets(st.sampled_from(sorted(target)))) if target else set()
    assert rules.extendable(sorted(sub), K)
    assert rules.extendable(sorted(target), len(target))
    assert rules.admits(sorted(target))


@settings(max_examples=150, deadline=None)
@given(problems())
def test_admits_agrees_with_the_specification(problem):
    roles, spec, K = problem
    rules = _rules(roles, spec)
    treatable = sorted(_treatable(roles))
    for size in range(0, min(K, len(treatable)) + 1):
        for T in itertools.combinations(treatable, size):
            assert rules.admits(T) == oracle_treatment_ok(T, roles, spec)


@settings(max_examples=150, deadline=None)
@given(problems())
def test_the_audit_never_refuses_a_feasible_problem(problem):
    roles, spec, K = problem
    forced = [j for j, r in enumerate(roles) if r == T_ONLY]
    if len(forced) > K:
        return
    rules = _rules(roles, spec)
    if _feasible_groups(roles, spec, K):
        rules.audit(forced, K)
