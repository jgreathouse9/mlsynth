"""Business rules in the certified search (#965, v1): the treatment-side rules and
the control-side removals.

certify_treatment takes the library's role sets (``eligibility``, one frozenset of
"treatment", "control" and "unassigned" per geo) and its DesignRules. A treatment
group is a design only when every geo in it may be treated, every geo that may only
be treated is in it, and the rules admit it (conflicts, stratum quotas, the budget,
the population tolerance). Its controls are drawn from the geos that may be controls,
outside the group, and not banned by it (exclude_bordering_donors). Every claim is
checked against enumeration of exactly those designs, scored by the library's
relative key. Geos that have to be controls and the control-size band are refused
here; they reach inside the certificate and come separately."""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.design.rules import CONTROL, TREATMENT, UNASSIGNED, DesignRules

from .brute import raw_panel, relative_optimum
from .treatment import _checked_roles, certify_treatment

N_TEST = 4
K = 2
ALL = frozenset({TREATMENT, CONTROL, UNASSIGNED})
SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _panel(seed=5, p=7):
    return raw_panel(seed, n=30, p=p)


def rules_optimum(M, k, n_test, roles, rules):
    """The best design under the roles and rules, by enumeration: (key, T, C)."""
    p = M.shape[1]
    forced = {j for j in range(p) if roles[j] == frozenset({TREATMENT})}
    treatable = [j for j in range(p) if TREATMENT in roles[j]]
    best = (-np.inf, None, None)
    for T in combinations(treatable, k):
        if not forced <= set(T) or (rules is not None and not rules.admits(T)):
            continue
        banned = rules.banned_controls(T) if rules is not None else frozenset()
        allowed = [j for j in range(p) if CONTROL in roles[j] and j not in T and j not in banned]
        y = M[:, list(T)].sum(axis=1)
        for r in range(1, len(allowed) + 1):
            for C in combinations(allowed, r):
                v = score_split(y, M[:, list(C)].sum(axis=1), objective="relative", n_test=n_test).key[0]
                if v > best[0]:
                    best = (v, set(T), set(C))
    return best


def _run(M, roles, rules=None, **kw):
    kw = dict(n_test=N_TEST, restarts=3, time_limit=None) | kw
    return certify_treatment(M, K, eligibility=roles, rules=rules, **kw)


def _agrees(out, M, roles, rules):
    best, T, C = rules_optimum(M, K, N_TEST, roles, rules)
    assert out.certified
    assert set(out.treatment) == T and set(out.controls) == C
    assert out.key == pytest.approx(best, rel=1e-9)
    if rules is not None:
        assert rules.admits_design(out.treatment, out.controls)
    return T, C


def _free(M):
    """The unconstrained optimum, which each rule below is built to bind against."""
    _, T, C = relative_optimum(M, K, N_TEST)
    return sorted(T), sorted(C)


# ---------------------------------------------------------------- smoke


def test_smoke_a_conflict_moves_the_design():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, conflict_pairs=[tuple(T)])
    out = _run(M, roles, rules)
    T2, _ = _agrees(out, M, roles, rules)
    assert T2 != set(T)


# ---------------------------------------------------------------- unit


def test_no_rules_is_the_unconstrained_search():
    M = _panel()
    out = _run(M, None, None)
    T, C = _free(M)
    assert out.treatment == T and out.controls == C


def test_a_geo_that_may_not_be_treated():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    roles[T[0]] = frozenset({CONTROL, UNASSIGNED})
    T2, _ = _agrees(_run(M, roles), M, roles, None)
    assert T[0] not in T2


def test_a_geo_that_has_to_be_treated():
    M = _panel()
    T, _ = _free(M)
    j = next(g for g in range(M.shape[1]) if g not in T)
    roles = [ALL] * M.shape[1]
    roles[j] = frozenset({TREATMENT})
    T2, C2 = _agrees(_run(M, roles), M, roles, None)
    assert j in T2


def test_a_geo_that_may_not_be_a_control():
    M = _panel()
    _, C = _free(M)
    roles = [ALL] * M.shape[1]
    roles[C[0]] = frozenset({TREATMENT, UNASSIGNED})
    _, C2 = _agrees(_run(M, roles), M, roles, None)
    assert C[0] not in C2


def test_a_stratum_maximum():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, strata=[(T, None, 1)])
    T2, _ = _agrees(_run(M, roles, rules), M, roles, rules)
    assert len(T2 & set(T)) <= 1


def test_a_stratum_minimum():
    M = _panel()
    T, _ = _free(M)
    others = [g for g in range(M.shape[1]) if g not in T]
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, strata=[(others[:2], 1, None)])
    T2, _ = _agrees(_run(M, roles, rules), M, roles, rules)
    assert T2 & set(others[:2])


def test_a_budget():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    costs = [10.0 if g in T else 1.0 for g in range(M.shape[1])]
    rules = DesignRules.build(roles, costs=costs, budget=5.0)
    T2, _ = _agrees(_run(M, roles, rules), M, roles, rules)
    assert not T2 & set(T)


def test_bordering_donors_are_excluded():
    M = _panel()
    T, C = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, conflict_pairs=[(T[0], C[0])], exclude_bordering_donors=True)
    T2, C2 = _agrees(_run(M, roles, rules), M, roles, rules)
    assert not (C2 & rules.banned_controls(T2))


class _NotThisGroup:
    """A population tolerance stand-in that admits every group but one."""

    def __init__(self, group):
        self.group = sorted(group)

    def admits(self, group):
        return sorted(group) != self.group

    def anchor(self, size):
        """No anchor group, which the library's greedy reads as nothing to filter."""
        return None


def test_the_population_tolerance_is_read():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles).with_tolerance(_NotThisGroup(T))
    T2, _ = _agrees(_run(M, roles, rules), M, roles, rules)
    assert T2 != set(T)


def test_inadmissible_groups_are_counted_not_searched():
    """The rules are read on the groups the floors leave; a group the floors drop
    cannot compete whether the rules admit it or not. The unconstrained optimum's
    group always survives the floors, so it is among those counted."""
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    roles[T[0]] = frozenset({CONTROL, UNASSIGNED})
    out = _run(M, roles)
    assert 1 <= out.inadmissible <= M.shape[1] - 1     # among the k = 2 groups holding T[0]
    assert all(T[0] not in s["treatment"] for s in out.survivors)


def test_the_pool_respects_the_rules():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, conflict_pairs=[tuple(T)])
    one = _run(M, roles, rules)
    two = _run(M, roles, rules, n_jobs=2)
    assert (two.treatment, two.controls, two.key) == (one.treatment, one.controls, one.key)


@st.composite
def _rule_sets(draw):
    p = 6
    roles = [ALL] * p
    for j in draw(st.lists(st.integers(0, p - 1), max_size=2, unique=True)):
        roles[j] = draw(st.sampled_from([frozenset({CONTROL, UNASSIGNED}), frozenset({TREATMENT, UNASSIGNED}),
                                         frozenset({TREATMENT}), frozenset({UNASSIGNED})]))
    pairs = draw(st.lists(st.tuples(st.integers(0, p - 1), st.integers(0, p - 1)).filter(lambda t: t[0] != t[1]),
                          max_size=3))
    hi = draw(st.sampled_from([None, 1]))
    strata = [(list(range(3)), None, hi)] if hi is not None else []
    border = draw(st.booleans())
    return roles, DesignRules.build(roles, conflict_pairs=pairs, strata=strata, exclude_bordering_donors=border)


def _audit_refuses(roles, rules):
    forced = [j for j, r in enumerate(roles) if r == frozenset({TREATMENT})]
    try:
        rules.audit(forced, K)
    except MlsynthDataError:
        return True
    return len(forced) > K


@given(seed=st.integers(0, 5_000), rr=_rule_sets())
@SETTINGS
def test_property_the_certified_design_is_the_enumerated_optimum_under_the_rules(seed, rr):
    """Rules the library's audit refuses have no design, and the search refuses them
    too; any others give the enumerated optimum, or no design when there is none."""
    roles, rules = rr
    M = raw_panel(seed, n=30, p=6)
    best, T, C = rules_optimum(M, K, N_TEST, roles, rules)
    if _audit_refuses(roles, rules):
        assert T is None
        with pytest.raises(MlsynthDataError):
            _run(M, roles, rules)
        return
    out = _run(M, roles, rules)
    assert out.certified
    if T is None:
        assert out.treatment is None
    else:
        assert set(out.treatment) == T and set(out.controls) == C
        assert rules.admits_design(out.treatment, out.controls)


# ---------------------------------------------------------------- edge


def test_no_admissible_group_is_reported_as_no_design():
    M = _panel(p=5)
    roles = [ALL] * 5
    rules = DesignRules.build(roles, conflict_pairs=list(combinations(range(5), 2)))
    out = _run(M, roles, rules)
    assert out.treatment is None and out.certified and out.inadmissible == 10


def test_a_group_whose_every_control_is_banned_is_inadmissible():
    M = _panel(p=5)
    roles = [frozenset({TREATMENT, UNASSIGNED})] * 2 + [ALL] * 3
    rules = DesignRules.build(roles, conflict_pairs=[(0, 2), (0, 3), (0, 4)], exclude_bordering_donors=True)
    out = _run(M, roles, rules)
    assert out.treatment is None or 0 not in out.treatment
    _agrees(out, M, roles, rules) if out.treatment is not None else None
    assert [0, 1] not in [s["treatment"] for s in out.survivors]
    adm = _checked_roles(roles, rules, 5)
    assert adm.controls((0, 1)) == [] and not adm.admits((0, 1))
    assert adm.admits((1, 2)) and adm.controls((1, 2)) == [3, 4]


# ---------------------------------------------------------------- failure


def test_eligibility_of_the_wrong_length_raises():
    with pytest.raises(MlsynthDataError, match="eligibility"):
        _run(_panel(), [ALL] * 3)


def test_an_unknown_role_raises():
    roles = [ALL] * 7
    roles[0] = frozenset({"treated"})
    with pytest.raises(MlsynthDataError, match="eligibility"):
        _run(_panel(), roles)


def test_rules_over_other_geos_raise():
    roles = [ALL] * 7
    with pytest.raises(MlsynthDataError, match="rules"):
        _run(_panel(), roles, DesignRules.build([ALL] * 5))


def test_rules_built_from_other_roles_raise():
    roles = [ALL] * 7
    other = [ALL] * 7
    other[0] = frozenset({CONTROL, UNASSIGNED})
    with pytest.raises(MlsynthDataError, match="rules"):
        _run(_panel(), roles, DesignRules.build(other))


@pytest.mark.parametrize("make", [
    lambda: ([frozenset({CONTROL})] + [ALL] * 6, None),
    lambda: ([ALL] * 7, DesignRules.build([ALL] * 7, min_controls=2)),
    lambda: ([ALL] * 7, DesignRules.build([ALL] * 7, max_controls=3)),
])
def test_rules_that_reach_inside_the_certificate_are_refused(make):
    """A geo that has to be a control, and the control-size band, need the
    certificate's own fixings and size limits; until they have them the search
    refuses them, naming the rule."""
    roles, rules = make()
    with pytest.raises(MlsynthDataError, match="has to be a control|min_controls|max_controls"):
        _run(_panel(), roles, rules)


def test_a_start_that_breaks_the_rules_raises():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, conflict_pairs=[tuple(T)])
    with pytest.raises(MlsynthDataError, match="start treatment group"):
        _run(M, roles, rules, start=T)


def test_a_start_the_rules_admit_reaches_the_same_design():
    M = _panel()
    T, _ = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, conflict_pairs=[tuple(T)])
    best = _run(M, roles, rules)
    other = next(G for G in combinations(range(M.shape[1]), 2) if rules.admits(G) and list(G) != best.treatment)
    out = _run(M, roles, rules, start=list(other))
    assert (out.treatment, out.controls, out.key) == (best.treatment, best.controls, best.key)


def test_rules_no_group_can_meet_are_refused_naming_the_rule():
    """DesignRules.audit's refusals: a problem no treatment group of the size admits
    is refused up front, with the rule that binds."""
    roles = [frozenset({TREATMENT})] * 2 + [ALL] * 5
    rules = DesignRules.build(roles, strata=[([0, 1], None, 1)])
    with pytest.raises(MlsynthDataError, match="max_per_stratum"):
        _run(_panel(), roles, rules)


def test_more_geos_that_have_to_be_treated_than_the_group_holds_are_refused():
    roles = [frozenset({TREATMENT})] * 3 + [ALL] * 4
    with pytest.raises(MlsynthDataError, match="have to be treated"):
        _run(_panel(), roles)


def test_an_impossible_rule_set_is_refused_before_the_floors(monkeypatch):
    """The refusal comes before any floor is computed, which on a large panel is the
    expensive step; the library's start would raise too, but only after the start."""
    from . import treatment

    def no_floors(*a, **k):
        raise AssertionError("the floors ran on a rule set no group can meet")

    monkeypatch.setattr(treatment, "_streamed_floor_rows", no_floors)
    monkeypatch.setattr(treatment, "_start", no_floors)
    roles = [frozenset({TREATMENT})] * 2 + [ALL] * 5
    rules = DesignRules.build(roles, strata=[([0, 1], None, 1)])
    with pytest.raises(MlsynthDataError, match="max_per_stratum"):
        _run(_panel(), roles, rules)
