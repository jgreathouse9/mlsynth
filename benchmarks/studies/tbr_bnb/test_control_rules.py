"""The control-side rules that reach inside the certificate (#965, v1): controls
fixed in, and a band on the control group's size.

A geo whose role set holds "control" but not "unassigned" has to be a control
whenever it is not treated; the library's ``min_controls`` and ``max_controls``
bound the control group's size. In the slope-free certificate the first are
fixings at every size's root and the second the sizes searched, and the RSS descent
that sets the incumbent keeps to both, so the incumbent is always a design the rules
allow. The group screen and the hull stage bound every design of every size, a
superset of the allowed ones, so they stay valid unchanged. Every claim is checked
against enumeration of exactly the allowed designs."""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.rules import CONTROL, TREATMENT, UNASSIGNED, DesignRules

from mlsynth.utils.tbr_helpers.design.exact import slopefree as sf
from .brute import panel, positive_optimum, raw_panel
from mlsynth.utils.tbr_helpers.design.exact.search import incumbent
from .test_rules import ALL, K, N_TEST, _agrees, _free, _panel, _run, rules_optimum
from mlsynth.utils.tbr_helpers.design.exact.treatment import _checked_roles

SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def constrained_optimum(y, X, fixed, band):
    """The least positive-slope RSS over control groups holding ``fixed`` with a
    size in ``band``, by enumeration: (rss, group), (inf, None) when none has a
    positive slope."""
    lo, hi = band
    best, arg = np.inf, None
    for r in range(max(lo, len(fixed), 1), min(hi, X.shape[1]) + 1):
        for C in combinations(range(X.shape[1]), r):
            if not set(fixed) <= set(C):
                continue
            x = X[:, list(C)].sum(axis=1)
            b = (y @ x) / (x @ x)
            if b > 0:
                v = float(((y - b * x) ** 2).sum())
                if v < best:
                    best, arg = v, set(C)
    return best, arg


# ---------------------------------------------------------------- smoke


def test_smoke_a_fixed_control_and_a_band_certify_the_enumerated_optimum():
    y, X = panel(0, p=6)
    best, C = constrained_optimum(y, X, {2}, (2, 3))
    out = sf.certify_slope_free(y, X, restarts=5, fixed={2}, band=(2, 3))
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


# ---------------------------------------------------------------- unit


@given(seed=st.integers(0, 5_000), fixed=st.sets(st.integers(0, 5), max_size=2),
       lo=st.integers(1, 3), width=st.integers(0, 3))
@SETTINGS
def test_property_the_descent_keeps_to_the_fixings_and_the_band(seed, fixed, lo, width):
    y, X = panel(seed, p=6)
    hi = min(6, max(lo, len(fixed)) + width)
    lo = min(lo, hi)
    _, C = incumbent(y, X, restarts=5, seed=seed, fixed=frozenset(fixed), band=(lo, hi))
    assert fixed <= C and lo <= len(C) <= hi


@given(seed=st.integers(0, 5_000), fixed=st.sets(st.integers(0, 5), max_size=2),
       lo=st.integers(1, 4), width=st.integers(0, 3))
@SETTINGS
def test_property_the_certificate_is_the_enumerated_optimum_under_fixings_and_a_band(seed, fixed, lo, width):
    y, X = panel(seed, p=6)
    hi = min(6, max(lo, len(fixed)) + width)
    lo = min(lo, hi)
    best, C = constrained_optimum(y, X, fixed, (lo, hi))
    if C is None:
        with pytest.raises(sf.NoDesignLeft):
            sf.certify_slope_free(y, X, restarts=5, fixed=frozenset(fixed), band=(lo, hi))
        return
    out = sf.certify_slope_free(y, X, restarts=5, fixed=frozenset(fixed), band=(lo, hi))
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


def test_only_the_band_is_searched():
    y, X = panel(1, p=6)
    out = sf.certify_slope_free(y, X, restarts=5, band=(3, 4), group_screen=False)
    assert set(out.sizes) <= {3, 4}


def test_a_start_outside_the_rules_is_not_taken():
    """The unconstrained optimum, given as the start where it breaks a fixing, beats
    every design the rules allow; it is not taken, and the certificate is the best
    allowed design."""
    for seed in range(20):
        y, X = panel(seed, p=6)
        _, free = positive_optimum(y, X)
        j = next((g for g in range(6) if g not in free), None)
        if j is not None and len(free) <= 3:
            break
    best, C = constrained_optimum(y, X, {j}, (1, 4))
    out = sf.certify_slope_free(y, X, restarts=5, fixed=frozenset({j}), band=(1, 4), start=set(free))
    assert j in out.controls and set(out.controls) == C


def test_the_best_neighbour_keeps_to_the_rules():
    y, X = panel(3, p=6)
    D = sf.ExactData(y, X)
    nb = sf._best_neighbour(D.c, D.G, [{0, 1}], [], ok=lambda S: 2 in S and len(S) <= 2)
    assert nb is None or (2 in nb and len(nb) <= 2)


# ---------------------------------------------------------------- treatment level


def test_a_geo_that_has_to_be_a_control_is_in_every_design():
    M = _panel()
    T, C = _free(M)
    j = next(g for g in range(M.shape[1]) if g not in T and g not in C)
    roles = [ALL] * M.shape[1]
    roles[j] = frozenset({CONTROL})
    T2, C2 = _agrees(_run(M, roles), M, roles, None)
    assert j in C2


def test_a_geo_treated_or_a_control():
    """Roles {treatment, control}: the geo is in the treatment group or in the
    control group, never held out."""
    M = _panel()
    T, C = _free(M)
    j = next(g for g in range(M.shape[1]) if g not in T and g not in C)
    roles = [ALL] * M.shape[1]
    roles[j] = frozenset({TREATMENT, CONTROL})
    T2, C2 = _agrees(_run(M, roles), M, roles, None)
    assert j in T2 | C2


def test_min_controls():
    M = _panel()
    _, C = _free(M)
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, min_controls=len(C) + 1)
    _, C2 = _agrees(_run(M, roles, rules), M, roles, rules)
    assert len(C2) >= len(C) + 1


def test_max_controls():
    M = _panel()
    _, C = _free(M)
    assert len(C) >= 2
    roles = [ALL] * M.shape[1]
    rules = DesignRules.build(roles, max_controls=len(C) - 1)
    _, C2 = _agrees(_run(M, roles, rules), M, roles, rules)
    assert len(C2) <= len(C) - 1


def test_a_required_control_banned_by_the_group_rules_the_group_out():
    """A geo that has to be a control but borders a treated geo under
    exclude_bordering_donors leaves that group no design."""
    M = _panel()
    T, C = _free(M)
    j = next(g for g in range(M.shape[1]) if g not in T)
    roles = [ALL] * M.shape[1]
    roles[j] = frozenset({CONTROL})
    rules = DesignRules.build(roles, conflict_pairs=[(T[0], j)], exclude_bordering_donors=True)
    T2, C2 = _agrees(_run(M, roles, rules), M, roles, rules)
    assert T[0] not in T2 and j in C2


@st.composite
def _control_rule_sets(draw):
    p = 6
    roles = [ALL] * p
    for j in draw(st.lists(st.integers(0, p - 1), max_size=2, unique=True)):
        roles[j] = draw(st.sampled_from([frozenset({CONTROL}), frozenset({TREATMENT, CONTROL}),
                                         frozenset({CONTROL, UNASSIGNED})]))
    lo = draw(st.sampled_from([None, 2]))
    hi = draw(st.sampled_from([None, 2, 3]))
    if lo is not None and hi is not None and lo > hi:
        hi = lo
    return roles, DesignRules.build(roles, min_controls=lo, max_controls=hi)


@given(seed=st.integers(0, 5_000), rr=_control_rule_sets())
@SETTINGS
def test_property_the_design_is_the_enumerated_optimum_under_control_rules(seed, rr):
    roles, rules = rr
    M = raw_panel(seed, n=30, p=6)
    best, T, C = rules_optimum(M, K, N_TEST, roles, rules)
    out = _run(M, roles, rules)
    assert out.certified
    if T is None:
        assert out.treatment is None
    else:
        assert set(out.treatment) == T and set(out.controls) == C
        assert rules.admits_design(out.treatment, out.controls)


# ---------------------------------------------------------------- edge


def test_a_band_of_one_size():
    y, X = panel(4, p=6)
    best, C = constrained_optimum(y, X, set(), (3, 3))
    out = sf.certify_slope_free(y, X, restarts=5, band=(3, 3))
    assert set(out.controls) == C and len(C) == 3


def test_every_control_fixed():
    y, X = panel(5, p=4)
    best, C = constrained_optimum(y, X, {0, 1, 2, 3}, (1, 4))
    if C is None:
        with pytest.raises(sf.NoDesignLeft):
            sf.certify_slope_free(y, X, restarts=5, fixed=frozenset(range(4)))
    else:
        out = sf.certify_slope_free(y, X, restarts=5, fixed=frozenset(range(4)))
        assert set(out.controls) == {0, 1, 2, 3}


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("fixed, band", [({7}, (1, 6)), ({0, 1, 2}, (1, 2)), (set(), (4, 3)), (set(), (0, 2)),
                                         (set(), (1, 7))])
def test_fixings_or_a_band_no_design_can_meet_raise(fixed, band):
    y, X = panel(0, p=6)
    with pytest.raises(MlsynthDataError, match="fixed|band"):
        incumbent(y, X, restarts=5, fixed=frozenset(fixed), band=band)


def test_an_excluded_incumbent_falls_back_within_the_rules():
    """With the best allowed design excluded, the incumbent is the best neighbour the
    rules allow, and the certificate is the best allowed design that is not
    excluded."""
    y, X = panel(6, p=6)
    fixed, band = frozenset({1}), (2, 3)
    _, C = constrained_optimum(y, X, fixed, band)
    best, arg = np.inf, None
    for r in range(2, 4):
        for S in combinations(range(6), r):
            if 1 in S and set(S) != C:
                x = X[:, list(S)].sum(axis=1)
                b = (y @ x) / (x @ x)
                if b > 0 and float(((y - b * x) ** 2).sum()) < best:
                    best, arg = float(((y - b * x) ** 2).sum()), set(S)
    out = sf.certify_slope_free(y, X, restarts=5, fixed=fixed, band=band, exclude=[C])
    assert set(out.controls) == arg and float(out.incumbent) == pytest.approx(best, rel=1e-9)


def test_a_group_that_cannot_fill_the_band_is_no_design():
    """min_controls 4 with two geos that may not be controls: a group that treats
    neither has three candidate controls and no design, so the design treats one."""
    M = _panel()
    roles = [frozenset({TREATMENT, UNASSIGNED})] * 2 + [ALL] * 5
    rules = DesignRules.build(roles, min_controls=4)
    out = _run(M, roles, rules)
    _agrees(out, M, roles, rules)
    assert set(out.treatment) & {0, 1}
    adm = _checked_roles(roles, rules, 7)
    assert not adm.admits((2, 3)) and adm.admits((0, 2))


def test_a_group_leaving_more_required_controls_than_the_band_is_no_design():
    """Three geos treated or controls, max_controls 2: a group that treats none of
    them would need all three as controls."""
    M = _panel()
    roles = [frozenset({TREATMENT, CONTROL})] * 3 + [ALL] * 4
    rules = DesignRules.build(roles, max_controls=2)
    out = _run(M, roles, rules)
    _agrees(out, M, roles, rules)
    assert set(out.treatment) & {0, 1, 2}
    adm = _checked_roles(roles, rules, 7)
    assert not adm.admits((3, 4)) and adm.admits((0, 3))
