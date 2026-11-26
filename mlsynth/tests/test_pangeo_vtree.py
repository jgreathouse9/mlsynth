"""The volume-bound exact-cover tree (:mod:`mlsynth.utils.pangeo_helpers.vtree`).

The tree enumerates exact covers as the cover search does, but bounds every node afresh: the Lagrangian bound of
the node's own set-partitioning problem (the groups left, the units left, the cardinality row) at multipliers
improved by the volume algorithm from the parent's, and drops the groups whose reduced cost already rules them out
below the node. The bound is evaluated in floating point with its rounding subtracted, so the tests check it
against exact rational arithmetic on ill-conditioned inputs, and check the tree against two oracles that share no
code with it: enumeration of every exact cover on small arms, and the plain cover search (:func:`.cover.search`
from the root's duals over every group, run to its proof), which ``test_pangeo_cover.py`` checks against the full
MIP.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, cover, matching, vtree


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _arm(seed, n, Q, min_pairs=1):
    Y = _panel(seed, n)
    pool = columns.score_columns(np.arange(n), Y, Q)
    k = columns.min_groups(n, Q, min_pairs)
    return pool, columns.root_lp(pool, k), k


def _units(pool, j):
    return frozenset(int(u) for u in pool.members[j] if u >= 0)


def _brute(pool, n, k):
    """The least exact cost over exact covers with at least k groups, by enumeration (small arms only)."""
    best = np.inf
    groups = [_units(pool, j) for j in range(len(pool))]
    by_unit = {i: [j for j, g in enumerate(groups) if min(g) == i] for i in range(n)}

    def rec(unc, count, cost):
        nonlocal best
        if not unc:
            if count >= k:
                best = min(best, cost)
            return
        for j in by_unit[min(unc)]:
            if groups[j] <= unc:
                rec(unc - groups[j], count + 1, cost + float(pool.cost[j]))

    rec(frozenset(range(n)), 0, 0.0)
    return best


def _plain_optimum(pool, root, k, Q):
    res = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return float(pool.cost[res.rows].sum())


def _assert_cover(pool, rows, n, k):
    units = sorted(int(u) for j in rows for u in pool.members[j] if u >= 0)
    assert units == list(range(n)), "the design covers every unit exactly once"
    assert len(rows) >= k


def _up(pool, rows):
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _close(a, b):
    return a == pytest.approx(b, rel=1e-9, abs=1e-12)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_tree_finds_and_proves_the_optimum_from_no_incumbent():
    """Smoke: on a 12-unit Q = 2 arm, over every group and with no incumbent, the tree proves a design at the
    optimum found by enumerating every exact cover."""
    pool, root, k = _arm(1, 12, 2)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7)
    assert isinstance(res, cover.CoverSearch) and res.stage == "volume" and res.status == "proved"
    _assert_cover(pool, res.rows, 12, k)
    assert _close(float(pool.cost[res.rows].sum()), _brute(pool, 12, k))
    assert np.isfinite(res.upper) and np.isfinite(res.lower) and res.nodes > 0


# --- unit ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed,n,Q", [(2, 10, 2), (3, 9, 3), (4, 10, 3), (5, 11, 2)])
def test_the_tree_reaches_the_enumerated_optimum(seed, n, Q):
    pool, root, k = _arm(seed, n, Q)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7)
    assert res.status == "proved"
    _assert_cover(pool, res.rows, n, k)
    z = _brute(pool, n, k)
    assert _close(float(pool.cost[res.rows].sum()), z)
    assert res.lower <= z * (1 + 1e-12) + 1e-12 <= res.upper * (1 + 1e-9) + 1e-9


def test_the_upper_bound_is_the_designs_rigorous_upper_bound():
    pool, root, k = _arm(6, 12, 2)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7)
    assert Fraction(res.upper) >= _up(pool, res.rows)
    assert res.upper <= float(_up(pool, res.rows)) * (1 + 1e-12)


def test_from_an_incumbent_at_the_optimum_the_tree_proves_it():
    """From the optimum's own upper bound over the groups it cannot exclude, as the proof runs, the tree proves:
    it finds nothing cheaper and its lower bound is at the optimum."""
    pool, root, k = _arm(7, 12, 3)
    z = _plain_optimum(pool, root, k, 3)
    z_up = columns._up(Fraction(z) * (1 + Fraction(1, 10**12)))
    keep = np.flatnonzero(root.reduced_costs <= columns._up(Fraction(z_up) - Fraction(root.bound)))
    res = vtree.search(pool, root, keep, k, float(z_up), 10**7)
    assert res.status == "proved"
    assert res.lower <= z * (1 + 1e-12) + 1e-12
    assert res.lower >= z * (1 - 1e-9)
    if res.rows.size:
        assert _close(float(pool.cost[res.rows].sum()), z)


def test_an_incumbent_below_the_optimum_leaves_nothing_to_find():
    pool, root, k = _arm(8, 11, 2)
    z = _plain_optimum(pool, root, k, 2)
    res = vtree.search(pool, root, np.arange(len(pool)), k, 0.9 * z, 10**7)
    assert res.status == "proved" and res.rows.size == 0
    assert res.upper == 0.9 * z and res.lower <= z


def test_reduced_cost_fixing_does_not_change_the_answer_and_cuts_the_tree():
    pool, root, k = _arm(9, 13, 3)
    on = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, fix=True)
    off = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, fix=False)
    assert on.status == off.status == "proved"
    assert _close(float(pool.cost[on.rows].sum()), float(pool.cost[off.rows].sum()))
    assert on.nodes < off.nodes


def test_no_volume_steps_still_proves_the_optimum():
    """With no volume steps each node is bounded at its parent's multipliers (the root's at the root): weaker, still
    a valid bound."""
    pool, root, k = _arm(10, 12, 2)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, iterations=0)
    assert res.status == "proved"
    assert _close(float(pool.cost[res.rows].sum()), _brute(pool, 12, k))


def test_volume_steps_tighten_the_tree():
    pool, root, k = _arm(11, 13, 3)
    none = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, iterations=0, fix=False)
    some = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, iterations=50, fix=False)
    assert some.nodes < none.nodes


def test_the_node_limit_stops_the_tree():
    pool, root, k = _arm(12, 13, 3)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 3)
    assert res.status == "node_limit" and res.nodes <= 3


def test_more_than_64_units_take_two_mask_words():
    """A 70-unit pairs arm (Q = 1): the units' masks need two 64-bit words. Pairs are a minimum-cost perfect
    matching, so the oracle is the blossom matcher (:func:`.matching.min_weight_matching`, certified by its duals);
    the plain cover search does not prove this arm in 10^9 nodes."""
    n = 70
    pool, root, k = _arm(13, n, 1)
    W = np.zeros((n, n))
    for j in range(len(pool)):
        a, b = (int(u) for u in pool.members[j] if u >= 0)
        W[a, b] = W[b, a] = float(pool.cost[j])
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7)
    assert res.status == "proved"
    _assert_cover(pool, res.rows, n, k)
    assert _close(float(pool.cost[res.rows].sum()), matching.min_weight_matching(W)["cost"])


def test_the_progress_array_receives_the_search():
    pool, root, k = _arm(14, 12, 2)
    prog = cover.new_progress()
    prog[cover.P_TOTAL] = 5.0
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, progress=prog)
    assert prog[cover.P_SEARCHES] == 1.0 and prog[cover.P_DONE] == 1.0
    assert prog[cover.P_NODES] == res.nodes and prog[cover.P_TOTAL] == 5.0 + res.nodes
    assert prog[cover.P_INCUMBENT] == res.upper and prog[cover.P_LOWEST] == res.lower


def test_the_progress_array_is_written_while_the_tree_runs(monkeypatch):
    """Every PROGRESS_EVERY nodes the kernel writes the node count and the incumbent."""
    pool, root, k = _arm(15, 13, 3)
    prog = cover.new_progress()
    monkeypatch.setattr(vtree, "PROGRESS_EVERY", 1)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7, progress=prog)
    assert prog[cover.P_NODES] == res.nodes
    assert prog[cover.P_ROOT_TOTAL] > 0 and prog[cover.P_ROOT_DONE] == prog[cover.P_ROOT_TOTAL]


# --- the bound against exact arithmetic ----------------------------------------------------------------------------

def _exact_bound(members, clo, pi, s, unc, kk):
    L = sum((Fraction(float(pi[u])) for u in range(pi.size) if unc[u]), Fraction(0)) + Fraction(float(s)) * kk
    rcs = []
    for j in range(members.shape[0]):
        rc = Fraction(float(clo[j])) - Fraction(float(s))
        for u in members[j]:
            if u >= 0:
                rc -= Fraction(float(pi[u]))
        rcs.append(rc)
        L += min(rc, Fraction(0))
    return L, rcs


def _bound(members, clo, pi, s, unc, kk):
    n = pi.size
    words = vtree._words(n)
    uncm = vtree._mask_of(np.flatnonzero(unc), words)
    opn = np.arange(members.shape[0], dtype=np.int64)
    rcl = np.empty(members.shape[0])
    L = vtree._bound_low(members, clo, opn, opn.size, pi, float(s), uncm, n, kk, rcl)
    return L, rcl


def _ill_conditioned(rng, n, C, level, spread):
    """Groups of 2 to 4 units over n units; costs at ``level`` plus small differences; multipliers that nearly
    cancel the costs, so the reduced costs are small differences of large numbers."""
    members = -np.ones((C, 4), dtype=np.int64)
    for j in range(C):
        size = int(rng.integers(2, 5))
        members[j, :size] = np.sort(rng.choice(n, size=size, replace=False))
    pi = level / 3.0 + rng.normal(size=n) * spread
    s = abs(float(rng.normal())) * spread
    clo = np.array([pi[members[j][members[j] >= 0]].sum() + s for j in range(C)]) + rng.normal(size=C) * spread
    return members, clo, pi, s


@pytest.mark.parametrize("level,spread", [(1.0, 1e-3), (1e8, 1e-6), (1e12, 1.0), (1e15, 1e-1)])
def test_the_float_bound_is_below_the_exact_bound_on_ill_conditioned_inputs(level, spread):
    rng = np.random.default_rng(int(level) % 97)
    n, C = 9, 60
    members, clo, pi, s = _ill_conditioned(rng, n, C, level, spread)
    unc = np.ones(n, dtype=bool)
    for kk in (0, 2):
        L, rcl = _bound(members, clo, pi, s, unc, kk)
        Lx, rcs = _exact_bound(members, clo, pi, s, unc, kk)
        assert Fraction(L) <= Lx
        assert all(Fraction(float(a)) <= b for a, b in zip(rcl, rcs))


def test_the_float_bound_is_tight_on_benign_inputs():
    """The slack is rounding-sized: within 1e-12 of the scale on well-scaled inputs, so the bound prunes as the
    exact one would."""
    rng = np.random.default_rng(3)
    n, C = 9, 60
    members, clo, pi, s = _ill_conditioned(rng, n, C, 1.0, 1e-1)
    unc = np.ones(n, dtype=bool)
    L, _ = _bound(members, clo, pi, s, unc, 2)
    Lx, _ = _exact_bound(members, clo, pi, s, unc, 2)
    scale = float(np.abs(pi).sum() + np.abs(clo).sum() + 2 * s)
    assert float(Lx) - L <= 1e-12 * scale


def test_the_multipliers_sum_is_rounded_down_when_no_reduced_cost_is_negative():
    """With every reduced cost positive the bound is the multipliers' sum alone, which no per-group slack covers.
    Adding 1.5e-16 to 1.0 rounds up each time (half an ulp of 1.0 is 1.1e-16), so the floating-point sum exceeds the
    exact one and only the outer slack brings the bound back under it."""
    n = 9
    pi = np.array([1.0] + [1.5e-16] * (n - 1))
    members = np.array([[0, 1, -1, -1], [2, 3, -1, -1], [4, 5, 6, 7]], dtype=np.int64)
    clo = np.array([10.0, 10.0, 10.0])
    unc = np.ones(n, dtype=bool)
    naive = 0.0
    for x in pi:
        naive += x
    assert Fraction(naive) > sum((Fraction(float(x)) for x in pi), Fraction(0))     # the hazard is real
    L, rcl = _bound(members, clo, pi, 0.0, unc, 0)
    Lx, _ = _exact_bound(members, clo, pi, 0.0, unc, 0)
    assert all(rcl > 0) and Fraction(L) <= Lx


def test_overlapping_groups_past_the_first_word_are_never_combined():
    """Units 64 and up sit in a second mask word. On a 66-unit pool whose only partition costs 82, two groups that
    share unit 64, (0, 64) and (1, 64, 65), with the pairs (2, 3) ... (62, 63) would cover every unit for 31.2. The
    tree must drop (1, 64, 65) once (0, 64) is chosen, and return the partition."""
    n = 66
    groups = [(2 * i, 2 * i + 1) for i in range(32)] + [(64, 65), (0, 64), (1, 64, 65)]
    cost = [1.0] * 32 + [50.0, 0.1, 0.1]
    w = max(len(g) for g in groups)
    members = -np.ones((len(groups), w), dtype=np.int64)
    for j, g in enumerate(groups):
        members[j, :len(g)] = sorted(g)
    c = np.asarray(cost)
    pool = columns.ColumnPool(np.arange(n), members, c, np.zeros(len(groups), dtype=np.int64), True, c.copy(), c.copy())

    class Root:
        pi, sigma = np.zeros(n), 0.0

    res = vtree.search(pool, Root(), np.arange(len(pool)), 1, np.inf, 10**6)
    assert res.status == "proved"
    _assert_cover(pool, res.rows, n, 1)
    assert _close(float(pool.cost[res.rows].sum()), 82.0)


def test_covered_units_carry_no_multiplier_in_the_bound():
    rng = np.random.default_rng(4)
    n, C = 8, 30
    members, clo, pi, s = _ill_conditioned(rng, n, C, 1.0, 1e-1)
    unc = np.ones(n, dtype=bool)
    unc[[0, 3]] = False
    keep = np.array([j for j in range(C) if not np.isin(members[j], [0, 3]).any()])
    L, _ = _bound(members[keep], clo[keep], pi, s, unc, 1)
    Lx, _ = _exact_bound(members[keep], clo[keep], pi, s, unc, 1)
    assert Fraction(L) <= Lx and float(Lx) - L <= 1e-12 * float(np.abs(pi).sum() + np.abs(clo).sum() + s)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), level=st.sampled_from([1.0, 1e4, 1e8, 1e12, 1e15]),
       spread=st.sampled_from([1e-9, 1e-6, 1e-3, 1.0]), kk=st.integers(0, 4), drop=st.integers(0, 3))
def test_property_the_float_bound_and_reduced_costs_never_exceed_the_exact_ones(seed, level, spread, kk, drop):
    """Property: over levels to 1e15 and differences to 1e-9 of them, the float bound is at most the exact bound of
    the same multipliers and each reduced cost's lower end at most the exact reduced cost."""
    rng = np.random.default_rng(seed)
    n, C = 10, 40
    members, clo, pi, s = _ill_conditioned(rng, n, C, level, spread * max(level, 1.0) ** 0.5)
    unc = np.ones(n, dtype=bool)
    gone = rng.choice(n, size=drop, replace=False)
    unc[gone] = False
    keep = np.array([j for j in range(C) if not np.isin(members[j], gone).any()], dtype=np.int64)
    if keep.size == 0:
        return
    L, rcl = _bound(members[keep], clo[keep], pi, s, unc, kk)
    Lx, rcs = _exact_bound(members[keep], clo[keep], pi, s, unc, kk)
    assert Fraction(L) <= Lx
    assert all(Fraction(float(a)) <= b for a, b in zip(rcl, rcs))


# --- edges --------------------------------------------------------------------------------------------------------

def test_no_groups_proves_nothing_to_find():
    pool, root, k = _arm(16, 10, 2)
    res = vtree.search(pool, root, np.zeros(0, dtype=np.int64), k, 7.0, 10**7)
    assert (res.status, res.rows.size, res.upper, res.lower, res.nodes) == ("proved", 0, 7.0, 7.0, 0)


def test_groups_that_cannot_cover_a_unit_prove_nothing_to_find():
    pool, root, k = _arm(17, 10, 2)
    keep = np.array([j for j in range(len(pool)) if 0 not in _units(pool, j)], dtype=np.int64)
    res = vtree.search(pool, root, keep, k, np.inf, 10**7)
    assert res.status == "proved" and res.rows.size == 0 and res.upper == np.inf


def test_a_cardinality_row_no_cover_meets_proves_nothing_to_find():
    pool, root, k = _arm(18, 8, 2)
    res = vtree.search(pool, root, np.arange(len(pool)), 5, np.inf, 10**7)    # 8 units hold at most 4 groups
    assert res.status == "proved" and res.rows.size == 0


def test_a_two_unit_arm_is_its_one_group():
    pool, root, k = _arm(19, 2, 1)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7)
    assert res.status == "proved" and list(res.rows) == [0]
    assert _close(res.upper, float(pool.cost_up[0])) and res.lower <= float(pool.cost[0])


def test_the_cardinality_row_is_met_when_it_binds():
    """With min_pairs forcing more groups than the cheapest cover holds, the design meets it."""
    n, Q = 10, 3
    pool, root, k = _arm(20, n, Q, min_pairs=5)
    res = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**7)
    assert res.status == "proved" and len(res.rows) >= k == 5
    assert _close(float(pool.cost[res.rows].sum()), _brute(pool, n, k))


# --- failures -----------------------------------------------------------------------------------------------------

def test_a_negative_node_limit_is_refused():
    pool, root, k = _arm(21, 8, 2)
    with pytest.raises(MlsynthConfigError, match="node_limit"):
        vtree.search(pool, root, np.arange(len(pool)), k, np.inf, -1)


def test_a_negative_step_count_is_refused():
    pool, root, k = _arm(21, 8, 2)
    with pytest.raises(MlsynthConfigError, match="iterations"):
        vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10, iterations=-1)


def test_a_malformed_progress_array_is_refused():
    pool, root, k = _arm(22, 8, 2)
    with pytest.raises(MlsynthConfigError, match="progress"):
        vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10, progress=np.zeros(3))


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 13), Q=st.integers(2, 3), min_pairs=st.integers(1, 3),
       incumbent=st.sampled_from(["none", "optimum", "above", "below"]), fix=st.booleans(),
       iterations=st.sampled_from([0, 5, 50]))
def test_property_the_tree_agrees_with_the_plain_search(seed, n, Q, min_pairs, incumbent, fix, iterations):
    """Property: over arms, cardinality rows, incumbents, fixing and step counts, a proved tree returns a design at
    the plain search's optimum when its incumbent is above it, none when the incumbent is below it, an upper bound at
    least its design's, and a lower bound at most the optimum."""
    if not columns.feasible(n, Q, min_pairs):
        return
    pool, root, k = _arm(seed, n, Q, min_pairs)
    z = _plain_optimum(pool, root, k, Q)
    best = dict(none=np.inf, optimum=float(columns._up(Fraction(z) * (1 + Fraction(1, 10**9)))),
                above=1.3 * z + 1e-9, below=0.7 * z)[incumbent]
    res = vtree.search(pool, root, np.arange(len(pool)), k, best, 10**7, fix=fix, iterations=iterations)
    assert res.status == "proved" and res.stage == "volume"
    assert res.lower <= z * (1 + 1e-12) + 1e-12
    if incumbent == "below":
        assert res.rows.size == 0 and res.upper == best
    else:
        _assert_cover(pool, res.rows, n, k)
        assert _close(float(pool.cost[res.rows].sum()), z)
        assert Fraction(res.upper) >= _up(pool, res.rows)
