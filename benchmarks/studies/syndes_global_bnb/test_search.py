"""The branch and bound returns the enumeration's optimum, under any cost model.

The reference is the library's own two-way search,
``solve_synthetic_design_exact``, which enumerates every admissible set below
its candidate limit.
"""
import numpy as np
import pytest

from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions

from .search import search


def _factor(N, T=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 3)) @ rng.standard_normal((3, N)) + 0.3 * rng.standard_normal((T, N))


def _no_closed_form():
    """Every set's closed-form minimiser breaks its sign conditions, so every
    price comes from the iterative solve (seed 2 of a probe over random panels)."""
    rng = np.random.default_rng(2)
    N, T = int(rng.integers(7, 11)), int(rng.integers(4, 15))
    rng.integers(2, N - 1)
    Y = rng.standard_normal((T, N)) * rng.uniform(0.1, 5, N) + rng.standard_normal(N) * 3
    return Y, 2, 1e-3


def _agree(res, ref):
    assert res.value == pytest.approx(float(ref.objective_value), rel=1e-9, abs=1e-14)
    assert res.lower <= res.value * (1 + 1e-12)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("N,K", [(8, 3), (10, 4), (12, 2), (12, 5)])
def test_the_optimum_is_the_enumerations(N, K, seed):
    Y = _factor(N, seed=seed)
    res = search(Y, K)
    _agree(res, solve_synthetic_design_exact(Y, K))
    assert res.certified


@pytest.mark.parametrize("costs", [dict(price_cost=1.0, bound_cost=0.0),
                                   dict(price_cost=0.0, bound_cost=1.0)])
def test_the_cost_model_changes_the_work_never_the_answer(costs):
    """Bound every node, or enumerate the root: the same optimum."""
    Y = _factor(14, seed=5)
    ref = solve_synthetic_design_exact(Y, 5)
    res = search(Y, 5, **costs)
    _agree(res, ref)
    if costs["bound_cost"] == 0.0:
        assert res.nodes_bounded > 1
    else:
        assert res.nodes_bounded == 0 and res.sets_priced == 2002


def test_sets_without_a_closed_form_are_solved():
    Y, K, lam = _no_closed_form()
    res = search(Y, K, lam=lam, price_cost=1.0, bound_cost=0.0)
    _agree(res, solve_synthetic_design_exact(Y, K, lam=lam))
    assert res.sets_solved > 0


def test_forced_and_forbidden_units_are_the_root():
    Y = _factor(12, seed=3)
    free = solve_synthetic_design_exact(Y, 4)
    banned = [int(i) for i in free.selected_unit_indices[:2]]
    rules = DesignRestrictions(forbidden=banned, forced_in=[11])
    ref = solve_synthetic_design_exact(Y, 4, restrictions=rules)
    res = search(Y, 4, forced=[11], forbidden=banned, price_cost=1.0, bound_cost=0.0)
    _agree(res, ref)
    assert 11 in res.treated and not set(banned) & set(res.treated)


def test_a_bad_incumbent_costs_time_not_the_answer():
    Y = _factor(12, seed=4)
    ref = solve_synthetic_design_exact(Y, 4)
    res = search(Y, 4, incumbent=np.array([0, 1, 2, 3]), restarts=0,
                 price_cost=1.0, bound_cost=0.0)
    _agree(res, ref)


def test_infeasible_restrictions_are_refused():
    from mlsynth.exceptions import MlsynthEstimationError
    with pytest.raises(MlsynthEstimationError, match="no admissible"):
        search(_factor(8), 2, forced=[0, 1, 2])


def test_lifting_every_node_gives_the_same_optimum():
    Y = _factor(12, seed=6)
    ref = solve_synthetic_design_exact(Y, 4)
    res = search(Y, 4, price_cost=1.0, bound_cost=0.0, lift_cost=0.0)
    _agree(res, ref)
    assert res.nodes_lifted > 0
    plain = search(Y, 4, price_cost=1.0, bound_cost=0.0, lift_cost=np.inf)
    assert plain.nodes_lifted == 0
    assert res.nodes_bounded <= plain.nodes_bounded


def test_a_subtree_past_the_table_limit_is_priced_in_full(monkeypatch):
    """Subtrees too large for a cached table are generated in chunks; every set
    is priced. ``_first`` once pulled one set past each chunk and dropped it:
    the 10-treated GeoLift enumeration priced 847,618,148 of 847,660,528."""
    from math import comb
    from . import search as mod
    monkeypatch.setattr(mod, "TABLE_LIMIT", 10)
    monkeypatch.setattr(mod, "CHUNK", 7)
    mod._TABLES.clear()
    Y = _factor(12, seed=7)
    res = search(Y, 4, price_cost=0.0, bound_cost=1.0, lift_cost=np.inf)
    assert res.sets_priced == comb(12, 4)
    _agree(res, solve_synthetic_design_exact(Y, 4))


@pytest.mark.parametrize("lift", ["dual", "products"])
@pytest.mark.parametrize("seed", [6, 8])
def test_the_regularised_dual_lift_gives_the_same_optimum(lift, seed):
    """The lift's bound by its regularised dual (``dual.py``), stopped as soon
    as it prunes, in place of the interior-point lift: the same optimum."""
    Y = _factor(12, seed=seed)
    ref = solve_synthetic_design_exact(Y, 4)
    res = search(Y, 4, price_cost=1.0, bound_cost=0.0, lift_cost=0.0, lift=lift)
    _agree(res, ref)
    assert res.nodes_lifted > 0


def test_an_unknown_lift_is_refused():
    with pytest.raises(ValueError, match="lift"):
        search(_factor(8), 2, lift="simplex")


def test_lifts_that_never_prune_are_weighted_down(monkeypatch):
    """The lift tier weighs its cost by its own prune rate, as the bound tier
    does: a lift that never prunes is tried a few times per subtree size and
    then left alone. Unweighted, every node above the cost threshold is lifted."""
    from . import search as mod
    calls = []

    def useless(R, I, O, F, K, target=None):
        calls.append(len(F))
        return mod.node_upper(R, I, O, F, K)        # what the cheap tier already had

    monkeypatch.setitem(mod.LIFTS, "ipm", useless)
    Y = _factor(14, seed=5)
    ref = solve_synthetic_design_exact(Y, 5)
    counts = {}
    for adaptive in (True, False):
        calls.clear()
        res = search(Y, 5, price_cost=1e-3, bound_cost=0.0, lift_cost=1e-3, adaptive=adaptive)
        _agree(res, ref)
        counts[adaptive] = len(calls)
    assert 0 < counts[True] < counts[False]


# ----------------------------------------------------------------------
# the lift's cost, learned from the lifts the search pays for
# ----------------------------------------------------------------------

from hypothesis import given, settings, strategies as st

from .search import LiftCost


def test_with_no_lift_observed_the_cost_is_the_calibrations():
    cost = LiftCost(lambda n: 2e-4 * n ** 3)
    assert cost(30) == pytest.approx(2e-4 * 30 ** 3, rel=1e-12)


def test_the_cost_moves_to_what_the_lifts_cost():
    """A lift that stops at its prune target costs less in the search than the
    calibration, which times it without one: the observed cost wins."""
    cost = LiftCost(lambda n: 1e-4 * n ** 3)
    for n in (30, 32, 35, 38, 40) * 40:
        cost.observe(n, 1e-6 * n ** 3)
    assert cost(36) == pytest.approx(1e-6 * 36 ** 3, rel=0.05)


@settings(max_examples=200, deadline=None)
@given(st.floats(1e-8, 1e-2), st.floats(1e-8, 1e-2), st.integers(1, 50), st.integers(5, 60))
def test_the_cost_lies_between_the_calibration_and_the_lifts(prior, seen, count, n):
    cost = LiftCost(lambda k: prior * k ** 3)
    for _ in range(count):
        cost.observe(n, seen * n ** 3)
    lo, hi = sorted((prior * n ** 3, seen * n ** 3))
    assert lo * (1 - 1e-12) <= cost(n) <= hi * (1 + 1e-12)


def test_every_lift_paid_for_is_observed(monkeypatch):
    from . import search as mod
    seen = []

    class Recording(mod.LiftCost):
        def observe(self, n, seconds):
            seen.append(n)
            super().observe(n, seconds)

    monkeypatch.setattr(mod, "LiftCost", Recording)
    # a calibration that makes the first lift pay, whatever the machine's load
    monkeypatch.setattr(mod, "_lift_seconds", lambda R, K, lift=None: (lambda n: 1e-9))
    Y = _factor(14, seed=5)
    res = search(Y, 5, price_cost=1e-3, bound_cost=0.0, lift="dual")
    _agree(res, solve_synthetic_design_exact(Y, 5))
    assert res.nodes_lifted > 0
    assert len(seen) == res.nodes_lifted


def test_the_lift_weight_can_be_turned_off_alone(monkeypatch):
    """``weigh_lift=False`` keeps the cheap tier's weight and lifts every node
    whose pricing costs more than a lift, as before the lift was weighted; the
    result counts the prunes the lift made."""
    from . import search as mod
    calls = []

    def useless(R, I, O, F, K, target=None):
        calls.append(len(F))
        return mod.node_upper(R, I, O, F, K)

    monkeypatch.setitem(mod.LIFTS, "ipm", useless)
    Y = _factor(14, seed=5)
    ref = solve_synthetic_design_exact(Y, 5)
    counts = {}
    for weigh in (True, False):
        calls.clear()
        res = search(Y, 5, price_cost=1e-3, bound_cost=0.0, lift_cost=1e-3, weigh_lift=weigh)
        _agree(res, ref)
        assert res.lifts_pruned == 0
        counts[weigh] = len(calls)
    assert 0 < counts[True] < counts[False]


def test_a_lift_that_always_prunes_is_counted(monkeypatch):
    from . import search as mod
    monkeypatch.setitem(mod.LIFTS, "ipm", lambda R, I, O, F, K, target=None: 0.0)
    Y = _factor(12, seed=6)
    res = search(Y, 4, price_cost=1.0, bound_cost=0.0, lift_cost=0.0)
    assert res.nodes_lifted > 0
    assert res.lifts_pruned == res.nodes_lifted


def test_a_four_unit_panel_calibrates_on_one_size():
    """``max(4, N // 4) == N`` leaves one size to time the bound at."""
    Y = _factor(4, seed=1)
    _agree(search(Y, 2), solve_synthetic_design_exact(Y, 2))


def test_the_buffer_is_priced_when_it_fills(monkeypatch):
    from . import search as mod
    monkeypatch.setattr(mod, "CHUNK", 50)
    Y = _factor(12, seed=3)
    res = search(Y, 4, price_cost=0.0, bound_cost=1.0, lift_cost=np.inf)
    assert res.sets_priced == 495
    _agree(res, solve_synthetic_design_exact(Y, 4))


def test_the_final_solve_can_beat_the_incumbent():
    """Where no set has a closed form every set waits for the final solve; from
    a poor incumbent and no seeds, that solve is what finds the optimum."""
    Y, K, lam = _no_closed_form()
    N = Y.shape[1]
    ref = solve_synthetic_design_exact(Y, K, lam=lam)
    worst = np.array([int(i) for i in range(N) if i not in set(ref.selected_unit_indices)][:K])
    res = search(Y, K, lam=lam, incumbent=worst, restarts=0, price_cost=1.0, bound_cost=0.0)
    _agree(res, ref)
    assert res.sets_solved > 0
