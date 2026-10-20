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
