"""Checks of the #839 two-way search copy: under the library's choices it is
the library, count for count; under every rule and incumbent it returns the
enumeration's optimum, certified; the ascent never lowers ``sigma'R sigma``;
and the pseudo-costs keep their books."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.exact import SubsetFilter, solve_synthetic_design_exact
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

from . import syndes_rules as sr


def _factor(N, T=14, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.35 * rng.standard_normal((T, N)))


def _noise(N, T=14, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1)) + rng.standard_normal((T, N))


def _setup(Y):
    problem = TwoWayProblem.from_panel(Y)
    return problem, BoundEngine.from_problem(problem)


@pytest.mark.parametrize("make,N,K,seed", [(_noise, 12, 4, 1), (_factor, 14, 5, 2),
                                           (_factor, 16, 6, 3)])
def test_the_library_choices_are_the_library(make, N, K, seed):
    problem, engine = _setup(make(N, seed=seed))
    lib = branch.branch_and_bound(problem, engine, K, SubsetFilter())
    got = sr.search(problem, engine, K)
    assert (got["nodes"], got["lifts"], got["pruned"], got["priced"]) == (
        lib.nodes, lib.lifts, lib.pruned, lib.priced)
    assert got["work"] == pytest.approx(lib.work, rel=1e-12)
    assert got["value"] == pytest.approx(lib.value, rel=1e-12)
    assert got["lower"] == pytest.approx(lib.lower, rel=1e-12)
    assert got["certified"] == lib.certified and got["trials"] == 0


@given(st.integers(8, 11), st.integers(0, 10 ** 6), st.booleans(),
       st.sampled_from(sr.RULES), st.sampled_from(sr.SEEDS), st.data())
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_every_rule_and_incumbent_returns_the_optimum(N, seed, kind, rule, seeds, data):
    K = data.draw(st.integers(2, N - 2))
    Y = (_noise if kind else _factor)(N, seed=seed % 1000)
    problem, engine = _setup(Y)
    optimum = solve_synthetic_design_exact(Y=Y, K=K, candidate_limit=10 ** 9).objective_value
    got = sr.search(problem, engine, K, rule=rule, seeds=seeds, optimum=optimum)
    assert got["closed"] and got["certified"]
    assert got["value"] == pytest.approx(optimum, rel=1e-9)
    assert got["near_work"] is not None and got["near_work"] >= 0.0


def test_the_swap_seed_is_near_before_any_work():
    problem, engine = _setup(_noise(12, seed=4))
    lib = branch.branch_and_bound(problem, engine, 4, SubsetFilter())
    got = sr.search(problem, engine, 4, optimum=lib.value)
    assert got["near_work"] == 0.0


def test_without_a_seed_the_search_finds_the_design_by_working():
    problem, engine = _setup(_factor(14, seed=5))
    lib = branch.branch_and_bound(problem, engine, 5, SubsetFilter())
    got = sr.search(problem, engine, 5, seeds="none", optimum=lib.value)
    assert got["near_work"] > 0.0 and got["value"] == pytest.approx(lib.value, rel=1e-12)


def test_pseudo_costs_try_units_and_charge_the_trials():
    problem, engine = _setup(_factor(16, seed=6))
    got = sr.search(problem, engine, 6, rule="pseudo")
    root = sr.search(problem, engine, 6)
    assert got["trials"] > 0 and got["value"] == pytest.approx(root["value"], rel=1e-12)


@given(st.integers(6, 14), st.integers(0, 10 ** 6), st.data())
@settings(max_examples=40, deadline=None)
def test_the_ascent_never_lowers_the_cut(N, seed, data):
    K = data.draw(st.integers(1, N - 1))
    _, engine = _setup(_noise(N, seed=seed % 1000))
    rng = np.random.default_rng(seed)
    start = np.sort(rng.choice(N, K, replace=False))
    S = sr.ascent(engine.R, K, start)

    def cut(T):
        sigma = -np.ones(N)
        sigma[T] = 1.0
        return float(sigma @ engine.R @ sigma)

    assert len(set(S.tolist())) == K and cut(S) >= cut(start) - 1e-9 * abs(cut(start))


def test_pseudo_costs_keep_their_books():
    costs = sr.PseudoCosts()
    assert costs.overall() == 1.0 and costs.mean(3, True, 0.5) == 0.5
    for g in range(sr.RELIABLE):
        costs.record(3, True, float(g))
        assert not costs.reliable(3)
    for _ in range(sr.RELIABLE):
        costs.record(3, False, -1.0)
    assert costs.reliable(3)
    assert costs.mean(3, True, 0.0) == pytest.approx(np.mean(range(sr.RELIABLE)))
    assert costs.mean(3, False, 9.0) == 0.0          # a loss counts as no gain
    assert sr.score(1.0, 7.0) == pytest.approx(1.0 + (7.0 - 1.0) / 6.0)


def test_an_unknown_rule_is_refused():
    problem, engine = _setup(_noise(8))
    with pytest.raises(ValueError):
        sr.search(problem, engine, 3, rule="random")
