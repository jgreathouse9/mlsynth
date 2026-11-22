"""Starting heuristics for the supergeo search: every one returns a valid design."""
import itertools
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from . import incumbent, dive


def valid(z, design):
    assert sorted(g for c in design for g in c) == list(range(len(z))), design
    assert all(2 <= len(c) <= 4 for c in design), design


def cost(z, design):
    return sum(dive._sc(np.asarray(z, float), tuple(c)) for c in design)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(2, 60), kind=st.integers(0, 3))
def test_greedy_returns_a_valid_design(seed, n, kind):
    rng = np.random.default_rng(seed)
    z = [rng.integers(1, 30, n).astype(float), rng.integers(1, 80, n) / 10.0,
         rng.lognormal(0, 1.3, n) * 100, np.full(n, 7.0)][kind]
    valid(z, incumbent.greedy(z))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(2, 30))
def test_grasp_returns_a_valid_design_no_worse_than_greedy(seed, n):
    z = np.random.default_rng(seed).lognormal(0, 1.3, n) * 100
    d = incumbent.grasp(z, seconds=0.2, seed=seed)
    valid(z, d)
    assert cost(z, d) <= cost(z, incumbent.greedy(z)) * (1 + 1e-12)


def test_greedy_balances_what_balances_exactly():
    # pairs of equal markets: the greedy gives every group zero loss
    z = np.repeat([3.0, 8.0, 21.0, 55.0], 2)
    assert cost(z, incumbent.greedy(z)) == 0.0
    # one large market equal to three small ones: one against three
    z = np.array([30.0, 10.0, 10.0, 10.0, 4.0, 4.0])
    assert cost(z, incumbent.greedy(z)) == 0.0


def test_greedy_never_strands_a_single_market():
    # odd n with every market the same: some group must be odd, none may be a singleton
    for n in (3, 5, 7, 9, 11):
        valid(np.full(n, 5.0), incumbent.greedy(np.full(n, 5.0)))


def test_lp_round_returns_a_valid_design_no_worse_than_its_start():
    rng = np.random.default_rng(4)
    for n in (9, 14, 20):
        z = rng.lognormal(0, 1.3, n) * 100
        start = incumbent.greedy(z)
        cols = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k)][:3000]
        d = incumbent.lp_round(z, cols, start, seconds=1.0)
        valid(z, d)
        assert cost(z, d) <= cost(z, start) * (1 + 1e-12)


def test_tiny_and_degenerate_inputs():
    valid([1.0, 2.0], incumbent.greedy([1.0, 2.0]))
    valid([5.0, 5.0, 5.0], incumbent.greedy([5.0, 5.0, 5.0]))
    with pytest.raises(ValueError):
        incumbent.greedy([1.0])
