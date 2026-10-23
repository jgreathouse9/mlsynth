"""Pentagonal inequalities past the triangles (#837, candidate 2).

The checks: every pentagonal row holds at every sign vector, separation finds
a planted violation and only violations, and the closure with pentagonal rows
lies between the subtree's maximum and the closure with triangles alone.
"""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

from . import cuts
from .test_dual import _best


def _lifted(x):
    v = np.concatenate([[1.0], x])
    return np.outer(v, v)


@settings(max_examples=40, deadline=None)
@given(st.integers(5, 9), st.integers(0, 10 ** 6))
def test_property_every_pentagon_holds_at_every_sign_vector(n, seed):
    x = np.random.default_rng(seed).choice([-1.0, 1.0], size=n)
    assert not len(cuts.pentagons(_lifted(x), gap=0.0, limit=10 ** 6))


def test_the_sign_patterns_are_the_sixteen_odd_ones_up_to_sign():
    P = cuts._PATTERNS
    assert P.shape == (16, 5)
    assert all(abs(p.sum()) in (1, 3, 5) for p in P)
    keys = {tuple(p) for p in P} | {tuple(-p) for p in P}
    assert len(keys) == 32


def test_separation_finds_a_planted_violation():
    """``X = I`` puts every pairwise product at zero, so every pentagon reads
    0 >= -2 and holds; pulling five entries to -0.6 breaks the all-plus one."""
    X = np.eye(7)
    idx = [1, 2, 3, 4, 5]
    for i, j in combinations(idx, 2):
        X[i, j] = X[j, i] = -0.6
    found = cuts.pentagons(X, gap=1e-9, limit=10)
    assert len(found) >= 1
    first = found[0]
    assert tuple(first[0]) == tuple(idx)
    assert cuts.value(X, first[0], first[1]) == pytest.approx(-6.0)


def test_the_most_violated_come_first_and_the_limit_holds():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((8, 8))
    X = np.corrcoef(M)
    found = cuts.pentagons(X, gap=0.0, limit=3)
    assert len(found) <= 3
    vals = [cuts.value(X, s, p) for s, p in found]
    assert vals == sorted(vals)


@pytest.fixture(scope="module")
def node():
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((6, 1)) + rng.standard_normal((6, 11))
    R = BoundEngine.from_problem(TwoWayProblem.from_panel(Y)).R
    I, O = [0], [1]
    F = [u for u in range(11) if u not in I + O]
    K = 5
    c, b, A = branch.node_parts(R, I, O, F)
    return R, I, O, F, K, (c, b, A, 2 * (K - len(I)) - len(F))


def test_the_closures_lie_between_the_maximum_and_each_other(node):
    R, I, O, F, K, parts = node
    tri = cuts.closure(*parts, pentagons=False)
    pent = cuts.closure(*parts, pentagons=True)
    top = _best(R, I, O, F, K)
    assert top * (1 - 1e-7) <= pent["U"] <= tri["U"] * (1 + 1e-7)
    assert pent["pentagons"] > 0 and tri["pentagons"] == 0


def test_a_closure_without_cuts_left_says_so(node):
    *_, parts = node
    out = cuts.closure(*parts, pentagons=True, rounds=20)
    assert out["violated"] == 0


def test_the_table_ranks_the_closures_under_the_subtrees_minimum():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 10)) + 0.3 * rng.standard_normal((6, 10))
    rows = cuts.closures(Y, 3, depths=(0, 2))
    assert [r["fixed"] for r in rows] == [0, 2]
    for r in rows:
        assert r["triangles"] <= r["pentagons"] + 1e-7 <= 1 + 2e-7
        assert 0 < r["need"] <= 1 + 1e-12


def test_fewer_than_five_lifted_indices_carry_no_pentagon():
    assert cuts.pentagons(-np.ones((4, 4)) + 2 * np.eye(4), gap=0.0) == []
