"""The certified persp+N bound: valid at every node, and as tight as the solver.

The property that matters is validity: at any node, the certified bound is at
most the smallest objective of the designs in the subtree, computed here by
enumeration. Tightness against the solver's value is checked on fixed panels.

From the repository root::

    python -m pytest benchmarks/studies/marex_bnb/test_persp_n.py -q
"""
import itertools
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.persp_n import _psd2, bound, certified
from mlsynth.utils.marex_helpers.exact import side_fit

warnings.filterwarnings("ignore")


def _panel(seed, n, T):
    rng = np.random.default_rng(seed)
    B = (np.cumsum(rng.normal(size=(T, 2)), 0) @ rng.uniform(0.4, 1.6, (2, n))
         + rng.normal(size=(T, n)))
    return B, B.mean(axis=1)


def _subtree_min(B, A, m, I, O):
    n = B.shape[1]
    best = math.inf
    for S in itertools.combinations(range(n), m):
        if any(j not in S for j in I) or any(j in S for j in O):
            continue
        C = [j for j in range(n) if j not in S]
        best = min(best, side_fit(B, A, list(S))[0] + side_fit(B, A, C)[0])
    return best


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(5, 8), T=st.integers(4, 20),
       m=st.integers(1, 3), side=st.sampled_from(["both", "treated"]),
       fix=st.integers(0, 2))
def test_the_certified_bound_never_exceeds_the_subtree_minimum(seed, n, T, m, side, fix):
    B, A = _panel(seed, n, T)
    pan = Panel(B - A[:, None])
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(n).tolist()
    I = order[:min(fix, m - 1)] if m > 1 else []
    O = order[len(I):len(I) + fix]
    F = [j for j in range(n) if j not in I + O]
    if len(F) <= m - len(I):
        return
    sub = _subtree_min(B, A, m, I, O)
    lb = certified(pan, m, I, O, F, side=side)[0]
    assert lb <= sub * (1 + 1e-12) + 1e-300


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_certified_bound_is_as_tight_as_the_solver(seed):
    B, A = _panel(seed, 9, 20)
    pan = Panel(B - A[:, None])
    lb, val, status, _, _ = certified(pan, 3, [], [], list(range(9)))
    assert status.startswith("optimal")
    assert lb <= val * (1 + 1e-6)
    assert lb >= val - 1e-5 * abs(val)


def test_the_two_sided_bound_dominates_the_treated_block():
    B, A = _panel(4, 9, 20)
    pan = Panel(B - A[:, None])
    both = certified(pan, 3, [], [], list(range(9)), side="both")[0]
    treated = certified(pan, 3, [], [], list(range(9)), side="treated")[0]
    assert both >= treated - 1e-9 * abs(both)


def test_a_node_whose_treated_set_is_complete():
    # r = 0: every free market is a control, and the bound must stay under
    # the one design left.
    B, A = _panel(5, 7, 12)
    pan = Panel(B - A[:, None])
    I, O = [0, 1], [2]
    F = [3, 4, 5, 6]
    sub = _subtree_min(B, A, 2, I, O)
    assert certified(pan, 2, I, O, F)[0] <= sub * (1 + 1e-12)


def test_the_solver_value_and_the_uncertified_bound_agree():
    B, A = _panel(6, 8, 15)
    Bc = B - A[:, None]
    val, status, _ = bound(Bc, 2, [], [], list(range(8)))
    lb, val2, _, _, _ = certified(Panel(Bc), 2, [], [], list(range(8)))
    assert val == pytest.approx(val2, rel=1e-6)
    assert lb <= val2 * (1 + 1e-6)


@given(a=st.floats(-2, 2), b=st.floats(-2, 2), c=st.floats(-2, 2))
def test_a_repaired_multiplier_is_positive_semidefinite(a, b, c):
    a2, b2, c2 = _psd2(np.array([[a, b], [b, c]]))
    assert a2 >= 0 and c2 >= 0 and b2 * b2 <= a2 * c2


def test_with_z_returns_the_indicators():
    B, A = _panel(7, 8, 15)
    out = certified(Panel(B - A[:, None]), 2, [], [], list(range(8)), with_z=True)
    zv = out[-1]
    assert set(zv) == set(range(8)) and abs(sum(zv.values()) - 2) < 1e-5


@pytest.mark.parametrize("seed, n, m", [(8, 9, 3), (9, 10, 2), (10, 9, 4)])
def test_the_search_with_persp_n_bounds_returns_the_optimum(seed, n, m):
    # lift_cost = 0 bounds every node with two or more sets, so the persp+N
    # path decides the whole tree; its answer must be the enumerated optimum.
    from benchmarks.studies.marex_bnb.hybrid import search
    B, A = _panel(seed, n, 20)
    best = min((side_fit(B, A, list(S))[0]
                + side_fit(B, A, [j for j in range(n) if j not in S])[0], S)
               for S in itertools.combinations(range(n), m))
    res = search(B, A, m, price_cost=1e-6, lift_cost=lambda K: 0.0, bounder="persp_n")
    assert res["stats"]["lifts"] > 0
    assert res["set"] == best[1]
    # best[0] is a floating-point sum of two fits; the bracket is exact
    assert float(res["lower"]) <= best[0] * (1 + 1e-12)
    assert best[0] <= float(res["hi"]) * (1 + 1e-12)
    assert res["open"] == 0
