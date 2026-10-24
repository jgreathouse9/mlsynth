"""Checks of the persp+N bound by a regularised dual (#833).

The dual maximises a smooth lower bound over the multipliers the library's
certificate reads. The checks are that its exact value is the library's
certificate for the same multipliers, that every value it reports lies under
every design of the node, that its gradient is the smooth bound's, and that
it reaches the interior-point relaxation on a node small enough to solve.
"""
import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.marex_helpers import branch
from mlsynth.utils.marex_helpers.certify import ExactData
from mlsynth.utils.marex_helpers.exact import side_fit

from . import dual


def _panel(n, T, seed):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = f @ rng.uniform(0.4, 1.6, size=(2, n)) + rng.normal(size=(T, n)) + 20.0
    return B, B.mean(axis=1)


def _split(n, I, O):
    return [j for j in range(n) if j not in I and j not in O]


def _subtree_min(B, A, m, I, O):
    n = B.shape[1]
    best = np.inf
    for S in itertools.combinations(range(n), m):
        if set(I) <= set(S) and not set(O) & set(S):
            C = [j for j in range(n) if j not in S]
            best = min(best, side_fit(B, A, list(S))[0] + side_fit(B, A, C)[0])
    return best


def _theta(rng, layout, spread=1.0):
    th = rng.normal(scale=spread, size=layout.size)
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in layout.bounds()])
    return np.maximum(th, lo)


NODES = [((), ()), ((1,), ()), ((), (0, 2)), ((3,), (1,))]


# ----------------------------------------------------------------------
# the node and its multipliers
# ----------------------------------------------------------------------


def test_the_node_is_set_up_as_the_library_sets_it_up():
    B, A = _panel(7, 12, 0)
    data = ExactData(B, A)
    node = dual.node_parts(data, 3, [1], [2], _split(7, [1], [2]))
    assert [s.U for s in node.sides] == [[0, 1, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
    assert [s.control for s in node.sides] == [False, True]
    assert node.r == 2 and node.n_free == 5
    gmax = max(data.gram.frac(i, i) for i in range(7))
    assert node.scale == Fraction(2) ** round(np.log2(float(gmax)))
    assert np.allclose(node.sides[0].G, data.Gf[np.ix_([0, 1, 3, 4, 5, 6], [0, 1, 3, 4, 5, 6])]
                       / float(node.scale))


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 3), st.integers(0, 10 ** 6), st.booleans())
def test_property_the_exact_value_is_the_librarys_certificate(which, seed, penalised):
    """The float exact bound and the library's exact-arithmetic certificate
    of the same multipliers agree to rounding."""
    B, A = _panel(6, 10, 2)
    I, O = NODES[which]
    rng = np.random.default_rng(seed)
    lin = (rng.uniform(0, 50, size=6), rng.uniform(0, 50, size=6)) if penalised else (None, None)
    node = dual.node_parts(ExactData(B, A), 3, I, O, _split(6, I, O), *lin)
    layout = dual.Layout(node)
    th = _theta(rng, layout, spread=0.3)
    exact, _, _ = dual.evaluate(th, node, layout, 1e-3)
    cert = dual.fullform.certificate(node.data.gram, node.scale, node.r, node.n_free,
                               dual.to_duals(th, node, layout))
    assert cert is not None
    assert float(cert) == pytest.approx(exact * float(node.scale), rel=1e-9, abs=1e-9)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 3), st.integers(0, 10 ** 6), st.floats(1e-6, 1.0))
def test_property_every_value_lies_under_every_design(which, seed, alpha):
    """For any multipliers of the right signs and any alpha: smooth <= exact
    <= the subtree's minimum."""
    B, A = _panel(6, 10, 3)
    I, O = NODES[which]
    node = dual.node_parts(ExactData(B, A), 3, I, O, _split(6, I, O))
    layout = dual.Layout(node)
    th = _theta(np.random.default_rng(seed), layout, spread=0.5)
    exact, smooth, _ = dual.evaluate(th, node, layout, alpha)
    sub = _subtree_min(B, A, 3, I, O) / float(node.scale)
    assert smooth <= exact + 1e-12
    assert exact <= sub * (1 + 1e-9) + 1e-12


@pytest.mark.parametrize("which", range(4))
def test_the_gradient_is_the_smooth_bounds(which):
    B, A = _panel(6, 10, 4)
    I, O = NODES[which]
    node = dual.node_parts(ExactData(B, A), 3, I, O, _split(6, I, O),
                           np.linspace(0, 30, 6), np.linspace(30, 0, 6))
    layout = dual.Layout(node)
    rng = np.random.default_rng(which)
    th = _theta(rng, layout, spread=0.2)
    th = np.maximum(th, np.array([0.05 if b[0] is not None else -np.inf
                                  for b in layout.bounds()]))   # off the bounds
    _, _, g = dual.evaluate(th, node, layout, 0.05)
    h = 1e-6
    for idx in rng.choice(layout.size, size=min(40, layout.size), replace=False):
        e = np.zeros(layout.size)
        e[idx] = h
        fd = (dual.evaluate(th + e, node, layout, 0.05)[1]
              - dual.evaluate(th - e, node, layout, 0.05)[1]) / (2 * h)
        assert g[idx] == pytest.approx(fd, rel=1e-4, abs=1e-6), idx


def test_the_capped_simplex_part_is_its_minimum_at_zero_smoothing():
    c = np.array([3.0, -1.0, 2.0, 0.5, -4.0])
    val, z = dual.capped_simplex(c, 2, 1e-12)
    assert val == pytest.approx(-5.0, abs=1e-9)
    assert z.sum() == pytest.approx(2.0) and np.all((z >= 0) & (z <= 1))


@pytest.mark.parametrize("r", [0, 3])
def test_the_capped_simplex_with_no_choice_is_its_one_point(r):
    val, z = dual.capped_simplex(np.array([1.0, -2.0, 0.5]), r, 0.1)
    assert np.array_equal(z, np.full(3, float(r == 3)))
    assert val == pytest.approx((-0.5 if r == 3 else 0.0) + 0.05 * (r - r), abs=1e-12)


def test_a_perspective_factor_at_zero_cannot_move_and_the_start_avoids_it():
    """The factor's gradient vanishes at ``p = q = s = 0``, which is why the
    dual starts off it."""
    B, A = _panel(6, 10, 6)
    node = dual.node_parts(ExactData(B, A), 3, [], [], list(range(6)))
    layout = dual.Layout(node)
    b = layout.blocks[0]
    _, _, g0 = dual.evaluate(np.zeros(layout.size), node, layout, 0.01)
    assert not np.any(g0[b["lam"][0]:b["lam"][1]])
    th = layout.start(0.1)
    lam = th[b["lam"][0]:b["lam"][1]].reshape(-1, 3)
    assert np.all(lam[:, 0] == 0.1) and np.all(lam[:, 1] == 0) and np.all(lam[:, 2] == 0.1)
    _, _, g = dual.evaluate(th, node, layout, 0.01)
    assert np.any(g[b["lam"][0]:b["lam"][1]])


def test_a_node_with_no_free_market_bounds_its_one_design():
    """Every market decided: no indicators, and the bound sits under the one
    design the node holds."""
    B, A = _panel(6, 10, 7)
    I, O = (0, 2, 4), (1, 3, 5)
    node = dual.node_parts(ExactData(B, A), 3, I, O, [])
    assert node.n_free == 0
    out = dual.dual_bound(node.data, 3, I, O, [], **FAST)
    value = side_fit(B, A, list(I))[0] + side_fit(B, A, list(O))[0]
    assert out["lower"] is not None and float(out["lower"]) <= value * (1 + 1e-9)


# ----------------------------------------------------------------------
# the bound
# ----------------------------------------------------------------------


#: A schedule that converges on eight markets in a few seconds.
FAST = dict(a0=1e-2, shrink=0.25, per_round=2000)


@pytest.fixture(scope="module")
def small():
    B, A = _panel(8, 14, 5)
    return B, A, ExactData(B, A)


@pytest.mark.parametrize("which", range(4))
def test_the_dual_reaches_the_interior_point_bound(small, which):
    B, A, data = small
    I, O = NODES[which]
    F = _split(8, I, O)
    ipm = branch.node_bound(data, 3, I, O, F)
    out = dual.dual_bound(data, 3, I, O, F, **FAST)
    sub = _subtree_min(B, A, 3, I, O)
    assert out["lower"] is not None
    assert float(out["lower"]) <= sub * (1 + 1e-9)
    assert float(out["lower"]) >= float(ipm.lower) * (1 - 2e-3) - 1e-9 * sub
    assert out["evaluations"] > 0 and out["seconds"] > 0


def test_a_target_already_met_stops_at_once(small):
    B, A, data = small
    F = _split(8, [], [])
    full = dual.dual_bound(data, 3, [], [], F, **FAST)
    low = dual.dual_bound(data, 3, [], [], F, target=float(full["lower"]) * 0.5, **FAST)
    assert low["evaluations"] < full["evaluations"]
    assert low["reached"] and float(low["lower"]) >= float(full["lower"]) * 0.5


def test_the_budget_stops_the_run(small):
    B, A, data = small
    out = dual.dual_bound(data, 3, [], [], _split(8, [], []), budget=50)
    assert out["evaluations"] == 50 and out["lower"] is not None and not out["reached"]


def test_a_node_whose_certificate_fails_reports_none(small, monkeypatch):
    B, A, data = small
    monkeypatch.setattr(dual.fullform, "lambda_min_lower", lambda D, *a, **k: None)
    out = dual.dual_bound(data, 3, [], [], _split(8, [], []), max_rounds=1)
    assert out["lower"] is None


def test_the_runner_appends_a_row(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(dual, "RESULTS", tmp_path)
    monkeypatch.setattr(dual, "compare", lambda name: [{"panel": name, "node": "root"}])
    assert dual.main(["factor12"]) == 0
    assert json.loads(open(tmp_path / "compare.jsonl").readline())["panel"] == "factor12"
