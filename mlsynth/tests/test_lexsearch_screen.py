"""The affine screen in front of LEXSCM's exact enumeration.

Enumeration prices every treated set ``S`` by the simplex QP

    L(S) = min_{w in simplex} w' Q w,      Q = G_SS,

and keeps the ``top_K`` smallest. Most sets cannot enter the top-K, and most
of the ones that can have their minimiser inside the simplex, where a closed
form answers. The screen computes, for each ``Q`` of a batch,

    v = (Q + eps I)^-1 1 / 1'(Q + eps I)^-1 1      (the affine minimiser),
    low = 2 min_j (Q v)_j - v'Q v,

``low`` is a lower bound on ``L(S)`` for every ``v`` whatever its accuracy,
because ``(w - v)'Q(w - v) >= 0`` gives ``w'Qw >= 2 (Qv)'w - v'Qv`` and
``(Qv)'w >= min_j (Qv)_j`` on the simplex. When ``v`` is on the simplex,
``v'Qv`` is attained, so a vanishing gap between the two is ``L(S)`` itself.
A set whose ``low`` exceeds the current K-th best loss cannot enter the
top-K and is never solved.

The contract the tests hold:

* ``low <= L(S)`` on every set, conditioning and rank deficiency included;
* where the screen answers, the answer is ``L(S)``;
* enumeration with the screen returns the top-K the QP alone returns;
* the QP sees only the sets the screen cannot settle, a small share;
* a block the screen cannot read (non-finite) reaches the QP, which reports it.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.fast_scm_helpers import lexsearch
from mlsynth.utils.fast_scm_helpers.lexsearch import (
    _enumerate,
    _losses_for,
    _screen_bounds,
    select_treated_designs,
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _panel(T, M, factor_strength, seed):
    """An f-centred, standardised panel's Gram, as ``build_X_tilde`` makes it.

    ``factor_strength`` runs from independent columns (0) to one common factor
    (1), the near-collinear regime of a geo panel.
    """
    rng = np.random.default_rng(seed)
    idio = rng.standard_normal((T, M))
    f = np.cumsum(rng.standard_normal(T))
    X = factor_strength * np.outer(f, rng.uniform(0.6, 1.6, M)) + (1.0 - factor_strength) * idio
    X = X - X.mean(1, keepdims=True)
    sd = np.std(X, 1, keepdims=True)
    X = X / np.where(sd < 1e-8, 1.0, sd)
    return X.T @ X


def _blocks(G, sets):
    sets = np.asarray(sets, dtype=int)
    return G[sets[:, :, None], sets[:, None, :]]


def _brute_top(G, cand, m, top_K):
    combs = np.array(list(combinations(cand, m)), dtype=int)
    L = _losses_for(G, combs)
    order = np.argsort(L, kind="stable")[:top_K]
    return combs[order], L[order]


# --- smoke --------------------------------------------------------------------


def test_screen_returns_one_bound_per_block():
    G = _panel(12, 8, 0.5, 0)
    Qs = _blocks(G, list(combinations(range(8), 3)))
    low, value, exact = _screen_bounds(Qs)
    assert low.shape == value.shape == exact.shape == (len(Qs),)
    assert np.isfinite(low).all()
    assert exact.dtype == bool


# --- unit: the invariants on fixtures with a known answer ---------------------


def test_identity_is_settled_at_the_barycentre():
    """``I_m``: the minimiser is the uniform weight, inside the simplex."""
    low, value, exact = _screen_bounds(np.eye(4)[None])
    assert exact[0]
    assert value[0] == pytest.approx(0.25, rel=1e-12)
    assert low[0] == pytest.approx(0.25, rel=1e-12)


def test_minimiser_off_the_simplex_is_bounded_and_left_to_the_qp():
    """``[[1, 2], [2, 10]]``: the affine minimiser is (8/7, -1/7), outside the
    simplex. The simplex optimum is the vertex (1, 0) with loss 1, and the
    certified bound there is 6/7."""
    Q = np.array([[[1.0, 2.0], [2.0, 10.0]]])
    low, value, exact = _screen_bounds(Q)
    assert not exact[0]
    assert np.isinf(value[0])
    assert low[0] == pytest.approx(6.0 / 7.0, rel=1e-9)
    assert low[0] <= _losses_for(Q[0], np.array([[0, 1]]))[0]


def test_a_single_unit_is_its_own_diagonal():
    Qs = np.array([[[3.5]], [[0.2]]])
    low, value, exact = _screen_bounds(Qs)
    assert exact.all()
    np.testing.assert_allclose(value, [3.5, 0.2], rtol=1e-12)


# --- edge: degenerate blocks ----------------------------------------------------


@pytest.mark.parametrize("Q", [
    np.zeros((3, 3)),                                         # every unit at the target
    np.ones((3, 3)),                                          # three copies of one unit
    np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),               # rank one
    np.diag([0.0, 1.0, 4.0]),                                 # one unit at the target
])
def test_singular_blocks_are_bounded_without_error(Q):
    low, value, exact = _screen_bounds(Q[None])
    L = _losses_for(Q, np.arange(len(Q))[None])[0]
    assert np.isfinite(low[0])
    assert low[0] <= L + 1e-12 * max(1.0, abs(L))
    if exact[0]:
        assert value[0] == pytest.approx(L, rel=1e-9, abs=1e-12)


def test_more_units_than_predictors_leaves_every_block_singular():
    """``m > T``: ``G_SS`` has rank at most ``T``, so no block is invertible."""
    G = _panel(3, 10, 0.3, 4)
    sets = np.array(list(combinations(range(10), 5)))
    low, value, exact = _screen_bounds(_blocks(G, sets))
    L = _losses_for(G, sets)
    assert (low <= L + 1e-10 * np.maximum(1.0, np.abs(L))).all()
    np.testing.assert_allclose(value[exact], L[exact], rtol=1e-8, atol=1e-11)


# --- failure: what the screen cannot read reaches the QP, which reports it ------


def test_a_non_finite_block_is_reported_not_skipped():
    """A NaN in ``G`` is the QP's to report. A screen that compared NaN to the
    threshold would find it false and drop the set without a word."""
    G = _panel(10, 7, 0.4, 2)
    G[2, 5] = G[5, 2] = np.nan
    low, _, exact = _screen_bounds(_blocks(G, [[0, 2, 5]]))
    assert not np.isfinite(low[0]) and not exact[0]
    with pytest.raises(ValueError, match="non-finite"):
        _enumerate(G, np.arange(7), 3, 2, None, None, 80)


# --- properties: over the input domain -------------------------------------------


@SETTINGS
@given(T=st.integers(2, 14), M=st.integers(3, 9), m=st.integers(1, 5),
       strength=st.floats(0.0, 0.999), ridge=st.sampled_from([0.0, 0.0, 0.3]),
       seed=st.integers(0, 10_000))
def test_the_bound_never_exceeds_the_loss(T, M, m, strength, ridge, seed):
    m = min(m, M)
    G = _panel(T, M, strength, seed) + ridge * np.eye(M)
    sets = np.array(list(combinations(range(M), m)))
    low, value, exact = _screen_bounds(_blocks(G, sets))
    L = _losses_for(G, sets)
    tol = 1e-10 * np.maximum(1.0, np.abs(L))
    assert (low <= L + tol).all()
    assert (value[exact] >= L[exact] - tol[exact]).all()          # attained, so never below
    np.testing.assert_allclose(value[exact], L[exact], rtol=1e-9, atol=1e-12)


@SETTINGS
@given(T=st.integers(2, 14), M=st.integers(3, 10), m=st.integers(1, 5),
       top_K=st.integers(1, 12), strength=st.floats(0.0, 0.999),
       ridge=st.sampled_from([0.0, 0.0, 0.3]), seed=st.integers(0, 10_000))
def test_enumeration_returns_the_qps_top_k(T, M, m, top_K, strength, ridge, seed):
    m = min(m, M)
    G = _panel(T, M, strength, seed) + ridge * np.eye(M)
    cand = np.arange(M)
    raw, n = _enumerate(G, cand, m, top_K, None, None, 80, chunk=7)
    sets, L = _brute_top(G, cand, m, top_K)
    got = np.array([loss for _, loss in raw])
    assert n == len(list(combinations(cand, m)))
    assert len(raw) == len(L)
    np.testing.assert_allclose(got, L, rtol=1e-9, atol=1e-12)
    for S, loss in raw:                                # each returned set has the loss it claims
        np.testing.assert_allclose(_losses_for(G, np.asarray(S)[None])[0], loss, rtol=1e-9, atol=1e-12)


# --- the screen does the work it exists for --------------------------------------


def _count_qp(monkeypatch):
    calls = {"sets": 0}
    real = lexsearch._losses_for

    def counting(G, subsets, iters=70):
        calls["sets"] += len(subsets)
        return real(G, subsets, iters=iters)

    monkeypatch.setattr(lexsearch, "_losses_for", counting)
    return calls


def test_the_qp_sees_a_small_share_of_the_sets(monkeypatch):
    G = _panel(40, 26, 0.6, 7)
    calls = _count_qp(monkeypatch)
    raw, n = _enumerate(G, np.arange(26), 4, 3, None, None, 80, chunk=2_000)
    assert n == 14_950
    assert calls["sets"] <= 0.02 * n
    sets, L = _brute_top(G, np.arange(26), 4, 3)
    np.testing.assert_allclose([loss for _, loss in raw], L, rtol=1e-9)


def test_the_search_reports_how_many_sets_reached_the_qp():
    G = _panel(30, 14, 0.5, 3)
    out = select_treated_designs(G, np.arange(14), 3, top_K=2, method="enumerate")
    s = out["stats"]["search"]
    assert s["subsets_evaluated"] == 364
    assert 0 <= s["subsets_to_qp"] <= s["subsets_evaluated"]


def test_constraints_still_filter_before_the_screen():
    """Budget, forced units and conflicts act on the sets the screen sees."""
    G = _panel(30, 12, 0.5, 9)
    costs = np.linspace(1.0, 3.0, 12)
    conflict = np.zeros((12, 12), bool); conflict[0, 1] = conflict[1, 0] = True
    out = select_treated_designs(G, np.arange(12), 3, top_K=4, method="enumerate",
                                 unit_costs=costs, budget=6.0, forced=[4], conflict=conflict)
    feasible = [S for S in combinations(range(12), 3)
                if 4 in S and costs[list(S)].sum() <= 6.0 and not (0 in S and 1 in S)]
    L = _losses_for(G, np.array(feasible))
    best = sorted(zip(L, feasible))[:4]
    assert [tuple(d.indices) for d in out["top_designs"]] == [tuple(S) for _, S in best]
