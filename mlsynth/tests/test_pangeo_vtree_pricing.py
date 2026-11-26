"""Pricing in the volume tree (:mod:`mlsynth.utils.pangeo_helpers.vtree`): only the groups whose reduced cost can be
negative are priced.

Each volume step prices the node's open groups at a trial point: a group's reduced cost, and the group enters the
Lagrangian minimiser when that is negative. After a pass over every open group at (pf, sf), a group's reduced cost
at (pi, s) differs from its value there by at most the group's size times max |pi - pf| plus |s - sf|, and each of
the two computed values lies within a rounding term of its exact one. A group whose computed reduced cost at the full
pass reaches that movement plus both rounding terms has a computed reduced cost of at least zero at the trial point,
so it adds nothing to the bound and stays out of the minimiser: skipping it changes no bit of any iterate. The
tests hold the pricing to that claim: every iterate of the volume algorithm, and every result of the tree, equal
with ``kappa = 0`` (a full pass every step, the algorithm with no skipping) bit for bit, on benign arms and on
ill-conditioned inputs; and the pricing must skip, or it is only the old algorithm.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, cover, vtree


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _arm(seed, n, Q, min_pairs=1):
    pool = columns.score_columns(np.arange(n), _panel(seed, n), Q)
    k = columns.min_groups(n, Q, min_pairs)
    return pool, columns.root_lp(pool, k), k


def _node(pool, root, k, rows=None):
    """The root node's arrays over the pool rows ``rows`` (every row when None)."""
    rows = np.arange(len(pool)) if rows is None else np.asarray(rows, dtype=np.int64)
    n = int(pool.unit_indices.size)
    mem = np.ascontiguousarray(pool.members[rows], dtype=np.int64)
    clo = np.ascontiguousarray(pool.cost_lo[rows], dtype=float)
    return mem, clo, n, k, np.asarray(root.pi, dtype=float).copy(), float(max(root.sigma, 0.0))


def _run(mem, clo, n, kk, pi0, s0, target, T, kappa, units=None):
    """One call of the volume algorithm over every group in ``mem``, the units ``units`` (all when None) left;
    returns every output and every work array it writes."""
    C = mem.shape[0]
    uncm = vtree._mask_of(np.arange(n) if units is None else units, vtree._words(n))
    out = dict(pib=np.zeros(n), xt=np.zeros(C), xbar=np.zeros(C), v=np.zeros(n), r0=np.zeros(n), r1=np.zeros(n),
               pit=np.zeros(n), rcv=np.zeros(C), act=np.zeros(C, dtype=np.int64), pf=np.zeros(n))
    s, steps, priced = vtree._volume(mem, clo, np.arange(C, dtype=np.int64), 0, C, uncm, n, kk, pi0, s0,
                                     float(target), int(T), out["pib"], out["xt"], out["xbar"], out["v"], out["r0"],
                                     out["r1"], out["pit"], out["rcv"], out["act"], out["pf"], float(kappa))
    out.update(sigma=s, steps=steps, priced=priced)
    return out


def _same_iterates(a, b):
    for key in ("pib", "xbar", "xt"):
        assert np.array_equal(a[key], b[key]), f"lazy pricing changed {key}"
    assert a["sigma"] == b["sigma"], "lazy pricing changed sigma"
    assert a["steps"] == b["steps"], "lazy pricing changed the number of steps"


def _ill_conditioned(rng, n, C, level, spread, W):
    """Groups of 1 to W units, costs at ``level`` with differences of ``spread``, and multipliers that make the
    reduced costs small differences of large terms."""
    mem = -np.ones((C, W), dtype=np.int64)
    for j in range(C):
        m = int(rng.integers(1, W + 1))
        mem[j, :m] = rng.choice(n, size=m, replace=False)
    size = (mem >= 0).sum(axis=1)
    pi = level / W * (1.0 + spread * rng.normal(size=n))
    clo = np.array([pi[mem[j, :size[j]]].sum() for j in range(C)]) + level * spread * rng.normal(size=C)
    return mem, clo, pi


# ----------------------------------------------------------------------------- smoke


def test_smoke_lazy_pricing_runs_the_volume_algorithm():
    pool, root, k = _arm(3, 12, 2)
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k)
    out = _run(mem, clo, n, kk, pi0, s0, np.inf, 50, vtree.PRICING_KAPPA)
    assert np.isfinite(out["pib"]).all() and np.isfinite(out["sigma"]) and out["sigma"] >= 0.0
    assert 0 < out["steps"] <= 50
    assert 0 < out["priced"] <= (out["steps"] + 1) * mem.shape[0]


# ----------------------------------------------------------------------------- the iterates are unchanged


@pytest.mark.parametrize("seed,n,Q", [(1, 12, 2), (2, 10, 3), (5, 14, 2), (7, 9, 4)])
@pytest.mark.parametrize("finite_target", [True, False])
def test_lazy_pricing_leaves_every_iterate_unchanged(seed, n, Q, finite_target):
    pool, root, k = _arm(seed, n, Q)
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k)
    target = float(root.bound) * 1.2 if finite_target else np.inf
    full = _run(mem, clo, n, kk, pi0, s0, target, 200, 0.0)
    lazy = _run(mem, clo, n, kk, pi0, s0, target, 200, vtree.PRICING_KAPPA)
    _same_iterates(lazy, full)


def test_lazy_pricing_prices_fewer_groups():
    pool, root, k = _arm(5, 14, 3)
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k)
    full = _run(mem, clo, n, kk, pi0, s0, np.inf, 200, 0.0)
    lazy = _run(mem, clo, n, kk, pi0, s0, np.inf, 200, vtree.PRICING_KAPPA)
    assert full["priced"] == (full["steps"] + 1) * mem.shape[0], "kappa 0 prices every group at every step"
    assert lazy["priced"] < full["priced"] / 2, "the pricing skips groups that cannot be negative"


def test_lazy_pricing_with_the_cardinality_row_moving_leaves_the_iterates_unchanged():
    pool, root, k = _arm(4, 12, 2, min_pairs=4)
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k)
    full = _run(mem, clo, n, kk, pi0, 0.0, np.inf, 200, 0.0)
    lazy = _run(mem, clo, n, kk, pi0, 0.0, np.inf, 200, vtree.PRICING_KAPPA)
    assert full["sigma"] > 0.0, "the cardinality row's multiplier moves on this arm"
    _same_iterates(lazy, full)


def test_lazy_pricing_on_a_node_leaves_the_iterates_unchanged():
    pool, root, k = _arm(6, 14, 3)
    first = int(np.argmin(pool.cost))
    taken = set(int(u) for u in pool.members[first] if u >= 0)
    rows = [j for j in range(len(pool)) if not taken & set(int(u) for u in pool.members[j] if u >= 0)]
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k, rows)
    units = np.array(sorted(set(range(n)) - taken))
    full = _run(mem, clo, n, kk - 1, pi0, s0, np.inf, 150, 0.0, units)
    lazy = _run(mem, clo, n, kk - 1, pi0, s0, np.inf, 150, vtree.PRICING_KAPPA, units)
    _same_iterates(lazy, full)
    assert np.all(lazy["pib"][sorted(taken)] == 0.0), "covered units carry no multiplier"


@pytest.mark.parametrize("level,spread", [(1.0, 1e-3), (1e8, 1e-6), (1e12, 1e-3), (1e15, 1e-1)])
def test_lazy_pricing_on_ill_conditioned_inputs_leaves_the_iterates_unchanged(level, spread):
    rng = np.random.default_rng(int(np.log10(level)) + 11)
    n, C, W = 14, 300, 6
    mem, clo, pi = _ill_conditioned(rng, n, C, level, spread, W)
    for kk, s0 in ((0, 0.0), (3, level * spread)):
        full = _run(mem, clo, n, kk, pi, s0, np.inf, 150, 0.0)
        lazy = _run(mem, clo, n, kk, pi, s0, np.inf, 150, vtree.PRICING_KAPPA)
        _same_iterates(lazy, full)


# ----------------------------------------------------------------------------- the skip test


def _rc(mem_row, c, pi, s):
    rc = c - s
    for u in mem_row:
        if u >= 0:
            rc -= pi[u]
    return rc


def _exact_rc(mem_row, c, pi, s):
    return Fraction(c) - Fraction(s) - sum((Fraction(float(pi[u])) for u in mem_row if u >= 0), Fraction(0))


def test_the_skip_margin_covers_the_rounding_of_both_evaluations():
    # Reduced costs that are differences of terms near 1e16, where one unit in the last place is 2: a group whose
    # computed reduced cost at the full pass sits a few units above the movement can turn negative at the trial
    # point by rounding alone.
    rng = np.random.default_rng(0)
    W, hits = 4, 0
    for _ in range(4000):
        pf = 2.5e15 + rng.integers(-8, 9, size=W).astype(float)
        c = float(pf.sum() + rng.integers(0, 16))
        pit = pf + rng.integers(-2, 3, size=W).astype(float)
        mf = float(np.abs(pit - pf).max())
        row = np.arange(W)
        rf = _rc(row, c, pf, 0.0)
        delta = W * mf + 1.0
        allowed = vtree._may_skip(W, mf, 0.0, delta, abs(c), float(np.abs(pf).max()), 0.0,
                                  float(np.abs(pit).max()), 0.0)
        if rf >= delta and not allowed:
            hits += 1
        if allowed and rf >= delta:
            assert _rc(row, c, pit, 0.0) >= 0.0, "a skipped group would have been negative at the trial point"
    assert hits > 0, "the rounding margin refuses some skips the movement alone would allow"


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 10**6), level=st.sampled_from([1.0, 1e6, 1e12, 1e15, 1e16]),
       spread=st.sampled_from([1e-12, 1e-6, 1e-3, 1.0]), W=st.integers(1, 8), s_move=st.booleans())
def test_property_a_skipped_group_is_never_negative_at_the_trial_point(seed, level, spread, W, s_move):
    rng = np.random.default_rng(seed)
    pf = level * (1.0 + spread * rng.normal(size=W))
    sf = abs(level * spread * rng.normal()) if s_move else 0.0
    c = float(pf.sum() + sf + level * spread * abs(rng.normal()) * W)
    step = level * spread * abs(rng.normal())
    pit = pf + step * rng.uniform(-1, 1, size=W)
    st_ = max(sf + (step * rng.uniform(-1, 1) if s_move else 0.0), 0.0)
    mf, ms = float(np.abs(pit - pf).max()), abs(st_ - sf)
    row = np.arange(W)
    delta = float(rng.uniform(0.0, 4.0)) * (W * mf + ms + level * spread)
    if vtree._may_skip(W, mf, ms, delta, abs(c), float(np.abs(pf).max()), sf, float(np.abs(pit).max()), st_):
        if _rc(row, c, pf, sf) >= delta:
            assert _rc(row, c, pit, st_) >= 0.0
            assert _exact_rc(row, c, pit, st_) >= 0, "the exact reduced cost is not negative either"


def test_no_movement_budget_allows_no_skip():
    assert not vtree._may_skip(3, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0)


# ----------------------------------------------------------------------------- the passes


def test_the_full_pass_prices_like_the_plain_pricing_and_lists_the_groups_below_delta():
    pool, root, k = _arm(2, 11, 3)
    mem, clo, n, kk, pi, s = _node(pool, root, k)
    C = mem.shape[0]
    uncm = vtree._mask_of(np.arange(n), vtree._words(n))
    opn = np.arange(C, dtype=np.int64)
    xt0, r0 = np.zeros(C), np.zeros(n)
    z0 = vtree._price(mem, clo, opn, 0, C, pi, s, uncm, n, kk, xt0)
    vs0 = vtree._resid(mem, opn, 0, C, xt0, uncm, n, kk, r0)
    xt, r, rcv, act = np.zeros(C), np.zeros(n), np.zeros(C), np.zeros(C, dtype=np.int64)
    pf = np.full(n, np.nan)
    delta = 0.05
    z, vs, na, cmax, pimax = vtree._price_full(mem, clo, opn, 0, C, pi, s, uncm, n, kk, xt, r, rcv, act, delta, pf)
    assert z == z0 and vs == vs0
    assert np.array_equal(pf, pi), "the full pass records the point it priced at"
    assert cmax == np.abs(clo).max(), "the full pass reports the largest cost magnitude, for the rounding margin"
    assert pimax == np.abs(pi).max(), "the full pass reports the largest multiplier magnitude"
    assert np.array_equal(xt, xt0) and np.array_equal(r, r0)
    rc = np.array([_rc(mem[j], clo[j], pi, s) for j in range(C)])
    assert np.array_equal(rcv, rc)
    assert np.array_equal(act[:na], np.flatnonzero(rc < delta)), "the active list is every group below delta, in order"


def test_the_active_pass_prices_only_the_listed_groups():
    pool, root, k = _arm(2, 11, 3)
    mem, clo, n, kk, pi, s = _node(pool, root, k)
    C = mem.shape[0]
    uncm = vtree._mask_of(np.arange(n), vtree._words(n))
    opn = np.arange(C, dtype=np.int64)
    rc = np.array([_rc(mem[j], clo[j], pi, s) for j in range(C)])
    listed = np.flatnonzero(rc < 0.0)[::2].astype(np.int64)
    xt, r = np.zeros(C), np.zeros(n)
    z, vs = vtree._price_act(mem, clo, opn, 0, listed.size, pi, s, uncm, n, kk, xt, r, listed)
    assert np.array_equal(np.flatnonzero(xt), listed)
    assert z == pytest.approx(float(pi.sum() + s * kk + rc[listed].sum()), rel=1e-12)
    assert vs == kk - listed.size


# ----------------------------------------------------------------------------- edges


def test_no_steps_prices_once():
    pool, root, k = _arm(3, 10, 2)
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k)
    out = _run(mem, clo, n, kk, pi0, s0, np.inf, 0, vtree.PRICING_KAPPA)
    assert out["steps"] == 0 and out["priced"] == mem.shape[0]
    assert np.array_equal(out["pib"], pi0)


def test_a_target_already_met_takes_no_step():
    pool, root, k = _arm(3, 10, 2)
    mem, clo, n, kk, pi0, s0 = _node(pool, root, k)
    out = _run(mem, clo, n, kk, pi0, s0, float(root.bound) - 1.0, 50, vtree.PRICING_KAPPA)
    assert out["steps"] == 0 and out["priced"] == mem.shape[0]


def test_a_single_group_and_no_cardinality_row():
    mem = np.array([[0, 1, -1, -1]], dtype=np.int64)
    full = _run(mem, np.array([1.0]), 2, 0, np.array([0.2, 0.3]), 0.0, np.inf, 60, 0.0)
    lazy = _run(mem, np.array([1.0]), 2, 0, np.array([0.2, 0.3]), 0.0, np.inf, 60, vtree.PRICING_KAPPA)
    _same_iterates(lazy, full)
    assert lazy["sigma"] == 0.0


def test_no_open_groups():
    mem = np.zeros((0, 4), dtype=np.int64)
    out = _run(mem, np.zeros(0), 3, 1, np.array([1.0, 1.0, 1.0]), 0.0, np.inf, 20, vtree.PRICING_KAPPA)
    assert out["priced"] == 0 and np.isfinite(out["pib"]).all()


# ----------------------------------------------------------------------------- the tree


@pytest.mark.parametrize("seed,n,Q,min_pairs", [(2, 12, 2, 1), (3, 11, 3, 1), (8, 14, 2, 3), (9, 70, 1, 1)])
def test_the_tree_with_lazy_pricing_is_the_tree_without_it(seed, n, Q, min_pairs):
    pool, root, k = _arm(seed, n, Q, min_pairs)
    a = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**8, kappa=0.0)
    b = vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**8)
    assert (a.status, a.nodes, a.upper, a.lower) == (b.status, b.nodes, b.upper, b.lower)
    assert np.array_equal(a.rows, b.rows)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 13), Q=st.integers(2, 3), min_pairs=st.integers(1, 3),
       iterations=st.integers(0, 120), kappa=st.floats(0.0, 64.0), fix=st.booleans())
def test_property_the_tree_does_not_depend_on_kappa(seed, n, Q, min_pairs, iterations, kappa, fix):
    if not columns.feasible(n, Q, min_pairs):
        return
    pool, root, k = _arm(seed, n, Q, min_pairs)
    keep = np.arange(len(pool))
    a = vtree.search(pool, root, keep, k, np.inf, 10**8, iterations=iterations, fix=fix, kappa=0.0)
    b = vtree.search(pool, root, keep, k, np.inf, 10**8, iterations=iterations, fix=fix, kappa=kappa)
    assert (a.status, a.nodes, a.upper, a.lower) == (b.status, b.nodes, b.upper, b.lower)
    assert np.array_equal(a.rows, b.rows)


def test_the_search_hands_kappa_to_the_kernel(monkeypatch):
    seen = []
    real = vtree._tree

    def spy(*args):
        seen.append(args[-1])
        return real(*args)

    monkeypatch.setattr(vtree, "_tree", spy)
    pool, root, k = _arm(2, 10, 2)
    vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**6)
    vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**6, kappa=7.5)
    assert seen == [vtree.PRICING_KAPPA, 7.5]


# ----------------------------------------------------------------------------- failures


@pytest.mark.parametrize("kappa", [-1.0, -1e-300, float("nan"), float("inf"), float("-inf")])
def test_a_kappa_that_is_negative_or_not_finite_is_refused(kappa):
    pool, root, k = _arm(2, 10, 2)
    with pytest.raises(MlsynthConfigError, match="kappa"):
        vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**6, kappa=kappa)


@settings(max_examples=40, deadline=None)
@given(kappa=st.one_of(st.floats(max_value=-1e-300, allow_nan=False), st.just(float("nan")),
                       st.just(float("inf"))))
def test_property_every_invalid_kappa_is_refused(kappa):
    pool, root, k = _arm(2, 8, 2)
    with pytest.raises(MlsynthConfigError, match="kappa"):
        vtree.search(pool, root, np.arange(len(pool)), k, np.inf, 10**6, kappa=kappa)
