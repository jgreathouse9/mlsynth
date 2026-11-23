"""PANGEO's group costs computed the way least squares problems are solved, and the elimination done with bounds
that hold under rounding.

A split's cost is the residual sum of squares of fitting a level to the gap between its halves' mean paths. Phase 2
first scored it as c'Gc with G = A'A, the cross-product (normal-equations) matrix of the centred paths, and for the
weighted objective with the one-pass formula on uncentred data. Both lose accuracy as the square of the conditioning
(Golub and Van Loan, Matrix Computations, 4th ed., section 5.3.7; Bjorck, Numerical Methods for Least Squares
Problems, 1996, section 2.2.1). On panels whose units share a large common factor and run nearly parallel in
clusters, the costs that decide the design came out with relative errors near 0.4, some negative, and the design was
above the exact optimum on 14 of 16 panels.

The oracle here is exact rational arithmetic on the panel's doubles.
"""
from __future__ import annotations

import itertools
from fractions import Fraction as Fr

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, mip
from mlsynth.utils.pangeo_helpers.pairs import solve_pairs

OBJECTIVES = ("ss_res", "r2", "weighted", "supergeo_total")
_EPS = Fr(1e-12)


# --- the exact oracle -----------------------------------------------------------------------------------------------

def _wmean(rows, weights, idx):
    W = sum(weights[i] for i in idx)
    return [sum(weights[i] * rows[i][t] for i in idx) / W for t in range(len(rows[idx[0]]))]


def _exact_split(Y, A, B, objective, w=None, cov=None, scales=None, cw=None, uw=None):
    n, T = Y.shape
    rows = [[Fr(float(v)) for v in Y[i]] for i in range(n)]
    u = [Fr(1)] * n if uw is None else [Fr(float(v)) for v in uw]
    if objective == "supergeo_total":
        d = sum(sum(rows[i]) for i in A) - sum(sum(rows[i]) for i in B)
        s = d * d
    else:
        ma, mb = _wmean(rows, u, A), _wmean(rows, u, B)
        g = [a - b for a, b in zip(ma, mb)]
        if objective == "weighted":
            W = [Fr(float(x)) for x in w]
            Ws = max(sum(W), _EPS)
            d = sum(wi * gi for wi, gi in zip(W, g)) / Ws
            s = sum(wi * (gi - d) ** 2 for wi, gi in zip(W, g))
        else:
            d = sum(g) / T
            s = sum((gi - d) ** 2 for gi in g)
            if objective == "r2":
                abar = sum(ma) / T
                tot = sum((a - abar) ** 2 for a in ma)
                if tot > _EPS:
                    s = s / tot
    if cov is not None:
        C = [[Fr(float(v)) for v in cov[i]] for i in range(n)]
        ca, cb = _wmean(C, u, A), _wmean(C, u, B)
        sc = [Fr(float(v)) for v in scales]
        om = [Fr(1)] * len(sc) if cw is None else [Fr(float(v)) for v in cw]
        s += sum(o * ((a - b) / q) ** 2 for o, a, b, q in zip(om, ca, cb, sc))
    return s


def _exact_group(Y, g, Q, objective, **kw):
    g = list(g)
    best = None
    for a in range(1, min(Q, len(g) - 1) + 1):
        if len(g) - a > Q:
            continue
        for rest in itertools.combinations(g[1:], a - 1):
            A = [g[0], *rest]
            B = [u for u in g if u not in A]
            c = _exact_split(Y, A, B, objective, **kw)
            best = c if best is None or c < best else best
    return best


def _exact_optimum(n, cost):
    best = [None]

    def go(left, acc):
        if best[0] is not None and acc >= best[0]:
            return
        if not left:
            best[0] = acc
            return
        i = min(left)
        for g, c in cost.items():
            if g[0] == i and set(g) <= left:
                go(left - set(g), acc + c)

    go(frozenset(range(n)), Fr(0))
    return best[0]


def _clustered(seed, n, T, F=1e6, LEV=1e7, EPS=1e-2):
    """Units in clusters of two sharing a common factor's loading, at levels LEV: the good groups run nearly
    parallel, so the costs that decide the design are small next to the paths."""
    rng = np.random.default_rng(seed)
    f = np.sin(np.arange(T) * 2 * np.pi / 12) + 0.3 * np.arange(T) / T
    load = np.repeat(1 + 0.01 * rng.normal(size=(n + 1) // 2), 2)[:n, None]
    return LEV * (1 + rng.uniform(size=(n, 1))) + F * f * load + EPS * rng.normal(size=(n, T))


def _kw(objective, n, T, seed, cov=False, uw=False, levels=0.0):
    rng = np.random.default_rng(seed + 7)
    kw = {}
    if objective == "weighted":
        kw["weights"] = rng.uniform(0.2, 2.0, T)
    if cov:
        kw["cov"] = levels + rng.normal(size=(n, 2))
        kw["cov_scales"] = rng.uniform(0.5, 2.0, 2)
        kw["cov_weights"] = rng.uniform(0.0, 2.0, 2)
    if uw:
        kw["unit_weights"] = rng.uniform(0.5, 3.0, n)
    return kw


def _oracle_kw(kw):
    return dict(w=kw.get("weights"), cov=kw.get("cov"), scales=kw.get("cov_scales"), cw=kw.get("cov_weights"),
                uw=kw.get("unit_weights"))


def _rows(pool):
    return [tuple(int(p) for p in r if p >= 0) for r in pool.members]


# --- the costs ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("LEV", [1e7, 1e10])
@pytest.mark.parametrize("objective", ["ss_res", "r2", "weighted"])
def test_near_parallel_costs_are_accurate(objective, LEV):
    """Unit: on a near-parallel clustered panel every group's computed cost is within 1e-6 of its exact cost,
    relative. The normal-equations form gave relative errors near 0.4 on the weighted objective at levels 1e7; with
    the levels left in the data (the one-pass form) levels of 1e10 cost about 1e-4."""
    n, T = 8, 16
    Y = _clustered(0, n, T, LEV=LEV)
    kw = _kw(objective, n, T, 0)
    pool = columns.score_columns(np.arange(n), Y, 2, objective=objective, **kw)
    for r, c in zip(_rows(pool), pool.cost):
        ex = _exact_group(Y, r, 2, objective, **_oracle_kw(kw))
        assert abs(Fr(float(c)) - ex) <= Fr(1e-6) * ex, (r, float(c), float(ex))


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_costs_are_never_negative(objective):
    """Unit: a sum of squares is not negative; the normal-equations form returned costs below zero."""
    n, T = 8, 16
    Y = _clustered(1, n, T)
    pool = columns.score_columns(np.arange(n), Y, 2, objective=objective, **_kw(objective, n, T, 1))
    assert (pool.cost >= 0).all() and (pool.cost_lo >= 0).all()


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 6), Q=st.integers(1, 3), T=st.integers(2, 10),
       objective=st.sampled_from(OBJECTIVES), cov=st.booleans(), uw=st.booleans(),
       F=st.sampled_from([0.0, 1e3, 1e6]), LEV=st.sampled_from([0.0, 1e4, 1e8]), EPS=st.sampled_from([1e-3, 1.0]),
       clev=st.sampled_from([0.0, 1e6]))
def test_property_exact_cost_lies_within_the_bounds(seed, n, Q, T, objective, cov, uw, F, LEV, EPS, clev):
    """Property: for every group, cost_lo <= exact cost <= cost_up, and the computed cost lies between them, over
    panels with large common factors, large levels, small noise, covariates at large levels and unit weights."""
    Y = _clustered(seed, n, T, F, LEV, EPS)
    kw = _kw(objective, n, T, seed, cov, uw, clev)
    pool = columns.score_columns(np.arange(n), Y, Q, objective=objective, **kw)
    for r, c, lo, up in zip(_rows(pool), pool.cost, pool.cost_lo, pool.cost_up):
        ex = _exact_group(Y, r, Q, objective, **_oracle_kw(kw))
        assert Fr(float(lo)) <= ex <= Fr(float(up)), (r, float(lo), float(ex), float(up))
        assert lo <= c <= up


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_bounds_are_narrow_on_a_benign_panel(objective):
    """Unit: on a panel with no common factor and no levels the bounds are within 1e-12 of the cost, relative to the
    largest cost, so they cost the elimination nothing."""
    n, T = 8, 16
    Y = np.random.default_rng(3).normal(size=(n, T)).cumsum(axis=1)
    pool = columns.score_columns(np.arange(n), Y, 2, objective=objective, **_kw(objective, n, T, 3))
    assert ((pool.cost_up - pool.cost_lo) <= 1e-12 * pool.cost.max()).all()


# --- the designs ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["ss_res", "weighted"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_design_is_the_exact_optimum_on_near_parallel_panels(objective, seed):
    """Unit: the design's exact cost is the exact optimum. With the normal-equations form it was above the optimum
    on 14 of 16 such panels, by up to 154%."""
    n, T = 8, 16
    Y = _clustered(seed, n, T)
    kw = _kw(objective, n, T, seed)
    cost = {g: _exact_group(Y, g, 2, objective, **_oracle_kw(kw))
            for m in (2, 3, 4) for g in itertools.combinations(range(n), m)}
    chosen, d = columns.solve_exact(np.arange(n), Y, 2, objective=objective, **kw)
    got = sum(cost[tuple(int(u) for u in c["members"])] for c in chosen)
    assert got == _exact_optimum(n, cost)
    assert d["objective_value"] >= 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pairs_path_is_the_exact_optimum_on_near_parallel_panels(seed):
    """Unit: the pairs matcher, fed the new scorer's candidates, returns the exact optimal pairing."""
    n, T = 8, 16
    Y = _clustered(seed, n, T)
    cost = {g: _exact_group(Y, g, 1, "ss_res") for g in itertools.combinations(range(n), 2)}
    cands = columns.score_columns(np.arange(n), Y, 1).candidates()
    chosen, _ = solve_pairs(cands, np.arange(n), min_pairs=1)
    got = sum(cost[tuple(sorted(int(u) for u in c["members"]))] for c in chosen)
    assert got == _exact_optimum(n, cost)


# --- the bound ------------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 6), Q=st.integers(1, 3), k=st.integers(1, 3),
       scale=st.sampled_from([1e-6, 1.0, 1e8]), F=st.sampled_from([0.0, 1e6]))
def test_property_the_bound_and_the_reduced_costs_are_rigorous(seed, n, Q, k, scale, F):
    """Property: for any duals, the reduced costs returned are at most the exact ones and the Lagrangian bound is at
    most the exact Lagrangian bound, computed in rationals from the exact costs."""
    T = 8
    Y = _clustered(seed, n, T, F, 1e7 if F else 0.0, 1e-2)
    pool = columns.score_columns(np.arange(n), Y, Q)
    if len(pool) == 0:
        return
    rng = np.random.default_rng(seed)
    med = float(np.median(pool.cost)) + 1.0
    pi = rng.normal(size=n) * scale * med
    sigma = abs(float(rng.normal())) * scale * med
    L, rc = columns.lagrangian_bound(pool, pi, sigma, k)
    P = [Fr(float(p)) for p in pi]
    S = Fr(float(sigma))
    exact_rc = []
    for r in _rows(pool):
        c = _exact_group(Y, r, Q, "ss_res")
        exact_rc.append(c - sum(P[i] for i in r) - S)
    for lo, ex in zip(rc, exact_rc):
        assert Fr(float(lo)) <= ex
    exact_L = sum(P) + S * k + sum(min(Fr(0), e) for e in exact_rc)
    assert Fr(float(L)) <= exact_L


def test_covariates_at_large_levels_are_accurate():
    """Unit: covariates near 1e13 that differ by about 1 across units give the cost to 1e-6 relative; they are
    centred over the arm first, which leaves every difference of means exact. Uncentred, the differences of means
    lose about 1e-3 to cancellation."""
    n, T = 6, 8
    Y = 0.1 * np.random.default_rng(5).normal(size=(n, T)).cumsum(axis=1)
    kw = dict(cov=1e13 + np.random.default_rng(6).normal(size=(n, 2)), cov_scales=np.array([1.0, 2.0]))
    pool = columns.score_columns(np.arange(n), Y, 2, **kw)
    for r, c in zip(_rows(pool), pool.cost):
        ex = _exact_group(Y, r, 2, "ss_res", **_oracle_kw(kw))
        assert abs(Fr(float(c)) - ex) <= Fr(1e-6) * ex


@pytest.mark.parametrize("weights", [np.r_[-1.0, np.ones(7)], np.zeros(8), np.r_[np.nan, np.ones(7)], np.ones(5)])
def test_weighted_needs_valid_weights(weights):
    """Failure: weights that are negative, sum to zero, are not finite or do not match the window are reported."""
    with pytest.raises(ValueError, match="weights"):
        columns.score_columns(np.arange(4), np.random.default_rng(0).normal(size=(4, 8)), 2, objective="weighted",
                              weights=weights)


def test_reduced_costs_are_bounded_below_on_a_near_parallel_panel():
    """Unit: with small duals on a near-parallel panel, every group's reduced-cost bound is at most its exact reduced
    cost. Reduced costs formed from the computed costs, which can lie above the exact ones by more than the rounding
    of the subtraction, would leave a group of a cheaper design out of the MIP."""
    n, T = 8, 16
    Y = _clustered(2, n, T, LEV=1e10)
    pool = columns.score_columns(np.arange(n), Y, 2)
    pi = np.full(n, 1e-9)
    _, rc = columns.lagrangian_bound(pool, pi, 0.0, 2)
    P = Fr(1e-9)
    for r, lo in zip(_rows(pool), rc):
        assert Fr(float(lo)) <= _exact_group(Y, r, 2, "ss_res") - len(r) * P
