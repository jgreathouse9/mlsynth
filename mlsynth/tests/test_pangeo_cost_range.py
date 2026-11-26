"""PANGEO's exact path across the whole range of the double format (#1170).

Two defects at the small end. ``mip._cost_scale`` formed the power of two 2^(30 - e) for the largest cost
m 2^e, which overflows once that cost is below about 2^-994, so every solve raised a bare ``OverflowError``. And
below the normal range the rigorous bounds stopped bracketing the exact costs. Every bound here is built from the
relative form of Rump, "Verification methods: rigorous results using floating-point arithmetic", Acta Numerica 19
(2010), Theorem 2.1, whose product bounds hold "provided no underflow has occurred" (p. 303). Rump's model of an
operation, (2.3) on p. 302, is a o b = fl(a o b)(1 + eps) + delta with |eps| <= u, |delta| <= eta / 2 and
eps * delta = 0, eta = 2^-1074. Three squares that form a cost bound lose that delta with no relative term to cover
it: the residual sum of squares of the trajectory objectives, the square of the difference of totals of
supergeo_total, and each covariate's weighted squared difference. The fix carries the delta at those three, adding
eta to an upper bound and subtracting it from a lower one, which is exact when the result is subnormal and rounds
away when it is normal (module docstring of :mod:`.columns`).

The oracle is exact rational arithmetic on the panel's doubles (``test_pangeo_columns_accuracy``). A panel is scaled
by 2^s: exactly, since a power of two multiplies a double without rounding while the product is normal, and costs
scale by 4^s.
"""
from __future__ import annotations

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover, mip
from mlsynth.tests.test_pangeo_columns_accuracy import (_exact_group, _exact_optimum, _kw, _oracle_kw, _rows)

ETA = 2.0 ** -1074
OBJECTIVES = ("ss_res", "r2", "weighted", "supergeo_total")
#: 0 is a benign panel; -480 to -510 keep every cost normal; -520 and below put them in the subnormal range;
#: -560 sends every square to zero.
SCALES = (0, -480, -510, -515, -520, -525, -530, -540, -560)


def _panel(seed, n, T=10):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * 3.0


def _scored(seed, n, Q, s, objective="ss_res", cov=False, uw=False, T=10):
    Y = np.ldexp(_panel(seed, n, T), s)
    kw = _kw(objective, n, T, seed, cov=cov, uw=uw)
    return Y, kw, columns.score_columns(np.arange(n), Y, Q, objective=objective, **kw)


def _brackets(Y, kw, pool, Q, objective):
    bad = []
    for r, g in enumerate(_rows(pool)):
        ex = _exact_group(Y, g, Q, objective, **_oracle_kw(kw))
        if not (Fr(float(pool.cost_lo[r])) <= ex <= Fr(float(pool.cost_up[r]))):
            bad.append((g, float(pool.cost_lo[r]), float(ex), float(pool.cost_up[r])))
    return bad


# --- smoke ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("s", [-510, -520, -530])
def test_smoke_a_panel_of_tiny_costs_is_solved_and_its_bounds_bracket_the_optimum(s):
    """Smoke: on largest costs from 7.7e-306 (normal) down to 7e-318 (subnormal) the exact path returns a design
    instead of an OverflowError, and its reported bounds bracket the exact optimum; the design is within the gap
    they report."""
    n, Q = 6, 2
    Y = np.ldexp(_panel(3, n), s)
    chosen, d = columns.solve_exact(np.arange(n), Y, Q)
    pool = columns.score_columns(np.arange(n), Y, Q)
    exact = {g: _exact_group(Y, g, Q, "ss_res") for g in _rows(pool)}
    opt = _exact_optimum(n, exact)
    design = sum((exact[tuple(sorted(int(u) for u in c["members"]))] for c in chosen), Fr(0))
    lower = d["dual_bound"] if d.get("dual_bound") is not None else d["lagrangian_bound"]
    assert Fr(float(lower)) <= opt <= design <= Fr(float(d["objective_upper"]))


# --- unit: the scorer's bounds bracket the exact cost at every scale -------------------------------------------------

@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("s", SCALES)
def test_every_bound_brackets_the_exact_cost_at_every_scale(objective, s):
    """Unit: every group's [cost_lo, cost_up] holds its exact cost, at normal and subnormal scales alike. On main
    154 of 154 groups failed from s = -520. For the trajectory objectives the cause is the residual sum of squares,
    whose T squares underflow; for supergeo_total it is the square of the difference of totals."""
    Y, kw, pool = _scored(5, 6, 2, s, objective)
    assert _brackets(Y, kw, pool, 2, objective) == []


@pytest.mark.parametrize("s", [-520, -540])
def test_covariates_and_unit_weights_keep_the_bracket_at_subnormal_scales(s):
    """Unit: the covariate term and unit weights are not scaled with the paths, so the cost mixes a subnormal
    trajectory term with a normal covariate term; the bracket holds for the sum."""
    Y, kw, pool = _scored(6, 5, 2, s, "ss_res", cov=True, uw=True)
    assert _brackets(Y, kw, pool, 2, "ss_res") == []


def test_tiny_covariate_differences_keep_the_bracket():
    """Unit: covariates near 1e-160 make the covariate term's products underflow; the bracket holds."""
    n, Q, T = 5, 2, 8
    Y = _panel(7, n, T)
    kw = _kw("ss_res", n, T, 7, cov=True)
    kw["cov"] = np.ldexp(kw["cov"], -530)
    pool = columns.score_columns(np.arange(n), Y, Q, **kw)
    assert _brackets(Y, kw, pool, Q, "ss_res") == []


@pytest.mark.parametrize("sy,sc,om", [(-530, -530, 1e3), (0, -540, 1e6), (-545, -520, 1e2), (-560, -535, 1.0)])
def test_subnormal_covariates_with_large_weights_keep_the_bracket(sy, sc, om):
    """Unit: the third cause. Each covariate adds om (v^2), and v^2 underflows by up to eta / 2, which the weight om
    then multiplies; with the paths and covariates both subnormal no relative term covers it. Without the squares'
    term, 501 of 2,160 groups over these four regimes fell outside their bounds."""
    n, Q, T, M = 4, 2, 2, 6
    for seed in range(6):
        rng = np.random.default_rng(seed)
        Y = np.ldexp(rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)), sy)
        kw = dict(cov=np.ldexp(rng.normal(size=(n, M)), sc), cov_scales=rng.uniform(0.5, 2.0, M),
                  cov_weights=om * rng.uniform(0.1, 2.0, M))
        pool = columns.score_columns(np.arange(n), Y, Q, **kw)
        assert _brackets(Y, kw, pool, Q, "ss_res") == []


def test_subnormal_period_weights_keep_the_bracket_at_a_normal_scale():
    """Unit: the weighted objective forms w d d as (w d) d. With some weights subnormal and the paths at a normal
    scale, w d underflows and the second d multiplies its error, so the allowance scales with the largest
    deviation. Weights are only validated to sum past 1e-12, so this is inside the domain."""
    n, Q, T = 5, 2, 8
    Y = 1e6 * _panel(11, n, T)
    w = np.array([1.0, 1e-310, 2.0, 5e-320, 1.0, 1e-315, 0.5, 1.0])
    kw = {"weights": w}
    pool = columns.score_columns(np.arange(n), Y, Q, objective="weighted", weights=w)
    assert _brackets(Y, kw, pool, Q, "weighted") == []


@pytest.mark.parametrize("s", [-500, -520, -540])
def test_the_lagrangian_bound_and_reduced_costs_are_rigorous_at_tiny_scales(s):
    """Unit: for duals on the costs' own scale, the reduced costs returned are at most the exact ones and the
    Lagrangian bound at most the exact Lagrangian bound."""
    n, Q, k = 6, 2, 2
    Y, _, pool = _scored(8, n, Q, s)
    rng = np.random.default_rng(8)
    med = float(np.median(pool.cost)) + ETA
    pi = rng.normal(size=n) * med
    sigma = abs(float(rng.normal())) * med
    L, rc = columns.lagrangian_bound(pool, pi, sigma, k)
    P = [Fr(float(p)) for p in pi]
    exact_rc = [_exact_group(Y, g, Q, "ss_res") - sum(P[i] for i in g) - Fr(float(sigma)) for g in _rows(pool)]
    assert all(Fr(float(lo)) <= ex for lo, ex in zip(rc, exact_rc))
    assert Fr(float(L)) <= sum(P) + Fr(float(sigma)) * k + sum(min(Fr(0), e) for e in exact_rc)


def test_the_cover_search_proves_only_what_the_exact_costs_allow_at_a_subnormal_scale():
    """Unit: the search's lower bound at a subnormal scale is at most the exact optimum, and its design's upper bound
    at least the design's exact cost."""
    n, Q = 6, 2
    Y, _, pool = _scored(9, n, Q, -525)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    res = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**6)
    exact = {g: _exact_group(Y, g, Q, "ss_res") for g in _rows(pool)}
    opt = _exact_optimum(n, exact)
    assert res.status == "proved"
    assert Fr(float(res.lower)) <= opt
    design = sum((exact[_rows(pool)[j]] for j in res.rows), Fr(0))
    assert design <= Fr(float(res.upper))


# --- unit: the cost scale ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("top", [1e-293, 1e-300, 2.0 ** -1022, 1e-310, ETA, 1.0, 1e300, np.finfo(float).max])
def test_the_cost_scale_is_a_finite_power_of_two_that_puts_the_largest_cost_at_most_2_to_the_30(top):
    scale = mip._cost_scale([top, top / 3])
    assert math.isfinite(scale) and scale > 0 and math.frexp(scale)[0] == 0.5
    assert top * scale < 2.0 ** 30


@pytest.mark.parametrize("top", [1e-293, 1.0, 12345.678, 1e300])
def test_the_cost_scale_is_unchanged_where_it_did_not_overflow(top):
    """Unit: wherever the old scale was finite it is unchanged: the largest cost lands in [2^29, 2^30)."""
    assert 2.0 ** 29 <= top * mip._cost_scale([top]) < 2.0 ** 30


def test_the_cost_scale_of_nothing_or_zero_is_one():
    assert mip._cost_scale([]) == mip._cost_scale([0.0, 0.0]) == 1.0


def test_the_mip_solves_an_arm_of_subnormal_costs():
    """Failure, as reported: solve_partition raised a bare OverflowError on a cost of 1e-300."""
    c = [{"members": np.array([0, 1]), "score": 1e-300, "side_a": np.array([0]), "side_b": np.array([1])},
         {"members": np.array([2, 3]), "score": 5e-324, "side_a": np.array([2]), "side_b": np.array([3])}]
    chosen = mip.solve_partition(c, np.arange(4))
    assert sorted(int(u) for g in chosen for u in g["members"]) == [0, 1, 2, 3]


# --- unit: nothing changes where nothing underflows ----------------------------------------------------------------

@pytest.mark.parametrize("objective", ["ss_res", "weighted", "supergeo_total"])
@pytest.mark.parametrize("s", [-200, -60, 0, 60, 200])
def test_bounds_scale_exactly_by_four_to_the_s_where_everything_is_normal(objective, s):
    """Unit: the underflow term costs nothing at normal scales. Scaling the paths by 2^s is exact, every relative
    error term scales with them, and an added eta rounds away, so each bound of the scaled panel is the unscaled one
    times 4^s exactly. A term that changed normal results would break this."""
    _, _, ref = _scored(10, 6, 2, 0, objective)
    _, _, pool = _scored(10, 6, 2, s, objective)
    for a, b in ((pool.cost, ref.cost), (pool.cost_lo, ref.cost_lo), (pool.cost_up, ref.cost_up)):
        assert np.array_equal(a, np.ldexp(b, 2 * s))


# --- property -------------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 6), Q=st.integers(1, 2), s=st.integers(-575, 0),
       objective=st.sampled_from(OBJECTIVES), T=st.integers(2, 12))
def test_property_every_bound_brackets_the_exact_cost_over_the_whole_range(seed, n, Q, s, objective, T):
    """Property: over panels, sizes, objectives, period counts and every scale from benign down to all-underflow,
    each group's bounds hold its exact cost."""
    Y, kw, pool = _scored(seed, n, Q, s, objective, T=T)
    assert _brackets(Y, kw, pool, Q, objective) == []
