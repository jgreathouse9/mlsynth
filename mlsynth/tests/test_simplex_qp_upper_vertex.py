"""The capped simplex QP at a vertex: every weight at zero or at its cap.

When the caps of the weights at their caps sum to exactly one, no weight lies
strictly between its bounds, and the sum-to-one multiplier nu is not fixed by
any free weight: the KKT conditions only bound it, max g over the capped
weights <= nu <= min g over the weights at zero that can still rise. The loop
used to unpin the largest capped weight, read nu from that one weight's
gradient, release a capped weight against it, take a step of length zero, and
repeat until its pivot cap. The answer was right where the vertex was optimal,
but the solve reported converged=False after 50 J pivots, and on 40 geos it
stopped short of the optimum (KKT error 2e-7) when it was not.

Caps of 1/m on m donors are the structure of the TBR hull (#965), whose
vertices are the designs, so the case is the common one there.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.tests.test_simplex_qp_upper import _brute, _obj
from mlsynth.utils.solvers.active_set import simplex_kkt_error, solve_simplex_qp

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _vertex_case(seed, J=6, T=12, m=None):
    """A target that pulls hard on m donors, under caps of 1/m: the optimum puts
    those m donors at their caps and the rest at zero."""
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, J)) if m is None else m
    B = rng.normal(size=(T, J))
    A = 3.0 * B[:, :m].sum(axis=1) + 0.1 * rng.normal(size=T)
    return B, A, np.full(J, 1.0 / m), m


def _solve(B, A, u, **kw):
    return solve_simplex_qp(B, A, upper=u, return_info=True, **kw)


# ---------------------------------------------------------------- smoke


def test_smoke_a_vertex_optimum_converges():
    B, A, u = np.eye(4), np.array([2.0, 1.0, 0.0, -1.0]), np.full(4, 0.5)
    w, info = _solve(B, A, u)
    assert info["converged"] and info["certified"]
    assert np.allclose(w, [0.5, 0.5, 0.0, 0.0], atol=1e-12)
    assert info["pivots"] <= 8


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", range(30))
def test_a_vertex_optimum_is_reached_in_few_pivots(seed):
    B, A, u, m = _vertex_case(seed)
    w, info = _solve(B, A, u)
    assert info["converged"] and info["certified"]
    assert info["pivots"] <= 4 * B.shape[1]
    assert _obj(B, A, w) == pytest.approx(_brute(B, A, u)[0], rel=1e-10, abs=1e-12)
    assert simplex_kkt_error(B, A, w, upper=u) < 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_a_vertex_that_is_not_optimal_is_left(seed):
    """Started at a vertex that is not the optimum, the solve moves mass off it."""
    B, A, u, m = _vertex_case(seed)
    start = np.zeros(B.shape[1]); start[-m:] = 1.0 / m           # the wrong m donors
    w, info = _solve(B, A, u, warm_start=start, accelerate=False)
    assert info["converged"] and info["certified"]
    assert _obj(B, A, w) == pytest.approx(_brute(B, A, u)[0], rel=1e-10, abs=1e-12)
    assert _obj(B, A, w) < _obj(B, A, start)


def test_the_vertex_is_left_along_the_most_violated_pair():
    """At a vertex the move is mass from the capped weight with the largest gradient
    to the zero weight with the smallest: the steepest edge of the capped simplex."""
    B = np.eye(4)
    A = np.array([0.0, 0.2, 5.0, 1.0])
    u = np.full(4, 0.5)
    start = np.array([0.5, 0.5, 0.0, 0.0])
    w, info = _solve(B, A, u, warm_start=start, accelerate=False)
    assert info["converged"]
    assert np.allclose(w, [0.0, 0.0, 0.5, 0.5], atol=1e-12)


def test_the_uncapped_path_changes_no_bit():
    """Without caps the loop never meets a capped vertex; its pivots are unchanged."""
    rng = np.random.default_rng(3)
    B = rng.normal(size=(10, 6)); A = B[:, :2].sum(1) + 0.1 * rng.normal(size=10)
    w0, i0 = solve_simplex_qp(B, A, return_info=True)
    w1, i1 = solve_simplex_qp(B, A, upper=np.full(6, 1.0), return_info=True)
    assert np.array_equal(w0, w1) and i0["pivots"] == i1["pivots"]


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 100_000), J=st.integers(3, 6), linear=st.booleans())
def test_property_vertex_optima_converge_to_the_exact_minimum(seed, J, linear):
    B, A, u, m = _vertex_case(seed, J=J)
    lin = np.random.default_rng(seed + 1).normal(size=J) if linear else None
    w, info = _solve(B, A, u, linear=lin)
    assert info["converged"] and info["certified"]
    assert info["pivots"] <= 4 * J
    assert _obj(B, A, w, lin) <= _brute(B, A, u, lin)[0] + 1e-10 * max(1.0, abs(_brute(B, A, u, lin)[0]))


@SETTINGS
@given(seed=st.integers(0, 100_000), J=st.integers(3, 6), m=st.integers(1, 3))
def test_property_caps_of_one_over_m_on_any_target(seed, J, m):
    """Caps of 1/m on a target that need not favour m donors: the exact minimum,
    converged, whether or not the optimum is a vertex."""
    m = min(m, J - 1)
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(8, J)); A = rng.normal(size=8) * rng.uniform(0.1, 5.0)
    u = np.full(J, 1.0 / m)
    w, info = _solve(B, A, u)
    assert info["converged"] and info["certified"]
    assert _obj(B, A, w) <= _brute(B, A, u)[0] + 1e-10 * max(1.0, _brute(B, A, u)[0])


# ---------------------------------------------------------------- edges


def test_a_donor_fixed_at_a_zero_cap_is_never_released():
    """A cap of zero fixes the weight: it carries no sign condition and cannot take mass."""
    B, A, _, _ = _vertex_case(4, J=5, m=2)
    u = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
    A = A + 10.0 * B[:, 4]                                       # pull toward the fixed donor
    w, info = _solve(B, A, u)
    assert info["converged"] and info["certified"]
    assert w[4] == 0.0
    assert _obj(B, A, w) == pytest.approx(_brute(B, A, u)[0], rel=1e-10)


def test_caps_that_fill_the_total_exactly_leave_one_point():
    """J caps of 1/J: the only feasible point is the uniform one, a vertex with no
    zero weight, optimal at once."""
    B, A, _, _ = _vertex_case(5, J=4, m=2)
    w, info = _solve(B, A, np.full(4, 0.25))
    assert info["converged"] and np.allclose(w, 0.25, atol=1e-15)
    assert info["pivots"] <= 2


def test_one_cap_of_one_is_a_vertex_on_one_donor():
    """m = 1: the cap of one cannot bind, and the optimum on one donor is a vertex
    of the plain simplex."""
    B, A, u, _ = _vertex_case(6, J=5, m=1)
    w, info = _solve(B, A, u)
    assert info["converged"] and info["certified"]
    assert _obj(B, A, w) == pytest.approx(_brute(B, A, u)[0], rel=1e-10)


# ---------------------------------------------------------------- failure


def test_an_unconverged_solve_is_reported():
    """A pivot limit too small to reach the vertex is reported as not converged, not
    passed off as an answer."""
    B, A, u, m = _vertex_case(7, J=6, m=3)
    w, info = _solve(B, A, u, max_iter=1, accelerate=False)
    assert not info["converged"]
