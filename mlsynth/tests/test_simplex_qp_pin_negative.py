"""A free weight that solves to a small negative is pinned and the free set re-solved.

``solve_simplex_qp`` accepted a full step to the free-set optimum when its smallest
weight was at least ``-tol`` (1e-9), then clamped that weight to zero without pinning
it or re-solving. The other weights kept the values the clamped donor's presence had
given them, so the returned point was off the optimum by about the clamped amount and
failed the solver's own certificate. Found through SparseSC's inner QP (#934): a 10-by-8
design where donor 3 solved to -7.67e-10 on the free set {0, 1, 2, 3, 5, 6}; the
clamped point scored 2.92e-10 on ``simplex_kkt_error`` against ``CERTIFY_TOL`` = 1e-10,
while the free set {0, 1, 2, 5, 6} solved to 2.7e-17 (agents_solvers.md, rules 2 and 8).

The construction below reproduces it: a target equal to ``B w`` for a ``w`` on the
plane ``1'w = 1`` with one entry at ``-delta``, ``delta`` inside ``tol``, so the first
free-set solve returns that ``w`` exactly.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mlsynth.utils.solvers.active_set import CERTIFY_TOL, simplex_kkt_error, solve_simplex_qp


def _near_vertex(seed, m=12, J=6, delta=5e-10, k=0):
    """A tall design and a target whose plane optimum has weight ``-delta`` on donor ``k``."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(m, J))
    w = rng.dirichlet(np.ones(J))
    w[k] = 0.0
    w = w / w.sum() * (1.0 + delta)
    w[k] = -delta
    return B, B @ w


# --------------------------------------------------------------------------- smoke
def test_a_small_negative_free_weight_is_pinned_and_the_answer_certified():
    B, A = _near_vertex(0)
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert info["converged"] and info["kkt_error"] <= CERTIFY_TOL
    assert w[0] == 0.0 and w.min() >= 0.0 and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- unit
def test_the_answer_is_the_optimum_on_the_remaining_donors():
    B, A = _near_vertex(1)
    w = solve_simplex_qp(B, A)
    rest = solve_simplex_qp(B[:, 1:], A)
    np.testing.assert_allclose(w[1:], rest, atol=1e-14)


def test_the_capped_program_pins_too():
    B, A = _near_vertex(2)
    w, info = solve_simplex_qp(B, A, upper=np.full(B.shape[1], 0.9), return_info=True)
    assert info["converged"]
    assert w[0] == 0.0


def test_a_weight_just_over_its_cap_is_pinned_at_the_cap():
    """The cap side of the same clamp: donor 0 solves to its cap plus delta."""
    rng = np.random.default_rng(6)
    B = rng.normal(size=(12, 6))
    cap, delta = 0.4, 6e-10
    w = rng.dirichlet(np.ones(5)) * (1.0 - cap - delta)
    w = np.r_[cap + delta, w]
    sol, info = solve_simplex_qp(B, B @ w, upper=np.full(6, cap), return_info=True)
    assert info["converged"] and info["kkt_error"] <= CERTIFY_TOL
    assert sol[0] == cap


# --------------------------------------------------------------------------- edge
@pytest.mark.parametrize("delta", [1e-15, 1e-12, 9.9e-10])
def test_across_the_tolerance_band(delta):
    B, A = _near_vertex(3, delta=delta)
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert info["converged"] and w[0] == 0.0


def test_two_negatives_at_once():
    rng = np.random.default_rng(4)
    B = rng.normal(size=(12, 6))
    w = np.array([-3e-10, -6e-10, 0.3, 0.3, 0.2, 0.2])
    w[2:] *= (1.0 + 9e-10) / w[2:].sum()
    sol, info = solve_simplex_qp(B, B @ w, return_info=True)
    assert info["converged"] and sol[0] == 0.0 and sol[1] == 0.0


def test_a_warm_start_at_the_clamped_point_is_still_certified():
    B, A = _near_vertex(5)
    start = np.linalg.lstsq(np.vstack([B, np.ones((1, B.shape[1]))]), np.r_[A, 1.0], rcond=None)[0]
    start = np.maximum(start, 0.0); start /= start.sum()
    w, info = solve_simplex_qp(B, A, warm_start=start, return_info=True)
    assert info["converged"]


# --------------------------------------------------------------------------- property
@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), J=st.integers(3, 10), k=st.integers(0, 9),
       delta=st.floats(1e-16, 9.9e-10), scale=st.integers(-6, 6))
@example(seed=0, J=6, k=0, delta=5e-10, scale=0)
@example(seed=9999, J=3, k=1, delta=1e-16, scale=0)
def test_property_a_clamped_weight_never_reaches_the_answer(seed, J, k, delta, scale):
    """Certified, and donor ``k`` at zero to the rounding of a weight (rule 4).

    Below rounding the free-set solve can return ``k`` at +2.2e-16 (seed 9999,
    ``delta`` 1e-16), which is a zero to the precision of the weights.
    """
    B, A = _near_vertex(seed, m=J + 6, J=J, delta=delta, k=k % J)
    s = 10.0 ** scale
    w, info = solve_simplex_qp(s * B, s * A, return_info=True)
    assert info["converged"]
    assert simplex_kkt_error(s * B, s * A, w) <= CERTIFY_TOL
    assert w[k % J] <= 64 * np.finfo(float).eps
