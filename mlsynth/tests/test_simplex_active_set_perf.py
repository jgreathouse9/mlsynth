"""Performance contract for the active-set simplex QP (PR #58).

Speed is asserted with **machine-independent** proxies -- pivot counts and
warm-start work reduction -- not wall-clock, which flakes in CI. The headline
win of the active-set method is *warm-starting*: across a sweep of related
problems (the conformal / market-selection pattern) a seeded start collapses the
work to near zero (~0 pivots), which on Kansas measures ~66x faster than the
cvxpy incumbent. A single cold solve is also ~2x faster (rank-revealing-QR inner
solve, no canonicalisation tax). The durable speed record (a Dolan-More
performance profile vs cvxpy) lives in ``benchmarks/`` rather than here.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.active_set import solve_simplex_qp


@pytest.mark.parametrize("m,J", [(20, 5), (50, 20), (89, 49), (10, 30), (8, 8)])
def test_pivot_count_bounded_and_converged(m, J):
    """The active set terminates in O(J) pivots -- a deterministic work bound
    that catches both non-termination and a perf regression."""
    rng = np.random.default_rng(7 + m + J)
    for _ in range(25):
        B = rng.normal(size=(m, J))
        A = rng.normal(size=m)
        _, info = solve_simplex_qp(B, A, return_info=True)
        assert info["converged"]
        assert info["pivots"] <= 2 * J          # observed: <= J


def test_warm_start_collapses_work_on_related_problems():
    """Sweeping a chain of nearby problems (one donor perturbed each step) --
    the conformal-refit / market-selection pattern -- a warm start from the
    previous solution does far less work than a cold start."""
    rng = np.random.default_rng(11)
    B = rng.normal(size=(40, 12))
    A = rng.normal(size=40)
    prev = solve_simplex_qp(B, A)
    cold_total = warm_total = 0
    for _ in range(20):
        B = B.copy()
        B[:, rng.integers(12)] += 0.05 * rng.normal(size=40)
        _, cold = solve_simplex_qp(B, A, return_info=True)
        warm_w, warm = solve_simplex_qp(B, A, warm_start=prev, return_info=True)
        cold_total += cold["pivots"]
        warm_total += warm["pivots"]
        # warm start must not change the optimum, only the work
        assert np.allclose(B @ warm_w, B @ solve_simplex_qp(B, A), atol=1e-7)
        prev = warm_w
    assert warm_total < cold_total              # strictly less work warm


def test_warm_start_from_garbage_is_ignored():
    """An infeasible / wrong-shape warm start is silently discarded (falls back
    to the uniform start) rather than corrupting the solve."""
    rng = np.random.default_rng(3)
    B = rng.normal(size=(15, 6))
    A = rng.normal(size=15)
    cold = solve_simplex_qp(B, A)
    for bad in (np.full(6, -1.0), np.zeros(6), np.ones(3), np.array([np.nan] * 6)):
        w = solve_simplex_qp(B, A, warm_start=bad)
        assert np.allclose(B @ w, B @ cold, atol=1e-7)
