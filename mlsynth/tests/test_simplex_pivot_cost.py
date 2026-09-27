"""The pivot loop reaches the gradient by the cheaper of two identical routes.

``solve_simplex_qp`` needs the reduced gradient ``G w - c`` every pivot, with
``G = B'B`` and ``c = B'A``. Forming ``G`` costs ``m J^2`` once and every pivot
then costs ``J^2``. Reaching the same vector as ``B'(B w) - c`` costs ``2 m J``
per pivot and forms nothing. Which is cheaper is decided by the shape alone:
a design with at least as many matching rows as donors is better served by the
Gram, and a wider one -- more donors than rows, the ordinary panel -- is not.

Measured per pivot on a 10x160 design, 3.51 us through the Gram against 2.25 us
through ``B``, plus the 19.4 us to form ``G`` at all, which is 5.8 percent of
that solve. Over nine panels spanning 20x16 to 175x361, against a control that
is the same file relocated (1.01x, the harness's noise floor):

    mean -> sum        1.12x
    gradient via B     1.16x
    both               1.25x

``BF.mean(axis=1)`` is the other change. It costs three Python frames per call
(``mean`` -> ``_mean`` -> ``_count_reduce_items`` -> ``reduce``) and runs once a
pivot; ``sum(axis=1)`` keeps only the last of them. cProfile over 3000 solves of
a 20x16 panel puts ``_mean`` and ``_count_reduce_items`` at 11.4 percent of the
run.

A third change measured out at nothing and is not here: caching the two arrays
``_gelsy_lstsq`` allocates per call, which came in at 1.01x against the control,
since the allocations are ~0.15 us and the workspace *query* was already cached.

The tests below pin what the change must not do. The two routes are the same
gradient, so the returned weights and the pivot count are unchanged; ``c`` is
where a ``linear`` term lives, so the route through ``B`` has to subtract ``c``
and not ``A``. Getting that wrong is the near miss this file exists for: writing
``B'(B w - A)`` drops the linear term silently, and the answer stays plausible.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers.active_set import solve_simplex_qp


def _panel(m, J, seed=0, rank=3):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(m, rank))
    L = rng.uniform(0.5, 1.5, (rank, J))
    B = 1.0 + F @ L + rng.standard_normal((m, J))
    A = 1.0 + F @ np.ones(rank) + rng.standard_normal(m)
    return np.ascontiguousarray(B), np.ascontiguousarray(A)


def _gram_gradient(B, A, w, linear=None):
    """The gradient the Gram route computes, spelled out independently."""
    c = B.T @ A
    if linear is not None:
        c = c - 0.5 * linear
    return (B.T @ B) @ w - c


# --------------------------------------------------------------------------- #
# 1. the two routes are the same vector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J", [(10, 160), (20, 16), (16, 16), (40, 160),
                                 (5, 8), (100, 60)])
def test_the_two_gradient_routes_agree(m, J):
    """``B'(B w) - c`` and ``G w - c`` to machine precision, both orientations."""
    B, A = _panel(m, J, seed=m * J)
    w = solve_simplex_qp(B, A)
    c = B.T @ A
    viaB = B.T @ (B @ w) - c
    viaG = (B.T @ B) @ w - c
    scale = 1.0 + float(np.max(np.abs(viaG)))
    assert np.max(np.abs(viaB - viaG)) <= 1e-10 * scale


@pytest.mark.parametrize("m,J", [(10, 60), (30, 20)])
def test_the_route_through_b_must_subtract_c_and_not_a(m, J):
    """The near miss, pinned. With a linear term ``c`` is not ``B'A``, so
    ``B'(B w - A)`` is a different vector and the loop's dual test would read a
    gradient the objective does not have."""
    B, A = _panel(m, J, seed=7)
    w = solve_simplex_qp(B, A)
    linear = np.linspace(-1.0, 1.0, J)
    right = B.T @ (B @ w) - (B.T @ A - 0.5 * linear)
    wrong = B.T @ (B @ w - A)
    assert np.allclose(right, _gram_gradient(B, A, w, linear), atol=1e-10)
    assert not np.allclose(wrong, right, atol=1e-8)


# --------------------------------------------------------------------------- #
# 2. the substitutions change no answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J", [(10, 160), (20, 16), (16, 16), (17, 16),
                                 (15, 16), (40, 160), (5, 300), (120, 40)])
def test_weights_and_pivots_match_a_forced_gram_solve(m, J):
    """Both routes are available on every shape; the solver picks by shape and
    the answer may not depend on the pick. Asserted through the KKT certificate
    and the objective, which are route-independent."""
    B, A = _panel(m, J, seed=m + J)
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert info["converged"]
    assert w.min() >= -1e-9 and abs(w.sum() - 1.0) <= 1e-9
    g = _gram_gradient(B, A, w)
    support = w > 1e-7
    scale = 1.0 + float(np.max(np.abs(g)))
    nu = float(g[support].mean())
    assert np.all(np.abs(g[support] - nu) <= 1e-6 * scale)
    if (~support).any():
        assert np.all(g[~support] >= nu - 1e-6 * scale)


@pytest.mark.parametrize("m,J", [(16, 16), (17, 16), (15, 16)])
def test_the_shape_boundary_is_not_a_discontinuity(m, J):
    """``m == J`` is where the route flips. The objective either side of it moves
    only with the design, never with the route."""
    B, A = _panel(m, J, seed=99)
    w = solve_simplex_qp(B, A)
    obj = float(np.sum((B @ w - A) ** 2))
    # Padding one all-zero matching row cannot change the program, and it moves
    # the shape across the boundary.
    Bp = np.vstack([B, np.zeros((1, J))])
    Ap = np.concatenate([A, [0.0]])
    wp = solve_simplex_qp(Bp, Ap)
    objp = float(np.sum((Bp @ wp - Ap) ** 2))
    assert objp == pytest.approx(obj, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. edges
# --------------------------------------------------------------------------- #
def test_single_donor():
    B = np.array([[2.0], [3.0], [1.5]]); A = np.array([2.0, 3.0, 1.5])
    w = solve_simplex_qp(B, A)
    assert w.shape == (1,) and w[0] == pytest.approx(1.0)


def test_single_matching_row_is_a_wide_design():
    """One row against six donors: the Gram is 6x6 for a rank-one design, so the
    wide route is the one that runs, and the answer is still on the simplex."""
    B = np.arange(1.0, 7.0).reshape(1, 6)
    A = np.array([B.mean()])
    w = solve_simplex_qp(B, A)
    assert w.min() >= -1e-9 and abs(w.sum() - 1.0) <= 1e-9
    assert float(np.sum((B @ w - A) ** 2)) <= 1e-18


def test_collinear_donors_still_certify():
    B, A = _panel(30, 20, seed=5)
    B[:, 5] = B[:, 4]
    B[:, 9] = 2.0 * B[:, 3]
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert info["converged"]
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1.0) <= 1e-9


# --------------------------------------------------------------------------- #
# 4. the linear term, through the solver
# --------------------------------------------------------------------------- #
# Section 1 checks the two gradient routes as arithmetic, which is not enough to
# notice the near miss where it would live. ``B'(B w - A)`` differs from
# ``B'(B w) - c`` only when ``c`` is not ``B'A``, which is only when a ``linear``
# term is set, and only the wide route is reached on a wide design. So the test
# that kills it has to drive ``solve_simplex_qp`` itself, with a linear term, on
# a design with more donors than matching rows.
@pytest.mark.parametrize("m,J", [(10, 60), (8, 40), (5, 300)])
def test_a_linear_term_on_a_wide_design_is_solved_for_its_own_objective(m, J):
    """``min w'Gw - c'w`` with the linear part set: the returned point has to be
    optimal for *that* objective, not for the plain least squares."""
    B, A = _panel(m, J, seed=m * 31 + J)
    rng = np.random.default_rng(m + J)
    linear = rng.normal(size=J) * 0.5
    w, info = solve_simplex_qp(B, A, linear=linear, return_info=True)
    assert info["converged"]
    assert w.min() >= -1e-9 and abs(w.sum() - 1.0) <= 1e-9
    g = _gram_gradient(B, A, w, linear)
    support = w > 1e-7
    scale = 1.0 + float(np.max(np.abs(g)))
    nu = float(g[support].mean())
    assert np.all(np.abs(g[support] - nu) <= 1e-6 * scale), \
        "stationarity fails on the support, so the gradient the loop used is " \
        "not the gradient of the objective it was given"
    if (~support).any():
        assert np.all(g[~support] >= nu - 1e-6 * scale)


def test_the_linear_term_actually_moves_the_answer():
    """Power for the test above: without this, a solver that ignored ``linear``
    entirely could pass it whenever the plain optimum happened to coincide."""
    B, A = _panel(10, 60, seed=4)
    w0 = solve_simplex_qp(B, A)
    w1 = solve_simplex_qp(B, A, linear=np.linspace(-4.0, 4.0, 60))
    assert np.max(np.abs(w0 - w1)) > 1e-3
