"""The PCR-SC convex weight solve, against the program it claims to solve.

``clustersc_helpers.pcr.convex.solve_simplex`` fits Abadie-Diamond-Hainmueller
weights to the HSVT-denoised donor block:

.. math::

   \\min_{w \\geq 0,\\, \\mathbf{1}^\\top w = 1} \\| x_0^- - \\widetilde{M}^- w \\|_2

The objective is a norm, not a squared norm, and ``solve_simplex_qp`` minimises
the square. Squaring is monotone on the non-negative reals, so the two have the
same argmin; these tests hold the implementation to that claim by solving the
non-squared program with cvxpy in the test and comparing.

The oracle is built here rather than imported so it keeps saying what the paper
says even if the module changes solver.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.clustersc_helpers.pcr.convex import solve_simplex
from mlsynth.utils.solvers.minnorm import simplex_point_is_optimal

TOL = 1e-6


def _cvxpy_oracle(M: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """The non-squared program, solved independently of the module."""
    w = cp.Variable(M.shape[1])
    prob = cp.Problem(
        cp.Minimize(cp.norm(x0 - M @ w, 2)), [w >= 0, cp.sum(w) == 1]
    )
    prob.solve(solver=cp.CLARABEL)
    assert w.value is not None, prob.status
    return np.asarray(w.value, dtype=float)


def _sse(M: np.ndarray, x0: np.ndarray, w: np.ndarray) -> float:
    r = x0 - M @ w
    return float(r @ r)


def _panel(rng, T0: int, J: int):
    M = rng.normal(size=(T0, J))
    x0 = rng.normal(size=T0)
    return M, x0


# ------------------------------------------------------------------ smoke
def test_returns_one_finite_weight_per_donor():
    rng = np.random.default_rng(0)
    M, x0 = _panel(rng, 8, 4)
    w = solve_simplex(M, x0)
    assert w.shape == (4,)
    assert w.dtype == np.float64
    assert np.all(np.isfinite(w))


# -------------------------------------------------------------- invariants
@pytest.mark.parametrize("T0,J", [(8, 3), (12, 5), (6, 6), (20, 2)])
def test_the_answer_is_on_the_simplex(T0, J):
    rng = np.random.default_rng(T0 * 100 + J)
    M, x0 = _panel(rng, T0, J)
    w = solve_simplex(M, x0)
    assert w.min() >= -TOL
    assert abs(w.sum() - 1.0) < TOL


@pytest.mark.parametrize("seed", range(6))
def test_it_solves_the_non_squared_program(seed):
    """Squaring the objective does not move the argmin."""
    rng = np.random.default_rng(seed)
    M, x0 = _panel(rng, 10, 4)
    got = solve_simplex(M, x0)
    want = _cvxpy_oracle(M, x0)
    # Compare objective values, which are well defined even where the
    # minimiser is a face, and the weights themselves where it is a point.
    assert _sse(M, x0, got) == pytest.approx(_sse(M, x0, want), abs=1e-6)
    # The weights are compared at the oracle's own accuracy: CLARABEL on the
    # cone form stops a few 1e-05 short, so a tighter bound here would be
    # asserting against its tolerance and not against the program.
    assert np.allclose(got, want, atol=2e-4)


def test_a_target_inside_the_hull_is_fitted_exactly():
    rng = np.random.default_rng(7)
    M = rng.normal(size=(9, 4))
    truth = np.array([0.5, 0.2, 0.3, 0.0])
    w = solve_simplex(M, M @ truth)
    assert _sse(M, M @ truth, w) < 1e-12
    assert np.allclose(w, truth, atol=1e-5)


@pytest.mark.parametrize("c", [0.1, 2.0, 3.5, 100.0])
def test_scaling_the_panel_leaves_the_weights_alone(c):
    """Scaling both sides multiplies the objective by ``c`` and moves nothing.

    The tolerance is what separates an exact solve from a tolerance-limited
    one. On seed 11 the active set reproduces its own answer to 1.7e-16 under
    a 3.5x scale, and its worst drift over 30 seeds crossed with the four
    scales here is 7.8e-16. CLARABEL on the second-order-cone form of the same
    program drifts 4.2e-05 on seed 11, at an optimum where every weight is
    strictly positive and the minimiser is therefore unique -- the objective
    values agree to 4e-09, so that is the conic solver stopping early, not an
    ambiguity in the program.
    """
    rng = np.random.default_rng(11)
    M, x0 = _panel(rng, 10, 4)
    assert np.allclose(solve_simplex(M, x0), solve_simplex(c * M, c * x0), atol=1e-12)


def test_relabelling_the_donors_permutes_the_weights():
    rng = np.random.default_rng(13)
    M, x0 = _panel(rng, 12, 5)
    perm = rng.permutation(5)
    base = solve_simplex(M, x0)
    moved = solve_simplex(M[:, perm], x0)
    assert np.allclose(moved, base[perm], atol=1e-5)


# ------------------------------------------------------------- edge cases
def test_a_single_donor_takes_all_the_weight():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(6, 1))
    w = solve_simplex(M, rng.normal(size=6))
    assert np.allclose(w, [1.0], atol=TOL)


def test_one_pre_period_still_solves():
    w = solve_simplex(np.array([[1.0, 2.0, 4.0]]), np.array([2.0]))
    assert abs(w.sum() - 1.0) < TOL
    assert w.min() >= -TOL


def test_duplicate_donors_reach_the_same_fit():
    """A repeated column makes the argmin a face; the fit is still pinned.

    Both solvers land on the same vertex here. The fits are compared at 1e-06
    and not tighter because the oracle's answer is slightly outside the
    feasible set: on this design CLARABEL returns weights of order -1e-10 on
    the two tied columns, which buys it an objective 1.5e-08 below the exact
    one. ``simplex_point_is_optimal`` certifies the active set's point and
    rejects the oracle's for exactly that reason, so the feasible answer is the
    one being asserted.
    """
    rng = np.random.default_rng(5)
    col = rng.normal(size=(8, 1))
    M = np.hstack([col, col, rng.normal(size=(8, 2))])
    x0 = rng.normal(size=8)
    got, want = solve_simplex(M, x0), _cvxpy_oracle(M, x0)
    assert _sse(M, x0, got) == pytest.approx(_sse(M, x0, want), abs=1e-6)
    assert got.min() >= 0.0
    assert simplex_point_is_optimal(M, x0, got)


def test_collinear_donors_do_not_break_the_solve():
    rng = np.random.default_rng(17)
    a = rng.normal(size=(10, 1))
    M = np.hstack([a, 2.0 * a, -0.5 * a])
    w = solve_simplex(M, rng.normal(size=10))
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < TOL


# ---------------------------------------------------------------- failures
def test_a_one_dimensional_donor_block_is_refused():
    with pytest.raises(MlsynthEstimationError, match="2D"):
        solve_simplex(np.ones(5), np.ones(5))


def test_a_length_mismatch_is_refused_and_names_both_sides():
    with pytest.raises(MlsynthEstimationError) as excinfo:
        solve_simplex(np.ones((5, 2)), np.ones(4))
    assert "5" in str(excinfo.value) and "4" in str(excinfo.value)


def test_an_empty_donor_pool_raises_a_translated_error():
    """No donors cannot satisfy sum(w) == 1.

    The guard is the estimator's, so the caller sees ``MlsynthEstimationError``
    and not whichever exception the underlying solver happens to raise.
    """
    with pytest.raises(MlsynthEstimationError):
        solve_simplex(np.ones((5, 0)), np.ones(5))
