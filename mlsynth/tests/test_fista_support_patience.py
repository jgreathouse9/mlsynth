"""The FISTA warm start stops once the support stops moving.

``fista_warm_start`` exists to name the *support* of the simplex optimum, not to
converge to it -- the exact active set does the converging, and it only needs to
be told which donors to start from. Its ``tol=1e-7`` stopping rule tests the
wrong thing for that job: on the two SDID programs of a 101x120 panel the
support is final by iteration ~150 while the iterate is still moving at
iteration 400, so more than half the work buys nothing.

The stop tested here is: once the support -- ``w > 1e-9``, the same threshold
``solve_simplex_qp`` reads when it seeds its active set -- has been unchanged
for ``support_patience`` consecutive iterations, return.

It is speed only. A warm start cannot change what ``solve_simplex_qp`` returns:
the active set certifies KKT optimality on the design whatever point it starts
from, and rejects a start it cannot use. So the tests here assert two separate
things -- that the stop fires, and that the certified answer is untouched.
"""

import numpy as np
import pytest

from mlsynth.utils.bilevel import accelerate
from mlsynth.utils.bilevel.accelerate import (
    SUPPORT_PATIENCE,
    SUPPORT_TOL,
    fista_warm_start,
)
from mlsynth.utils.bilevel.active_set import solve_simplex_qp


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _factor_panel(J, T0, seed=0):
    """A factor-structure SC design: two AR-free factors, ``J`` donors."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T0, 2))
    Mu = np.zeros((J + 1, 2))
    half = J // 2
    Mu[: 1 + half, 0] = 1.0
    Mu[1 + half:, 1] = 1.0
    Y = F @ Mu.T + rng.standard_normal((T0, J + 1))
    return Y[:, 0], Y[:, 1:]                       # A (T0,), B (T0, J)


def _count_iterations(B, A, **kwargs):
    """Run ``fista_warm_start`` and report how many iterations it took.

    Counted by wrapping the projection, which the loop calls exactly once per
    iteration, so the count needs no new return value on the public function.
    """
    calls = {"n": 0}
    real = accelerate.simplex_project

    def counting(v):
        calls["n"] += 1
        return real(v)

    accelerate.simplex_project = counting
    try:
        w = fista_warm_start(B, A, **kwargs)
    finally:
        accelerate.simplex_project = real
    return w, calls["n"]


def _assert_feasible(w, J):
    w = np.asarray(w, float).ravel()
    assert w.shape == (J,)
    assert np.all(np.isfinite(w))
    assert w.min() >= -1e-12
    assert abs(w.sum() - 1.0) <= 1e-9


# --------------------------------------------------------------------------- #
# 1. smoke
# --------------------------------------------------------------------------- #
class TestSmoke:
    def test_returns_a_simplex_point(self):
        A, B = _factor_panel(100, 200)
        _assert_feasible(fista_warm_start(B, A), 100)

    def test_default_patience_is_exposed_and_positive(self):
        assert isinstance(SUPPORT_PATIENCE, int)
        assert SUPPORT_PATIENCE > 0
        assert 0.0 < SUPPORT_TOL < 1e-6


# --------------------------------------------------------------------------- #
# 2. the stop fires, and it is what makes the warm start cheap
# --------------------------------------------------------------------------- #
class TestTheStopFires:
    def test_stops_well_short_of_max_iter(self):
        """On a design whose support settles, the loop must not run to 400."""
        A, B = _factor_panel(120, 240)
        _, n_iter = _count_iterations(B, A)
        assert n_iter < 400, "support-patience stop never fired"

    def test_disabling_patience_restores_the_full_run(self):
        """``support_patience=None`` is the pre-change behaviour: only ``tol``
        and ``max_iter`` stop the loop."""
        A, B = _factor_panel(120, 240)
        _, n_patient = _count_iterations(B, A)
        _, n_full = _count_iterations(B, A, support_patience=None)
        assert n_full == 400
        assert n_patient < n_full

    def test_patience_above_max_iter_never_fires(self):
        A, B = _factor_panel(120, 240)
        _, n_iter = _count_iterations(B, A, max_iter=60, support_patience=500)
        assert n_iter == 60

    def test_larger_patience_never_stops_earlier(self):
        """Monotone in the knob: waiting longer for the support to hold cannot
        make the loop exit sooner."""
        A, B = _factor_panel(120, 240)
        counts = [_count_iterations(B, A, support_patience=p)[1]
                  for p in (10, 20, 40, 80)]
        assert counts == sorted(counts)


# --------------------------------------------------------------------------- #
# 3. speed only -- the certified answer does not move
# --------------------------------------------------------------------------- #
class TestAnswerUnchanged:
    @pytest.mark.parametrize("J,T0", [(100, 200), (120, 240), (90, 90), (150, 120)])
    def test_seeded_solve_equals_cold_solve(self, J, T0):
        A, B = _factor_panel(J, T0)
        cold = solve_simplex_qp(B, A)
        seeded = solve_simplex_qp(B, A, warm_start=fista_warm_start(B, A))
        np.testing.assert_allclose(seeded, cold, rtol=0, atol=1e-12)

    def test_seeded_solve_equals_solve_from_the_full_run(self):
        """Stopping early changes the seed; it must not change the answer the
        seed leads to."""
        A, B = _factor_panel(150, 300)
        early = fista_warm_start(B, A)
        full = fista_warm_start(B, A, support_patience=None)
        assert not np.array_equal(early, full), "the two seeds should differ"
        w_early = solve_simplex_qp(B, A, warm_start=early)
        w_full = solve_simplex_qp(B, A, warm_start=full)
        np.testing.assert_allclose(w_early, w_full, rtol=0, atol=1e-12)

    def test_still_deterministic(self):
        A, B = _factor_panel(120, 240)
        assert np.array_equal(fista_warm_start(B, A), fista_warm_start(B, A))

    def test_seed_is_feasible_so_the_active_set_can_use_it(self):
        """An infeasible seed is silently discarded, which would make the stop
        a pure loss instead of a saving."""
        A, B = _factor_panel(120, 240)
        seed = fista_warm_start(B, A)
        _assert_feasible(seed, 120)
        _, info = solve_simplex_qp(B, A, warm_start=seed, return_info=True)
        _, cold_info = solve_simplex_qp(B, A, return_info=True, accelerate=False)
        assert info["converged"]
        assert info["pivots"] < cold_info["pivots"]


# --------------------------------------------------------------------------- #
# 4. edges
# --------------------------------------------------------------------------- #
class TestEdges:
    def test_single_donor_returns_the_vertex_without_iterating(self):
        rng = np.random.default_rng(0)
        B = rng.standard_normal((20, 1))
        A = rng.standard_normal(20)
        w, n_iter = _count_iterations(B, A)
        assert n_iter == 0
        np.testing.assert_array_equal(w, np.ones(1))

    def test_degenerate_design_returns_uniform(self):
        """All-zero donors: the Lipschitz bound collapses and the warm start
        falls back to the uniform point before any iteration."""
        B = np.zeros((10, 5))
        A = np.arange(10.0)
        w, n_iter = _count_iterations(B, A)
        assert n_iter == 0
        np.testing.assert_allclose(w, np.full(5, 0.2))

    def test_support_that_never_settles_still_terminates(self):
        """Duplicate donors leave the support free to swap between equivalent
        optima; the loop must still stop at ``max_iter``."""
        rng = np.random.default_rng(5)
        core = rng.standard_normal((30, 6))
        B = np.hstack([core, core, core])           # every donor has two twins
        A = rng.standard_normal(30)
        w, n_iter = _count_iterations(B, A, max_iter=120)
        assert n_iter <= 120
        _assert_feasible(w, 18)

    def test_patience_of_one_still_returns_a_usable_seed(self):
        A, B = _factor_panel(100, 200)
        w = fista_warm_start(B, A, support_patience=1)
        _assert_feasible(w, 100)
        np.testing.assert_allclose(
            solve_simplex_qp(B, A, warm_start=w), solve_simplex_qp(B, A),
            rtol=0, atol=1e-12)

    def test_target_reachable_exactly_by_one_donor(self):
        """More rows than donors, so ``B`` has full column rank and the vertex
        is the *unique* exact reproduction -- the seed has to find it."""
        rng = np.random.default_rng(9)
        B = rng.standard_normal((120, 100))
        A = B[:, 7].copy()
        w = fista_warm_start(B, A)
        _assert_feasible(w, 100)
        assert np.argmax(w) == 7


# --------------------------------------------------------------------------- #
# 5. failure -- a bad knob is reported, not absorbed
# --------------------------------------------------------------------------- #
class TestFailures:
    @pytest.mark.parametrize("bad", [0, -1, -25])
    def test_non_positive_patience_raises(self, bad):
        A, B = _factor_panel(20, 40)
        with pytest.raises(ValueError, match="support_patience"):
            fista_warm_start(B, A, support_patience=bad)

    def test_non_integer_patience_raises(self):
        A, B = _factor_panel(20, 40)
        with pytest.raises(ValueError, match="support_patience"):
            fista_warm_start(B, A, support_patience=2.5)
