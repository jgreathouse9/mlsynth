"""Tests for SYNDES two-way's branch and bound above the candidate limit (#803).

Contract. Above ``candidate_limit`` the exact two-way backend searches treated
sets by a branch and bound and returns the optimum the enumeration would, with
its design certified. A node fixes units into the treated set (``I``) and out
of it (``O``) and leaves ``F`` free; it is priced set by set or bounded and
split. A node is discarded only on a bound computed in exact arithmetic from the
closed form's own data, so the answer does not depend on the costs that decide
where the search spends time, nor on any bound being tight.

The levels:

- the node's parts reproduce every completion (unit, property);
- the certificate bounds every completion for any multipliers, and is tight at
  the trust-region's and the lift's (unit, property);
- the trust-region bound and the lift bound every completion, and the lift is
  never looser than the trust region (unit, property);
- the search returns the enumeration's optimum, under any cost model, with any
  restriction, from a poor incumbent, when a certificate or a lift fails, and
  at any scale and level of the outcome (smoke, property, edge);
- a work limit returns the best design with a valid bound and a warning, every
  kind of work counts toward it, and an invalid limit is refused (failure).
"""
from __future__ import annotations

import warnings
from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.branch import (
    Cuts,
    ExactR,
    LiftCost,
    Multipliers,
    PruneRates,
    branch_and_bound,
    lift,
    node_parts,
    separate,
    trust_region,
)
from mlsynth.utils.syndes_helpers.certificate import syndes_certificate
from mlsynth.utils.syndes_helpers.exact import (
    SubsetFilter,
    solve_synthetic_design_exact,
    solve_synthetic_design_exact_pool,
)
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem
from mlsynth.utils.syndes_helpers.optimization import SYNDESSolveWarning
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions

SETTINGS = dict(max_examples=25, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])

#: Every node is bounded and every bounded node that survives the trust region
#: is lifted: the costs that make the search do the most bounding.
EAGER = dict(price_seconds=1.0, bound_seconds=lambda n: 0.0,
             lift_seconds=lambda n: 0.0)


def _panel(N=10, T=14, rank=3, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _noise(N=10, T=14, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1)) + rng.standard_normal((T, N))


def _engine(Y, lam=None):
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    return problem, BoundEngine.from_problem(problem)


def _completions(N, K, I, O, F):
    r = K - len(I)
    for extra in combinations(F, r):
        sigma = -np.ones(N)
        sigma[list(I) + list(extra)] = 1.0
        yield sigma


def _float_max(R, N, K, I, O, F):
    return max(float(s @ R @ s) for s in _completions(N, K, I, O, F))


def _exact_max(xr: ExactR, N, K, I, O, F) -> Fraction:
    Nint, e = xr.exact.N, xr.exact.e
    best = None
    for s in _completions(N, K, I, O, F):
        si = [int(v) for v in s]
        total = sum(si[i] * si[j] * int(Nint[i, j]) for i in range(N) for j in range(N))
        value = Fraction(total, 1 << e)
        best = value if best is None or value > best else best
    return best


def _node(N, K, seed):
    """A random node with at least one completion: ``I``, ``O``, ``F``."""
    rng = np.random.default_rng(seed)
    units = rng.permutation(N)
    n_i = int(rng.integers(0, K + 1))
    free_min = K - n_i
    n_o = int(rng.integers(0, N - n_i - max(free_min, 1) + 1))
    I = sorted(int(u) for u in units[:n_i])
    O = sorted(int(u) for u in units[n_i:n_i + n_o])
    F = sorted(int(u) for u in units[n_i + n_o:])
    return I, O, F


def _enumerated(Y, K, **kwargs):
    return solve_synthetic_design_exact(Y=Y, K=K, candidate_limit=10 ** 9, **kwargs)


def _branched(Y, K, **kwargs):
    return solve_synthetic_design_exact(Y=Y, K=K, candidate_limit=1, **kwargs)


panels = st.builds(
    lambda N, seed, kind: (_noise if kind else _panel)(N=N, seed=seed),
    st.integers(6, 9), st.integers(0, 10 ** 6), st.booleans())


# ----------------------------------------------------------------------
# the node and its certificate
# ----------------------------------------------------------------------


class TestNodeParts:
    @given(panels, st.data())
    @settings(**SETTINGS)
    def test_parts_reproduce_every_completion(self, Y, data):
        """``c + 2 b'x + x'A x = sigma'R sigma`` for every completion."""
        _, engine = _engine(Y)
        N = engine.N
        K = data.draw(st.integers(1, N - 1))
        I, O, F = _node(N, K, data.draw(st.integers(0, 10 ** 6)))
        c, b, A = node_parts(engine.R, I, O, F)
        for sigma in _completions(N, K, I, O, F):
            x = sigma[F]
            assert c + 2 * b @ x + x @ A @ x == pytest.approx(
                float(sigma @ engine.R @ sigma), rel=1e-9, abs=1e-12)

    def test_a_decided_node_has_no_free_part(self):
        _, engine = _engine(_panel())
        c, b, A = node_parts(engine.R, [0, 1], list(range(2, 10)), [])
        sigma = -np.ones(10); sigma[[0, 1]] = 1.0
        assert b.shape == (0,) and A.shape == (0, 0)
        assert c == pytest.approx(float(sigma @ engine.R @ sigma))


def _random_multipliers(n, rng, scale):
    size = n + 1
    trip = [(i, j, k) for i in range(size) for j in range(i + 1, size)
            for k in range(j + 1, size)]
    m = min(len(trip), int(rng.integers(0, 6)))
    pick = rng.choice(len(trip), size=m, replace=False) if m else []
    tri = np.array([trip[p] for p in pick], dtype=np.int64).reshape(-1, 3)
    cuts = Cuts(tri[:, 0], tri[:, 1], tri[:, 2],
                rng.integers(1, 5, size=m).astype(np.int64))
    return Multipliers(diag=scale * rng.standard_normal(size),
                       rows=scale * rng.standard_normal(size), cuts=cuts,
                       weights=scale * np.abs(rng.standard_normal(m)))


class TestCertificate:
    @given(panels, st.data())
    @settings(**SETTINGS)
    def test_any_multipliers_bound_every_completion(self, Y, data):
        """Weak duality: whatever the multipliers, the certificate is an upper
        bound on ``sigma'R sigma`` over the subtree, in exact arithmetic."""
        _, engine = _engine(Y)
        xr = ExactR(engine)
        N = engine.N
        K = data.draw(st.integers(1, N - 1))
        I, O, F = _node(N, K, data.draw(st.integers(0, 10 ** 6)))
        rng = np.random.default_rng(data.draw(st.integers(0, 10 ** 6)))
        scale = float(np.abs(engine.R).max()) * 10.0 ** data.draw(st.integers(-3, 3))
        m = _random_multipliers(len(F), rng, scale)
        U = xr.certify(I, O, F, K, m)
        assert U is not None
        assert U >= _exact_max(xr, N, K, I, O, F)

    def test_negative_cut_weights_are_not_used_as_given(self):
        """A cut's multiplier must be nonnegative. At a node whose bound is tight
        at a design, with the plane's normal pushed out of the spectrum, a
        negative weight on a cut the design satisfies strictly would lower the
        bound by four times the weight, under the maximum; it is read as zero."""
        _, engine = _engine(_panel())
        xr = ExactR(engine)
        I, F = [2], [6, 8]
        O = [u for u in range(10) if u not in (2, 6, 8)]
        parts = node_parts(engine.R, I, O, F)
        U, x, m = trust_region(*parts, 0)
        assert U == pytest.approx(_float_max(engine.R, 10, 2, I, O, F), rel=1e-12)
        # the cut whose value at the maximising design is 4, not 0
        t = 2 if x[0] > 0 else 3
        scale = float(np.abs(engine.R).max())
        rows = m.rows + 10.0 * scale * np.array([0.0, 1.0, 1.0])
        bad = Multipliers(m.diag, rows, Cuts(np.array([0]), np.array([1]),
                                             np.array([2]), np.array([t])),
                          np.array([-1e-3 * scale]))
        assert xr.certify(I, O, F, 2, bad) >= _exact_max(xr, 10, 2, I, O, F)

    def test_the_trust_region_multipliers_certify_its_bound(self):
        _, engine = _engine(_panel(N=12, seed=4))
        xr = ExactR(engine)
        I, O, F = [3], [0, 7], [u for u in range(12) if u not in (3, 0, 7)]
        K = 4
        U, _, m = trust_region(*node_parts(engine.R, I, O, F), 2 * (K - 1) - len(F))
        cert = float(xr.certify(I, O, F, K, m))
        assert cert >= float(_exact_max(xr, 12, K, I, O, F))
        assert cert == pytest.approx(U, rel=1e-9)

    def test_the_lift_multipliers_certify_its_bound(self):
        _, engine = _engine(_noise(N=12, seed=2))
        xr = ExactR(engine)
        I, O, F = [], [5], [u for u in range(12) if u != 5]
        K = 4
        U, m, _ = lift(*node_parts(engine.R, I, O, F), 2 * K - len(F))
        cert = float(xr.certify(I, O, F, K, m))
        assert cert >= float(_exact_max(xr, 12, K, I, O, F))
        assert cert == pytest.approx(U, rel=1e-8)

    def test_a_decided_node_certifies_its_one_set(self):
        _, engine = _engine(_panel())
        xr = ExactR(engine)
        I, O = [0, 4], [u for u in range(10) if u not in (0, 4)]
        m = Multipliers(np.zeros(1), np.zeros(1), Cuts.empty(), np.zeros(0))
        U, value = xr.certify(I, O, [], 2, m), _exact_max(xr, 10, 2, I, O, [])
        assert U >= value
        assert float(U) == pytest.approx(float(value), rel=1e-9)

    def test_no_eigenvalue_bound_means_no_certificate(self, monkeypatch):
        _, engine = _engine(_panel())
        xr = ExactR(engine)
        monkeypatch.setattr(branch, "lambda_min_lower", lambda D: None)
        m = Multipliers(np.zeros(3), np.zeros(3), Cuts.empty(), np.zeros(0))
        assert xr.certify([0], list(range(3, 10)), [1, 2], 2, m) is None


class TestSeparation:
    @given(st.integers(3, 7), st.integers(0, 10 ** 6))
    @settings(max_examples=40, deadline=None)
    def test_a_design_violates_no_triangle(self, n, seed):
        """Every triangle inequality holds at a design, so separation never
        cuts one off."""
        x = np.random.default_rng(seed).choice([-1.0, 1.0], size=n)
        v = np.concatenate([[1.0], x])
        assert len(separate(np.outer(v, v), gap=0.0)) == 0

    @given(st.integers(3, 7), st.integers(0, 10 ** 6))
    @settings(max_examples=40, deadline=None)
    def test_what_it_finds_is_violated_most_first_within_the_limit(self, size, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.0, 1.0, (size, size))
        X = 0.5 * (X + X.T)
        np.fill_diagonal(X, 1.0)
        cuts = separate(X, gap=0.05, limit=7)
        values = cuts.values(X)
        assert len(cuts) <= 7
        assert np.all(values < -0.05)
        assert np.all(np.diff(values) >= 0.0)

    def test_a_matrix_too_small_for_a_triangle(self):
        assert len(separate(np.eye(2))) == 0


# ----------------------------------------------------------------------
# the node bounds
# ----------------------------------------------------------------------


class TestTrustRegion:
    @given(panels, st.data())
    @settings(**SETTINGS)
    def test_it_bounds_every_completion(self, Y, data):
        _, engine = _engine(Y)
        N = engine.N
        K = data.draw(st.integers(1, N - 1))
        I, O, F = _node(N, K, data.draw(st.integers(0, 10 ** 6)))
        s = 2 * (K - len(I)) - len(F)
        U, x, _ = trust_region(*node_parts(engine.R, I, O, F), s)
        top = _float_max(engine.R, N, K, I, O, F)
        assert U >= top - 1e-10 * abs(top)
        assert x.shape == (len(F),)
        if len(F):
            assert x.sum() == pytest.approx(s, abs=1e-8)

    def test_it_is_exact_with_one_of_two_free_units_to_choose(self):
        """With ``n = 2`` and ``s = 0`` the sphere meets the plane at the two
        designs, so the relaxation is exact."""
        _, engine = _engine(_panel())
        I, F = [2], [6, 8]
        O = [u for u in range(10) if u not in (2, 6, 8)]
        U, _, _ = trust_region(*node_parts(engine.R, I, O, F), 0)
        assert U == pytest.approx(_float_max(engine.R, 10, 2, I, O, F), rel=1e-10)

    def test_a_decided_node_returns_its_value(self):
        _, engine = _engine(_panel())
        I, O = [1, 3], [u for u in range(10) if u not in (1, 3)]
        U, x, _ = trust_region(*node_parts(engine.R, I, O, []), 0)
        assert x.shape == (0,)
        assert U == pytest.approx(_float_max(engine.R, 10, 2, I, O, []))

    def test_all_free_units_on_one_side(self):
        """``r = n``: the free units are all treated, ``x = 1``."""
        _, engine = _engine(_panel())
        I, F = [0], [4, 5]
        O = [u for u in range(10) if u not in (0, 4, 5)]
        U, x, _ = trust_region(*node_parts(engine.R, I, O, F), 2)
        assert np.allclose(x, 1.0)
        assert U == pytest.approx(_float_max(engine.R, 10, 3, I, O, F), rel=1e-10)

    def test_a_zero_quadratic_and_linear_part(self):
        U, x, _ = trust_region(1.5, np.zeros(4), np.zeros((4, 4)), 0)
        assert U == pytest.approx(1.5)
        assert x.sum() == pytest.approx(0.0)


class TestLift:
    @given(panels, st.data())
    @settings(**SETTINGS)
    def test_it_bounds_every_completion_and_never_exceeds_the_trust_region(self, Y, data):
        _, engine = _engine(Y)
        N = engine.N
        K = data.draw(st.integers(1, N - 1))
        I, O, F = _node(N, K, data.draw(st.integers(0, 10 ** 6)))
        parts = node_parts(engine.R, I, O, F)
        s = 2 * (K - len(I)) - len(F)
        U, _, _ = lift(*parts, s)
        top = _float_max(engine.R, N, K, I, O, F)
        assert U >= top - 1e-10 * abs(top)
        assert U <= trust_region(*parts, s)[0] * (1 + 1e-12)

    def test_it_is_tighter_than_the_trust_region_on_a_steep_panel(self):
        _, engine = _engine(_noise(N=12, seed=2))
        I, O, F = [], [5], [u for u in range(12) if u != 5]
        parts = node_parts(engine.R, I, O, F)
        U, _, evaluations = lift(*parts, 2 * 4 - len(F))
        assert evaluations > 0
        assert U < trust_region(*parts, 2 * 4 - len(F))[0] * (1 - 1e-3)

    def test_a_target_stops_it_early(self):
        _, engine = _engine(_noise(N=12, seed=2))
        I, O, F = [], [5], [u for u in range(12) if u != 5]
        parts = node_parts(engine.R, I, O, F)
        s = 2 * 4 - len(F)
        full, _, full_evals = lift(*parts, s)
        tr = trust_region(*parts, s)[0]
        target = full + 0.5 * (tr - full)
        U, _, evals = lift(*parts, s, target=target)
        assert U <= target
        assert evals < full_evals

    def test_a_target_the_trust_region_meets_costs_nothing(self):
        _, engine = _engine(_panel())
        I, O, F = [0], [1], list(range(2, 10))
        parts = node_parts(engine.R, I, O, F)
        tr = trust_region(*parts, 2 * 2 - 8)[0]
        U, _, evals = lift(*parts, 2 * 2 - 8, target=2.0 * tr)
        assert evals == 0 and U == pytest.approx(tr)

    @pytest.mark.parametrize("n", [0, 1])
    def test_tiny_nodes_take_the_trust_region(self, n):
        _, engine = _engine(_panel())
        F = list(range(10 - n, 10))
        I = [0] if n == 0 else []
        O = [u for u in range(10) if u not in I and u not in F]
        K = len(I) + (1 if n else 0)
        parts = node_parts(engine.R, I, O, F)
        U, _, evals = lift(*parts, 2 * (K - len(I)) - n)
        assert evals == 0
        assert U == pytest.approx(trust_region(*parts, 2 * (K - len(I)) - n)[0])

    def test_a_zero_node(self):
        U, _, evals = lift(0.0, np.zeros(5), np.zeros((5, 5)), 1)
        assert U == 0.0 and evals == 0

    def test_a_round_cut_short_by_its_budget_is_still_a_bound(self, monkeypatch):
        """L-BFGS-B returns at its evaluation budget, not on convergence; the
        bound is whatever the evaluations reached, and still a bound."""
        monkeypatch.setattr(branch, "EVALUATIONS", 3)
        _, engine = _engine(_noise(N=10, seed=5))
        xr = ExactR(engine)
        I, O, F = [], [2], [u for u in range(10) if u != 2]
        U, m, evals = lift(*node_parts(engine.R, I, O, F), 2 * 3 - len(F))
        assert evals > 0
        assert xr.certify(I, O, F, 3, m) >= _exact_max(xr, 10, 3, I, O, F)
        assert U >= _float_max(engine.R, 10, 3, I, O, F) * (1 - 1e-12)


def _scale(b, A):
    return max(float(np.abs(b).max(initial=0.0)), float(np.abs(A).max(initial=0.0)))


class TestLiftConstant:
    """``c`` is the fixed units' value against each other. Every relaxation the
    lift bounds has ``X_00 = 1``, so ``c`` adds to the bound and shapes nothing:
    the lift is translation equivariant in it. Its scale was once taken from
    ``c`` too, and the fixed units' count grew it, so a node deep in the tree
    ran its dual on a free block shrunk toward zero against the fixed ``alpha``
    schedule, and stopped at the trust region where the relaxation was tighter
    (/rca)."""

    def test_a_deep_node_passes_the_trust_region(self):
        """Forty units with twelve fixed, where the constant is ten times the
        free block's largest entry: the relaxation is tighter than the trust
        region, and the lift that took its scale from the constant stopped at
        the trust region."""
        _, engine = _engine(_noise(N=40, T=10, seed=0))
        I, O, F = list(range(6)), list(range(6, 12)), list(range(12, 40))
        c, b, A = node_parts(engine.R, I, O, F)
        s = 2 * (20 - len(I)) - len(F)
        assert abs(c) > 8 * _scale(b, A)
        U, _, _ = lift(c, b, A, s)
        assert U < trust_region(c, b, A, s)[0] * (1 - 1e-4)

    @given(panels, st.data())
    @settings(**SETTINGS)
    def test_the_constant_adds_to_the_bound_and_shapes_nothing(self, Y, data):
        _, engine = _engine(Y)
        N = engine.N
        K = data.draw(st.integers(1, N - 1))
        I, O, F = _node(N, K, data.draw(st.integers(0, 10 ** 6)))
        c, b, A = node_parts(engine.R, I, O, F)
        s = 2 * (K - len(I)) - len(F)
        t = data.draw(st.floats(-100.0, 100.0)) * _scale(b, A)
        U, _, evaluations = lift(c, b, A, s)
        shifted, _, shifted_evaluations = lift(c + t, b, A, s)
        assert shifted_evaluations == evaluations
        assert shifted - t == pytest.approx(U, rel=1e-12, abs=1e-12 * (abs(t) + abs(U)))

    def test_a_node_with_no_free_part_left_costs_no_evaluation(self):
        """A constant over zero parts is the node's value: nothing to lift."""
        U, _, evals = lift(3.5, np.zeros(5), np.zeros((5, 5)), 1)
        assert U == 3.5 and evals == 0

    def test_its_multipliers_certify_a_deep_node(self):
        """The certificate reads the constant from the multipliers, unscaled,
        whatever the scale the dual ran at."""
        _, engine = _engine(_noise(N=10, seed=5))
        xr = ExactR(engine)
        I, O, F = [0, 1], [2, 3], list(range(4, 10))
        U, m, evals = lift(*node_parts(engine.R, I, O, F), 2 * (5 - 2) - len(F))
        assert evals > 0
        assert xr.certify(I, O, F, 5, m) >= _exact_max(xr, 10, 5, I, O, F)
        assert U >= _float_max(engine.R, 10, 5, I, O, F) * (1 - 1e-12)


def _blas_threads():
    """The thread count of every BLAS pool the process has loaded."""
    from threadpoolctl import threadpool_info
    return [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]


@pytest.fixture
def two_threads():
    """The caller runs BLAS on two threads, so a lift that forgot to pin
    itself would show it even where the environment pins BLAS to one."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(2, user_api="blas"):
        threads = _blas_threads()
        if max(threads) < 2:  # pragma: no cover - CI and development machines run several threads; this guards a single-core runner
            pytest.skip("BLAS cannot run two threads here")
        yield threads


def _steep_node():
    _, engine = _engine(_noise(N=12, seed=2))
    I, O, F = [], [5], [u for u in range(12) if u != 5]
    return node_parts(engine.R, I, O, F), 2 * 4 - len(F)


class TestLiftThreads:
    """#857: the lift's matrices have at most ``N + 1`` rows, where a BLAS
    thread pool costs more to coordinate than it saves. On four threads a
    root lift took 16 s at 60 units against 0.8 s on one, on a quiet machine.
    The lift runs on one thread and gives the caller back its own setting."""

    def test_the_dual_is_evaluated_on_one_thread(self, two_threads, monkeypatch):
        seen = []
        theta = branch._theta

        def spy(*args, **kwargs):
            if len(seen) < 2:
                seen.append(_blas_threads())
            return theta(*args, **kwargs)

        monkeypatch.setattr(branch, "_theta", spy)
        parts, s = _steep_node()
        _, _, evaluations = lift(*parts, s)
        assert evaluations > 0 and seen
        assert all(set(threads) == {1} for threads in seen)

    def test_the_callers_setting_is_restored(self, two_threads):
        parts, s = _steep_node()
        lift(*parts, s)
        assert _blas_threads() == two_threads

    def test_a_lift_that_raises_still_restores_it(self, two_threads, monkeypatch):
        def broken(*args, **kwargs):
            raise FloatingPointError("the dual failed")

        monkeypatch.setattr(branch, "_theta", broken)
        parts, s = _steep_node()
        with pytest.raises(FloatingPointError, match="the dual failed"):
            lift(*parts, s)
        assert _blas_threads() == two_threads

    def test_pinning_changes_neither_the_bound_nor_the_work(self):
        """The lift is deterministic on one thread; run under the caller's own
        one-thread limit, it takes the same evaluations to the same bound."""
        from threadpoolctl import threadpool_limits
        parts, s = _steep_node()
        U, m, evaluations = lift(*parts, s)
        with threadpool_limits(1):
            U1, m1, evaluations1 = lift(*parts, s)
        assert (U, evaluations) == (U1, evaluations1)
        np.testing.assert_array_equal(m.diag, m1.diag)

    @given(panels, st.data())
    @settings(max_examples=8, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_any_node_leaves_the_callers_threads_as_it_found_them(self, two_threads, Y, data):
        _, engine = _engine(Y)
        N = engine.N
        K = data.draw(st.integers(1, N - 1))
        I, O, F = _node(N, K, data.draw(st.integers(0, 10 ** 6)))
        lift(*node_parts(engine.R, I, O, F), 2 * (K - len(I)) - len(F))
        assert _blas_threads() == two_threads


# ----------------------------------------------------------------------
# the cost model
# ----------------------------------------------------------------------


class TestPruneRates:
    def test_an_unseen_size_starts_at_one_half(self):
        assert PruneRates().rate(7) == pytest.approx(0.5)

    def test_hits_raise_and_misses_lower_a_size(self):
        rates = PruneRates()
        for _ in range(5):
            rates.record(3, True)
        high = rates.rate(3)
        for _ in range(20):
            rates.record(4, False)
        assert high > 0.5 > rates.rate(4)

    def test_misses_at_other_sizes_lower_an_unseen_one(self):
        """The estimate is pooled across sizes: where no bound has pruned at any
        size, an unseen size does not start at one half."""
        rates = PruneRates()
        for size in range(10):
            for _ in range(5):
                rates.record(size, False)
        assert rates.rate(30) < 0.2

    @given(st.lists(st.tuples(st.integers(0, 40), st.booleans()), max_size=60),
           st.integers(0, 40))
    @settings(max_examples=60, deadline=None)
    def test_a_rate_is_a_probability(self, history, size):
        rates = PruneRates()
        for s, hit in history:
            rates.record(s, hit)
        assert 0.0 < rates.rate(size) < 1.0


class TestTheCostModel:
    """The costs are the one-thread calibration of #845, at the free units the
    search's nodes have: a modelled second is a second of the clock on the
    machine that measured them, to within the spread of its nodes."""

    @pytest.mark.parametrize("n, seconds", [(10, 0.51e-3), (25, 0.80e-3),
                                            (40, 1.03e-3), (60, 1.28e-3)])
    def test_a_lift_evaluation_costs_what_it_measured(self, n, seconds):
        assert branch.evaluation_seconds(n) == pytest.approx(seconds, rel=0.1)

    @pytest.mark.parametrize("n, seconds", [(10, 0.26e-3), (40, 0.53e-3),
                                            (60, 0.63e-3)])
    def test_a_trust_region_bound_costs_what_it_measured(self, n, seconds):
        assert branch.trust_region_seconds(n) == pytest.approx(seconds, rel=0.15)

    @pytest.mark.parametrize("n, seconds", [(10, 1.71e-3), (40, 13.4e-3),
                                            (60, 30.7e-3)])
    def test_a_certificate_costs_what_it_measured(self, n, seconds):
        assert branch.certificate_seconds(n) == pytest.approx(seconds, rel=0.1)

    def test_a_set_priced_costs_what_it_measured(self):
        # 2.47 µs a set at 40 units, on the enumeration's path
        assert branch.PRICE_SECONDS_PER_UNIT * 40 == pytest.approx(2.47e-6, rel=0.1)

    def test_the_default_limit_leaves_the_documented_searches_room(self):
        """At forty units with ten treated, the documented GeoLift and
        structureless searches do 59.6 and 60.4 modelled seconds of work under
        these costs (#845); the default is about twice the larger."""
        from mlsynth.utils.syndes_helpers.exact import WORK_LIMIT
        assert WORK_LIMIT >= 1.95 * 60.4

    @given(st.integers(1, 120))
    @settings(max_examples=40, deadline=None)
    def test_every_cost_grows_with_the_node(self, n):
        for cost in (branch.evaluation_seconds, branch.trust_region_seconds,
                     branch.certificate_seconds):
            assert 0.0 < cost(n) < cost(n + 1)


class TestLiftCost:
    def test_it_starts_at_its_prior(self):
        cost = LiftCost(evaluations=300.0)
        assert cost(20) == pytest.approx(300.0 * branch.evaluation_seconds(20))

    def test_observed_lifts_move_it_in_logarithms(self):
        """Two hundred lifts at ten evaluations outweigh a prior of three hundred
        counted as two lifts: an arithmetic mean would sit near thirteen."""
        cost = LiftCost(evaluations=300.0, weight=2.0)
        for _ in range(200):
            cost.observe(10)
        per = cost(20) / branch.evaluation_seconds(20)
        assert 10.0 < per < 11.0

    @given(st.lists(st.integers(0, 5000), max_size=50), st.integers(2, 80))
    @settings(max_examples=60, deadline=None)
    def test_a_cost_is_positive_and_grows_with_the_node(self, observed, n):
        cost = LiftCost()
        for e in observed:
            cost.observe(e)
        assert 0.0 < cost(n) < cost(n + 1)


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------


class TestSearch:
    def test_smoke_above_the_limit(self):
        """Above the limit the answer is the enumeration's, certified."""
        Y = _panel(N=14, T=20, seed=5)
        got = _branched(Y, 5)
        want = _enumerated(Y, 5)
        assert got.raw_results["search"] == "branch_and_bound"
        assert want.raw_results["search"] == "enumeration"
        assert tuple(got.selected_unit_indices) == tuple(want.selected_unit_indices)
        assert got.objective_value == pytest.approx(want.objective_value, rel=1e-9)
        assert got.raw_results["certified"] is True
        assert got.raw_results["lower_bound"] <= got.objective_value
        assert got.raw_results["lower_bound"] >= got.objective_value * (1 - 1e-8)
        assert np.isfinite(got.contrast_weights).all()

    @given(panels, st.data())
    @settings(**SETTINGS)
    def test_it_returns_the_enumerations_optimum(self, Y, data):
        N = Y.shape[1]
        K = data.draw(st.integers(1, N - 1))
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, K, SubsetFilter(), **EAGER)
        want = _enumerated(Y, K)
        assert out.closed
        assert out.value == pytest.approx(want.objective_value, rel=1e-9)
        assert out.lower <= out.value

    @pytest.mark.parametrize("costs", [
        dict(),
        EAGER,
        dict(price_seconds=1.0, bound_seconds=lambda n: 0.0,
             lift_seconds=lambda n: np.inf),
        dict(price_seconds=0.0),
    ])
    def test_the_answer_does_not_depend_on_the_costs(self, costs):
        Y = _noise(N=12, seed=8)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), **costs)
        want = _enumerated(Y, 4)
        assert tuple(out.treated) == tuple(want.selected_unit_indices)

    def test_eager_costs_bound_lift_and_prune(self):
        Y = _noise(N=12, seed=8)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), **EAGER)
        assert out.nodes > 0 and out.lifts > 0 and out.pruned > 0

    def test_a_root_the_forced_units_decide(self):
        Y = _panel(N=11, seed=1)
        problem, engine = _engine(Y)
        filt = SubsetFilter(forced_in=frozenset({1, 4, 7}))
        out = branch_and_bound(problem, engine, 3, filt, **EAGER)
        assert tuple(out.treated) == (1, 4, 7)
        assert out.closed and out.nodes == 0 and out.priced == 1

    def test_no_bound_means_every_set_is_priced(self):
        Y = _panel(N=11, seed=1)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), price_seconds=0.0)
        assert out.nodes == 0 and out.priced == comb(11, 4)

    def test_a_poor_incumbent_does_not_change_the_answer(self):
        Y = _panel(N=12, seed=3)
        problem, engine = _engine(Y)
        enumerated = _enumerated(Y, 4)
        worst = np.array([[0, 1, 2, 3]])
        out = branch_and_bound(problem, engine, 4, SubsetFilter(),
                               incumbent_sets=worst, restarts=0, **EAGER)
        assert tuple(out.treated) == tuple(enumerated.selected_unit_indices)

    def test_the_sets_whose_closed_form_breaks_are_solved(self):
        """At a small penalty on a panel with no structure the sign conditions
        bind for nearly every set, so the optimum is one the search must solve,
        not read off the closed form."""
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((12, 10))
        problem = TwoWayProblem.from_panel(Y, lam=0.05)
        engine = BoundEngine.from_problem(problem)
        out = branch_and_bound(problem, engine, 3, SubsetFilter(), restarts=0, **EAGER)
        want = _enumerated(Y, 3, lam=0.05)
        assert out.solved > 0
        assert tuple(out.treated) == tuple(want.selected_unit_indices)
        assert out.value == pytest.approx(want.objective_value, rel=1e-9)

    def test_a_failed_certificate_keeps_the_search_exact(self, monkeypatch):
        """A node whose bound cannot be certified is split, never discarded."""
        Y = _noise(N=11, seed=4)
        problem, engine = _engine(Y)
        monkeypatch.setattr(ExactR, "certify", lambda self, *a, **k: None)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), **EAGER)
        assert out.pruned == 0
        assert tuple(out.treated) == tuple(_enumerated(Y, 4).selected_unit_indices)

    def test_a_failed_lift_keeps_the_trust_region_bound(self, monkeypatch):
        Y = _noise(N=11, seed=4)
        problem, engine = _engine(Y)
        monkeypatch.setattr(branch, "lift", lambda *a, **k: (np.nan, None, 7))
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), **EAGER)
        assert tuple(out.treated) == tuple(_enumerated(Y, 4).selected_unit_indices)

    def test_a_certificate_that_does_not_clear_the_incumbent_prunes_nothing(
            self, monkeypatch):
        """The float bound says prune; the exact one disagrees; the node is
        split."""
        Y = _noise(N=11, seed=4)
        problem, engine = _engine(Y)
        monkeypatch.setattr(ExactR, "certify",
                            lambda self, *a, **k: Fraction(10 ** 30))
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), **EAGER)
        assert out.pruned == 0
        assert tuple(out.treated) == tuple(_enumerated(Y, 4).selected_unit_indices)

    @given(st.integers(0, 10 ** 6), st.sampled_from([1e-4, 1.0, 1e4]),
           st.sampled_from([0.0, 1e6]))
    @settings(max_examples=10, deadline=None)
    def test_the_design_does_not_depend_on_the_outcomes_units(self, seed, scale, level):
        Y = _noise(N=9, seed=seed)
        base = _branched(Y, 3)
        moved = _branched(scale * Y + level, 3)
        assert tuple(moved.selected_unit_indices) == tuple(base.selected_unit_indices)
        assert moved.objective_value == pytest.approx(
            scale ** 2 * base.objective_value, rel=1e-6)

    def test_every_size_when_K_is_free(self):
        Y = _panel(N=8, seed=6)
        got = _branched(Y, None)
        want = _enumerated(Y, None)
        assert tuple(got.selected_unit_indices) == tuple(want.selected_unit_indices)
        assert got.raw_results["certified"] is True

    def test_the_pool_above_the_limit(self):
        Y = _panel(N=10, seed=2)
        got = solve_synthetic_design_exact_pool(Y, 3, top_K=3, candidate_limit=1)
        want = solve_synthetic_design_exact_pool(Y, 3, top_K=3, candidate_limit=10 ** 9)
        assert [tuple(d.selected_unit_indices) for d in got] == [
            tuple(d.selected_unit_indices) for d in want]

    def test_the_certificate_reads_the_search(self):
        Y = _panel(N=12, seed=7)
        design = _branched(Y, 4)
        cert = syndes_certificate(Y, 4, "global_2way", design.objective_value,
                                  lam=design.lambda_value, search=design)
        assert cert.certified
        assert cert.optimality_gap == pytest.approx(0.0, abs=1e-8)

    def test_a_clean_search_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _branched(_panel(N=12, seed=9), 4)


class TestCutoff:
    """Searching every size, the best design of one size bounds the others."""

    def test_a_size_the_cutoff_dominates_returns_no_design(self):
        Y = _noise(N=11, seed=6)
        problem, engine = _engine(Y)
        at_4 = _enumerated(Y, 4).objective_value
        cutoff = 0.999 * at_4
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), cutoff=cutoff, **EAGER)
        assert out.treated is None and out.value == np.inf
        assert out.closed and out.certified
        assert out.lower <= at_4 * (1 + 1e-9)
        assert out.lower >= engine.global_lower_bound(4) * (1 - 1e-12)

    def test_a_cutoff_the_size_beats_changes_nothing(self):
        Y = _noise(N=11, seed=6)
        problem, engine = _engine(Y)
        want = _enumerated(Y, 4)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(),
                               cutoff=1.5 * want.objective_value, **EAGER)
        assert tuple(out.treated) == tuple(want.selected_unit_indices)

    def test_every_size_reports_the_smallest_bound(self):
        """With K free the bound is the smallest over the sizes, the dominated
        ones included, and never above the optimum."""
        Y = _noise(N=9, seed=3)
        got = _branched(Y, None)
        want = _enumerated(Y, None)
        assert got.raw_results["lower_bound"] <= want.objective_value * (1 + 1e-9)
        assert got.raw_results["certified"] is True

    def test_every_size_starts_from_the_swap_searchs_best(self):
        """With K free the swap search's best design over the sizes past the
        limit is every size's incumbent; when no size beats it, it is the design,
        with the work of its own size's search."""
        Y = _panel(N=9, seed=4)
        got = _branched(Y, None)
        want = _enumerated(Y, None)
        assert tuple(got.selected_unit_indices) == tuple(want.selected_unit_indices)
        assert got.raw_results["search"] == "branch_and_bound"
        assert got.raw_results["n_designs"] == comb(9, len(got.selected_unit_indices))
        assert got.raw_results["certified"] is True

    def test_every_size_with_restrictions(self):
        Y = _noise(N=9, seed=5)
        r = DesignRestrictions(forbidden=[0, 4], conflict_pairs=[(1, 2)])
        got = _branched(Y, None, restrictions=r)
        want = _enumerated(Y, None, restrictions=r)
        assert tuple(got.selected_unit_indices) == tuple(want.selected_unit_indices)
        assert got.raw_results["certified"] is True

    def test_a_search_stopped_at_its_limit_is_certified_when_its_bound_reaches(self):
        """Certification reads the bound, not whether the tree closed: with the
        cutoff between the root's bound and both children's, the root is split,
        the limit leaves the children open, and their certified bounds clear
        the cutoff."""
        Y = _noise(N=12, seed=3)
        problem, engine = _engine(Y)
        K, N, alpha = 4, 12, engine.alpha
        everyone = list(range(N))
        U, x, _ = trust_region(*node_parts(engine.R, [], [], everyone), 2 * K - N)
        u = int(np.argsort(-x, kind="stable")[0])
        rest = [v for v in everyone if v != u]
        root = 4 * alpha / U
        treat = 4 * alpha / trust_region(*node_parts(engine.R, [u], [], rest),
                                         2 * (K - 1) - (N - 1))[0]
        skip = 4 * alpha / trust_region(*node_parts(engine.R, [], [u], rest),
                                        2 * K - (N - 1))[0]
        assert root < min(treat, skip)
        cutoff = 0.5 * (root + min(treat, skip))
        out = branch_and_bound(problem, engine, K, SubsetFilter(), work_limit=1e-12,
                               cutoff=cutoff, restarts=0, price_seconds=1.0,
                               bound_seconds=lambda n: 0.0,
                               lift_seconds=lambda n: np.inf)
        assert out.treated is None and not out.closed
        assert out.certified
        assert out.lower >= cutoff * (1 - 1e-8)

    def test_a_dominated_size_at_its_work_limit_warns(self):
        Y = _noise(N=20, seed=13)
        best = _enumerated(Y, 8).objective_value
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 8, SubsetFilter(), work_limit=1e-12,
                               cutoff=0.999 * best, restarts=0)
        assert out.treated is None and not out.closed
        with pytest.warns(SYNDESSolveWarning, match="work limit"):
            from mlsynth.utils.syndes_helpers.exact import _search_by_branching
            got = _search_by_branching(problem, engine, 8, SubsetFilter(), seed=0,
                                       n_designs=1, incumbent_sets=None,
                                       work_limit=1e-12, cutoff=0.999 * best)
        assert got.value == np.inf and not got.certified


class TestRestrictions:
    N, K = 11, 4

    @pytest.mark.parametrize("restrictions, extra", [
        (DesignRestrictions(forced_in=[0, 1]), {}),
        (DesignRestrictions(forbidden=[2, 3, 4]), {}),
        (DesignRestrictions(conflict_pairs=[(0, 1), (2, 5), (5, 7)]), {}),
        (DesignRestrictions(strata=[((0, 1, 2, 3, 4), 1, 2)]), {}),
        (DesignRestrictions(forced_in=[6], strata=[((0, 1, 2), None, 1)],
                            conflict_pairs=[(6, 8)]), {}),
        (None, dict(costs=np.r_[np.full(5, 3.0), np.ones(6)], budget=6.0)),
    ])
    def test_the_restricted_optimum(self, restrictions, extra):
        Y = _noise(N=self.N, seed=11)
        got = _branched(Y, self.K, restrictions=restrictions, **extra)
        want = _enumerated(Y, self.K, restrictions=restrictions, **extra)
        assert tuple(got.selected_unit_indices) == tuple(want.selected_unit_indices)
        assert got.objective_value == pytest.approx(want.objective_value, rel=1e-9)
        assert got.raw_results["certified"] is True

    def test_eager_costs_honour_the_restrictions(self):
        Y = _noise(N=self.N, seed=11)
        r = DesignRestrictions(forced_in=[3], forbidden=[0], conflict_pairs=[(3, 9)],
                               strata=[((4, 5, 6, 7), 1, 1)])
        problem, engine = _engine(Y)
        filt = SubsetFilter.build(r, N=self.N)
        out = branch_and_bound(problem, engine, self.K, filt, **EAGER)
        want = _enumerated(Y, self.K, restrictions=r)
        assert tuple(out.treated) == tuple(want.selected_unit_indices)
        assert filt.accepts(tuple(int(u) for u in out.treated))

    def test_no_admissible_design_is_reported(self):
        Y = _noise(N=self.N, seed=11)
        r = DesignRestrictions(forced_in=[0, 1], conflict_pairs=[(0, 1)])
        with pytest.raises(MlsynthEstimationError, match="no feasible design"):
            _branched(Y, self.K, restrictions=r)

    def test_forced_and_forbidden_that_overlap_admit_nothing(self):
        Y = _noise(N=self.N, seed=11)
        problem, engine = _engine(Y)
        filt = SubsetFilter(forced_in=frozenset({1}), forbidden=frozenset({1}))
        with pytest.raises(MlsynthEstimationError, match="no feasible design"):
            branch_and_bound(problem, engine, self.K, filt)


class TestNodeOrder:
    """Best-first: the open node whose bound is smallest is bounded next, and
    each open node carries the certified bound of its nearest lifted
    ancestor."""

    def test_nodes_leave_the_queue_in_order_of_their_bound(self, monkeypatch):
        import heapq
        keys = []
        real = heapq.heappop

        def recording(heap):
            entry = real(heap)
            keys.append(entry[0])
            return entry

        monkeypatch.setattr(heapq, "heappop", recording)
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), restarts=0, **EAGER)
        assert out.closed and len(keys) > 3
        assert all(a <= b for a, b in zip(keys, keys[1:]))

    def test_a_stopped_search_keeps_its_lifted_nodes_bounds(self):
        """Stopped after a few lifts, every open node descends from the lifted
        root, so the bound is at least the root lift's certified bound: above
        the Rayleigh bound, and under the optimum."""
        Y = _noise(N=16, seed=13)
        problem, engine = _engine(Y)
        N, K, alpha = 16, 8, engine.alpha
        everyone = list(range(N))
        _, m, _ = lift(*node_parts(engine.R, [], [], everyone), 2 * K - N, target=None)
        root = 4 * alpha / float(ExactR(engine).certify([], [], everyone, K, m))
        out = branch_and_bound(problem, engine, K, SubsetFilter(), work_limit=1.5,
                               restarts=0, **EAGER)
        assert not out.closed and out.lifts > 1
        assert out.lower >= root * (1 - 1e-9)
        assert root > engine.global_lower_bound(K) * (1 + 1e-9)
        assert out.lower <= _enumerated(Y, K).objective_value * (1 + 1e-9)


    def test_with_no_incumbent_it_dives_to_a_design_first(self):
        """No seed and no cutoff: best first would spread its limit over the
        top of the tree; the search dives until it has a design."""
        Y = _noise(N=16, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 8, SubsetFilter(), work_limit=0.05,
                               restarts=0, price_seconds=1.0,
                               bound_seconds=lambda n: 0.0,
                               lift_seconds=lambda n: np.inf)
        assert not out.closed and out.treated is not None
        assert out.value >= _enumerated(Y, 8).objective_value * (1 - 1e-9)


class TestWorkLimit:
    def test_a_limit_returns_the_best_design_with_a_valid_bound(self):
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=1e-12,
                               **EAGER)
        want = _enumerated(Y, 4)
        assert not out.closed
        assert out.lower <= want.objective_value * (1 + 1e-9)
        assert out.lower >= engine.global_lower_bound(4) * (1 - 1e-12)
        assert out.value >= want.objective_value * (1 - 1e-9)

    def test_a_limit_reached_from_a_poor_incumbent_keeps_a_valid_bound(self):
        """The incumbent at the limit is the worst set; the bound is the open
        nodes', under the optimum, not the incumbent's."""
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=1e-12,
                               incumbent_sets=np.array([[0, 1, 2, 3]]), restarts=0,
                               **EAGER)
        best = _enumerated(Y, 4).objective_value
        assert out.value > best * (1 + 1e-6)
        assert out.lower <= best * (1 + 1e-9)
        assert not out.certified

    def test_a_loose_prune_tolerance_is_charged_to_the_bound(self, monkeypatch):
        """Nodes discarded within a tolerance of the incumbent may hold a better
        design; the bound reports theirs, so it never passes the optimum."""
        monkeypatch.setattr(branch, "PRUNE_RTOL", 0.9)
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(),
                               incumbent_sets=np.array([[0, 1, 2, 3]]), restarts=0,
                               **EAGER)
        best = _enumerated(Y, 4).objective_value
        assert out.value > best * (1 + 1e-6)
        assert out.lower <= best * (1 + 1e-9)
        assert not out.certified

    def test_the_design_says_so_and_warns(self):
        """Twenty units and eight treated are enough sets that the default
        costs bound the root."""
        Y = _noise(N=20, seed=13)
        with pytest.warns(SYNDESSolveWarning, match="work limit"):
            got = solve_synthetic_design_exact(Y=Y, K=8, candidate_limit=1,
                                               work_limit=1e-12)
        assert got.raw_results["certified"] is False
        assert got.raw_results["lower_bound"] <= got.objective_value

    def test_pricing_counts_toward_the_limit(self):
        """Where no bound is paid for, the search prices; the sets it would price
        cost the model's seconds, so the limit stops a search whose bounds do
        not prune."""
        Y = _noise(N=20, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 8, SubsetFilter(), work_limit=0.01,
                               price_seconds=0.0)
        assert comb(20, 8) * branch.PRICE_SECONDS_PER_UNIT * 20 > 0.01
        assert not out.closed and out.nodes == 0 and out.priced == 0
        assert out.lower <= _enumerated(Y, 8).objective_value * (1 + 1e-9)

    def test_an_open_node_with_one_set_left_is_priced(self):
        """With one treated unit, the child that includes the first unit is
        decided; the limit leaves it open, and it is priced."""
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 1, SubsetFilter(), work_limit=1e-12,
                               restarts=0, **EAGER)
        assert not out.closed
        assert out.priced >= 1
        assert out.lower <= _enumerated(Y, 1).objective_value * (1 + 1e-9)

    def test_a_limit_before_any_design_is_reported(self):
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        with pytest.raises(MlsynthEstimationError, match="work limit"):
            branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=1e-12,
                             restarts=0, **EAGER)

    def test_an_open_node_whose_bound_cannot_be_certified_falls_back(
            self, monkeypatch):
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        monkeypatch.setattr(ExactR, "certify", lambda self, *a, **k: None)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=1e-12,
                               **EAGER)
        assert out.lower == pytest.approx(engine.global_lower_bound(4))

    @pytest.mark.parametrize("cost, lifts", [("trust_region_seconds", True),
                                             ("evaluation_seconds", True),
                                             ("certificate_seconds", False)])
    def test_every_kind_of_work_counts_toward_the_limit(self, monkeypatch, cost, lifts):
        """Priced at two seconds against a one-second limit, one bound, one lift
        evaluation or one certificate stops a search that closes when it costs
        what the model says. Without lifts the only certificates in the search
        are the ones that discard a node."""
        costs = EAGER if lifts else dict(EAGER, lift_seconds=lambda n: np.inf)
        Y = _noise(N=12, seed=13)
        problem, engine = _engine(Y)
        free = branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=1.0,
                                **costs)
        assert free.closed and (free.lifts > 0) == lifts and free.pruned > 0
        monkeypatch.setattr(branch, cost, lambda n: 2.0)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=1.0,
                               **costs)
        assert not out.closed and out.nodes < free.nodes
        assert out.lower <= _enumerated(Y, 4).objective_value * (1 + 1e-9)

    @given(st.floats(1e-6, 1e-1), st.integers(0, 10 ** 6))
    @settings(max_examples=12, deadline=None)
    def test_at_any_limit_the_design_and_bound_bracket_the_optimum(self, limit, seed):
        Y = _noise(N=11, seed=seed % 1000)
        problem, engine = _engine(Y)
        out = branch_and_bound(problem, engine, 4, SubsetFilter(), work_limit=limit,
                               **EAGER)
        best = _enumerated(Y, 4).objective_value
        assert out.lower <= best * (1 + 1e-9)
        assert out.value >= best * (1 - 1e-9)
        if out.closed:
            assert out.value == pytest.approx(best, rel=1e-9)

    @pytest.mark.parametrize("limit", [0, -3.0, np.inf, np.nan, True, "1"])
    def test_an_invalid_limit_is_refused(self, limit):
        with pytest.raises(MlsynthConfigError, match="work_limit"):
            solve_synthetic_design_exact(Y=_panel(), K=3, work_limit=limit)


class TestStillRefused:
    def test_a_singular_panel_past_the_limit(self):
        rng = np.random.default_rng(9)
        Y = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 24))
        with pytest.raises(MlsynthEstimationError, match="ill-conditioned"):
            solve_synthetic_design_exact(Y=Y, K=8, lam=1e-14, candidate_limit=1000)
