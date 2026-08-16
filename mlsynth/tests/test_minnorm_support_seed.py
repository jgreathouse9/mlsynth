"""The batched min-norm solver is told its support instead of discovering it.

Wolfe's min-norm-point method adds one vertex per iteration and solves a corral
system that grows with each addition, so an optimum supported on ``k`` donors
costs at least ``k`` iterations of a ``(k, k)`` solve. That cost model assumes a
sparse optimum. SDID's unit-weight program is ridged by ``zeta``, and the ridge
is there to push weight off the vertices onto a face, so the optimum comes out
around 70 percent dense -- median 67 of 99 donors on a 101-unit panel, 26 of 37
on Prop 99. The solver spent 87 iterations rediscovering, one element at a time,
a support a first-order pass names in one go (#461).

Two kinds of test live here, red for different reasons.

The work assertions start red. They count active-set iterations and hold a
dense-support batch to a handful, and they check that the seed is not paid for
where it cannot earn its keep -- a narrow pool, a caller-supplied warm start, or
the accelerator switched off.

The answer tests pass before the change and must pass after it. The seed is
speed only: Wolfe's optimality test stays the arbiter, so a seed that is late,
wrong, or actively adversarial may cost iterations and may not move the
objective. That is checked against seeds chosen to be bad on purpose, because
"the answer does not depend on the seed" is the property the whole design rests
on -- if it fails, the accelerator is not an accelerator, it is a second
estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel import accelerate as ACC
from mlsynth.utils.bilevel.accelerate import (
    fista_warm_start_batch,
    simplex_project,
    simplex_project_batch,
)
from mlsynth.utils.bilevel.minnorm import solve_simplex_minnorm_batch


# --------------------------------------------------------------------------- #
# Grams
# --------------------------------------------------------------------------- #
def ridged_batch(S=40, J=90, T=30, ridge=0.5, seed=0):
    """A stack shaped like SDID's placebo unit-weight family.

    Short and wide (``T < J``) with a ridge, which is what makes the optimum a
    dense face instead of a vertex -- the regime the seed is for.
    """
    rng = np.random.default_rng(seed)
    G = np.empty((S, J, J))
    for s in range(S):
        B = rng.standard_normal((T, J))
        B -= B.mean(axis=0)
        G[s] = B.T @ B + ridge * np.eye(J)
    return G


def sparse_batch(S=40, J=90, T=200, seed=1):
    """Tall and unridged: the optimum sits on a few donors."""
    rng = np.random.default_rng(seed)
    G = np.empty((S, J, J))
    for s in range(S):
        B = rng.standard_normal((T, J)) ** 2
        G[s] = B.T @ B
    return G


def obj(G, W):
    return np.einsum("sj,sjk,sk->s", W, G, W)


def support_density(W):
    return (W > 1e-9).sum(axis=1) / W.shape[1]


# =========================================================================== #
# the projection the seed is built on
# =========================================================================== #
class TestBatchProjection:
    @pytest.mark.parametrize("shape", [(1, 1), (1, 5), (7, 3), (40, 90)])
    def test_rows_land_on_the_simplex(self, shape):
        V = np.random.default_rng(0).standard_normal(shape) * 5
        P = simplex_project_batch(V)
        assert P.shape == shape
        assert (P >= 0).all()
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_matches_the_scalar_projection_row_by_row(self):
        """Differential against the shipped one-vector routine."""
        V = np.random.default_rng(1).standard_normal((25, 12)) * 3
        P = simplex_project_batch(V)
        for i, row in enumerate(V):
            np.testing.assert_allclose(P[i], simplex_project(row), atol=1e-12)

    def test_a_point_already_on_the_simplex_is_fixed(self):
        W = simplex_project_batch(np.random.default_rng(2).standard_normal((10, 8)))
        np.testing.assert_allclose(simplex_project_batch(W), W, atol=1e-12)

    def test_projection_is_the_nearest_simplex_point(self):
        rng = np.random.default_rng(3)
        V = rng.standard_normal((6, 5)) * 2
        P = simplex_project_batch(V)
        for i in range(V.shape[0]):
            d = np.linalg.norm(V[i] - P[i])
            for _ in range(200):
                q = rng.dirichlet(np.ones(5))
                assert np.linalg.norm(V[i] - q) >= d - 1e-12

    def test_single_column_is_the_only_point(self):
        np.testing.assert_allclose(
            simplex_project_batch(np.array([[-4.0], [7.0]])), [[1.0], [1.0]])


# =========================================================================== #
# the seed itself
# =========================================================================== #
class TestSeed:
    def test_returns_feasible_rows(self):
        G = ridged_batch(S=8, J=40, T=15)
        W = fista_warm_start_batch(G)
        assert W.shape == (8, 40)
        assert (W >= 0).all()
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)

    def test_is_deterministic(self):
        G = ridged_batch(S=6, J=30, T=12)
        np.testing.assert_array_equal(fista_warm_start_batch(G),
                                      fista_warm_start_batch(G))

    def test_finds_a_dense_support_on_a_ridged_family(self):
        G = ridged_batch(S=10, J=60, T=20)
        assert support_density(fista_warm_start_batch(G)).mean() > 0.3

    def test_a_zero_gram_still_returns_a_feasible_row(self):
        """A zero Gram is PSD with a flat objective: every simplex point is
        optimal, and the seed may not return a NaN for it."""
        G = np.zeros((3, 5, 5))
        W = fista_warm_start_batch(G)
        assert np.isfinite(W).all()
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)

    def test_single_donor(self):
        np.testing.assert_allclose(fista_warm_start_batch(np.ones((4, 1, 1))), 1.0)

    def test_single_problem(self):
        W = fista_warm_start_batch(ridged_batch(S=1, J=20, T=8))
        assert W.shape == (1, 20)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "many", np.nan])
    def test_a_nonsensical_patience_is_refused(self, bad):
        with pytest.raises(ValueError, match="support_patience"):
            fista_warm_start_batch(ridged_batch(S=2, J=10, T=5),
                                   support_patience=bad)

    def test_a_bool_patience_is_accepted_as_the_integer_it_is(self):
        """``True`` is an ``Integral`` equal to 1, so it is a patience of one.

        Asserted rather than left implicit because the scalar
        ``fista_warm_start`` takes it the same way, and the two validators are
        meant to agree.
        """
        W = fista_warm_start_batch(ridged_batch(S=2, J=10, T=5),
                                   support_patience=True)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)

    def test_none_patience_is_allowed(self):
        W = fista_warm_start_batch(ridged_batch(S=2, J=10, T=5),
                                   support_patience=None)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)

    def test_the_support_stop_shortens_a_long_budget(self, monkeypatch):
        """The patience rule, exercised where it can reach.

        At the default budget it cannot fire: patience is 100 sampled every 10,
        so it wants 100 iterations and the budget is 50. It is there for a
        caller that raises the budget, and this is that caller. Counted in
        projections, since that is one per iteration and the function reports no
        iteration count.
        """
        G = ridged_batch(S=6, J=40, T=14)
        counts = {}
        real = ACC.simplex_project_batch
        for label, patience in (("patience", 100), ("none", None)):
            n = []
            monkeypatch.setattr(ACC, "simplex_project_batch",
                                lambda V: (n.append(1) or real(V)))
            fista_warm_start_batch(G, max_iter=600, support_patience=patience)
            monkeypatch.undo()
            counts[label] = len(n)
        assert counts["patience"] < counts["none"], counts


# =========================================================================== #
# work assertions -- these start red
# =========================================================================== #
def corral_rows_solved(monkeypatch, fn):
    """Total rows passed through the corral solve -- ``sum(live)`` over the run.

    The work assertion counts this and not iterations. Iterations are the wrong
    metric here: a seeded batch can take the same number and still cost a
    fraction, because the candidates certify early so the arrays shrink, and the
    cost is ``sum(live * c^3)`` and not the iteration count. Measured on a
    ridged 500 x 40 family, cold and seeded both run 45 iterations and the
    seeded one is 3.6x faster.
    """
    rows = []
    real = np.linalg.solve
    monkeypatch.setattr(np.linalg, "solve",
                        lambda A, b: (rows.append(A.shape[0]) or real(A, b)))
    out = fn()
    monkeypatch.undo()
    return sum(rows), out


class TestWorkDone:
    def test_a_dense_support_batch_solves_far_fewer_corral_rows(self, monkeypatch):
        G = ridged_batch(S=40, J=90, T=30)
        cold_rows, (_, cold) = corral_rows_solved(
            monkeypatch,
            lambda: solve_simplex_minnorm_batch(G, accelerate=False,
                                                return_info=True))
        warm_rows, (_, warm) = corral_rows_solved(
            monkeypatch,
            lambda: solve_simplex_minnorm_batch(G, return_info=True))
        assert warm["converged"].all() and cold["converged"].all()
        assert warm_rows * 3 < cold_rows, (
            f"seeded solved {warm_rows} corral rows against the cold "
            f"{cold_rows}: the seed is not naming the support")

    def test_the_seed_is_what_shrinks_the_live_set(self, monkeypatch):
        """Live candidates fall away immediately when the support is supplied."""
        G = ridged_batch(S=40, J=90, T=30)
        _, (_, warm) = corral_rows_solved(
            monkeypatch, lambda: solve_simplex_minnorm_batch(G, return_info=True))
        _, (_, cold) = corral_rows_solved(
            monkeypatch,
            lambda: solve_simplex_minnorm_batch(G, accelerate=False,
                                                return_info=True))
        assert warm["iterations"] < cold["iterations"]

    def test_a_narrow_pool_is_not_seeded(self, monkeypatch):
        calls = []
        real = ACC.fista_warm_start_batch
        monkeypatch.setattr(
            "mlsynth.utils.bilevel.minnorm.fista_warm_start_batch",
            lambda G, **kw: (calls.append(G.shape) or real(G, **kw)))
        solve_simplex_minnorm_batch(ridged_batch(S=10, J=4, T=6))
        assert calls == []

    def test_a_caller_warm_start_is_not_overridden(self, monkeypatch):
        calls = []
        real = ACC.fista_warm_start_batch
        monkeypatch.setattr(
            "mlsynth.utils.bilevel.minnorm.fista_warm_start_batch",
            lambda G, **kw: (calls.append(G.shape) or real(G, **kw)))
        G = ridged_batch(S=10, J=90, T=30)
        solve_simplex_minnorm_batch(G, warm_start=np.full((10, 90), 1 / 90))
        assert calls == []

    def test_accelerate_false_disables_the_seed(self, monkeypatch):
        calls = []
        real = ACC.fista_warm_start_batch
        monkeypatch.setattr(
            "mlsynth.utils.bilevel.minnorm.fista_warm_start_batch",
            lambda G, **kw: (calls.append(G.shape) or real(G, **kw)))
        solve_simplex_minnorm_batch(ridged_batch(S=10, J=90, T=30), accelerate=False)
        assert calls == []


# =========================================================================== #
# the answer does not depend on the seed
# =========================================================================== #
BATCHES = {
    "ridged-dense": lambda: ridged_batch(S=30, J=70, T=25),
    "sparse": lambda: sparse_batch(S=20, J=90, T=200),
    "narrow": lambda: ridged_batch(S=15, J=6, T=20),
    "square": lambda: ridged_batch(S=12, J=40, T=40),
    "wide-ridge": lambda: ridged_batch(S=12, J=100, T=10, ridge=5.0),
    "tiny-ridge": lambda: ridged_batch(S=12, J=50, T=20, ridge=1e-8),
}


def scale_of(G):
    """The problem magnitude the solver's own KKT test is relative to."""
    return np.einsum("sjj->sj", G).mean(axis=1)


class TestAnswerUnchanged:
    @pytest.mark.parametrize("name", sorted(BATCHES))
    def test_seeded_and_unseeded_reach_the_same_optimum(self, name):
        """Same optimum, measured the way the solver measures optimality.

        The comparison is against the problem's own magnitude and not against
        the objective value, because the solver's optimality test is: a donor
        enters when it improves the fit by more than ``_KKT_TOL * scale``, with
        ``scale`` the mean diagonal of ``G``. On a near-singular family the
        optimum is a face whose objective is numerically zero -- the tiny-ridge
        batch here reaches 1e-10 against a scale of ~20 -- so two certified
        answers can sit 60 percent apart *relatively* while being the same
        answer by every standard the solver applies. Ratios of zeros are not the
        property under test.

        Where the optimum is not numerically zero the check is also made
        relatively, so the loose absolute bound cannot hide a real divergence.
        """
        G = BATCHES[name]()
        W_cold, cold = solve_simplex_minnorm_batch(G, accelerate=False,
                                                   return_info=True)
        W_warm, warm = solve_simplex_minnorm_batch(G, return_info=True)
        o_c, o_w = obj(G, W_cold), obj(G, W_warm)
        s = scale_of(G)
        gap = np.abs(o_w - o_c) / s
        assert gap.max() < 1e-8, f"objective moved by {gap.max():.2e} of scale"

        real = o_c > 1e-6 * s          # optima that are not numerically zero
        if real.any():
            rel = np.abs(o_w[real] - o_c[real]) / np.abs(o_c[real])
            assert rel.max() < 1e-9, f"objective moved by {rel.max():.2e}"
        assert warm["converged"].sum() >= cold["converged"].sum()

    def test_a_face_of_optima_is_not_a_divergence(self):
        """The near-singular case the scaled comparison above exists for."""
        G = ridged_batch(S=12, J=50, T=20, ridge=1e-8)
        W_cold, cold = solve_simplex_minnorm_batch(G, accelerate=False,
                                                   return_info=True)
        W_warm, warm = solve_simplex_minnorm_batch(G, return_info=True)
        assert cold["converged"].all() and warm["converged"].all()
        o_c, o_w = obj(G, W_cold), obj(G, W_warm)
        s = scale_of(G)
        assert (o_c / s < 1e-9).all(), "fixture must land on a near-zero optimum"
        # Certified means no donor improves the fit by more than the tolerance,
        # so a seeded answer is allowed to be better, and here it is.
        assert (o_w <= o_c + 1e-10 * s).all()

    @pytest.mark.parametrize("name", sorted(BATCHES))
    def test_the_returned_weights_are_feasible(self, name):
        W = solve_simplex_minnorm_batch(BATCHES[name]())
        assert (W >= 0).all()
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)


class TestAdversarialSeeds:
    """A seed may cost iterations. It may not move the answer."""

    @staticmethod
    def _bad_seeds(S, J, rng):
        yield "uniform", np.full((S, J), 1.0 / J)
        yield "wrong-vertex", np.eye(J)[rng.integers(0, J, S)]
        yield "random-face", rng.dirichlet(np.ones(J), size=S)
        one_hot = np.zeros((S, J)); one_hot[:, -1] = 1.0
        yield "last-donor", one_hot
        yield "not-normalised", rng.random((S, J)) * 7.0
        yield "wrong-shape", np.full((S + 3, J), 1.0 / J)
        yield "negative", -rng.random((S, J))
        yield "non-finite", np.full((S, J), np.nan)

    def test_a_bad_seed_never_moves_the_optimum(self):
        rng = np.random.default_rng(7)
        G = ridged_batch(S=12, J=50, T=18)
        S, J = G.shape[0], G.shape[-1]
        W_ref = solve_simplex_minnorm_batch(G, accelerate=False)
        o_ref = obj(G, W_ref)
        for label, seed in self._bad_seeds(S, J, rng):
            W, info = solve_simplex_minnorm_batch(G, warm_start=seed,
                                                  return_info=True)
            o = obj(G, W)
            rel = np.abs(o - o_ref) / np.maximum(np.abs(o_ref), 1e-300)
            assert rel.max() < 1e-9, f"{label} seed moved the objective by {rel.max():.2e}"
            assert (W >= 0).all() and np.allclose(W.sum(axis=1), 1.0, atol=1e-9), label

    def test_an_exhausted_budget_is_reported_not_hidden(self):
        G = ridged_batch(S=8, J=60, T=20)
        _, info = solve_simplex_minnorm_batch(G, max_iter=1, return_info=True)
        assert info["iterations"] == 1
        assert not info["converged"].all()


# =========================================================================== #
# the SDID programs this was found on
# =========================================================================== #
class TestSdidShapedFamily:
    def test_placebo_shaped_batch_agrees_with_the_cold_solve(self):
        """The (S, J, J) shape and ridge of an SDID placebo unit-weight family."""
        G = ridged_batch(S=100, J=99, T=35, ridge=0.7, seed=11)
        W_cold = solve_simplex_minnorm_batch(G, accelerate=False)
        W_warm, info = solve_simplex_minnorm_batch(G, return_info=True)
        o_c, o_w = obj(G, W_cold), obj(G, W_warm)
        np.testing.assert_allclose(o_w, o_c, rtol=1e-9)
        assert info["converged"].all()
        assert support_density(W_warm).mean() > 0.3, (
            "fixture must reproduce the dense-support regime the issue is about")
