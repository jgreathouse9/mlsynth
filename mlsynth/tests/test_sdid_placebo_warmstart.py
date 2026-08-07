"""Warm-start chaining through the batched solver's one-at-a-time fallback.

``solve_intercept_simplex_many`` batches a family of simplex programs, but a
design that fails ``gram_reduction_is_safe`` is solved one at a time instead.
SDID's time-weight program is exactly that case -- its design is ``(N0, T0)``
with ``T0 > N0``, so it is rank deficient and always falls back -- and it is also
the pivot-heavy half of the placebo loop, since it carries ``T0`` variables
against the unit program's ``N0``.

Consecutive placebo draws differ by only the treated columns, so the previous
draw's solution is a near-feasible active set for the next. These tests pin the
contract that chaining it changes the cost and not the answer.
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.minnorm import gram_reduction_is_safe
from mlsynth.utils.sdid_helpers.weights import (
    _solve_intercept_simplex,
    fit_time_weights,
    solve_intercept_simplex_many,
)


def _objective(design, target, intercept, weights, ridge=0.0):
    """``||a + design @ w - target||^2 + ridge * ||w||^2``."""
    resid = intercept + design @ weights - target
    return float(resid @ resid + ridge * (weights @ weights))


def _time_weight_family(n_draws=12, n_donors=14, n_pre=20, seed=0):
    """A family of SDID time-weight programs, as the placebo loop poses them.

    Design is ``(n_donors, n_pre)`` -- rank deficient whenever
    ``n_pre > n_donors``, which is the fallback case under test.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_pre + 8, n_donors + 2)).cumsum(axis=0)
    problems = []
    for d in range(n_draws):
        idx = rng.choice(n_donors + 2, size=2, replace=False)
        mask = np.ones(n_donors + 2, dtype=bool)
        mask[idx] = False
        pre = base[:n_pre][:, mask]
        post_mean = base[n_pre:][:, mask].mean(axis=0)
        design = pre.T
        problems.append((design - design.mean(axis=0)[None, :],
                         post_mean - post_mean.mean(), 0.0))
    return problems


class TestFallbackIsTheCaseUnderTest:
    """The premise: SDID time-weight designs really do bypass the batch."""

    def test_time_weight_design_is_rank_deficient(self):
        problems = _time_weight_family()
        design = problems[0][0]
        assert design.shape[1] > design.shape[0], (
            "time-weight design should have more variables than equations")
        assert not gram_reduction_is_safe(design), (
            "this family must take the one-at-a-time fallback, or the test "
            "is not exercising the path it claims to")

    def test_unit_weight_design_is_batched(self):
        rng = np.random.default_rng(1)
        pre = rng.normal(size=(30, 8)).cumsum(axis=0)
        centred = pre - pre.mean(axis=0)[None, :]
        augmented = np.vstack([centred, np.sqrt(4.0) * np.eye(8)])
        assert gram_reduction_is_safe(augmented)


class TestChainingPreservesTheAnswer:
    """Chaining must change the cost, not the result."""

    def test_matches_cold_one_at_a_time(self):
        problems = _time_weight_family()
        cold = [_solve_intercept_simplex(d, t, r) for d, t, r in problems]
        got = solve_intercept_simplex_many(problems)
        assert len(got) == len(cold)
        for (a_c, w_c), (a_g, w_g) in zip(cold, got):
            np.testing.assert_allclose(w_g, w_c, atol=1e-8)
            assert a_g == pytest.approx(a_c, abs=1e-8)

    def test_objective_is_not_worsened(self):
        problems = _time_weight_family(seed=3)
        cold = [_solve_intercept_simplex(d, t, r) for d, t, r in problems]
        got = solve_intercept_simplex_many(problems)
        for (design, target, ridge), (a_c, w_c), (a_g, w_g) in zip(
                problems, cold, got):
            obj_cold = _objective(design, target, a_c, w_c, ridge)
            obj_warm = _objective(design, target, a_g, w_g, ridge)
            scale = max(1.0, abs(obj_cold))
            assert obj_warm <= obj_cold + 1e-8 * scale

    def test_weights_stay_on_the_simplex(self):
        for _, w in solve_intercept_simplex_many(_time_weight_family(seed=5)):
            assert w.min() >= -1e-9
            assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_repeated_calls_are_deterministic(self):
        problems = _time_weight_family(seed=7)
        first = solve_intercept_simplex_many(problems)
        second = solve_intercept_simplex_many(problems)
        for (_, a), (_, b) in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_fit_time_weights_unchanged(self):
        """The public entry point must be unmoved by the internal change."""
        rng = np.random.default_rng(11)
        pre = rng.normal(size=(24, 9)).cumsum(axis=0)
        post_mean = rng.normal(size=9)
        intercept, lam = fit_time_weights(pre, post_mean)
        expect_a, expect_w = _solve_intercept_simplex(pre.T, post_mean, 0.0)
        np.testing.assert_allclose(lam, expect_w, atol=1e-9)
        assert intercept == pytest.approx(expect_a, abs=1e-9)


class TestEdgeCases:
    def test_single_problem_has_no_chain_to_build(self):
        problems = _time_weight_family(n_draws=1)
        got = solve_intercept_simplex_many(problems)
        cold = _solve_intercept_simplex(*problems[0])
        np.testing.assert_allclose(got[0][1], cold[1], atol=1e-9)

    def test_mixed_shapes_do_not_cross_contaminate(self):
        """A warm start of the wrong length must never be applied."""
        a = _time_weight_family(n_draws=3, n_donors=14, n_pre=20, seed=2)
        b = _time_weight_family(n_draws=3, n_donors=10, n_pre=16, seed=4)
        mixed = [a[0], b[0], a[1], b[1], a[2], b[2]]
        got = solve_intercept_simplex_many(mixed)
        for (design, target, ridge), (_, w) in zip(mixed, got):
            assert w.shape[0] == design.shape[1]
            cold = _solve_intercept_simplex(design, target, ridge)[1]
            np.testing.assert_allclose(w, cold, atol=1e-8)

    def test_single_column_problem(self):
        design = np.array([[2.0], [3.0], [4.0]])
        target = np.array([1.0, 2.0, 3.0])
        (_, w), = solve_intercept_simplex_many([(design, target, 0.0)])
        np.testing.assert_allclose(w, [1.0])

    def test_empty_family(self):
        assert solve_intercept_simplex_many([]) == []


class TestWarmStartRobustness:
    """A warm start is a hint; a bad one must not corrupt the solve."""

    @pytest.mark.parametrize("bad", ["nan", "negative", "unnormalised",
                                     "wrong_length"])
    def test_infeasible_warm_start_is_ignored(self, bad):
        rng = np.random.default_rng(13)
        design = rng.normal(size=(12, 20))
        target = rng.normal(size=12)
        reference = solve_simplex_qp(design, target)
        seeds = {
            "nan": np.full(20, np.nan),
            "negative": np.full(20, -1.0),
            "unnormalised": np.full(20, 5.0),
            "wrong_length": np.full(7, 1.0 / 7),
        }
        got = solve_simplex_qp(design, target, warm_start=seeds[bad])
        np.testing.assert_allclose(got, reference, atol=1e-9)

    def test_degenerate_duplicate_columns_keep_the_same_fit(self):
        """Duplicated columns make the argmin non-unique.

        The weights may then differ between starting points -- the "same fit,
        other weights" case -- but the fitted values must not.
        """
        rng = np.random.default_rng(17)
        design = rng.normal(size=(10, 16))
        design[:, 8:] = design[:, :8]
        target = rng.normal(size=10)
        cold = solve_simplex_qp(design, target)
        neighbour = solve_simplex_qp(
            design + 1e-3 * rng.normal(size=design.shape), target)
        warm = solve_simplex_qp(design, target, warm_start=neighbour)
        np.testing.assert_allclose(design @ warm, design @ cold, atol=1e-8)

    def test_rejects_malformed_design(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros((0, 3)), np.zeros(3))


class TestChainingReducesWork:
    """The point of the change: fewer active-set pivots, same answer.

    Counted through the solver's own ``return_info`` diagnostics, so the
    assertion is on work done and not on wall-clock.
    """

    @staticmethod
    def _pivots(problems):
        """Total pivots spent solving ``problems`` through the batched entry."""
        import mlsynth.utils.sdid_helpers.weights as weights_module

        seen = []
        original = weights_module.solve_simplex_qp

        def counting(B, A, **kwargs):
            w, info = original(B, A, **{**kwargs, "return_info": True})
            seen.append(info["pivots"])
            return w

        weights_module.solve_simplex_qp = counting
        try:
            solve_intercept_simplex_many(problems)
        finally:
            weights_module.solve_simplex_qp = original
        return sum(seen), len(seen)

    def test_chain_costs_fewer_pivots_than_cold_starts(self):
        problems = _time_weight_family(n_draws=16, seed=21)
        chained, n_solves = self._pivots(problems)
        assert n_solves == len(problems), (
            "every problem in this family should hit the fallback solver")

        cold = sum(
            solve_simplex_qp(
                design - design.mean(axis=0)[None, :],
                target - target.mean(),
                return_info=True,
            )[1]["pivots"]
            for design, target, _ in problems
        )
        assert chained < cold, (
            f"chained fallback spent {chained} pivots, cold spent {cold}: "
            "the warm start is not being threaded through the fallback")

    def test_first_problem_is_unaffected(self):
        """Nothing to chain from, so the first solve is the cold solve."""
        problems = _time_weight_family(n_draws=4, seed=23)
        design, target, ridge = problems[0]
        got = solve_intercept_simplex_many(problems)[0]
        cold = _solve_intercept_simplex(design, target, ridge)
        np.testing.assert_allclose(got[1], cold[1], atol=1e-9)
