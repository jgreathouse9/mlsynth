"""Many-design simplex least squares, stated in Gram form.

``simplex_lstsq_batch`` already handles one shared design against many targets.
The forward-selection scan is the transpose of that situation: one shared
target (the treated unit) against many *different* designs, one per candidate
donor. What makes it batchable anyway is that FISTA never touches the design --
only ``A'A`` and ``A'b`` -- so a stack of Gram systems is enough, and the
candidate Grams are all submatrices of one donor Gram.

The contract asserted here is agreement with the scalar ``simplex_lstsq``,
problem by problem, plus the simplex geometry every row has to satisfy.

One thing these tests deliberately do NOT assert. On a rank-deficient Gram the
optimum is a face, not a point, so the batched and scalar solvers may stop at
different members of it. The objectives must agree; the weights need not, and
pinning them would pin a solver artefact -- the same stance
``test_bilevel_simplex_batch`` takes for the shared-design solver.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.bilevel.simplex import (
    _lipschitz_rows,
    _project_simplex_rows,
    project_simplex,
    simplex_lstsq,
    simplex_lstsq_gram_many,
)


def _grams(designs, b):
    """Stack ``(A'A, A'b)`` for a list of designs sharing the target ``b``."""
    AtA = np.stack([A.T @ A for A in designs])
    Atb = np.stack([A.T @ b for A in designs])
    return AtA, Atb


def _objective(AtA, Atb, W, bb):
    """``||A w - b||^2`` for each row, computed in Gram space."""
    return (bb - 2.0 * np.einsum("mi,mi->m", W, Atb)
            + np.einsum("mi,mij,mj->m", W, AtA, W))


def _candidates(T0=24, n_sel=3, M=9, seed=0):
    """A shared selected block plus one distinct extra column per problem."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(T0, n_sel))
    extra = rng.normal(size=(T0, M))
    designs = [np.column_stack([base, extra[:, j]]) for j in range(M)]
    return designs, rng.normal(size=T0)


# ------------------------------------------------- the row-wise projection
class TestTheRowProjection:
    """The batched twin of ``project_simplex``, asserted against it directly.

    ``simplex_lstsq_gram_many`` short-circuits the one-donor case before the
    projection is reached, so its degenerate branch has no route in from there.
    """

    @pytest.mark.parametrize("n,M,scale", [(1, 6, 1.0), (2, 4, 10.0),
                                           (9, 30, 1e3), (40, 7, 1e-3)])
    def test_it_agrees_with_the_scalar_version_row_by_row(self, n, M, scale):
        rng = np.random.default_rng(n * 31 + M)
        V = rng.normal(size=(M, n)) * scale
        got = _project_simplex_rows(V)
        for m in range(M):
            assert np.allclose(got[m], project_simplex(V[m]), atol=1e-12)

    def test_a_single_column_is_the_degenerate_branch(self):
        got = _project_simplex_rows(np.array([[-4.0], [0.0], [9.0]]))
        assert np.allclose(got, 1.0)

    def test_a_point_already_on_the_simplex_is_left_alone(self):
        V = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        assert np.allclose(_project_simplex_rows(V), V, atol=1e-12)

    def test_ties_take_the_last_qualifying_index(self):
        """Where several indices satisfy the threshold condition the scalar
        projection takes the last; taking the first lands on the wrong face."""
        V = np.array([[0.5, 0.5, 0.5, 0.5]])
        assert np.allclose(_project_simplex_rows(V)[0], project_simplex(V[0]))


class TestTheStepSize:
    def test_it_matches_the_scalar_lipschitz_constant(self):
        from mlsynth.utils.bilevel.simplex import _lipschitz_constant

        rng = np.random.default_rng(21)
        designs = [rng.normal(size=(25, 4)) for _ in range(5)]
        got = _lipschitz_rows(np.stack([A.T @ A for A in designs]))
        want = np.array([_lipschitz_constant(A) for A in designs])
        assert np.allclose(got, want, rtol=1e-8)

    def test_a_zero_gram_does_not_divide_by_zero(self):
        out = _lipschitz_rows(np.zeros((3, 4, 4)))
        assert np.isfinite(out).all()
        assert (out > 0).all()


# ------------------------------------------------------------------- smoke
class TestItRuns:
    def test_the_smallest_possible_batch(self):
        AtA = np.ones((1, 1, 1))
        Atb = np.ones((1, 1))
        W = simplex_lstsq_gram_many(AtA, Atb)
        assert W.shape == (1, 1)
        assert np.isfinite(W).all()

    def test_a_realistic_batch_returns_one_row_per_problem(self):
        designs, b = _candidates()
        AtA, Atb = _grams(designs, b)
        W = simplex_lstsq_gram_many(AtA, Atb)
        assert W.shape == (len(designs), designs[0].shape[1])
        assert np.isfinite(W).all()


# -------------------------------------------------------------------- unit
class TestItAgreesWithTheScalarSolver:
    @pytest.mark.parametrize("T0,n_sel,M", [(24, 3, 9), (40, 1, 5), (15, 6, 12)])
    def test_weights_match_problem_by_problem(self, T0, n_sel, M):
        designs, b = _candidates(T0, n_sel, M, seed=T0 + M)
        AtA, Atb = _grams(designs, b)
        W = simplex_lstsq_gram_many(AtA, Atb)
        for j, A in enumerate(designs):
            assert np.allclose(W[j], simplex_lstsq(A, b), atol=1e-8)

    def test_on_a_rank_deficient_gram_the_objectives_agree(self):
        # More donors than rows: the optimum is a face, so compare objectives.
        rng = np.random.default_rng(3)
        base = rng.normal(size=(5, 20))
        designs = [np.column_stack([base, rng.normal(size=5)]) for _ in range(6)]
        b = rng.normal(size=5) + 8.0
        AtA, Atb = _grams(designs, b)
        W = simplex_lstsq_gram_many(AtA, Atb)
        got = _objective(AtA, Atb, W, float(b @ b))
        want = np.array([float(np.sum((A @ simplex_lstsq(A, b) - b) ** 2))
                         for A in designs])
        assert np.allclose(got, want, atol=1e-6)

    def test_one_problem_is_the_scalar_solver(self):
        rng = np.random.default_rng(5)
        A = rng.normal(size=(30, 4))
        b = rng.normal(size=30)
        W = simplex_lstsq_gram_many(A.T @ A, (A.T @ b))
        assert np.allclose(W[0], simplex_lstsq(A, b), atol=1e-8)


# ---------------------------------------------------------------- geometry
class TestTheGeometry:
    def test_every_row_lands_on_the_simplex(self):
        designs, b = _candidates(seed=7)
        AtA, Atb = _grams(designs, b)
        W = simplex_lstsq_gram_many(AtA, Atb)
        assert (W >= -1e-12).all()
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-9)

    def test_it_beats_uniform_weights(self):
        designs, b = _candidates(seed=8)
        AtA, Atb = _grams(designs, b)
        W = simplex_lstsq_gram_many(AtA, Atb)
        n = W.shape[1]
        U = np.full_like(W, 1.0 / n)
        bb = float(b @ b)
        assert (_objective(AtA, Atb, W, bb)
                <= _objective(AtA, Atb, U, bb) + 1e-9).all()


# ------------------------------------------------------------- edge cases
class TestDegenerateInput:
    def test_a_single_donor_takes_everything(self):
        rng = np.random.default_rng(9)
        A = rng.normal(size=(12, 1))
        b = rng.normal(size=12)
        W = simplex_lstsq_gram_many((A.T @ A)[None], (A.T @ b)[None])
        assert W.shape == (1, 1)
        assert W[0, 0] == pytest.approx(1.0)

    def test_an_empty_batch_returns_an_empty_result(self):
        W = simplex_lstsq_gram_many(np.empty((0, 3, 3)), np.empty((0, 3)))
        assert W.shape == (0, 3)

    def test_a_dead_donor_contributes_a_zero_row_and_column(self):
        # A column of zeros makes the Gram singular; the solve must still land
        # on the simplex rather than emit NaN.
        rng = np.random.default_rng(10)
        A = rng.normal(size=(20, 3))
        A[:, 2] = 0.0
        b = rng.normal(size=20)
        W = simplex_lstsq_gram_many((A.T @ A)[None], (A.T @ b)[None])
        assert np.isfinite(W).all()
        assert W.sum() == pytest.approx(1.0)

    def test_duplicated_problems_get_identical_rows(self):
        rng = np.random.default_rng(11)
        A = rng.normal(size=(18, 4))
        b = rng.normal(size=18)
        AtA = np.stack([A.T @ A] * 4)
        Atb = np.stack([A.T @ b] * 4)
        W = simplex_lstsq_gram_many(AtA, Atb)
        assert np.allclose(W, W[0], atol=1e-12)

    def test_collinear_donors_still_produce_simplex_weights(self):
        rng = np.random.default_rng(12)
        A = rng.normal(size=(20, 3))
        A[:, 1] = A[:, 0]                       # exact duplicate donor
        b = rng.normal(size=20)
        W = simplex_lstsq_gram_many((A.T @ A)[None], (A.T @ b)[None])
        assert np.isfinite(W).all()
        assert W.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- failures
class TestBadInputIsReported:
    def test_a_non_square_gram_is_rejected(self):
        with pytest.raises(MlsynthEstimationError, match="square"):
            simplex_lstsq_gram_many(np.zeros((2, 3, 4)), np.zeros((2, 3)))

    def test_mismatched_batch_lengths_are_rejected(self):
        with pytest.raises(MlsynthEstimationError, match="same number"):
            simplex_lstsq_gram_many(np.zeros((3, 2, 2)), np.zeros((2, 2)))

    def test_a_target_of_the_wrong_width_is_rejected(self):
        with pytest.raises(MlsynthEstimationError, match="width"):
            simplex_lstsq_gram_many(np.zeros((2, 3, 3)), np.zeros((2, 4)))

    def test_a_non_finite_gram_is_rejected(self):
        AtA = np.zeros((1, 2, 2))
        AtA[0, 0, 0] = np.nan
        with pytest.raises(MlsynthEstimationError, match="finite"):
            simplex_lstsq_gram_many(AtA, np.zeros((1, 2)))

    def test_a_wrongly_ranked_gram_is_rejected(self):
        with pytest.raises(MlsynthEstimationError, match="3-D"):
            simplex_lstsq_gram_many(np.zeros((2, 2, 2, 2)), np.zeros((2, 2)))
