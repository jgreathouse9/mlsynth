"""Multiple-right-hand-side simplex least squares.

A stacked synthetic-control design fits one weight vector per treated unit,
which looks like N independent problems. It is not: every treated unit in a
cohort solves against the *same* donor block ``A`` and differs only in its own
target ``b_j``. Stacking the targets as columns makes it one program with many
right-hand sides, so the Gram matrix and step size are formed once instead of
per unit -- 5.2s against 81.7s on the Walmart panel's 566 counties.

Two things these tests deliberately do NOT assert.

Batch and loop are not interchangeable on a rank-deficient design. With few
pre-treatment periods against many donors the optimum is a face rather than a
point, so two solvers can both reach it and land in different places. The
objectives must agree; the weights need not, and asserting they do would pin a
solver artefact.

The batched solver stops on the objective, while the scalar ``simplex_lstsq``
stops on the step norm. That is on purpose -- along a flat optimal face the
iterate keeps moving after the objective has settled, so a step-norm rule never
fires and the solver burns its whole iteration budget.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.simplex import project_simplex, simplex_lstsq


def _rank_deficient(T0=5, J=39, k=40, seed=0):
    """Fewer rows than donors: the Walmart cohorts' shape, null space >= 34.

    The targets are offset well outside the donors' convex hull. Without that
    the simplex fit is exact, every objective is zero, and comparing them says
    nothing -- the residuals differ only in their last bits.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T0, J)), rng.normal(size=(T0, k)) + 8.0


def _well_conditioned(m=30, n=8, k=25, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, n)), rng.normal(size=(m, k))


# --------------------------------------------------------------- projection
class TestTheBatchedProjection:
    @pytest.mark.parametrize("n,k,scale", [(1, 7, 1.0), (2, 5, 10.0),
                                           (39, 200, 1e3), (100, 50, 1e-3)])
    def test_it_agrees_with_the_scalar_version_column_by_column(self, n, k, scale):
        from mlsynth.utils.bilevel.simplex import project_simplex_cols

        rng = np.random.default_rng(n * 100 + k)
        V = rng.normal(size=(n, k)) * scale
        batched = project_simplex_cols(V)
        scalar = np.column_stack([project_simplex(V[:, j]) for j in range(k)])
        np.testing.assert_allclose(batched, scalar, atol=1e-12)

    def test_every_column_lands_on_the_simplex(self):
        from mlsynth.utils.bilevel.simplex import project_simplex_cols

        rng = np.random.default_rng(2)
        P = project_simplex_cols(rng.normal(size=(12, 40)) * 100.0)
        assert (P >= -1e-12).all()
        np.testing.assert_allclose(P.sum(axis=0), 1.0, atol=1e-12)

    def test_a_point_already_on_the_simplex_is_left_alone(self):
        from mlsynth.utils.bilevel.simplex import project_simplex_cols

        rng = np.random.default_rng(3)
        W = rng.dirichlet(np.ones(6), size=9).T          # (6, 9)
        np.testing.assert_allclose(project_simplex_cols(W), W, atol=1e-12)

    def test_the_radius_is_honoured(self):
        from mlsynth.utils.bilevel.simplex import project_simplex_cols

        rng = np.random.default_rng(4)
        P = project_simplex_cols(rng.normal(size=(5, 8)), z=3.0)
        np.testing.assert_allclose(P.sum(axis=0), 3.0, atol=1e-12)

    def test_a_single_row_is_the_degenerate_branch(self):
        from mlsynth.utils.bilevel.simplex import project_simplex_cols

        rng = np.random.default_rng(5)
        P = project_simplex_cols(rng.normal(size=(1, 6)))
        np.testing.assert_allclose(P, np.ones((1, 6)), atol=1e-12)


# ------------------------------------------------------------ least squares
class TestTheBatchedSolve:
    def test_it_matches_the_loop_on_a_well_conditioned_design(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _well_conditioned()
        W = simplex_lstsq_batch(A, B)
        loop = np.column_stack([simplex_lstsq(A, B[:, j], max_iter=20000,
                                              tol=1e-12)
                                for j in range(B.shape[1])])
        np.testing.assert_allclose(W, loop, atol=1e-4)

    def test_on_a_rank_deficient_design_the_objectives_agree(self):
        """Not the weights. The optimum is a face, so where on it a solver
        lands is an artefact; the loss is the identified quantity."""
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _rank_deficient()
        assert np.linalg.matrix_rank(A) < A.shape[1]     # the premise
        W = simplex_lstsq_batch(A, B)
        loop = np.column_stack([simplex_lstsq(A, B[:, j], max_iter=20000,
                                              tol=1e-12)
                                for j in range(B.shape[1])])
        obj_b = ((A @ W - B) ** 2).sum(axis=0)
        obj_l = ((A @ loop - B) ** 2).sum(axis=0)
        # Guard against the comparison going vacuous: if the fit were exact the
        # objectives would both be ~0 and agreeing would mean nothing.
        assert obj_b.min() > 1.0
        np.testing.assert_allclose(obj_b, obj_l, rtol=1e-3)

    def test_the_solution_is_on_the_simplex(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _well_conditioned()
        W = simplex_lstsq_batch(A, B)
        assert (W >= -1e-9).all()
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-8)

    def test_it_beats_uniform_weights(self):
        """A minimiser must not do worse than the point it starts from."""
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _well_conditioned()
        W = simplex_lstsq_batch(A, B)
        U = np.full_like(W, 1.0 / A.shape[1])
        assert (((A @ W - B) ** 2).sum(0) <= ((A @ U - B) ** 2).sum(0) + 1e-9).all()

    def test_the_ridge_shrinks_toward_uniform(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _rank_deficient()
        uniform = np.full(A.shape[1], 1.0 / A.shape[1])
        plain = simplex_lstsq_batch(A, B, ridge=0.0)
        heavy = simplex_lstsq_batch(A, B, ridge=1e3)
        d_plain = np.abs(plain - uniform[:, None]).max()
        d_heavy = np.abs(heavy - uniform[:, None]).max()
        assert d_heavy < d_plain

    def test_one_column_agrees_with_the_scalar_solver(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _well_conditioned(k=1)
        W = simplex_lstsq_batch(A, B)
        assert W.shape == (A.shape[1], 1)
        np.testing.assert_allclose(
            W[:, 0], simplex_lstsq(A, B[:, 0], max_iter=20000, tol=1e-12),
            atol=1e-4)

    def test_a_single_donor_takes_everything(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        rng = np.random.default_rng(7)
        W = simplex_lstsq_batch(rng.normal(size=(6, 1)), rng.normal(size=(6, 4)))
        np.testing.assert_allclose(W, np.ones((1, 4)), atol=1e-12)

    def test_a_one_dimensional_target_is_accepted_as_one_column(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        A, B = _well_conditioned(k=1)
        np.testing.assert_allclose(simplex_lstsq_batch(A, B[:, 0]),
                                   simplex_lstsq_batch(A, B), atol=1e-12)


# ---------------------------------------------------------------- failures
class TestFailuresAreReported:
    def test_mismatched_shapes_are_rejected(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_batch

        rng = np.random.default_rng(8)
        with pytest.raises(ValueError):
            simplex_lstsq_batch(rng.normal(size=(6, 3)), rng.normal(size=(5, 4)))

    def test_a_negative_radius_is_rejected(self):
        from mlsynth.utils.bilevel.simplex import project_simplex_cols

        with pytest.raises(ValueError):
            project_simplex_cols(np.zeros((3, 2)), z=-1.0)
