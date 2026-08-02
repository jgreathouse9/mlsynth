"""Leave-one-out simplex least squares: the in-space placebo test's shape.

An in-space placebo test refits the synthetic control with each donor cast as
treated against the others. That looks like ``J`` independent programs. It is
not: every one of them is the *same* matrix with one column deleted, and each
one's target is itself a column of that matrix. So a single Gram ``M'M`` yields
every subproblem's Gram (a submatrix) and every subproblem's ``A'b`` (a column
of the same Gram), and no product with the data is needed inside the iteration
at all.

Running the family as one masked multiple-right-hand-side program instead of a
loop is 28x on the Walmart panel's cohort shape -- 18.0s to 0.65s for one
cohort's 39 placebos, with the objectives within 0.01 percent.

Two things these tests deliberately do NOT assert, following
``test_bilevel_simplex_batch.py``.

Weights are not comparable across solvers on a rank-deficient design. With
fewer rows than columns the optimum is a face, so two solvers can both reach it
and land in different places. Objectives must agree; weights need not.

Nor is the excluded column's absence a matter of degree. It is exactly zero, so
that is asserted exactly rather than to a tolerance.

The border
----------

``M``'s columns past ``n_targets`` are donors that are never themselves cast as
treated. That is what the permutation form of a placebo test needs: the treated
unit joins every placebo's donor pool but is not itself a placebo.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.simplex import (project_simplex,
                                           project_simplex_cols,
                                           simplex_lstsq)


def _rank_deficient(m=6, n=40, seed=0):
    """Fewer rows than columns: the Walmart cohorts' shape, null space >= 34."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(m, 3))
    return 100.0 + F @ rng.normal(size=(3, n)) + 0.5 * rng.normal(size=(m, n))


def _well_conditioned(m=40, n=6, seed=1):
    """More rows than columns and a target outside the hull, so the optimum is
    a point and weights are comparable across solvers."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, n))


def _loop(M, n_targets):
    """The reference: one ``simplex_lstsq`` per left-out column."""
    n = M.shape[1]
    W = np.zeros((n, n_targets))
    for j in range(n_targets):
        keep = [k for k in range(n) if k != j]
        W[keep, j] = simplex_lstsq(M[:, keep], M[:, j],
                                   max_iter=20000, tol=1e-11)
    return W


def _objectives(M, W, n_targets):
    R = M @ W - M[:, :n_targets]
    return (R * R).sum(axis=0)


# ------------------------------------------------------------- the mask
class TestTheMaskedProjection:
    def test_masked_coordinates_come_out_exactly_zero(self):
        rng = np.random.default_rng(0)
        V = rng.normal(size=(8, 5)) * 50.0
        mask = rng.random((8, 5)) < 0.4
        mask[:, 2] = False                       # leave one column unmasked
        W = project_simplex_cols(V, mask=mask)
        assert (W[mask] == 0.0).all()

    def test_the_unmasked_block_is_still_on_the_simplex(self):
        rng = np.random.default_rng(1)
        V = rng.normal(size=(8, 5))
        mask = np.zeros((8, 5), bool)
        mask[0, :] = True
        W = project_simplex_cols(V, mask=mask)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-12)
        assert (W >= -1e-12).all()

    def test_masking_a_row_is_the_same_as_deleting_it(self):
        """The point of the mask: it must not perturb the projection of what
        is left, only remove what is masked."""
        rng = np.random.default_rng(2)
        V = rng.normal(size=(9, 4)) * 3.0
        mask = np.zeros((9, 4), bool)
        mask[3, :] = True
        masked = project_simplex_cols(V, mask=mask)
        deleted = project_simplex_cols(np.delete(V, 3, axis=0))
        np.testing.assert_allclose(np.delete(masked, 3, axis=0), deleted,
                                   atol=1e-12)

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3, 1e9])
    def test_it_does_not_depend_on_the_data_scale(self, scale):
        """A sentinel value big enough for one panel and too small for another
        would be a latent bug, so the threshold is derived from the column."""
        rng = np.random.default_rng(3)
        V = rng.normal(size=(7, 3)) * scale
        mask = np.zeros((7, 3), bool)
        mask[1, :] = mask[5, :] = True
        W = project_simplex_cols(V, mask=mask)
        assert (W[mask] == 0.0).all()
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-10)
        keep = [0, 2, 3, 4, 6]
        np.testing.assert_allclose(
            W[keep], project_simplex_cols(V[keep]), atol=1e-10 * max(scale, 1))

    def test_no_mask_is_the_old_behaviour(self):
        rng = np.random.default_rng(4)
        V = rng.normal(size=(6, 4))
        np.testing.assert_array_equal(project_simplex_cols(V),
                                      project_simplex_cols(V, mask=None))

    def test_a_fully_masked_column_is_an_error(self):
        """There is no projection onto an empty simplex, and returning zeros
        would silently violate the sum constraint."""
        V = np.zeros((4, 2))
        mask = np.zeros((4, 2), bool)
        mask[:, 1] = True
        with pytest.raises(ValueError, match="mask"):
            project_simplex_cols(V, mask=mask)

    def test_a_mismatched_mask_is_an_error(self):
        with pytest.raises(ValueError, match="shape"):
            project_simplex_cols(np.zeros((4, 2)), mask=np.zeros((3, 2), bool))


# ------------------------------------------------------------- smoke
class TestSmoke:
    def test_it_returns_one_column_per_target(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient(n=12)
        W = simplex_lstsq_loo(M)
        assert W.shape == (12, 12)

    def test_every_column_is_on_the_simplex(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        W = simplex_lstsq_loo(_rank_deficient(n=12))
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-9)
        assert (W >= -1e-12).all()

    def test_the_left_out_column_gets_exactly_no_weight(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        W = simplex_lstsq_loo(_rank_deficient(n=12))
        assert (np.diag(W) == 0.0).all()


# ------------------------------------------------------------- correctness
class TestItSolvesTheSameProblem:
    def test_the_masked_solution_is_feasible_for_the_deleted_problem(self):
        """The exact form of "masking is deletion".

        Two approximate solvers on a flat design cannot be compared tightly, so
        the claim is made structurally instead: the weights this returns, read
        against the explicitly deleted design, give bit-for-bit the objective
        they give against the full one. That can only hold if the excluded
        column contributes nothing.
        """
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient(n=10)
        W = simplex_lstsq_loo(M)
        for j in range(10):
            keep = [k for k in range(10) if k != j]
            np.testing.assert_allclose(M @ W[:, j], M[:, keep] @ W[keep, j],
                                       rtol=0, atol=1e-12)
            assert W[j, j] == 0.0                    # exactly
            assert W[keep, j].sum() == pytest.approx(1.0, abs=1e-9)

    def test_it_does_no_worse_than_solving_each_deletion_separately(self):
        """The comparison that survives non-identification.

        Where the optimum is a face, two solvers land in different places and
        neither is wrong -- but one can still be stopped earlier than the other,
        and batching is where that goes wrong. Judging the *summed* objective
        lets a column that has plateaued be stopped while its neighbours are
        still improving; on the Walmart cohort shape that cost up to 3 percent
        of an individual column's fit. Hence the per-column rule, and hence this
        test, which is what would catch a regression back to the summed one.

        The claim is one-sided and absolute: no meaningfully worse. It cannot be
        relative, because a column that already fits to 1e-6 makes any ratio
        meaningless.
        """
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient(n=10)
        mine = _objectives(M, simplex_lstsq_loo(M), 10)
        loop = _objectives(M, _loop(M, 10), 10)
        assert (mine <= loop + 1e-3).all()
        assert mine.sum() == pytest.approx(loop.sum(), rel=1e-3)

    def test_on_a_unique_optimum_the_weights_match_the_loop(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _well_conditioned()
        np.testing.assert_allclose(simplex_lstsq_loo(M), _loop(M, M.shape[1]),
                                   atol=1e-5)

    def test_on_a_flat_optimum_the_two_reach_the_same_total_fit(self):
        """The design the placebo test actually faces. Weights are not
        identified there, so pinning them would pin a solver artefact -- and
        per-column ratios are meaningless where a column fits to 1e-6. The
        total is what a permutation distribution is built from.
        """
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient()
        n = M.shape[1]
        mine = _objectives(M, simplex_lstsq_loo(M), n).sum()
        loop = _objectives(M, _loop(M, n), n).sum()
        assert mine == pytest.approx(loop, rel=2e-3)

    def test_a_column_that_reproduces_itself_from_a_twin_finds_it(self):
        """A donor with an exact duplicate elsewhere must be reconstructed by
        that duplicate, which is the one case where the optimum is a corner and
        so is identified."""
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        rng = np.random.default_rng(7)
        M = rng.normal(size=(10, 5))
        M[:, 4] = M[:, 0]                        # column 4 twins column 0
        W = simplex_lstsq_loo(M)
        assert W[4, 0] == pytest.approx(1.0, abs=1e-6)
        assert W[0, 4] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------- the border
class TestTheBorder:
    def test_columns_past_n_targets_are_donors_but_never_targets(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient(n=10)
        W = simplex_lstsq_loo(M, n_targets=7)
        assert W.shape == (10, 7)
        assert (np.diag(W[:7, :7]) == 0.0).all()

    def test_a_border_column_can_carry_weight(self):
        """The permutation form needs the treated unit available to every
        placebo. A border that could never be used would make the option
        vacuous."""
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        rng = np.random.default_rng(11)
        M = rng.normal(size=(12, 5))
        M[:, 4] = M[:, 0] + 1e-9 * rng.normal(size=12)   # near-twin of col 0
        W = simplex_lstsq_loo(M, n_targets=4)
        assert W[4, 0] > 0.99

    def test_the_default_is_every_column(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient(n=8)
        np.testing.assert_array_equal(simplex_lstsq_loo(M),
                                      simplex_lstsq_loo(M, n_targets=8))


# ------------------------------------------------------------- edges
class TestEdgeCases:
    def test_two_columns_leaves_a_single_donor(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        W = simplex_lstsq_loo(_rank_deficient(m=5, n=2))
        np.testing.assert_allclose(W, [[0.0, 1.0], [1.0, 0.0]], atol=1e-12)

    def test_one_target_against_a_border(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        W = simplex_lstsq_loo(_rank_deficient(m=5, n=4), n_targets=1)
        assert W.shape == (4, 1)
        assert W[0, 0] == 0.0
        assert W.sum() == pytest.approx(1.0, abs=1e-9)

    def test_a_ridge_shrinks_the_weights_toward_uniform(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        M = _rank_deficient(n=12)
        plain = simplex_lstsq_loo(M)
        ridged = simplex_lstsq_loo(M, ridge=1.0)
        spread = lambda W: float(np.mean(W.max(axis=0)))      # noqa: E731
        assert spread(ridged) < spread(plain)


# ------------------------------------------------------------- failures
class TestFailuresAreReported:
    def test_a_single_column_has_nothing_to_fit_against(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        with pytest.raises(ValueError, match="at least two"):
            simplex_lstsq_loo(np.ones((4, 1)))

    def test_more_targets_than_columns_is_an_error(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        with pytest.raises(ValueError, match="n_targets"):
            simplex_lstsq_loo(_rank_deficient(n=5), n_targets=6)

    def test_no_targets_is_an_error(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        with pytest.raises(ValueError, match="n_targets"):
            simplex_lstsq_loo(_rank_deficient(n=5), n_targets=0)

    def test_a_one_dimensional_input_is_an_error(self):
        from mlsynth.utils.bilevel.simplex import simplex_lstsq_loo

        with pytest.raises(ValueError, match="2-D"):
            simplex_lstsq_loo(np.arange(5.0))
