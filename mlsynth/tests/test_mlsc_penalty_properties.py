"""Generative property tests for the mlSC block penalty matrix.

Reference: Bottmer, L., "Synthetic Control with Disaggregated Data" (JMP,
Stanford), Equation 5.2. The penalty is

    sum_s sum_c (omega_sc - v_sc w_s)^2 = omega' Q omega,

with ``build_block`` returning ``Q = I - v 1' - 1 v' + (v'v) J`` per aggregate
and ``build_penalty_matrix`` gluing the blocks. The claims below hold for every
``v`` on the simplex, so they are asserted over the domain instead of at a
fixture.

Why this layer. Property tests generate inside the specification, which is the
region a fault of omission lives in -- see the instrument contract in
``agents/agents_tests.md``. These helpers are the first-choice target for it:
pure functions, no solver, no DGP, microseconds per call.

The kernel property is the one that carries the estimator's meaning. ``Qv = 0``
says the penalty vanishes exactly on population-proportional weights, so
``lambda`` prices departure from the classical aggregation and nothing else.
Assert it at one ``v`` and you have tested a vector; assert it over the simplex
and you have tested the claim.

Named corners are kept as ``@example`` instead of left to the generator: a
block with one disaggregate, a county with no population, and all mass on one
county are the degenerate shares a real panel produces, and one of them
(``v = [1, 0]``, where ``Q = diag(0, 2)``) is subtle enough that a probe built
as a basis vector silently tests the kernel it meant to avoid.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from mlsynth.utils.mlsc_helpers.penalty import (
    build_block,
    build_penalty_matrix,
    build_sqrt_factor,
)

_TOL = 1e-10

# Raw non-negative shares. Normalising these covers the interior of the simplex
# and its boundary (zeros survive normalisation) without a separate strategy.
_raw_share = st.floats(min_value=0.0, max_value=1e3,
                       allow_nan=False, allow_infinity=False)


@st.composite
def simplex_vectors(draw, min_size: int = 1, max_size: int = 12) -> np.ndarray:
    """A population-share vector: non-negative, summing to one."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    raw = np.asarray(draw(st.lists(_raw_share, min_size=size, max_size=size)))
    total = raw.sum()
    assume(total > 1e-6)                  # all-zero shares are not a panel
    v = raw / total
    assume(np.isfinite(v).all())
    return v


@st.composite
def block_structures(draw, max_blocks: int = 5, max_block_size: int = 6):
    """``(v, disagg_to_agg)`` for an arbitrary set of aggregates."""
    n_blocks = draw(st.integers(min_value=1, max_value=max_blocks))
    blocks = [draw(simplex_vectors(1, max_block_size)) for _ in range(n_blocks)]
    v = np.concatenate(blocks)
    idx = np.repeat(np.arange(n_blocks), [b.size for b in blocks])
    return v, idx


# ---------------------------------------------------------------------------
# One block
# ---------------------------------------------------------------------------

@given(v=simplex_vectors())
@example(v=np.array([1.0]))                    # a block with one disaggregate
@example(v=np.array([1.0, 0.0]))               # a county with no population
@example(v=np.array([0.0, 0.0, 1.0]))          # all mass on one county
@example(v=np.array([0.5, 0.5]))
def test_block_is_symmetric_and_psd(v):
    Q = build_block(v)
    assert Q.shape == (v.size, v.size)
    np.testing.assert_allclose(Q, Q.T, atol=_TOL)
    assert np.linalg.eigvalsh(Q).min() >= -1e-9


@given(v=simplex_vectors())
@example(v=np.array([1.0, 0.0]))
@example(v=np.array([0.0, 0.0, 1.0]))
def test_population_shares_lie_in_the_kernel(v):
    """The claim that gives ``lambda`` its meaning."""
    np.testing.assert_allclose(build_block(v) @ v, 0.0, atol=1e-12)


@given(v=simplex_vectors(min_size=2))
@example(v=np.array([1.0, 0.0]))
def test_the_kernel_is_exactly_the_population_ray(v):
    """``R = I - v 1'`` drops one dimension, so ``rank(Q) = C - 1`` exactly.

    The kernel being no larger than the population ray is what makes every
    other weight direction penalised. Written as equality: the weaker ``<=``
    was the cautious guess, and 3000 draws at block sizes up to 12, including
    shares within 1e-16 of a corner, produce no numerical-rank violation.
    """
    assert np.linalg.matrix_rank(build_block(v), tol=1e-9) == v.size - 1


@given(v=simplex_vectors(min_size=2), scale=st.floats(0.1, 50.0))
@example(v=np.array([1.0, 0.0]), scale=1.0)
@example(v=np.array([0.2, 0.3, 0.5]), scale=1.0)
def test_penalty_vanishes_on_proportional_weights_and_not_off_them(v, scale):
    """The penalty is zero on the population ray and positive off it.

    The off-ray probe is built by projecting ``v`` out. Picking a basis vector
    instead is wrong at a corner such as ``v = [1, 0]``, where that vector is
    ``v`` and the assertion would be testing the kernel it meant to avoid.
    """
    Q = build_block(v)
    on_ray = scale * v
    assert abs(float(on_ray @ Q @ on_ray)) < 1e-10

    probe = np.arange(1.0, v.size + 1.0)
    probe = probe - (probe @ v / (v @ v)) * v
    assume(np.linalg.norm(probe) > 1e-6)
    assert float(probe @ Q @ probe) > 0.0


@given(size=st.integers(min_value=1, max_value=30))
def test_uniform_shares_give_the_centering_matrix(size):
    """``v = 1/C`` collapses ``Q`` to ``I - J/C``."""
    Q = build_block(np.full(size, 1.0 / size))
    np.testing.assert_allclose(
        Q, np.eye(size) - np.full((size, size), 1.0 / size), atol=_TOL
    )


@given(v=simplex_vectors(), permutation_seed=st.integers(0, 2**32 - 1))
def test_relabelling_the_disaggregates_permutes_the_penalty(v, permutation_seed):
    """Column order is a labelling choice, so ``Q`` must permute with it."""
    order = np.random.default_rng(permutation_seed).permutation(v.size)
    np.testing.assert_allclose(
        build_block(v[order]), build_block(v)[np.ix_(order, order)], atol=_TOL
    )


# ---------------------------------------------------------------------------
# Assembly across blocks
# ---------------------------------------------------------------------------

@given(structure=block_structures())
@settings(max_examples=150)
def test_assembly_is_block_diagonal_and_matches_its_blocks(structure):
    v, idx = structure
    Q = build_penalty_matrix(v, idx)

    assert Q.shape == (v.size, v.size)
    np.testing.assert_allclose(Q[idx[:, None] != idx[None, :]], 0.0, atol=_TOL)
    for block in np.unique(idx):
        mask = idx == block
        np.testing.assert_allclose(
            Q[np.ix_(mask, mask)], build_block(v[mask]), atol=_TOL
        )


@given(structure=block_structures())
@settings(max_examples=150)
def test_assembled_penalty_annihilates_the_population_vector(structure):
    v, idx = structure
    np.testing.assert_allclose(build_penalty_matrix(v, idx) @ v, 0.0, atol=1e-12)


@given(structure=block_structures())
@settings(max_examples=150)
def test_square_root_factor_reproduces_the_penalty(structure):
    """``R' R == Q``.

    The solver augments the design with ``sqrt(lambda) R`` instead of forming
    ``Q``, so these two representations agreeing is what makes the augmented
    least-squares path equal the penalised program.
    """
    v, idx = structure
    R = build_sqrt_factor(v, idx)
    np.testing.assert_allclose(R.T @ R, build_penalty_matrix(v, idx), atol=1e-10)
