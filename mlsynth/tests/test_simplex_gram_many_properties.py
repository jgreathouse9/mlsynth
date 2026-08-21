"""Invariants of the many-design simplex solver, over generated Grams.

The example tests pin agreement with the scalar solver on designs someone chose.
These assert the properties that have to hold for *every* Gram, which is where a
solver that happens to agree on the chosen examples gets caught: the answer must
sit on the simplex, must not depend on the order the problems are stacked in,
must be equivariant to relabelling the donors, and must never be beaten by the
uniform point it starts from.
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.bilevel.simplex import simplex_lstsq_gram_many

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _batch(seed, M, n, T0):
    """A well-posed stack: each Gram comes from a real design, so it is PSD."""
    rng = np.random.default_rng(seed)
    designs = [rng.normal(size=(T0, n)) for _ in range(M)]
    b = rng.normal(size=T0)
    AtA = np.stack([A.T @ A for A in designs])
    Atb = np.stack([A.T @ b for A in designs])
    return AtA, Atb, float(b @ b)


def _objective(AtA, Atb, W, bb):
    return (bb - 2.0 * np.einsum("mi,mi->m", W, Atb)
            + np.einsum("mi,mij,mj->m", W, AtA, W))


@SETTINGS
@given(seed=st.integers(0, 2 ** 20), M=st.integers(1, 8),
       n=st.integers(1, 7), T0=st.integers(3, 30))
def test_every_row_is_a_convex_combination(seed, M, n, T0):
    AtA, Atb, _ = _batch(seed, M, n, T0)
    W = simplex_lstsq_gram_many(AtA, Atb)
    assert W.shape == (M, n)
    assert np.isfinite(W).all()
    assert (W >= -1e-9).all()
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-8)


@SETTINGS
@given(seed=st.integers(0, 2 ** 20), M=st.integers(2, 8),
       n=st.integers(1, 6), T0=st.integers(4, 30))
def test_the_stacking_order_does_not_change_any_answer(seed, M, n, T0):
    """Problem ``m`` must get the same weights wherever it sits in the stack.

    A solver that let one problem's convergence state leak into another -- a
    shared step size, a shared stopping flag -- would fail exactly here.
    """
    AtA, Atb, _ = _batch(seed, M, n, T0)
    W = simplex_lstsq_gram_many(AtA, Atb)
    perm = np.random.default_rng(seed + 1).permutation(M)
    W_perm = simplex_lstsq_gram_many(AtA[perm], Atb[perm])
    assert np.allclose(W_perm, W[perm], atol=1e-8)


@SETTINGS
@given(seed=st.integers(0, 2 ** 20), M=st.integers(1, 6),
       n=st.integers(2, 6), T0=st.integers(4, 30))
def test_relabelling_the_donors_permutes_the_weights(seed, M, n, T0):
    """Donor order is a labelling choice, not information."""
    AtA, Atb, _ = _batch(seed, M, n, T0)
    p = np.random.default_rng(seed + 2).permutation(n)
    W = simplex_lstsq_gram_many(AtA, Atb)
    W_p = simplex_lstsq_gram_many(AtA[:, p][:, :, p], Atb[:, p])
    assert np.allclose(W_p, W[:, p], atol=1e-7)


@SETTINGS
@given(seed=st.integers(0, 2 ** 20), M=st.integers(1, 8),
       n=st.integers(1, 7), T0=st.integers(3, 30))
def test_it_never_loses_to_the_point_it_starts_from(seed, M, n, T0):
    """FISTA starts at the uniform point; a descent method may not end worse."""
    AtA, Atb, bb = _batch(seed, M, n, T0)
    W = simplex_lstsq_gram_many(AtA, Atb)
    U = np.full((M, n), 1.0 / n)
    assert (_objective(AtA, Atb, W, bb)
            <= _objective(AtA, Atb, U, bb) + 1e-8).all()


@SETTINGS
@given(seed=st.integers(0, 2 ** 20), n=st.integers(1, 6),
       T0=st.integers(4, 30), scale=st.floats(0.1, 10.0))
def test_scaling_the_problem_leaves_the_weights_alone(seed, n, T0, scale):
    """``A -> cA`` and ``b -> cb`` scale the objective, not its argmin."""
    AtA, Atb, _ = _batch(seed, 3, n, T0)
    W = simplex_lstsq_gram_many(AtA, Atb)
    W_s = simplex_lstsq_gram_many(AtA * scale ** 2, Atb * scale ** 2)
    assert np.allclose(W_s, W, atol=1e-6)
