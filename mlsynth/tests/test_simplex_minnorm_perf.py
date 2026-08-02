"""Work contract for the batched Gram-form simplex QP.

Speed is asserted through machine-independent proxies -- how many batched linear
solves the family costs -- not wall-clock, which flakes in CI. Two claims:

1. A batch of ``S`` problems costs a number of solves set by the *hardest*
   member, not by ``S``: the whole population moves in lockstep, so the work
   does not grow when more candidates are added.
2. A warm start from a neighbouring solution collapses the work, which is what
   makes the MSCMT outer search -- tens of thousands of inner solves over a
   slowly-moving population -- affordable.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.minnorm import (
    simplex_gram,
    solve_simplex_minnorm,
    solve_simplex_minnorm_batch,
)


def _batch(rng, S, m, J):
    return np.stack([simplex_gram(rng.normal(size=(m, J)), rng.normal(size=m))
                     for _ in range(S)])


@pytest.mark.parametrize("m,J", [(20, 5), (13, 17), (10, 30), (8, 8)])
def test_iteration_count_bounded(m, J):
    """The active set terminates in ``O(J)`` iterations -- a deterministic work
    bound that catches both non-termination and a work regression."""
    rng = np.random.default_rng(7 + m + J)
    G = _batch(rng, 40, m, J)
    _, info = solve_simplex_minnorm_batch(G, return_info=True)
    assert info["converged"].all()
    assert info["iterations"] <= 3 * J


def test_work_does_not_grow_with_batch_size():
    """Ten times the candidates for (about) the same number of solves."""
    rng = np.random.default_rng(3)
    small = _batch(rng, 20, 13, 17)
    rng = np.random.default_rng(3)
    large = _batch(rng, 200, 13, 17)
    _, i_small = solve_simplex_minnorm_batch(small, return_info=True)
    _, i_large = solve_simplex_minnorm_batch(large, return_info=True)
    assert i_large["iterations"] <= i_small["iterations"] + 6


def test_batch_costs_far_fewer_solves_than_a_loop():
    rng = np.random.default_rng(4)
    G = _batch(rng, 120, 13, 17)
    _, info = solve_simplex_minnorm_batch(G, return_info=True)
    loop = sum(solve_simplex_minnorm(Gi, return_info=True)[1]["iterations"]
               for Gi in G)
    assert info["iterations"] * 4 < loop


def test_warm_start_collapses_work_on_a_moving_population():
    """The MSCMT pattern: a population that drifts a little each generation. A
    warm start from the previous generation's weights must do strictly less work
    than starting cold, and reach the same optimum."""
    rng = np.random.default_rng(11)
    B = rng.normal(size=(13, 17))
    A = rng.normal(size=13)
    V = rng.uniform(0.1, 1.0, size=(30, 13))
    R = B - A[:, None]

    def grams(V):
        return np.einsum("sk,kj,kl->sjl", V, R, R)

    warm = None
    cold_total = warm_total = 0
    for _ in range(8):
        V = np.clip(V + 0.02 * rng.normal(size=V.shape), 1e-8, 1.0)
        G = grams(V)
        W_cold, ic = solve_simplex_minnorm_batch(G, return_info=True)
        W_warm, iw = solve_simplex_minnorm_batch(G, warm_start=warm,
                                                 return_info=True)
        cold_total += ic["iterations"]
        warm_total += iw["iterations"]
        for Gi, a, b in zip(G, W_cold, W_warm):
            assert np.isclose(float(a @ Gi @ a), float(b @ Gi @ b), atol=1e-10)
        warm = W_warm
    assert warm_total < cold_total
