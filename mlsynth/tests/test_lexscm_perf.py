"""Work contracts for the two LEXSCM hot loops.

Profiling one design fit on a 211-market weekly panel put 85% of the runtime in
two places: ``block_resample_windows`` (45s of 80s) and ``_afw_batched`` (23s).
Underneath both sat 5.47 million ``np.take`` calls, 5.47 million ``np.arange``
calls and 409k ``einsum`` calls -- Python dispatch, not arithmetic.

Speed is asserted the way ``test_simplex_active_set_perf`` asserts it: with
machine-independent proxies for the work done, not wall-clock, which flakes in
CI. Two proxies are used here.

``_afw_batched`` contracts an ``(N, m, m)`` batch against an ``(N, m)`` matrix
twice per iteration -- once for the gradient and once for ``Q @ D``. The second
is avoidable: ``D`` is always ``-W + e_s`` or ``W - e_a``, so ``Q @ D`` is the
already-computed ``Q @ W`` plus or minus one column of ``Q``. The contract is
one batched contraction per iteration.

``block_resample_windows`` builds every block with its own ``arange`` and
``take`` inside a double loop, so its Python-level call count scales with
``n_draws``. The contract is that it does not: the resampled index set is one
array computation whatever the number of draws.

The losses themselves are pinned against values recorded from the pre-optimised
implementation, so a refactor that changes the answer fails here rather than
silently moving every MDE the estimator reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.lexpower import block_resample_windows
from mlsynth.utils.fast_scm_helpers.lexsearch import _afw_batched


def _batch(N, m, T, seed):
    """A batch of ``N`` Gram sub-matrices, centered as lexsearch centers them."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, m * 3))
    X = X - X.mean(1, keepdims=True)
    G = X.T @ X
    subs = rng.choice(m * 3, size=(N, m), replace=True)
    return G[subs[:, :, None], subs[:, None, :]]


# Recorded from the implementation profiled above, before any optimisation.
GOLDEN = {
    "A": np.array([1.614039799678, 1.660303357212, 3.101118588921,
                   0.884245273217, 0.198609158605, 1.393357103436,
                   0.90353872171]),
    "B": np.array([0.574636648828, 0.399664475488, 0.359102974093,
                   0.726309084709, 0.311711618787]),
}


# --- the answer may not move ----------------------------------------------


@pytest.mark.parametrize("tag, shape", [("A", (7, 4, 9)), ("B", (5, 6, 8))])
def test_afw_losses_match_the_recorded_values(tag, shape):
    """Reusing the gradient to build ``Q @ D`` is an algebraic identity.

    It reassociates the arithmetic, so the result moves at floating-point
    level and no further. A tolerance that admits a genuinely different
    trajectory would defeat the purpose of recording the values at all.
    """
    got = _afw_batched(_batch(*shape, seed=11), iters=80)
    assert np.allclose(got, GOLDEN[tag], rtol=1e-9, atol=1e-11)


# --- and the work must come down ------------------------------------------


class _CountingArray(np.ndarray):
    """Counts batched ``(N, m, m) x (N, m)`` contractions performed on it."""


def _count_batched_contractions(monkeypatch):
    """Return a dict tallying ``einsum`` / ``matmul`` calls with a 3-D operand."""
    tally = {"n": 0}
    real_einsum, real_matmul = np.einsum, np.matmul

    def einsum(subscripts, *operands, **kw):
        if any(np.ndim(o) == 3 for o in operands):
            tally["n"] += 1
        return real_einsum(subscripts, *operands, **kw)

    def matmul(a, b, *args, **kw):
        if np.ndim(a) == 3 or np.ndim(b) == 3:
            tally["n"] += 1
        return real_matmul(a, b, *args, **kw)

    monkeypatch.setattr(np, "einsum", einsum)
    monkeypatch.setattr(np, "matmul", matmul)
    return tally


def test_afw_does_one_batched_contraction_per_iteration(monkeypatch):
    """Two per iteration is one more than the algebra needs."""
    tally = _count_batched_contractions(monkeypatch)
    iters = 40
    _afw_batched(_batch(7, 4, 9, seed=11), iters=iters)
    # one per iteration for the gradient, plus the closing loss evaluation
    assert tally["n"] <= iters + 1, (
        f"{tally['n']} batched contractions for {iters} iterations; "
        "Q @ D should be built from the gradient, not recomputed"
    )


def test_block_resampling_cost_does_not_scale_with_draws(monkeypatch):
    """The index set is one array computation however many windows are asked for."""
    calls = {"n": 0}
    real_take = np.take

    def take(*a, **kw):
        calls["n"] += 1
        return real_take(*a, **kw)

    monkeypatch.setattr(np, "take", take)
    rng = np.random.default_rng(0)
    series = [rng.normal(size=39) for _ in range(211)]

    calls["n"] = 0
    block_resample_windows(series, n_post=12, n_draws=100, block_len=3,
                           rng=np.random.default_rng(1))
    few = calls["n"]
    calls["n"] = 0
    block_resample_windows(series, n_post=12, n_draws=2000, block_len=3,
                           rng=np.random.default_rng(1))
    many = calls["n"]

    assert many <= few + 4, (
        f"{few} take() calls at 100 draws, {many} at 2000 -- the cost is "
        "per-block, so a Monte Carlo pays it n_draws times over"
    )


# --- the resampler must still resample ------------------------------------


def test_resampled_windows_come_from_the_pool_and_keep_its_scale():
    """Vectorising the index arithmetic must not change what is drawn."""
    rng = np.random.default_rng(3)
    series = [rng.normal(size=39) for _ in range(40)]
    pool = np.concatenate(series)
    out = block_resample_windows(series, n_post=12, n_draws=4000, block_len=3,
                                 rng=np.random.default_rng(5))

    assert out.shape == (4000, 12)
    assert np.isin(out, pool).all()                  # every value is a real draw
    assert abs(out.std() / pool.std() - 1.0) < 0.05  # and the scale survives


def test_resampled_windows_preserve_within_block_dependence():
    """Moving blocks exist to carry serial correlation; that is the invariant."""
    rng = np.random.default_rng(4)
    ar = np.empty((30, 60))
    ar[:, 0] = rng.normal(size=30)
    for t in range(1, 60):                            # strongly persistent pool
        ar[:, t] = 0.8 * ar[:, t - 1] + rng.normal(scale=0.6, size=30)
    series = list(ar)

    out = block_resample_windows(series, n_post=24, n_draws=3000, block_len=8,
                                 rng=np.random.default_rng(6))
    lag1 = np.corrcoef(out[:, :-1].ravel(), out[:, 1:].ravel())[0, 1]
    assert lag1 > 0.4, f"lag-1 correlation {lag1:.3f} -- blocks are not holding"
