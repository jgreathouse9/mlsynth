"""Generative property tests for the CSC-IPCA fast paths.

The example tests in ``test_cscipca_fast_paths.py`` fix a residual vector or a
panel and compare the fast path to the implementation it replaces. That catches
a wrong answer at the fixture. These assert the same equivalences over the
input domain, which is where a fault of omission lives: an off-by-one in the
window slice that happens to agree at even ``T``, a fallback that is correct
only when exactly one period is degenerate, a path cache that is right for the
shapes someone thought to type.

Three claims carry the meaning.

The p-value equals the roll form bit for bit, for every residual vector and
every admissible block length. Bit equality is the right relation because the
window form sums the same values in the same order; ``allclose`` here would
accept a reordering that changes a tie, and a tie is exactly what the ``>=``
in the p-value turns into a different answer.

The statistics are the circular block means, so rotating the residual vector
permutes them. The p-value itself is not rotation-invariant -- the rotation
decides which block is the post block -- so the invariant is asserted on the
multiset, which is what "circular" means operationally.

The factors solve the per-period normal equations. That is the specification
``solve_factors`` promises, independent of whether an LU or an SVD produced it,
and it holds on the degenerate periods too once stated in the minimum-norm form
the fallback guarantees.
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from mlsynth.utils.cscipca_helpers import als
from mlsynth.utils.cscipca_helpers.als import solve_factors, solve_gamma
from mlsynth.utils.cscipca_helpers.inference import _moving_block_pvalue

_SETTINGS = settings(max_examples=250, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow])

# Residuals span the magnitudes a real gap series produces, including exact
# zeros and exact ties -- the inputs where a reordered sum changes the answer.
_resid = st.floats(min_value=-1e6, max_value=1e6,
                   allow_nan=False, allow_infinity=False, width=64)


def _roll_stats(u, block):
    return np.array([np.roll(u, s)[-block:].mean() for s in range(u.shape[0])])


def _roll_pvalue(resid, block):
    u = np.abs(np.asarray(resid, dtype=float))
    stats = _roll_stats(u, block)
    return float(np.mean(stats >= stats[0]))


def _loop_factors(Y, X, gamma):
    T = X.shape[1]
    XG = np.einsum("itl,lk->itk", X, gamma, optimize=True)
    denom = np.einsum("itk,itj->tkj", XG, XG, optimize=True)
    numer = np.einsum("itk,it->tk", XG, Y, optimize=True)
    F = np.empty((T, XG.shape[2]))
    for t in range(T):
        F[t], *_ = np.linalg.lstsq(denom[t], numer[t], rcond=None)
    return F.T


# ======================================================================
# the moving-block p-value
# ======================================================================
@_SETTINGS
@given(st.lists(_resid, min_size=1, max_size=48), st.integers(min_value=1))
@example([1.0, 1.0, 1.0], 1)            # every statistic tied
@example([0.0, 0.0], 2)                 # zero residuals, full-length block
@example([3.0, -3.0, 1.0], 2)           # the tie is created by the absolute value
def test_the_window_form_equals_the_roll_form(values, block):
    u = np.asarray(values, dtype=float)
    block = 1 + (block - 1) % u.shape[0]            # any admissible length
    assert _moving_block_pvalue(u, block) == _roll_pvalue(u, block)


@_SETTINGS
@given(st.lists(_resid, min_size=1, max_size=32), st.integers(min_value=1),
       st.integers(min_value=0, max_value=47))
def test_the_statistics_are_a_rotation_invariant_multiset(values, block, shift):
    u = np.asarray(values, dtype=float)
    block = 1 + (block - 1) % u.shape[0]
    a = np.sort(_roll_stats(np.abs(u), block))
    b = np.sort(_roll_stats(np.abs(np.roll(u, shift)), block))
    assert np.allclose(a, b, rtol=0, atol=1e-9 * max(1.0, np.abs(a).max()))


@_SETTINGS
@given(st.lists(_resid, min_size=1, max_size=32), st.integers(min_value=1))
def test_the_p_value_is_a_share_of_the_blocks(values, block):
    u = np.asarray(values, dtype=float)
    T = u.shape[0]
    block = 1 + (block - 1) % T
    p = _moving_block_pvalue(u, block)
    # A share of T blocks, and the post block is always one of them.
    assert 1.0 / T - 1e-12 <= p <= 1.0
    assert abs(p * T - round(p * T)) < 1e-9


@_SETTINGS
@given(st.lists(_resid, min_size=1, max_size=32), st.integers(min_value=1),
       st.floats(min_value=1e-6, max_value=1e6, allow_nan=False,
                 allow_infinity=False))
def test_the_p_value_ignores_sign_and_positive_scale(values, block, scale):
    u = np.asarray(values, dtype=float)
    block = 1 + (block - 1) % u.shape[0]
    base = _moving_block_pvalue(u, block)
    assert _moving_block_pvalue(-u, block) == base
    # Scaling is exact only up to the rounding the multiplication introduces,
    # so compare the decisions the scaled statistics make, not their bits.
    assert _moving_block_pvalue(scale * u, block) == _roll_pvalue(scale * u, block)


@_SETTINGS
@given(st.lists(_resid, min_size=1, max_size=24))
@example([1.8685236347373575, 48576.86852363474, 999999.530581009])
def test_a_full_length_block_scores_every_shift_at_the_series_mean(values):
    """A full-length block carries no information, up to summation order.

    Every circular block of length ``T`` is the whole series, so in exact
    arithmetic all ``T`` statistics equal ``mean(|u|)`` and nothing is more
    extreme than the post block. In floating point they are summed from
    different starting offsets and can differ in the last bits, which the ``>=``
    turns into a p-value below 1 -- 2/3 on the generated vector kept as the
    ``@example`` above. That is a property of the statistic and not of this
    implementation: the roll form it replaced answers 2/3 there too, and the
    bit-identity property above pins the two together at ``block == T``.

    So the claim asserted is the one that holds: each statistic is the series
    mean to within rounding.
    """
    u = np.abs(np.asarray(values, dtype=float))
    T = u.shape[0]
    stats = _roll_stats(u, T)
    target = u.mean()
    scale = max(1.0, float(np.abs(target)))
    assert np.allclose(stats, target, rtol=0, atol=1e-12 * scale)
    # And it is still a share of T blocks that includes the post block.
    p = _moving_block_pvalue(u, T)
    assert 1.0 / T - 1e-12 <= p <= 1.0


# ======================================================================
# the batched F-step
# ======================================================================
_cell = st.floats(min_value=-20.0, max_value=20.0,
                  allow_nan=False, allow_infinity=False, width=64)


@st.composite
def _panels(draw, max_n=5, max_t=6, max_l=4):
    N = draw(st.integers(min_value=1, max_value=max_n))
    T = draw(st.integers(min_value=1, max_value=max_t))
    L = draw(st.integers(min_value=1, max_value=max_l))
    K = draw(st.integers(min_value=1, max_value=L))
    flat = draw(st.lists(_cell, min_size=N * T * L, max_size=N * T * L))
    X = np.asarray(flat, dtype=float).reshape(N, T, L)
    g = draw(st.lists(_cell, min_size=L * K, max_size=L * K))
    gamma = np.asarray(g, dtype=float).reshape(L, K)
    y = draw(st.lists(_cell, min_size=N * T, max_size=N * T))
    Y = np.asarray(y, dtype=float).reshape(N, T)
    return Y, X, gamma, K


@_SETTINGS
@given(_panels())
def test_the_factors_solve_the_per_period_normal_equations(panel):
    Y, X, gamma, K = panel
    F = solve_factors(Y, X, gamma)
    assert F.shape == (K, X.shape[1])
    assert np.all(np.isfinite(F))
    XG = np.einsum("itl,lk->itk", X, gamma, optimize=True)
    for t in range(X.shape[1]):
        A = XG[:, t, :].T @ XG[:, t, :]
        b = XG[:, t, :].T @ Y[:, t]
        scale = max(1.0, float(np.abs(A).max()), float(np.abs(b).max()))
        assert np.allclose(A @ F[:, t], b, rtol=0, atol=1e-7 * scale)


@_SETTINGS
@given(_panels())
@example((np.array([[0., 0., 0., 0., 0., 1.]]),
          np.array([[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [3., 0.]]]),
          np.array([[2.57296425e-245, 1.0], [0.0, 0.0]]), 2))
def test_the_batched_solve_agrees_with_the_per_period_loop(panel):
    """The LU fast path never disagrees with the minimum-norm loop.

    The ``@example`` is the panel that found the defect this gate exists for. Its
    second period is near-singular but not singular, so ``np.linalg.solve``
    neither raises nor regularizes: it returned 7.2e227 where the minimum-norm
    solution is 0.33. A ``try/except LinAlgError`` around the batched solve
    passes that straight through to the counterfactual.
    """
    Y, X, gamma, _K = panel
    got, want = solve_factors(Y, X, gamma), _loop_factors(Y, X, gamma)
    scale = max(1.0, float(np.abs(want).max()))
    # Where every period is well conditioned the LU runs and agrees to rounding
    # (the gate bounds this at about 1e-12); otherwise the fallback *is* the
    # loop, so they agree exactly.
    assert np.allclose(got, want, rtol=0, atol=1e-9 * scale)


@_SETTINGS
@given(_panels())
def test_a_degenerate_period_is_answered_by_the_minimum_norm_fallback(panel):
    # Force at least one exactly singular period and demand the loop's answer.
    Y, X, gamma, _K = panel
    X = X.copy()
    X[:, 0, :] = 0.0
    got = solve_factors(Y, X, gamma)
    assert np.all(np.isfinite(got))
    assert np.allclose(got[:, 0], 0.0, atol=1e-12)
    assert np.array_equal(got, _loop_factors(Y, X, gamma))


# ======================================================================
# the cached einsum paths
# ======================================================================
@_SETTINGS
@given(_panels(max_n=4, max_t=5, max_l=3))
def test_the_cached_path_changes_no_contraction(panel):
    Y, X, gamma, K = panel
    F = solve_factors(Y, X, gamma)
    for sub, ops in (("it,itl,kt->lk", (Y, X, F)),
                     ("itl,itm->tlm", (X, X)),
                     ("itl,lk->itk", (X, gamma))):
        cached = np.einsum(sub, *ops, optimize=als._einsum_path(sub, *ops))
        assert np.array_equal(cached, np.einsum(sub, *ops, optimize=True))


@_SETTINGS
@given(_panels(max_n=4, max_t=5, max_l=3))
def test_solve_gamma_matches_the_uncached_contraction(panel):
    Y, X, gamma, K = panel
    L = X.shape[2]
    F = solve_factors(Y, X, gamma)
    numer = np.einsum("it,itl,kt->lk", Y, X, F, optimize=True).reshape(L * K)
    G = np.einsum("itl,itm->tlm", X, X, optimize=True)
    denom = np.einsum("tlm,kt,jt->lkmj", G, F, F, optimize=True).reshape(L * K, L * K)
    want, *_ = np.linalg.lstsq(denom, numer, rcond=None)
    assert np.array_equal(solve_gamma(Y, X, F, K), want.reshape(L, K))
