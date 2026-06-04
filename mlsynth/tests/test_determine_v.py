"""Tests for the MSCMT ``determine_v`` predictor-weight canonicalisation."""

import warnings

import numpy as np
import pytest

from mlsynth.utils.fscm_helpers.bilevel import (
    BilevelProblem,
    canonical_v,
    canonical_v_diagnostics,
    check_v,
    kkt_matrix,
    max_order_v,
    min_loss_w_v,
    solve_bilevel,
)
from mlsynth.utils.fscm_helpers.bilevel.mscmt import _inner_weights


def _problem(seed: int = 0, K: int = 6, J: int = 8, T: int = 14) -> BilevelProblem:
    """A bilevel problem whose donors are mildly collinear (so ``V`` is
    non-identified) and whose inner optimum is sparse.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(K, 3))
    X0 = base @ rng.normal(size=(3, J)) + 0.1 * rng.normal(size=(K, J))  # near rank-3
    X1 = rng.normal(size=K)
    Y0 = rng.normal(size=(T, J))
    y1 = rng.normal(size=T)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def _W_for(prob: BilevelProblem, seed: int = 0) -> np.ndarray:
    """A donor weight vector that is, by construction, the inner optimum for a
    known ``V`` -- hence in the KKT polytope of some predictor weights.
    """
    rng = np.random.default_rng(seed)
    V = rng.uniform(0.1, 1.0, size=prob.n_predictors)
    return _inner_weights(prob, V), V


# --------------------------------------------------------------------------- #
# kkt_matrix
# --------------------------------------------------------------------------- #
def test_kkt_matrix_shape():
    prob = _problem()
    W, _ = _W_for(prob)
    M = kkt_matrix(prob, W)
    assert M.shape == (prob.n_donors, prob.n_predictors)


def test_kkt_matrix_stationary_at_inner_optimum():
    # The V that produced W must satisfy the KKT stationarity it encodes:
    # (M v)_j == 0 on the support of W.
    prob = _problem(seed=1)
    W, V = _W_for(prob, seed=1)
    M = kkt_matrix(prob, W)
    active = W > 1e-9
    np.testing.assert_allclose(M[active] @ V, 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# check_v
# --------------------------------------------------------------------------- #
def test_check_v_accepts_reproducing_v():
    prob = _problem(seed=2)
    W, V = _W_for(prob, seed=2)
    assert check_v(prob, V, W)


def test_check_v_rejects_mismatched_w():
    # A V that reproduces W cannot also certify a *different* donor vector.
    prob = _problem(seed=3)
    W, V = _W_for(prob, seed=3)
    rng = np.random.default_rng(99)
    W_wrong = rng.dirichlet(np.ones(prob.n_donors))
    assert not check_v(prob, V, W_wrong)


def test_check_v_rejects_nan():
    prob = _problem()
    W, _ = _W_for(prob)
    assert not check_v(prob, np.full(prob.n_predictors, np.nan), W)


# --------------------------------------------------------------------------- #
# min_loss_w_v / canonical_v
# --------------------------------------------------------------------------- #
def test_min_loss_w_certifies_and_is_normalised():
    prob = _problem(seed=4)
    W, _ = _W_for(prob, seed=4)
    v = min_loss_w_v(prob, W)
    assert np.all(np.isfinite(v))
    assert v.max() == pytest.approx(1.0)        # reported on max(v) = 1
    assert check_v(prob, v, W)


def test_min_loss_w_is_deterministic():
    prob = _problem(seed=5)
    W, _ = _W_for(prob, seed=5)
    np.testing.assert_array_equal(min_loss_w_v(prob, W), min_loss_w_v(prob, W))


def test_canonical_v_returns_simplex_and_ok():
    prob = _problem(seed=6)
    W, _ = _W_for(prob, seed=6)
    v, ok = canonical_v(prob, W)
    assert ok
    assert v.sum() == pytest.approx(1.0)         # reported on the simplex
    assert np.all(v >= 0)


def test_canonical_v_rejects_unknown_method():
    prob = _problem()
    W, _ = _W_for(prob)
    with pytest.raises(ValueError, match="min.loss.w|max.order"):
        canonical_v(prob, W, method="bogus")


# --------------------------------------------------------------------------- #
# max_order_v (leximin) + PUFAS
# --------------------------------------------------------------------------- #
def test_max_order_certifies():
    prob = _problem(seed=10)
    W, _ = _W_for(prob, seed=10)
    v, unique = max_order_v(prob, W)
    assert np.all(np.isfinite(v))
    assert v.max() == pytest.approx(1.0)
    assert isinstance(unique, bool)
    assert check_v(prob, v, W)


def test_max_order_lifts_the_floor_vs_min_loss():
    # Leximin maximises the smallest weight, so its minimum should be >= the
    # smallest *nonzero* weight that min.loss.w produces (it refuses to zero
    # predictors the data don't force out).
    prob = _problem(seed=11)
    W, _ = _W_for(prob, seed=11)
    v_min = min_loss_w_v(prob, W)
    v_max, _ = max_order_v(prob, W)
    assert v_max.min() >= v_min[v_min > 1e-6].min() - 1e-6


def test_max_order_is_deterministic():
    prob = _problem(seed=12)
    W, _ = _W_for(prob, seed=12)
    a, _ = max_order_v(prob, W)
    b, _ = max_order_v(prob, W)
    np.testing.assert_array_equal(a, b)


def test_canonical_v_max_order_method():
    prob = _problem(seed=13)
    W, _ = _W_for(prob, seed=13)
    v, ok = canonical_v(prob, W, method="max.order")
    assert ok
    assert v.sum() == pytest.approx(1.0)


def test_canonical_v_diagnostics_structure():
    prob = _problem(seed=14)
    W, _ = _W_for(prob, seed=14)
    d = canonical_v_diagnostics(prob, W)
    assert set(d) == {"min.loss.w", "min.loss.w_ok", "max.order",
                      "max.order_ok", "agreement"}
    assert d["agreement"] >= 0.0


def test_mscmt_max_order_metadata():
    prob = _problem(seed=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=60,
                            canonical_v="max.order")
    assert sol.metadata["v_method"] in ("max.order", "optimizer-fallback")
    assert "v_agreement" in sol.metadata


# --------------------------------------------------------------------------- #
# integration through solve_mscmt
# --------------------------------------------------------------------------- #
def test_mscmt_canonical_v_preserves_donor_weights():
    prob = _problem(seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = solve_bilevel(prob, method="mscmt", seed=0, maxiter=60)
        can = solve_bilevel(prob, method="mscmt", seed=0, maxiter=60,
                            canonical_v=True)
    # Canonicalisation must not change the estimate (donor weights / fit).
    np.testing.assert_allclose(raw.W, can.W, atol=1e-6)
    assert can.metadata["v_method"] in ("min.loss.w", "optimizer-fallback")


def test_mscmt_canonical_v_reduces_predictor_weight_spread():
    # Across seeds the raw optimiser V wobbles in the non-identified null space;
    # the canonical V should be markedly more stable.
    prob = _problem(seed=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.array([solve_bilevel(prob, method="mscmt", seed=s, maxiter=60).V
                        for s in range(4)])
        can = np.array([solve_bilevel(prob, method="mscmt", seed=s, maxiter=60,
                                      canonical_v=True).V for s in range(4)])
    raw_spread = float(np.max(raw.max(0) - raw.min(0)))
    can_spread = float(np.max(can.max(0) - can.min(0)))
    assert can_spread <= raw_spread + 1e-9
