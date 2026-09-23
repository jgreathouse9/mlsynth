"""Property tests for the TWSF kernels.

TWSF's forecast is a product of two estimated objects and a recursion, so a
defect in any one of them still returns a finite number of roughly the right
size. The example tests pin exactness on one noiseless fixture; these assert
the same claims over the input domain, which is what separates "the fixture
happens to work" from "the identity holds".

Three groups, in increasing order of how much machinery they exercise.

The companion recursion is exact algebra and its identities hold for every
coefficient vector: one step ahead of the state is the rule itself, the
Jacobian at lead one is the identity, and iterating the map agrees with
applying the companion matrix. These need no oracle.

The Page construction is a reindexing, so its content is checkable directly:
every response really is the observation one step past its own lag block, and
the forecast state really is the donors' terminal window. A transposition or an
off-by-one in the block layout is invisible to a shape assertion and caught
here.

The estimator's metamorphic relations are where property testing earns its keep
on statistical code. Scaling every outcome scales the forecast and leaves both
weight vectors alone; shifting the donors' labels permutes the unit weights and
leaves the forecast alone. Neither needs a known truth, and both would be
violated by a defect that no fixture-based assertion on magnitudes would see.

``derandomize=True`` throughout: a mutation run scores each mutant by whether
the suite fails, and a flakily-killed mutant corrupts that score.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from mlsynth.utils.twsf_helpers.pipeline import (
    _truncated_pinv,
    companion,
    fit_twsf,
    lead_jacobian,
    lead_map,
    page_blocks,
)

_SETTINGS = settings(derandomize=True, deadline=None, max_examples=60,
                     suppress_health_check=[HealthCheck.too_slow])

_FINITE = dict(allow_nan=False, allow_infinity=False,
               min_value=-3.0, max_value=3.0, width=64)


def _vec(min_size=1, max_size=6):
    return hnp.arrays(np.float64, st.integers(min_size, max_size),
                      elements=st.floats(**_FINITE))


def _matrix(rows, cols):
    return hnp.arrays(np.float64, (rows, cols), elements=st.floats(**_FINITE))


# --------------------------------------------------------------------------
# the companion recursion -- exact algebra, no oracle needed
# --------------------------------------------------------------------------

@_SETTINGS
@given(x=_vec(min_size=2))
def test_one_step_ahead_is_the_rule_itself(x):
    """g_1(x) = x: the lead-one map of the state is the coefficient vector."""
    assert np.allclose(lead_map(x, 1), x, atol=1e-12)


@_SETTINGS
@given(x=_vec(min_size=2))
def test_the_jacobian_at_lead_one_is_the_identity(x):
    """g_1 is the identity map, so its derivative is I."""
    assert np.allclose(lead_jacobian(x, 1), np.eye(x.size), atol=1e-12)


@_SETTINGS
@given(x=_vec(min_size=2), ell=st.integers(1, 5))
def test_lead_map_is_the_companion_iterated(x, ell):
    """g_ell advances the state exactly as the companion matrix does."""
    state = np.arange(1.0, x.size + 1.0)
    stepped = state.copy()
    for _ in range(ell):
        stepped = companion(x) @ stepped
    assert np.allclose(lead_map(x, ell) @ state, stepped[-1], atol=1e-8)


@_SETTINGS
@given(x=_vec(min_size=2))
def test_companion_carries_the_rule_in_its_last_row(x):
    P = companion(x)
    assert np.allclose(P[-1], x, atol=1e-12)
    assert np.allclose(P[:-1, 1:], np.eye(x.size - 1), atol=1e-12)
    assert np.allclose(P[:-1, 0], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# the Page construction -- a reindexing, so check the content not the shape
# --------------------------------------------------------------------------

@_SETTINGS
@given(n_donors=st.integers(1, 4), L=st.integers(1, 5),
       n_blocks=st.integers(2, 5), lead=st.integers(1, 3))
def test_every_response_is_the_observation_past_its_own_block(n_donors, L,
                                                              n_blocks, lead):
    """The Page layout must pair each lag vector with its own future value."""
    width = L + lead
    T1 = width * n_blocks
    Y = np.arange(n_donors * T1, dtype=float).reshape(n_donors, T1)
    Z, z, W = page_blocks(Y, L, lead=lead)

    assert Z.shape == (L, n_donors * (n_blocks - 1))
    assert z.shape == (n_donors * (n_blocks - 1),)
    assert W.shape == (n_donors, L)

    col = 0
    for j in range(n_donors):
        for b in range(n_blocks - 1):
            seg = Y[j, b * width:(b + 1) * width]
            assert np.allclose(Z[:, col], seg[:L])
            assert z[col] == seg[-1]
            col += 1
    assert np.allclose(W, Y[:, T1 - L:])


@_SETTINGS
@given(n_donors=st.integers(1, 3), L=st.integers(1, 4))
def test_one_usable_block_is_refused(n_donors, L):
    """The final block is held back for the forecast state, so one is not enough."""
    from mlsynth.exceptions import MlsynthConfigError
    Y = np.zeros((n_donors, L + 1))          # exactly one block
    with pytest.raises(MlsynthConfigError):
        page_blocks(Y, L)


# --------------------------------------------------------------------------
# the truncated pseudo-inverse
# --------------------------------------------------------------------------

@_SETTINGS
@given(A=_matrix(5, 7), k=st.integers(1, 4))
def test_truncated_pinv_satisfies_moore_penrose_on_the_truncation(A, k):
    """A_k P A_k = A_k, where A_k is the rank-k truncation P inverts."""
    from mlsynth.utils.pcr.core import hsvt
    A_k, _, s, _ = hsvt(A, k)
    assume(s.min() > 1e-6 * max(s.max(), 1e-12))     # else the inverse is ill-posed
    P = _truncated_pinv(A, k)
    assert np.allclose(A_k @ P @ A_k, A_k, atol=1e-6)


# --------------------------------------------------------------------------
# metamorphic relations on the estimator
# --------------------------------------------------------------------------

def _panel(seed, n_donors=5, T0=30, T1=60, rank=2):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_donors, rank))
    lam = rng.dirichlet(np.ones(n_donors))
    t = np.arange(1, T0 + T1 + 1)
    basis = np.vstack([np.sin(2 * np.pi * t / 9), np.cos(2 * np.pi * t / 9)])
    V0 = rng.standard_normal((rank, 2)) @ basis
    V1 = rng.standard_normal((rank, 2)) @ basis
    y_pre = (lam @ U) @ V0[:, :T0]
    Y_pre = U @ V0[:, :T0]
    Y_post = U @ V1[:, T0:]
    return y_pre, Y_pre, Y_post


def _fit(y_pre, Y_pre, Y_post, horizon=3, **kw):
    return fit_twsf(y_pre, Y_pre, Y_post, L=6, k_y=2, k_z=4, horizon=horizon,
                    **kw)


@_SETTINGS
@given(seed=st.integers(0, 200),
       scale=st.floats(min_value=0.2, max_value=5.0, allow_nan=False,
                       allow_infinity=False))
def test_scaling_every_outcome_scales_the_forecast_and_leaves_the_weights(seed,
                                                                         scale):
    """Units are a choice of measurement, not of method."""
    y, Yp, Yq = _panel(seed)
    base = _fit(y, Yp, Yq)
    scaled = _fit(y * scale, Yp * scale, Yq * scale)
    assert np.allclose(scaled.forecast, base.forecast * scale, rtol=1e-6,
                       atol=1e-9)
    assert np.allclose(scaled.beta, base.beta, rtol=1e-6, atol=1e-9)
    assert np.allclose(scaled.alpha, base.alpha, rtol=1e-6, atol=1e-9)
    assert np.allclose(scaled.std_error, base.std_error * abs(scale),
                       rtol=1e-6, atol=1e-9)


@_SETTINGS
@given(seed=st.integers(0, 200))
def test_relabelling_donors_permutes_the_weights_and_moves_nothing_else(seed):
    """Donor order is bookkeeping; the forecast must not depend on it."""
    y, Yp, Yq = _panel(seed)
    perm = np.random.default_rng(seed + 1).permutation(Yp.shape[0])
    base = _fit(y, Yp, Yq)
    shuffled = _fit(y, Yp[perm], Yq[perm])
    assert np.allclose(shuffled.forecast, base.forecast, rtol=1e-5, atol=1e-8)
    assert np.allclose(shuffled.beta, base.beta[perm], rtol=1e-5, atol=1e-8)


@_SETTINGS
@given(seed=st.integers(0, 200))
def test_direct_and_recursive_agree_at_one_step_over_the_domain(seed):
    """The paper states the two estimators coincide at h = 1."""
    y, Yp, Yq = _panel(seed)
    a = _fit(y, Yp, Yq, horizon=1, multistep="direct")
    b = _fit(y, Yp, Yq, horizon=1, multistep="recursive")
    assert np.allclose(a.forecast, b.forecast, rtol=1e-6, atol=1e-9)


@_SETTINGS
@given(seed=st.integers(0, 200))
def test_the_prediction_interval_is_never_narrower_than_the_confidence_one(seed):
    """The prediction interval carries a future innovation the CI omits."""
    y, Yp, Yq = _panel(seed)
    ci = _fit(y, Yp, Yq, interval="confidence")
    pi = _fit(y, Yp, Yq, interval="prediction")
    assert np.all(pi.std_error >= ci.std_error - 1e-12)


@_SETTINGS
@given(seed=st.integers(0, 200))
def test_the_interval_is_a_symmetric_band_about_the_forecast(seed):
    y, Yp, Yq = _panel(seed)
    fit = _fit(y, Yp, Yq)
    assert np.allclose(fit.upper - fit.forecast, fit.forecast - fit.lower,
                       atol=1e-12)
    assert np.all(fit.std_error >= 0.0)
