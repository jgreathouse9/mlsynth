"""Properties of the R stream and the Section 5 sample over R's seed range."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtri

from mlsynth.utils.marex_helpers.simulation import (_RStream,
                                                    generate_section5_sample)

_SEED = st.integers(0, 2 ** 31 - 1)


@settings(max_examples=200, deadline=None)
@given(_SEED, st.integers(0, 40), st.integers(0, 40))
def test_the_stream_is_one_sequence_however_it_is_read(seed, a, b):
    s = _RStream(seed)
    split = np.r_[s.runif(a), s.runif(b)]
    whole = _RStream(seed).runif(a + b)
    np.testing.assert_array_equal(split, whole)
    assert ((whole > 0) & (whole < 1)).all()


@settings(max_examples=200, deadline=None)
@given(_SEED, st.integers(0, 30))
def test_every_normal_draw_is_rs_inversion_of_two_uniforms(seed, n):
    u = _RStream(seed).runif(2 * n)
    big = 2.0 ** 27
    np.testing.assert_array_equal(
        _RStream(seed).rnorm(n), ndtri((np.floor(big * u[0::2]) + u[1::2]) / big))


@settings(max_examples=40, deadline=None)
@given(_SEED)
def test_a_sample_is_a_section5_sample(seed):
    s = generate_section5_sample(seed)
    t = generate_section5_sample(seed)
    np.testing.assert_array_equal(s.Y_N, t.Y_N)
    assert ((s.Z > 0) & (s.Z < 1)).all() and ((s.mu > 0) & (s.mu < 1)).all()
    np.testing.assert_allclose(s.tau[25:], (s.Y_I[:, 25:] - s.Y_N[:, 25:]).mean(axis=0),
                               rtol=0, atol=1e-12)
    assert np.isnan(s.Y_I[:, :25]).all() and np.isfinite(s.Y_I[:, 25:]).all()
