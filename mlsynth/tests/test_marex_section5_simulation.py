"""Abadie and Zhao's Section 5 simulation, draw for draw.

``generate_section5_sample(seed)`` reproduces what ``set.seed(seed)`` followed
by the authors' ``Generate_Model_Primitives`` gives in R. The anchors are R's
own outputs: the documented first draws after ``set.seed`` and values from the
capture in ``benchmarks/reference/marex_scdesign_sim/``, printed by R to ten
decimals.
"""

import numpy as np
import pytest
from scipy.special import ndtri

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers.simulation import (MAREXSample, _RStream,
                                                    _fixup,
                                                    generate_marex_sample,
                                                    generate_section5_sample)

# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


def test_a_seed_gives_a_section5_sample():
    s = generate_section5_sample(1)
    assert isinstance(s, MAREXSample)
    assert s.Y_N.shape == (15, 30) and s.Y_I.shape == (15, 30)
    assert s.Z.shape == (15, 7) and s.mu.shape == (15, 11)
    assert s.T0 == 25 and s.tau.shape == (30,)
    assert np.isfinite(s.Y_N).all() and np.isfinite(s.Y_I[:, 25:]).all()


# --------------------------------------------------------------------------- #
# R's stream
# --------------------------------------------------------------------------- #

#: R 4.x, default kinds ("Mersenne-Twister", "Inversion"), printed to 7 digits.
_R_DRAWS = [
    (1, "runif", [0.2655087, 0.3721239, 0.5728534]),
    (1, "rnorm", [-0.6264538, 0.1836433, -0.8356286]),
    (123, "runif", [0.2875775, 0.7883051]),
    (123, "rnorm", [-0.5604756, -0.2301775]),
    (42, "runif", [0.9148060]),
]


@pytest.mark.parametrize("seed,kind,expected", _R_DRAWS)
def test_the_stream_gives_rs_first_draws(seed, kind, expected):
    got = getattr(_RStream(seed), kind)(len(expected))
    np.testing.assert_allclose(got, expected, atol=5e-8)


def test_a_normal_draw_spends_two_uniforms():
    """R's inversion: ``qnorm((floor(2^27 u1) + u2) / 2^27)``."""
    u = _RStream(9).runif(8)
    big = 2.0 ** 27
    np.testing.assert_array_equal(
        _RStream(9).rnorm(4), ndtri((np.floor(big * u[0::2]) + u[1::2]) / big))


def test_a_raw_zero_is_moved_inside_the_interval():
    """R's ``fixup``: a draw of exactly 0 becomes half of ``i2_32m1``, the
    literal ``2.328306437080797e-10`` in R's RNG.c (``1 / (2^32 - 1)`` to the
    digits printed)."""
    np.testing.assert_array_equal(
        _fixup(np.array([0.0, 0.25])), [0.5 * 2.328306437080797e-10, 0.25])


# --------------------------------------------------------------------------- #
# the authors' panels
# --------------------------------------------------------------------------- #

#: (seed, array, unit, period or covariate, value) from the R capture, 1-based
#: as R printed them: ``yn_s1_u1_t1`` is ``Y_N[0, 0]`` of seed 1.
_CAPTURED = [
    (1, "Y_N", 1, 1, 46.0999461535),
    (1, "Y_N", 15, 30, 68.8032235789),
    (1, "Y_N", 7, 21, 69.0848867157),
    (1, "Y_I", 1, 26, 56.5777090949),
    (1, "Y_I", 15, 30, 63.1339113841),
    (1, "Z", 1, 1, 0.6684667382),
    (1, "Z", 15, 7, 0.5076418223),
    (12, "Y_N", 3, 5, 48.6963857269),
    (12, "Y_I", 9, 28, 51.8648521364),
    (12, "Z", 2, 4, 0.0950279189),
]


@pytest.mark.parametrize("seed,name,unit,col,value", _CAPTURED)
def test_the_sample_is_the_authors_draw(seed, name, unit, col, value):
    s = generate_section5_sample(seed)
    assert getattr(s, name)[unit - 1, col - 1] == pytest.approx(value, abs=1e-9)


def test_the_effect_is_the_unit_average_of_the_contrast():
    s = generate_section5_sample(3)
    np.testing.assert_array_equal(s.tau[:25], 0.0)
    np.testing.assert_allclose(s.tau[25:], (s.Y_I[:, 25:] - s.Y_N[:, 25:]).mean(axis=0),
                               rtol=0, atol=1e-12)


def test_treated_outcomes_are_undefined_before_treatment():
    """The authors define ``Y.I`` only on the experimental periods."""
    assert np.isnan(generate_section5_sample(2).Y_I[:, :25]).all()


def test_the_covariates_are_drawn_in_the_authors_order():
    """30 intercepts, 5 treated intercepts, then ``Z`` unit by unit (7 each),
    then ``mu`` unit by unit (11 each), all Uniform(0, 1)."""
    s = generate_section5_sample(1)
    u = _RStream(1).runif(35 + 15 * 7 + 15 * 11)
    np.testing.assert_array_equal(s.Z, u[35:140].reshape(15, 7))
    np.testing.assert_array_equal(s.mu, u[140:].reshape(15, 11))


def test_the_generic_sample_carries_its_mu():
    s = generate_marex_sample(J=4, R=2, F=3, T=6, T0=4, rng=np.random.default_rng(0))
    assert s.mu.shape == (4, 3) and ((s.mu >= 0) & (s.mu <= 1)).all()


def test_a_sample_built_without_mu_defaults_to_none():
    s = MAREXSample(Y_N=np.zeros((2, 3)), Y_I=np.zeros((2, 3)), tau=np.zeros(3),
                    T0=2, Z=np.zeros((2, 1)))
    assert s.mu is None


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def test_seed_zero_is_a_seed():
    a, b = generate_section5_sample(0), generate_section5_sample(1)
    assert not np.allclose(a.Y_N, b.Y_N)


def test_no_draws_are_no_draws():
    s = _RStream(4)
    assert s.runif(0).size == 0 and s.rnorm(0).size == 0
    np.testing.assert_array_equal(s.runif(2), _RStream(4).runif(2))


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [-1, 2 ** 31, 1.0, True, "1", None])
def test_a_seed_outside_rs_range_is_refused(seed):
    with pytest.raises(MlsynthConfigError, match="seed"):
        generate_section5_sample(seed)


@pytest.mark.parametrize("seed", [-1, 2 ** 31, 2.5, False])
def test_the_stream_refuses_the_same_seeds(seed):
    with pytest.raises(MlsynthConfigError, match="seed"):
        _RStream(seed)
