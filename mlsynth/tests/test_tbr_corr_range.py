"""The TBR key's correlation across the range of doubles (#990).

The correlation is s_xy / sqrt(s_xx s_yy). The product of the two sums of
squares leaves the range of doubles long before either sum does: below a scale of
about 2^-270 it is subnormal or zero, and above 2^270 it is infinite, so the key
read 1.0 or 0.0 for a correlation of 0.81. The correlation does not depend on the
scale of the series, and neither do the key's four gates; the inverse required
impact scales as one over it.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.tbr_helpers.design.batched import design_cache, score_designs
from mlsynth.utils.tbr_helpers.design.objective import score_split

N, N_TEST = 40, 8
SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _pair(seed, slope=2.0, noise=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(10.0, 1.0, N)
    return slope * x + rng.normal(0.0, noise, N), x


def _key(y, x, s):
    return score_split(np.ldexp(y, s), np.ldexp(x, s), objective="reference", n_test=N_TEST)


def test_smoke_the_correlation_is_the_sample_correlation():
    y, x = _pair(0)
    got = _key(y, x, 0)
    assert got.detail["corr"] == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
    assert got.key[4] == round(got.detail["corr"], 2)


@pytest.mark.parametrize("s", [-300, -270, 270, 300])
def test_the_correlation_holds_where_the_product_leaves_the_range(s):
    y, x = _pair(0)
    base, got = _key(y, x, 0), _key(y, x, s)
    assert got.detail["corr"] == pytest.approx(base.detail["corr"], rel=1e-12)
    assert got.key[:5] == base.key[:5]


@SETTINGS
@given(seed=st.integers(0, 10_000), s=st.integers(-480, 480),
       slope=st.floats(-3.0, 3.0).filter(lambda v: abs(v) > 0.05), noise=st.floats(0.1, 5.0))
def test_property_the_key_does_not_see_the_scale(seed, s, slope, noise):
    """Scaling both series by 2^s keeps the gates and the rounded correlation, and
    divides the inverse required impact by 2^s."""
    y, x = _pair(seed, slope, noise)
    base, got = _key(y, x, 0), _key(y, x, s)
    assert got.detail["corr"] == pytest.approx(base.detail["corr"], rel=1e-12, abs=1e-15)
    assert got.key[:4] == base.key[:4]
    assert got.key[5] == pytest.approx(base.key[5] * 2.0 ** -s, rel=1e-9)


@pytest.mark.parametrize("s", [-300, 300])
def test_the_batched_scorer_holds_where_the_product_leaves_the_range(s):
    """The batched scorer computes the same correlation and gives the same key."""
    rng = np.random.default_rng(3)
    Y = np.ldexp(rng.normal(10.0, 1.0, (N, 5)).cumsum(axis=0) / 4 + 50.0, s)
    cache = design_cache(Y, n_test=N_TEST)
    UT = np.array([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
    UC = np.array([[0, 1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0, 1.0]])
    got = score_designs(cache, UT, UC)
    for row, ut, uc in zip(got, UT, UC):
        direct = score_split(Y @ ut, Y @ uc, objective="reference", n_test=N_TEST)
        base = score_split(np.ldexp(Y, -s) @ ut, np.ldexp(Y, -s) @ uc, objective="reference", n_test=N_TEST)
        assert float(row[4]) == direct.key[4] == base.key[4]
        assert tuple(int(v) for v in row[:4]) == direct.key[:4] == base.key[:4]


def test_a_constant_series_has_zero_correlation_at_every_scale():
    y, x = _pair(0)
    for s in (-480, 0, 480):
        got = score_split(np.ldexp(y, s), np.full(N, 2.0 ** s), objective="reference", n_test=N_TEST)
        assert got.detail["corr"] == 0.0
