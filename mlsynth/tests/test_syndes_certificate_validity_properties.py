"""The certificate does not depend on the outcome's units (#804)."""

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mlsynth.utils.syndes_helpers.certificate import _gap, syndes_certificate

_SCALE = st.sampled_from([1e-6, 1e-3, 1e3, 1e6])


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2 ** 31 - 1), _SCALE, st.integers(1, 4))
def test_the_two_way_certificate_scales_with_the_panel(seed, c, K):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((16, 3)) @ rng.standard_normal((3, 6)) \
        + 0.3 * rng.standard_normal((16, 6))
    lam = float(np.mean(np.var(Y, axis=0, ddof=1)))
    inc = 5.0 * lam
    a = syndes_certificate(Y, K, "global_2way", inc, lam=lam)
    b = syndes_certificate(c * Y, K, "global_2way", c * c * inc, lam=c * c * lam)
    assert abs(b.lower_bound - c * c * a.lower_bound) <= 1e-8 * c * c * abs(a.lower_bound)
    assert abs(b.optimality_gap - a.optimality_gap) <= 1e-8


# Values whose every scaled copy (c down to 1e-9) is still a normal float. A
# subnormal does not scale: 0.5 * 5e-324 underflows to 0, so (c x, c y) is not
# a scaled copy of (x, y), and the gap of (0, 5e-324), -1, is not the gap of
# (0, 0), 0. Hypothesis drew that pair on a run of the SYNDES suite.
_SCALABLE = st.floats(-1e6, 1e6).filter(lambda v: v == 0.0 or abs(v) >= 1e-290)


@settings(max_examples=200, deadline=None)
@example(x=0.0, y=1e-290, c=1e-9)
@given(_SCALABLE, _SCALABLE, st.floats(1e-9, 1e9))
def test_the_gap_is_scale_free(x, y, c):
    assert abs(_gap(c * x, c * y) - _gap(x, y)) <= 1e-9
