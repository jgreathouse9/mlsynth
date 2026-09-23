"""Metamorphic invariants of the block permutation reference.

The unit suite pins the gather against the loop on fixed shapes. These
properties assert what has to hold for every path and every post block, and the
first is the one the optimisation rests on:

* equivalence -- the gather reproduces the loop bit for bit at ``q == 1``. Not
  ``allclose``: the p-value counts how many reference statistics reach the
  observed one, so a value moved by an ULP can cross a tie and change a rank;
* fidelity at other exponents -- where the guard sends the call back to the
  loop, the answer is still the loop's, which is what makes the guard invisible
  to a caller;
* structure -- one statistic per period of the path, all finite and
  non-negative, with shift zero equal to the observed statistic;
* equivariance -- rescaling the path rescales every reference statistic by the
  same factor, since the statistic is homogeneous of degree one at ``q == 1``;
* invariance -- rotating the whole path permutes the reference set and leaves it
  the same multiset, which is what makes the block scheme a permutation scheme.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from mlsynth.utils.bilevel.ridge_inference import _reference_stats, _stat

_SETTINGS = settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def _oracle(resids, post_slice, q):
    """The loop, transcribed, as the reference the gather must reproduce."""
    n = np.asarray(resids).shape[0]
    return np.array(
        [_stat(np.roll(resids, s)[post_slice], q) for s in range(n)], dtype=float)


_PATHS = npst.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=60),
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False,
                       allow_infinity=False, width=64),
)


@given(u=_PATHS, frac=st.floats(min_value=0.0, max_value=0.95))
@_SETTINGS
def test_the_gather_reproduces_the_loop_bit_for_bit(u, frac):
    pre = int(frac * u.size)
    got = _reference_stats(u, slice(pre, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, _oracle(u, slice(pre, None), 1.0))


@given(u=_PATHS, frac=st.floats(min_value=0.0, max_value=0.95),
       q=st.sampled_from([0.5, 1.5, 2.0, 3.0]))
@_SETTINGS
def test_other_exponents_still_give_the_loop(u, frac, q):
    pre = int(frac * u.size)
    got = _reference_stats(u, slice(pre, None), q, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, _oracle(u, slice(pre, None), q))


@given(u=_PATHS, frac=st.floats(min_value=0.0, max_value=0.95))
@_SETTINGS
def test_the_reference_set_has_the_shape_a_permutation_scheme_needs(u, frac):
    pre = int(frac * u.size)
    got = _reference_stats(u, slice(pre, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert got.shape == (u.size,)
    assert np.isfinite(got).all()
    assert (got >= 0.0).all()
    assert got[0] == _stat(u[pre:], 1.0)


@given(u=_PATHS, frac=st.floats(min_value=0.0, max_value=0.95),
       scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
@_SETTINGS
def test_rescaling_the_path_rescales_every_reference_statistic(u, frac, scale):
    """At ``q == 1`` the statistic is a scaled sum of absolute values, so it is
    homogeneous of degree one."""
    pre = int(frac * u.size)
    kw = dict(conformal_type="block", ns=0, seed=0)
    base = _reference_stats(u, slice(pre, None), 1.0, **kw)
    scaled = _reference_stats(u * scale, slice(pre, None), 1.0, **kw)
    assert np.allclose(scaled, base * scale, rtol=1e-9, atol=1e-12)


@given(u=_PATHS, frac=st.floats(min_value=0.0, max_value=0.95),
       shift=st.integers(min_value=0, max_value=59))
@_SETTINGS
def test_rotating_the_path_permutes_the_reference_set(u, frac, shift):
    """The block scheme's reference set is the orbit of the path under rotation,
    so rotating the input reorders that set without changing what is in it."""
    pre = int(frac * u.size)
    kw = dict(conformal_type="block", ns=0, seed=0)
    base = _reference_stats(u, slice(pre, None), 1.0, **kw)
    rotated = _reference_stats(np.roll(u, shift % u.size), slice(pre, None),
                               1.0, **kw)
    assert np.allclose(np.sort(rotated), np.sort(base), rtol=1e-9, atol=1e-12)
