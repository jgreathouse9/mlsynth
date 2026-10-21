"""Properties of the signed statistic and the reference window, over their domain.

Each property is an invariant the example tests check at one point:

* the block reference for the signed statistic is the statistic of every cyclic
  shift, shift zero is the observed value exactly, and so the p-value never
  falls below one over the window;
* with one post period the signed statistic and ``S_1`` coincide;
* the iid reference sums the same permutations the ``S_q`` reference draws;
* the signed statistic depends on the post block only through its sum.
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlsynth.utils.conformal.ridge_inference import _reference_stats, _stat

_FLOATS = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False,
                    allow_infinity=False)


@st.composite
def _path_and_post(draw, min_size=3, max_size=40):
    n = draw(st.integers(min_size, max_size))
    r = draw(arrays(np.float64, n, elements=_FLOATS))
    h = draw(st.integers(1, n - 1))
    return r, h


@settings(max_examples=200, deadline=None)
@given(_path_and_post())
def test_the_signed_block_reference_is_every_shift_and_floors_the_p_value(case):
    r, h = case
    n = r.size
    ref = _reference_stats(r, slice(n - h, None), 1.0, conformal_type="block",
                           ns=0, seed=0, statistic="signed")
    loop = np.array([_stat(np.roll(r, s)[n - h:], 1.0, "signed") for s in range(n)])
    np.testing.assert_array_equal(ref, loop)
    obs = _stat(r[n - h:], 1.0, "signed")
    assert ref[0] == obs
    assert np.mean(obs <= ref) >= 1.0 / n


@settings(max_examples=200, deadline=None)
@given(_path_and_post(), st.sampled_from(["block", "iid"]),
       st.integers(0, 2 ** 31 - 1))
def test_one_post_period_makes_signed_and_absolute_coincide(case, scheme, seed):
    r, _ = case
    n = r.size
    a = _reference_stats(r, slice(n - 1, None), 1.0, conformal_type=scheme,
                         ns=30, seed=seed, statistic="signed")
    b = _reference_stats(r, slice(n - 1, None), 1.0, conformal_type=scheme,
                         ns=30, seed=seed, statistic="sq")
    np.testing.assert_array_equal(a, b)


@settings(max_examples=150, deadline=None)
@given(_path_and_post(), st.integers(0, 2 ** 31 - 1))
def test_the_signed_iid_reference_sums_the_permutations_sq_draws(case, seed):
    r, h = case
    n = r.size
    ref = _reference_stats(r, slice(n - h, None), 1.0, conformal_type="iid",
                           ns=25, seed=seed, statistic="signed")
    rng = np.random.default_rng(seed)
    want = np.array([abs(rng.permutation(r)[n - h:].sum()) / np.sqrt(h)
                     for _ in range(25)])
    np.testing.assert_allclose(ref, want, rtol=1e-12, atol=1e-9)


@settings(max_examples=200, deadline=None)
@given(arrays(np.float64, st.integers(1, 30), elements=_FLOATS),
       st.integers(0, 2 ** 31 - 1))
def test_the_signed_statistic_reads_the_block_only_through_its_sum(x, seed):
    perm = np.random.default_rng(seed).permutation(x)
    assert _stat(perm, statistic="signed") == \
        abs(perm.sum()) / np.sqrt(x.size)
    shifted = x.copy()
    if x.size > 1:
        # moving mass between periods leaves the sum, and the statistic, alone
        shifted[0] += 7.0
        shifted[1] -= 7.0
        assert np.isclose(_stat(shifted, statistic="signed"),
                          _stat(x, statistic="signed"), rtol=1e-9, atol=1e-6)
