"""Properties of MAREX's cyclic global test, over its input domain.

* the p-value is a multiple of ``1 / n`` in ``[1 / n, 1]``, ``n`` the blank and
  post periods;
* across the ``n`` rotations of any blank and post contrast, at most
  ``floor(alpha * n)`` reject at ``alpha``, which is the test's validity under
  a contrast whose law is invariant to rotation;
* the p-value ignores the fitting periods, the sign of the contrast, and its
  scale (by powers of two, which floating point multiplies exactly);
* the p-value is one when there is no blank period to compare with.
"""
import math
import warnings

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlsynth.utils.marex_helpers.inference import compute_inference

_VALUES = st.floats(min_value=-50, max_value=50, allow_nan=False,
                    allow_infinity=False)


@st.composite
def _window(draw, max_tb=30, max_post=8):
    tb = draw(st.integers(1, max_tb))
    h = draw(st.integers(1, max_post))
    return draw(arrays(np.float64, tb + h, elements=_VALUES)), tb, h


def _p(window, tb, fit=None):
    fit = np.zeros(3) if fit is None else np.asarray(fit, float)
    y = np.concatenate([fit, window])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_inference(y, np.zeros_like(y), fit.size + tb, fit.size,
                                 tb, global_test="cyclic").global_p_value


@settings(max_examples=300, deadline=None)
@given(_window())
def test_the_p_value_is_on_the_grid_of_the_rotations(case):
    w, tb, h = case
    n = tb + h
    p = _p(w, tb)
    assert 1 / n - 1e-12 <= p <= 1.0
    assert abs(p * n - round(p * n)) < 1e-9


@settings(max_examples=200, deadline=None)
@given(_window(), st.sampled_from([0.05, 0.1, 0.2]))
def test_at_most_alpha_of_the_rotations_reject(case, alpha):
    w, tb, h = case
    n = tb + h
    rejections = sum(_p(np.roll(w, s), tb) <= alpha for s in range(n))
    assert rejections <= math.floor(alpha * n)


@settings(max_examples=200, deadline=None)
@given(_window(), arrays(np.float64, 6, elements=_VALUES))
def test_the_fitting_periods_do_not_move_it(case, fit):
    w, tb, _ = case
    assert _p(w, tb, fit) == _p(w, tb)


@settings(max_examples=200, deadline=None)
@given(_window(), st.integers(-4, 4))
def test_sign_and_power_of_two_scale_do_not_move_it(case, k):
    w, tb, _ = case
    assert _p(-w, tb) == _p(w, tb)
    assert _p(w * 2.0 ** k, tb) == _p(w, tb)


@settings(max_examples=200, deadline=None)
@given(arrays(np.float64, st.integers(1, 12), elements=_VALUES))
def test_with_no_blank_period_it_is_one(post):
    assert _p(post, 0) == 1.0
