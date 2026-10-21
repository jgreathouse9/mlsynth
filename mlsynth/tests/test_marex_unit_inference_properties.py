"""Properties of per-unit inference, over its input domain.

* The joint p-value lies on the arrangement grid and at or above ``1 / |Pi|``
  when exact, at or above ``1 / (1 + B)`` when drawn, and ignores each unit's
  scale and the order of the units.
* Each unit's control is a simplex weighting of its cluster's untreated units;
  eq. (11)'s effect is the weighted sum of the unit contrasts; Bonferroni is
  ``min(1, m p)``.
"""
import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlsynth.utils.marex_helpers.unit_inference import (joint_p_value,
                                                        unit_level_inference)

#: No subnormals: rescaling one by a factor below 1/2 rounds it to zero, which
#: erases a unit's variation in floating point while leaving it in the reals.
_X = st.floats(-50, 50, allow_nan=False, allow_infinity=False,
               allow_subnormal=False)


@st.composite
def _window(draw):
    m = draw(st.integers(1, 4))
    n = draw(st.integers(2, 9))
    h = draw(st.integers(1, n - 1))
    U = draw(arrays(np.float64, (m, n), elements=_X))
    return U, h


@settings(max_examples=200, deadline=None)
@given(_window(), st.integers(0, 2 ** 31 - 1))
def test_the_exact_joint_p_value_is_on_the_grid(case, seed):
    U, h = case
    n_arr = math.comb(U.shape[1], h)
    p = joint_p_value(U, h, max_combinations=n_arr,
                      rng=np.random.default_rng(seed))
    assert 1 / n_arr - 1e-15 <= p <= 1
    assert abs(p * n_arr - round(p * n_arr)) < 1e-9


@settings(max_examples=100, deadline=None)
@given(_window(), st.integers(1, 40), st.integers(0, 2 ** 31 - 1))
def test_a_drawn_joint_p_value_counts_the_observed(case, B, seed):
    U, h = case
    p = joint_p_value(U, h, max_combinations=B, rng=np.random.default_rng(seed))
    assert 1 / (B + 1) - 1e-15 <= p <= 1


@settings(max_examples=150, deadline=None)
@given(_window(), st.integers(0, 2 ** 31 - 1))
def test_the_joint_p_value_ignores_scale_and_unit_order(case, seed):
    U, h = case
    n_arr = math.comb(U.shape[1], h)
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.01, 100, (U.shape[0], 1))
    order = rng.permutation(U.shape[0])
    base = joint_p_value(U, h, max_combinations=n_arr, rng=np.random.default_rng(0))
    assert joint_p_value(U * scales, h, max_combinations=n_arr,
                         rng=np.random.default_rng(0)) == base
    assert joint_p_value(U[order], h, max_combinations=n_arr,
                         rng=np.random.default_rng(0)) == base


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2 ** 31 - 1), st.integers(1, 3))
def test_controls_eq11_and_bonferroni_hold_on_any_panel(seed, n_treated):
    rng = np.random.default_rng(seed)
    N, T0, Tb, H = 8, 30, 12, 5
    T = T0 + H
    Y = rng.normal(10, 2, (N, T))
    members = np.arange(N)
    treated = rng.choice(N, size=n_treated, replace=False)
    w = np.zeros(N)
    w[treated] = rng.dirichlet(np.ones(n_treated))
    lvl = unit_level_inference(
        Y, [("c0", members, w)], w, [f"u{i}" for i in range(N)],
        T0=T0, TcE=T0 - Tb, Tb=Tb, alpha=0.1, max_combinations=200,
        random_state=0, T_post=None)
    m = len(lvl.units)
    assert m == n_treated
    pool = {f"u{i}" for i in range(N) if i not in set(treated)}
    total = np.zeros(T)
    for label, ui in lvl.units.items():
        v = np.array(list(ui.control_weights.values()))
        assert set(ui.control_weights) == pool
        assert np.all(v >= -1e-12) and abs(v.sum() - 1) < 1e-9
        assert ui.p_bonferroni == min(1.0, m * ui.inference.global_p_value)
        total += w[int(label[1:])] * (Y[int(label[1:])] - ui.synthetic_control)
    np.testing.assert_allclose(lvl.eq11.fulltreated_effects, total,
                               rtol=1e-10, atol=1e-10)



from hypothesis import assume

from mlsynth.utils.marex_helpers.unit_inference import blank_check

_B = st.floats(-100, 100, allow_nan=False, allow_infinity=False)
#: A blank stretch with a spread well above ``_ar1_rho``'s zero-variance cut.
_BLANK = arrays(np.float64, st.integers(2, 30), elements=_B).filter(
    lambda b: np.ptp(b) > 1e-3)


def _close(x, y, tol):
    return abs(x - y) <= tol * (1.0 + abs(y))


@settings(max_examples=200, deadline=None)
@given(_BLANK, st.floats(min_value=0.01, max_value=100))
def test_the_blank_check_scales_with_the_units_and_its_t_does_not(b, c):
    level = np.abs(b) + 50.0
    a, s = blank_check(b, level), blank_check(c * b, c * level)
    assert _close(s.mean, c * a.mean, 1e-9)
    assert _close(s.se, c * a.se, 1e-7)
    assert _close(s.t, a.t, 1e-6)
    assert _close(s.relative, a.relative, 1e-9)
    assert s.rho == a.rho or _close(s.rho, a.rho, 1e-9)


@settings(max_examples=200, deadline=None)
@given(_BLANK, _B)
def test_a_shift_moves_the_mean_and_not_the_spread(b, k):
    level = np.full(b.size, 10.0)
    a, s = blank_check(b, level), blank_check(b + k, level)
    assert _close(s.mean, a.mean + k, 1e-9)
    assert _close(s.se, a.se, 1e-6)
    assert _close(s.rho, a.rho, 1e-6)


@settings(max_examples=200, deadline=None)
@given(arrays(np.float64, st.integers(1, 30), elements=_B))
def test_flipping_the_sign_flips_the_mean_and_t_and_keeps_the_p_value(b):
    level = np.full(b.size, 10.0)
    a, s = blank_check(b, level), blank_check(-b, level)
    assert s.mean == -a.mean and s.relative == -a.relative
    assert s.se == a.se or (np.isnan(s.se) and np.isnan(a.se))
    assert s.t == -a.t or (np.isnan(s.t) and np.isnan(a.t))
    assert s.p_value == a.p_value or (np.isnan(s.p_value) and np.isnan(a.p_value))
    assert np.isnan(s.p_value) or 0.0 <= s.p_value <= 1.0
