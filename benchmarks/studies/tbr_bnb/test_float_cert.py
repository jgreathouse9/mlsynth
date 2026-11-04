"""The hull's first-order certificate in floating point with a rigorous rounding
bound (#965, step 3), and the stage's decision with the exact one as fallback."""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from .brute import panel
from .certificate import (ExactData, dual_bound, gradient_bound, hull_bound, hull_bound_float, hull_clears,
                          inflate, knapsack_dual, linear_bound, objective_bound, perturbed_knapsack_bound,
                          qarr, residual_bound, round_down)
from .hull import Band

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _knapsack_exact(g, lo, hi, m):
    """min g'x over {0 <= x <= hi, m lo <= sum x <= m hi} in rationals, by the greedy fill."""
    g = [Q(float(v)) for v in g]
    lo, hi = Q(float(lo)), Q(float(hi))
    A, B = m * lo, m * hi
    s, val = Q(0), Q(0)
    for j in sorted(range(len(g)), key=lambda j: g[j]):
        if s >= B or (g[j] >= 0 and s >= A):
            break
        t = min(hi, (B - s) if g[j] < 0 else (A - s))
        s += t; val += g[j] * t
    return val


def _draw(lo, width):
    return lo * width if lo > 0 else width - 1.0


def _point(y, X, lo, hi, m, shift, seed):
    _, w = Band(y, X).value(lo, hi, m)
    return w + shift * np.random.default_rng(seed).normal(size=X.shape[1])


# ---------------------------------------------------------------- smoke


def test_smoke_the_float_bound_is_a_finite_float():
    y, X = panel(0)
    w = _point(y, X, 0.3, 0.9, 3, 0.0, 0)
    b = hull_bound_float(y, X, w, 0.3, 0.9, 3)
    assert isinstance(b, float) and np.isfinite(b)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", range(6))
def test_the_float_bound_is_the_exact_one_to_rounding(seed):
    """Below the exact certificate, and within 1e-10 of it relative to the data's
    size: the rounding bound costs tightness only at the scale of eps."""
    y, X = panel(seed)
    D = ExactData(y, X)
    for lo, hi, m in ((0.3, 0.9, 3), (0.0, 0.2, 2), (1.5, 2.0, 5), (0.05, 0.08, 6)):
        w = _point(y, X, lo, hi, m, 0.0, seed)
        exact = hull_bound(D, w, lo, hi, m)
        fl = hull_bound_float(y, X, w, lo, hi, m)
        assert Q(fl) <= exact
        assert float(exact) - fl <= 1e-10 * float(y @ y)


def test_the_knapsack_dual_is_tight_at_the_greedy_multiplier():
    """At the greedy fill's marginal cost the LP dual equals the knapsack minimum."""
    g = np.array([-3.0, -1.0, 0.5, 2.0, -0.25])
    for lo, hi, m in ((0.1, 0.3, 2), (0.0, 0.4, 3), (0.2, 0.2, 4), (0.05, 1.0, 1)):
        exact = float(_knapsack_exact(g, lo, hi, m))
        assert knapsack_dual(g, lo, hi, m) == pytest.approx(exact, rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("seed", range(4))
def test_the_clearing_decision_is_the_exact_one(seed):
    """hull_clears prunes exactly the pairs the exact certificate prunes: the float
    bound decides when it clears, and the exact one otherwise."""
    y, X = panel(seed)
    D = ExactData(y, X)
    band = Band(y, X)
    for lo, hi, m in ((0.3, 0.9, 3), (0.1, 0.25, 2), (0.05, 0.6, 5), (1.0, 1.2, 4)):
        _, w = band.value(lo, hi, m)
        exact = hull_bound(D, w, lo, hi, m)
        for UB in (exact * Q(99, 100), exact, exact + Q(1, 10 ** 9), exact * Q(101, 100)):
            stats = {}
            assert hull_clears(D, y, X, w, lo, hi, m, UB, stats) == (exact >= UB)


def test_the_exact_certificate_is_reached_only_when_the_float_one_falls_short():
    y, X = panel(1)
    D = ExactData(y, X)
    _, w = Band(y, X).value(0.3, 0.9, 3)
    exact = hull_bound(D, w, 0.3, 0.9, 3)
    stats = {}
    assert hull_clears(D, y, X, w, 0.3, 0.9, 3, exact * Q(9, 10), stats)
    assert stats.get("exact", 0) == 0 and stats["float"] == 1
    assert hull_clears(D, y, X, w, 0.3, 0.9, 3, exact, stats)          # a tie: only the exact one decides
    assert stats["exact"] == 1


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 6), shift=st.floats(-1.0, 1.0), k=st.integers(-30, 30))
@example(seed=0, lo=0.3, width=3.0, m=3, shift=0.0, k=30)
@example(seed=0, lo=0.3, width=3.0, m=3, shift=0.0, k=-30)
def test_property_the_float_bound_never_exceeds_the_exact_one(seed, lo, width, m, shift, k):
    """At any point, on data scaled by 2^k (a scale the rounding bound must follow),
    the float bound read as a rational is at most the exact certificate."""
    y, X = panel(seed)
    y, X = y * 2.0 ** k, X * 2.0 ** k
    hi = _draw(lo, width)
    w = _point(y, X, lo, hi, m, shift, seed)
    assert Q(hull_bound_float(y, X, w, lo, hi, m)) <= hull_bound(ExactData(y, X), w, lo, hi, m)


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 6), lo=st.floats(0.0, 1.0),
       width=st.floats(1.0, 4.0), lam=st.floats(-50.0, 50.0))
def test_property_the_knapsack_dual_is_a_lower_bound_at_any_multiplier(seed, m, lo, width, lam):
    """Weak duality: every multiplier on the sum gives a value at most the minimum,
    up to the double rounding of its own sum (which hull_bound_float bounds)."""
    g = np.random.default_rng(seed).normal(size=6) * 10
    hi = _draw(lo, width) if width > 1.0 else lo
    exact = _knapsack_exact(g, lo, hi, m)
    assert Q(knapsack_dual(g, lo, hi, m, lam=lam)) <= exact + Q(1, 10 ** 12) * (1 + abs(exact))


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 6), cut=st.floats(0.5, 1.5))
def test_property_the_clearing_decision_is_the_exact_one(seed, lo, width, m, cut):
    y, X = panel(seed)
    hi = _draw(lo, width)
    D = ExactData(y, X)
    _, w = Band(y, X).value(lo, hi, m)
    exact = hull_bound(D, w, lo, hi, m)
    UB = exact * Q(cut)
    assert hull_clears(D, y, X, w, lo, hi, m, UB, {}) == (exact >= UB)


@SETTINGS
@given(num=st.integers(-10 ** 30, 10 ** 30), den=st.integers(1, 10 ** 30))
@example(num=1, den=3)
@example(num=-1, den=3)
@example(num=2, den=3)
def test_property_round_down_is_the_largest_double_below(num, den):
    """The rounded double never exceeds the rational, and the next double up does."""
    x = Q(num, den)
    d = round_down(x)
    assert Q(d) <= x
    assert Q(float(np.nextafter(d, np.inf))) > x


# ---------------------------------------------------------------- the components, against exact rationals
#
# Each error term is a worst-case bound, orders of magnitude above the rounding
# random data produce, so the whole certificate has no power to see one dropped.
# Each component is checked on its own, on inputs that make its term's rounding
# real: a check that fails for the component without its term.


def _case(seed, k=0, tiny=False):
    y, X = panel(seed)
    if tiny:                                  # X w far below y: the subtraction's rounding is y's
        X = X * 1e-12
    y, X = y * 2.0 ** k, X * 2.0 ** k
    w = np.random.default_rng(seed).uniform(0, 0.4, size=X.shape[1])
    return y, X, w


def _exact_residual(y, X, w):
    return qarr(X) @ qarr(w) - qarr(y)


@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(-20, 20), tiny=st.booleans())
@example(seed=0, k=0, tiny=True)
def test_property_the_residual_bound_holds(seed, k, tiny):
    y, X, w = _case(seed, k, tiny)
    r, e = residual_bound(y, X, w)
    exact = _exact_residual(y, X, w)
    assert all(abs(Q(float(a)) - b) <= Q(inflate(float(c))) for a, b, c in zip(r, exact, e))


@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(-20, 20), tiny=st.booleans())
def test_property_the_objective_bound_holds(seed, k, tiny):
    y, X, w = _case(seed, k, tiny)
    r, e = residual_bound(y, X, w)
    f, err = objective_bound(r, e)
    exact = _exact_residual(y, X, w)
    assert Q(f) - Q(inflate(err)) <= exact @ exact


@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(-20, 20), tiny=st.booleans())
def test_property_the_gradient_bound_holds(seed, k, tiny):
    y, X, w = _case(seed, k, tiny)
    r, e = residual_bound(y, X, w)
    g, d = gradient_bound(X, r, e)
    exact = 2 * (qarr(X).T @ _exact_residual(y, X, w))
    assert all(abs(Q(float(a)) - b) <= Q(inflate(float(c))) for a, b, c in zip(g, exact, d))


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 6), lo=st.floats(0.0, 1.0),
       width=st.floats(1.0, 4.0), scale=st.floats(1e-3, 1e6))
@example(seed=0, m=2, lo=0.0, width=2.0, scale=10.0)    # the dual's sum rounds above the minimum here
def test_property_the_dual_bound_holds(seed, m, lo, width, scale):
    g = np.random.default_rng(seed).normal(size=6) * scale
    hi = _draw(lo, width) if width > 1.0 else lo
    v, err = dual_bound(g, lo, hi, m)
    assert Q(v) - Q(inflate(err)) <= _knapsack_exact(g, lo, hi, m)


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 6), lo=st.floats(0.0, 0.5),
       width=st.floats(1.5, 4.0), size=st.floats(0.1, 5.0))
def test_property_the_perturbed_knapsack_bound_holds(seed, m, lo, width, size):
    """For the gradient lowered by its whole error bound, the worst case for a
    filled point, and a fill that runs to m hi."""
    rng = np.random.default_rng(seed)
    gt = -np.abs(rng.normal(size=6)) * 10            # all negative: the fill ends at m hi
    d = np.full(6, size)
    hi = _draw(lo, width)
    v, err = perturbed_knapsack_bound(gt, d, lo, hi, m)
    assert Q(v) - Q(inflate(err)) <= _knapsack_exact(gt - d, lo, hi, m)


@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(-20, 20), size=st.floats(0.0, 5.0))
@example(seed=0, k=0, size=3.0)       # d'|w| rounds below its exact value: only the inflation covers it
def test_property_the_linear_bound_holds(seed, k, size):
    """For the gradient raised by its whole error bound along the sign of w, the worst case."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=6) * 2.0 ** k; w = rng.uniform(-1, 1, size=6)
    d = np.full(6, size * 2.0 ** k)
    v, err = linear_bound(g, d, w)
    gt = [Q(float(a)) + Q(float(b)) * (1 if c >= 0 else -1) for a, b, c in zip(g, d, w)]
    assert sum(a * Q(float(c)) for a, c in zip(gt, w)) <= Q(v) + Q(inflate(err))


# ---------------------------------------------------------------- edges


def test_one_candidate_control():
    y, X = panel(2, p=1)
    w = np.array([0.5])
    assert Q(hull_bound_float(y, X, w, 0.2, 0.9, 1)) <= hull_bound(ExactData(y, X), w, 0.2, 0.9, 1)


def test_the_origin_and_a_band_from_zero():
    y, X = panel(3)
    w = np.zeros(6)
    assert Q(hull_bound_float(y, X, w, 0.0, 0.5, 4)) <= hull_bound(ExactData(y, X), w, 0.0, 0.5, 4)


def test_a_point_band():
    y, X = panel(4)
    w = np.full(6, 0.4) / 2
    assert Q(hull_bound_float(y, X, w, 0.4, 0.4, 3)) <= hull_bound(ExactData(y, X), w, 0.4, 0.4, 3)


def test_collinear_controls():
    y, X = panel(5)
    X = np.column_stack([X, X[:, :2]])
    w = np.full(8, 0.1)
    assert Q(hull_bound_float(y, X, w, 0.1, 0.6, 4)) <= hull_bound(ExactData(y, X), w, 0.1, 0.6, 4)


def test_a_band_wider_than_the_controls_fill():
    """m hi above what p caps of hi can hold: the fill takes every negative cost."""
    g = np.array([-2.0, -1.0, 3.0])
    assert knapsack_dual(g, 0.0, 0.5, 9) == pytest.approx(float(_knapsack_exact(g, 0.0, 0.5, 9)))


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_point_is_refused(bad):
    y, X = panel(0)
    w = np.full(6, 0.1); w[2] = bad
    with pytest.raises(ValueError, match="finite"):
        hull_bound_float(y, X, w, 0.3, 0.9, 3)


def test_a_point_of_the_wrong_shape_is_refused():
    y, X = panel(0)
    with pytest.raises(ValueError, match="one weight per control"):
        hull_bound_float(y, X, np.full(5, 0.1), 0.3, 0.9, 3)
