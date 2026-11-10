"""The smoothed dual reads its unsmoothed value off its own eigendecomposition, and
L-BFGS-B gets its bounds built once (slopefree.dual_multipliers).

Each evaluation used to build S(y) twice and decompose it twice: eigh for the
smoothing, eigvalsh for the value b'y + T max(0, lambda_max) that decides whether
the dual has closed. The largest eigenvalue is in eigh's output, so the value comes
with the smoothing. The bounds were a list of a few thousand (lo, hi) pairs that
every minimize call converted; they are one Bounds object per dual. Neither changes
an iterate, so the closures and evaluation counts are the old ones.
"""
import numpy as np
import pytest
import scipy.optimize
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import slopefree as sf
from .slopefree import Rows, dyadic_below, objective
from .test_cascade import _at_optimum

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _dual(seed=0, m=3):
    c, G, lam, C = _at_optimum(seed)
    A = objective(c, G, dyadic_below(lam))
    R = Rows(c, m, cuts=[C])
    return A, R, m, c


def test_smoke_the_smoothing_comes_with_the_value():
    A, R, m, c = _dual()
    D = sf._Dual(A, R, m, c)
    y = np.abs(np.random.default_rng(0).normal(size=D.size)) * 1e-2
    f, g, v = D.smoothed_and_value(y, 1e-2)
    f0, g0 = D.smoothed(y, 1e-2)
    assert f == f0 and np.array_equal(g, g0) and np.isfinite(v)


@SETTINGS
@given(seed=st.integers(0, 10_000), a=st.floats(1e-4, 1.0), scale=st.floats(1e-4, 1e2))
def test_property_the_value_is_the_unsmoothed_certificate(seed, a, scale):
    """eigh's largest eigenvalue is eigvalsh's to rounding, so the value is value()'s."""
    A, R, m, c = _dual()
    D = sf._Dual(A, R, m, c)
    rng = np.random.default_rng(seed)
    y = rng.normal(size=D.size) * scale
    y[D.ne:] = np.abs(y[D.ne:])
    _, _, v = D.smoothed_and_value(y, a)
    want = D.value(y)
    lam = np.linalg.eigvalsh(D.S(y))
    assert abs(v - want) <= 1e-12 * (abs(float(D.b @ y)) + D.T * float(np.abs(lam).max())) + 1e-300


def test_each_evaluation_decomposes_once(monkeypatch):
    """value() is read once, at the start; the evaluations take it from the smoothing."""
    calls = []
    real = sf._Dual.value
    monkeypatch.setattr(sf._Dual, "value", lambda self, y: calls.append(1) or real(self, y))
    A, R, m, c = _dual()
    w, info = sf.dual_multipliers(A, R, m, c)
    assert info["evals"] > 1 and len(calls) == 1


def test_the_bounds_are_built_once(monkeypatch):
    seen = []
    real = scipy.optimize.minimize

    def spy(fun, x0, **kw):
        seen.append(kw["bounds"])
        return real(fun, x0, **kw)

    monkeypatch.setattr(scipy.optimize, "minimize", spy)
    sf.dual_multipliers(*_dual_open())
    assert seen and all(isinstance(b, scipy.optimize.Bounds) for b in seen)
    assert all(b is seen[0] for b in seen)
    D = sf._Dual(*_dual_open())
    assert np.all(seen[0].lb[:D.ne] == -np.inf) and np.all(seen[0].lb[D.ne:] == 0.0)
    assert np.all(seen[0].ub == np.inf)


def _dual_open(seed=0):
    """A size the dual cannot close (below the optimum's ratio), so every stage runs."""
    from fractions import Fraction as Q
    c, G, lam, C = _at_optimum(seed)
    A = objective(c, G, dyadic_below(lam * Q(1, 2)))
    R = Rows(c, len(C), cuts=[])
    return A, R, len(C), c


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("m", [2, 3])
def test_the_closures_and_evaluations_are_the_old_ones(monkeypatch, seed, m):
    """The old evaluation, smoothing then value(), against the new: the same closure,
    the same number of evaluations."""
    A, R, mm, c = _dual(seed, m)
    new = sf.dual_multipliers(A, R, mm, c)

    def old_evaluation(self, y, a):               # as it was: eigh for the smoothing, value() after
        lam, Q = np.linalg.eigh(self.S(y))
        z = sf._capped(lam / a, self.T)
        Z = (Q * z) @ Q.T
        return (float(self.b @ y + z @ lam - a / 2 * (z @ z)), self.b - self.M @ Z.reshape(-1, order="F"),
                self.value(y))

    monkeypatch.setattr(sf._Dual, "smoothed_and_value", old_evaluation)
    old = sf.dual_multipliers(A, R, mm, c)
    assert (new[0] is None) == (old[0] is None) and new[1]["evals"] == old[1]["evals"]
    if new[0] is not None:
        assert np.allclose(new[0].y_in, old[0].y_in) and np.allclose(new[0].y_eq, old[0].y_eq)


@SETTINGS
@given(seed=st.integers(0, 10_000), shift=st.floats(-1e3, 1e3))
def test_property_the_value_is_floored_at_b_y(seed, shift):
    """F(y) = b'y + T max(0, lambda_max): with S(y) shifted below zero the value is b'y,
    never below it, and it is value()'s on either side of zero."""
    A, R, m, c = _dual()
    D = sf._Dual(A, R, m, c)
    S0 = D.S
    D.S = lambda y: S0(y) - shift * np.eye(D.n)
    y = np.abs(np.random.default_rng(seed).normal(size=D.size)) * 1e-2
    _, _, v = D.smoothed_and_value(y, 1e-2)
    by = float(D.b @ y)
    assert v >= by and abs(v - D.value(y)) <= 1e-12 * (abs(by) + D.T * (abs(shift) + D.scale)) + 1e-300


def test_a_negative_definite_dual_has_value_b_y():
    A, R, m, c = _dual()
    D = sf._Dual(A, R, m, c)
    D.S = lambda y: -np.eye(D.n)
    y = np.zeros(D.size)
    y[D.ne:] = 1.0
    _, _, v = D.smoothed_and_value(y, 1e-2)
    assert v == float(D.b @ y) == D.value(y)
