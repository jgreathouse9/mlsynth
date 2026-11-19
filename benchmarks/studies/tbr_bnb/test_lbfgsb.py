"""The first-order dual's L-BFGS-B without scipy's wrapper (#1068).

dual_multipliers ran scipy.optimize.minimize(method="L-BFGS-B") once per smoothing
stage. Around each call the wrapper validates and standardizes the bounds, converts
them to the Fortran routine's codes in a Python loop over every multiplier (18,627 at
96 candidate controls), and wraps the objective in a ScalarFunction. _lbfgsb is the
same loop over scipy's own setulb with that machinery inlined: one evaluation at the
clipped start, an evaluation only when the point moves, the limit on evaluations
checked at each new iteration. It returns the point minimize returns, bit for bit,
and calls the objective at the same points in the same order; on the five beds it
took 12% off the time (#1068). Outside the scipy versions whose loop it reproduces
(_lbfgsb_supported) the dual runs through minimize as before."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.optimize import Bounds, minimize

from mlsynth.utils.tbr_helpers.design.exact import slopefree
from . import beds
from mlsynth.utils.tbr_helpers.design.exact.slopefree import _lbfgsb, _lbfgsb_supported

SETTINGS = settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])


class _Quadratic:
    """f(x) = x'Hx / 2 - c'x, H = L L' + shift I, recording every point it is called at."""

    def __init__(self, n, seed, shift=0.1):
        rng = np.random.default_rng(seed)
        L = rng.standard_normal((n, n))
        self.H, self.c = L @ L.T + shift * np.eye(n), rng.standard_normal(n) * 3
        self.calls = []

    def __call__(self, x, *args):
        self.calls.append(np.array(x))
        g = self.H @ x - self.c
        return float(x @ self.H @ x / 2 - self.c @ x), g


def _reference(fun, x0, lo, hi, maxcor, maxfun, args=()):
    return minimize(fun, x0, args=args, jac=True, method="L-BFGS-B", bounds=Bounds(lo, hi),
                    options=dict(maxfun=maxfun, maxcor=maxcor)).x


def _same_run(n, seed, lo, hi, x0, maxcor, maxfun):
    a, b = _Quadratic(n, seed), _Quadratic(n, seed)
    want = _reference(a, x0, lo, hi, maxcor, maxfun)
    got = _lbfgsb(b, x0, lo, hi, maxcor, maxfun)
    assert got.tobytes() == want.tobytes()
    assert len(a.calls) == len(b.calls)
    assert all(p.tobytes() == q.tobytes() for p, q in zip(a.calls, b.calls))


@st.composite
def _problems(draw):
    n = draw(st.integers(1, 25))
    kinds = draw(st.lists(st.sampled_from(["free", "lower", "upper", "box"]), min_size=n, max_size=n))
    lo = np.array([0.0 if k in ("lower", "box") else -np.inf for k in kinds])
    hi = np.array([1.5 if k in ("upper", "box") else np.inf for k in kinds])
    x0 = np.array(draw(st.lists(st.floats(-3, 3), min_size=n, max_size=n)))   # some outside the bounds
    return n, draw(st.integers(0, 10_000)), lo, hi, x0, draw(st.sampled_from([3, 5, 10])), draw(st.integers(1, 60))


# ---------------------------------------------------------------- smoke


def test_smoke_the_loop_returns_minimizes_point():
    n = 6
    _same_run(n, 1, np.r_[np.full(3, -np.inf), np.zeros(3)], np.full(n, np.inf), np.ones(n), 5, 150)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(_problems())
def test_property_the_loop_is_minimizes_loop(problem):
    """Same point, and the objective called at the same points in the same order,
    over free, lower, upper and boxed variables, starts outside the bounds, every
    correction count the dual uses and limits on evaluations that cut runs short."""
    _same_run(*problem)


@SETTINGS
@given(_problems(), st.integers(1, 12))
def test_property_an_error_in_the_objective_leaves_at_the_same_call(problem, k):
    """dual_multipliers stops a run by raising from the objective (_Closed, _Spent):
    the k-th call raises in both, after the same calls."""
    n, seed, lo, hi, x0, maxcor, maxfun = problem

    class Stop(Exception):
        pass

    def stopping(q):
        def f(x, *args):
            out = q(x)
            if len(q.calls) == k:
                raise Stop
            return out
        return f

    a, b = _Quadratic(n, seed), _Quadratic(n, seed)
    for run, q in ((lambda f: _reference(f, x0, lo, hi, maxcor, maxfun), a),
                   (lambda f: _lbfgsb(f, x0, lo, hi, maxcor, maxfun), b)):
        try:
            run(stopping(q))
        except Stop:
            pass
    assert len(a.calls) == len(b.calls)
    assert all(p.tobytes() == q.tobytes() for p, q in zip(a.calls, b.calls))


# ---------------------------------------------------------------- the dual


def _captured_duals(name, groups):
    """The inputs of dual_multipliers over a few groups of a bed."""
    seen, real = [], slopefree.dual_multipliers

    def spy(A, R, m, c, warm=None, **kw):
        seen.append((A, R, m, c, warm, kw))
        return real(A, R, m, c, warm=warm, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(slopefree, "dual_multipliers", spy)
    try:
        for i in groups:
            beds.run_group(name, i)
    finally:
        mp.undo()
    return seen


def _same_dual(out_a, out_b):
    (ma, ia), (mb, ib) = out_a, out_b
    assert ia["evals"] == ib["evals"]
    assert (ma is None) == (mb is None)
    if ma is not None:
        assert ma.level == mb.level
        assert np.asarray(ma.y_eq).tobytes() == np.asarray(mb.y_eq).tobytes()
        assert np.asarray(ma.y_in).tobytes() == np.asarray(mb.y_in).tobytes()
    pa, pb = ia["point"], ib["point"]
    assert (pa is None) == (pb is None)
    if pa is not None:
        assert pa.tobytes() == pb.tobytes()


def test_the_dual_over_the_loop_is_the_dual_over_minimize(monkeypatch):
    calls = _captured_duals("s20_k4", [0, 1])
    assert len(calls) >= 5
    for A, R, m, c, warm, kw in calls:
        direct = slopefree.dual_multipliers(A, R, m, c, warm=warm, **kw)
        monkeypatch.setattr(slopefree, "_LBFGSB_DIRECT", False)
        wrapped = slopefree.dual_multipliers(A, R, m, c, warm=warm, **kw)
        monkeypatch.setattr(slopefree, "_LBFGSB_DIRECT", True)
        _same_dual(direct, wrapped)


def test_a_bed_over_the_loop_is_the_bed_over_minimize(monkeypatch):
    fields = ("certified", "controls", "nodes", "full", "shor", "open", "rounded")
    direct = [beds.run_group("s15_k3", i) for i in range(4)]
    monkeypatch.setattr(slopefree, "_LBFGSB_DIRECT", False)
    wrapped = [beds.run_group("s15_k3", i) for i in range(4)]
    for a, b in zip(direct, wrapped):
        assert all(getattr(a, f) == getattr(b, f) for f in fields)


def test_the_search_uses_the_loop_on_this_scipy(monkeypatch):
    import scipy

    assert slopefree._LBFGSB_DIRECT == _lbfgsb_supported(scipy.__version__)
    seen, real = [], slopefree._lbfgsb
    monkeypatch.setattr(slopefree, "_lbfgsb", lambda *a, **k: seen.append(1) or real(*a, **k))
    beds.run_group("s20_k4", 0)
    assert seen if slopefree._LBFGSB_DIRECT else not seen


# ---------------------------------------------------------------- edges


def test_one_variable():
    _same_run(1, 3, np.zeros(1), np.full(1, np.inf), np.array([2.0]), 5, 150)


def test_a_start_at_the_optimum_stops_after_its_first_evaluation():
    q = _Quadratic(3, 0)
    x = np.linalg.solve(q.H, q.c)
    a, b = _Quadratic(3, 0), _Quadratic(3, 0)
    want = _reference(a, x, np.full(3, -np.inf), np.full(3, np.inf), 5, 150)
    got = _lbfgsb(b, x, np.full(3, -np.inf), np.full(3, np.inf), 5, 150)
    assert got.tobytes() == want.tobytes() and len(a.calls) == len(b.calls) <= 2


def test_one_evaluation_allowed():
    _same_run(4, 2, np.zeros(4), np.full(4, np.inf), np.full(4, 2.0), 5, 1)


def test_the_start_is_not_changed():
    x0 = np.array([-1.0, 2.0])
    keep = x0.copy()
    _lbfgsb(_Quadratic(2, 0), x0, np.zeros(2), np.full(2, np.inf), 5, 150)
    assert np.array_equal(x0, keep)


# ---------------------------------------------------------------- the version guard


@pytest.mark.parametrize("version,ok", [("1.15.0", True), ("1.16.2", True), ("1.18.1", True),
                                        ("1.18.0rc1", True), ("1.14.1", False), ("1.19.0", False),
                                        ("2.0.0", False), ("1.20.0.dev0+123", False)])
def test_the_loop_is_used_only_on_the_scipy_versions_it_reproduces(version, ok):
    assert _lbfgsb_supported(version) is ok


@pytest.mark.parametrize("version", ["", "one.two", "1"])
def test_an_unreadable_scipy_version_takes_minimize(version):
    assert _lbfgsb_supported(version) is False
