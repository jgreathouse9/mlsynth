"""The slope-free certificate in floating point (#965, fbound.py).

bound() sums S and k in rationals and bounds lambda_max(S) exactly; fbound sums them
in doubles with an entrywise bound on the rounding, and bounds lambda_max of the
exact S through Weyl's inequality (Rump 2010, Lemma 10.12). The tests check the
enclosures against the rational sums, the upper bound against every design and
against an exact lower bound on the dual function, the lower bound against
bound(), and the decision: the floats close a size alone, rule it out alone, and
hand it to bound() only when they straddle zero."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import fbound, slopefree as sf
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import Rows, _slack, bound, certify_slope_free, dyadic_below, objective, size_certificate

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(sf, "PRICE_LIMIT", 0)


def _instance(seed=2, p=5, frac=Q(1, 2)):
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    lam = dyadic_below(D.yy * frac)
    return D, lam, objective(D.c, D.G, lam)


def _rows(D, m, face=True, fixes=(), cuts=(), triangles=(), box=False):
    return Rows(D.c, m, cuts=cuts, fixes=fixes, triangles=triangles, face=face, box=box)


def _multipliers(R, seed, scale=1.0, negative=True):
    rng = np.random.default_rng(seed)
    y_eq = rng.normal(size=R.n_eq) * scale
    y_in = rng.normal(size=R.n_in) * scale
    if not negative:
        y_in = np.abs(y_in)
    return y_eq, y_in


def _designs(D, m, fixes=()):
    """The designs of size m within the fixings with c'u >= 0: every one is a point
    of the relaxation."""
    fx = dict(fixes)
    for C in itertools.combinations(range(len(D.c)), m):
        if all((j in C) == bool(v) for j, v in fx.items()) and sum((D.c[j] for j in C), Q(0)) >= 0:
            yield C


def _value(A, C):
    return sum((A[i, j] for i in C for j in C), Q(0))


def _exact_face(R, S):
    T = R.face.integer_basis()
    return T.T.dot(S).dot(T)


def _rayleigh_low(S):
    """An exact lower bound on lambda_max of the rational S: the Rayleigh quotient at
    a double vector, in rationals."""
    _, V = np.linalg.eigh(np.array(S, dtype=float))
    v = [Q(float(x)) for x in V[:, -1]]
    num = sum((v[i] * S[i, j] * v[j] for i in range(len(v)) for j in range(len(v))), Q(0))
    return num / sum((x * x for x in v), Q(0))


def _dual_low(A, R, y_eq, y_in, m):
    """An exact lower bound on the dual function bound() bounds above: k + T
    max(0, lambda_max), on the full space and on the face, the smaller."""
    S, k = _slack(A, R, y_eq, y_in)
    full = k + (1 + m) * max(Q(0), _rayleigh_low(S))
    if R.face is None:
        return full
    return min(full, k + R.face.trace * max(Q(0), _rayleigh_low(_exact_face(R, S))))


def _at_optimum(seed=2, p=5, m=2):
    """A size closed above the best ratio, and the multipliers that closed it."""
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    _, C = positive_optimum(y, X)
    u = np.zeros(p, dtype=object); u[:] = Q(0); u[list(C)] = Q(1)
    lam = dyadic_below((D.c @ u) ** 2 / (u @ D.G @ u) * Q(11, 10))
    res = size_certificate(D.c, D.G, lam, m)
    return D, lam, objective(D.c, D.G, lam), res


# ---------------------------------------------------------------- smoke


def test_smoke_the_float_bound_is_a_finite_double():
    D, lam, A = _instance()
    R = _rows(D, 2)
    y_eq, y_in = _multipliers(R, 0)
    up = fbound.bound_float(fbound.objective_float(A), R, y_eq, y_in, 2)
    assert isinstance(up, float) and np.isfinite(up)


def test_smoke_the_certificate_closes_a_size_its_relaxation_closes():
    D, lam, A, res = _at_optimum()
    assert res.certified
    w = res.multipliers
    R = Rows(D.c, 2, face=True, level=w.level)
    b = fbound.certificate(A, R, w.y_eq, w.y_in, 2)
    assert isinstance(b, Q) and b <= 0


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("face", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_the_float_bound_is_the_exact_one_to_rounding(seed, face):
    """Within 1e-9 of bound() relative to the dual's scale, and never below the
    exact lower bound on the dual function."""
    D, lam, A = _instance(seed)
    R = _rows(D, 2, face=face)
    y_eq, y_in = _multipliers(R, seed)
    up = fbound.bound_float(fbound.objective_float(A), R, y_eq, y_in, 2)
    exact = bound(A, R, y_eq, y_in, 2)
    scale = float(np.abs(np.array(_slack(A, R, y_eq, y_in)[0], dtype=float)).sum()) + 1.0
    assert abs(up - float(exact)) <= 1e-9 * scale * 3
    assert Q(up) >= _dual_low(A, R, y_eq, y_in, 2)


def test_the_slack_encloses_the_rational_sums():
    D, lam, A = _instance()
    R = _rows(D, 3, cuts=[{0, 1, 2}])
    y_eq, y_in = _multipliers(R, 1)
    S, k, E, ek = fbound.slack_float(fbound.objective_float(A), fbound.float_rows(R), y_eq, y_in)
    Sx, kx = _slack(A, R, y_eq, y_in)
    assert all(abs(Q(float(S[i, j])) - Sx[i, j]) <= Q(float(E[i, j])) for i in range(R.n) for j in range(R.n))
    assert abs(Q(k) - kx) <= Q(ek)
    assert np.array_equal(S, S.T)


def test_negative_inequality_multipliers_are_clipped_as_bound_clips_them():
    D, lam, A = _instance()
    R = _rows(D, 2)
    y_eq, y_in = _multipliers(R, 3)
    clipped = np.maximum(y_in, 0.0)
    a = fbound.slack_float(fbound.objective_float(A), fbound.float_rows(R), y_eq, y_in)
    b = fbound.slack_float(fbound.objective_float(A), fbound.float_rows(R), y_eq, clipped)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert (y_in < 0).any()


def test_the_face_encloses_the_rational_face():
    D, lam, A = _instance(p=6)
    R = _rows(D, 3, fixes=((0, 1), (4, 0)))
    y_eq, y_in = _multipliers(R, 4)
    F = fbound.float_rows(R)
    S, k, E, ek = fbound.slack_float(fbound.objective_float(A), F, y_eq, y_in)
    M, EM = fbound.face_float(F, S, E)
    Mx = _exact_face(R, _slack(A, R, y_eq, y_in)[0])
    n = M.shape[0]
    assert all(abs(Q(float(M[i, j])) - Mx[i, j]) <= Q(float(EM[i, j])) for i in range(n) for j in range(n))
    assert np.array_equal(M, M.T)


def test_the_lower_bound_is_below_the_exact_bound_and_close_to_it():
    D, lam, A = _instance()
    R = _rows(D, 2)
    y_eq, y_in = _multipliers(R, 5)
    af = fbound.objective_float(A)
    lo, up = fbound.bound_lower(af, R, y_eq, y_in, 2), fbound.bound_float(af, R, y_eq, y_in, 2)
    exact = bound(A, R, y_eq, y_in, 2)
    assert Q(lo) <= exact <= Q(up) + Q(abs(up)) * Q(1, 10**8)
    assert up - lo <= 1e-8 * (abs(up) + 1.0)


def test_the_floats_decide_without_the_exact_bound(monkeypatch):
    """A size the float bound closes, and one its lower bound rules out, cost no
    exact bound."""
    D, lam, A, res = _at_optimum()
    w = res.multipliers
    R = Rows(D.c, 2, face=True, level=w.level)
    monkeypatch.setattr(fbound, "bound", lambda *a, **k: pytest.fail("the exact bound was computed"))
    assert fbound.certificate(A, R, w.y_eq, w.y_in, 2) <= 0
    y_eq = np.zeros(R.n_eq); y_eq[0] = 1.0       # the row Z_00 = 1: k = 1, so every bound is at least 1
    open_ = fbound.certificate(A, R, y_eq, np.zeros(R.n_in), 2)
    assert open_ > 0


def test_between_the_float_bounds_the_exact_bound_decides(monkeypatch):
    D, lam, A, res = _at_optimum()
    w = res.multipliers
    R = Rows(D.c, 2, face=True, level=w.level)
    monkeypatch.setattr(fbound, "bound_float", lambda *a, **k: 1.0)
    monkeypatch.setattr(fbound, "bound_lower", lambda *a, **k: -1.0)
    calls = []
    real = fbound.bound
    monkeypatch.setattr(fbound, "bound", lambda *a, **k: calls.append(1) or real(*a, **k))
    b = fbound.certificate(A, R, w.y_eq, w.y_in, 2)
    assert calls == [1] and b == bound(A, R, w.y_eq, w.y_in, 2) and b <= 0


def test_an_open_size_reports_the_float_bound():
    D, lam, A = _instance()
    R = _rows(D, 2)
    z_eq, z_in = np.zeros(R.n_eq), np.zeros(R.n_in)
    z_eq[0] = 1.0
    b = fbound.certificate(A, R, z_eq, z_in, 2)
    assert b == Q(fbound.bound_float(fbound.objective_float(A), R, z_eq, z_in, 2)) and b > 0


def test_the_float_rows_are_built_once_per_rows():
    D, lam, A = _instance()
    R = _rows(D, 2)
    assert fbound.float_rows(R) is fbound.float_rows(R)


# ---------------------------------------------------------------- properties


_mag = st.sampled_from([0.0, 1e-320, 1e-200, 1e-8, 1.0, 1e3, 1e8])


@SETTINGS
@given(seed=st.integers(0, 10**6), p=st.integers(3, 5), m=st.integers(1, 3), face=st.booleans(),
       scale=_mag, spread=st.floats(0.0, 6.0), fix=st.booleans())
@example(seed=0, p=4, m=2, face=True, scale=1e8, spread=6.0, fix=True)
@example(seed=1, p=3, m=1, face=False, scale=1e-320, spread=0.0, fix=False)
def test_property_the_enclosures_and_bounds_hold(seed, p, m, face, scale, spread, fix):
    """Over random panels, sizes, fixings and multipliers spanning subnormal to large
    with mixed signs: the slack and the face enclose the rational sums, the upper
    bound is at least every design's value and the exact lower bound on the dual,
    and the lower bound is at most bound()."""
    m = min(m, p)
    D, lam, A = _instance(seed % 7, p)
    fixes = ((0, 1),) if fix and m >= 1 else ()
    R = _rows(D, m, face=face, fixes=fixes)
    rng = np.random.default_rng(seed)
    y_eq = rng.normal(size=R.n_eq) * scale * 10.0 ** rng.uniform(-spread, spread, R.n_eq)
    y_in = rng.normal(size=R.n_in) * scale * 10.0 ** rng.uniform(-spread, spread, R.n_in)
    af = fbound.objective_float(A)
    F = fbound.float_rows(R)
    S, k, E, ek = fbound.slack_float(af, F, y_eq, y_in)
    Sx, kx = _slack(A, R, y_eq, y_in)
    assert all(abs(Q(float(S[i, j])) - Sx[i, j]) <= Q(float(E[i, j])) for i in range(R.n) for j in range(R.n))
    assert abs(Q(k) - kx) <= Q(ek)
    if R.face is not None:
        M, EM = fbound.face_float(F, S, E)
        Mx = _exact_face(R, Sx)
        assert all(abs(Q(float(M[i, j])) - Mx[i, j]) <= Q(float(EM[i, j]))
                   for i in range(M.shape[0]) for j in range(M.shape[0]))
    up = fbound.bound_float(af, R, y_eq, y_in, m)
    lo = fbound.bound_lower(af, R, y_eq, y_in, m)
    assert Q(up) >= _dual_low(A, R, y_eq, y_in, m)
    for C in _designs(D, m, fixes):
        assert Q(up) >= _value(A, C)
    exact = bound(A, R, y_eq, y_in, m)
    assert exact is None or Q(lo) <= exact


@SETTINGS
@given(seed=st.integers(0, 10**6), p=st.integers(3, 5), m=st.integers(1, 3), frac=st.sampled_from([Q(1, 4), Q(1, 2), Q(9, 10)]))
def test_property_the_certificate_closes_whatever_bound_closes(seed, p, m, frac):
    """At a relaxation's own multipliers and at perturbed ones, the certificate is
    at least every design's value, and it closes every size bound() closes."""
    m = min(m, p)
    D, lam, A = _instance(seed % 7, p, frac)
    R = _rows(D, m)
    try:
        sol = sf.solve(D.c, D.G, lam, R)
    except Exception:  # pragma: no cover - Clarabel solves every panel of this size
        return
    rng = np.random.default_rng(seed)
    for y_eq, y_in in ((sol.y_eq, sol.y_in),
                       (sol.y_eq * (1 + 1e-9 * rng.normal(size=R.n_eq)), sol.y_in * (1 + 1e-9 * rng.normal(size=R.n_in)))):
        b = fbound.certificate(A, R, y_eq, y_in, m)
        exact = bound(A, R, y_eq, y_in, m)
        if exact is not None and exact <= 0:
            assert b <= 0
        for C in _designs(D, m):
            assert b >= _value(A, C)


# ---------------------------------------------------------------- edge cases


def test_a_single_control():
    D, lam, A = _instance(p=1)
    R = _rows(D, 1)
    y_eq, y_in = _multipliers(R, 0)
    up = fbound.bound_float(fbound.objective_float(A), R, y_eq, y_in, 1)
    assert Q(up) >= _dual_low(A, R, y_eq, y_in, 1)


def test_zero_multipliers_give_the_objective_alone():
    D, lam, A = _instance()
    R = _rows(D, 2, face=False)
    up = fbound.bound_float(fbound.objective_float(A), R, np.zeros(R.n_eq), np.zeros(R.n_in), 2)
    lam_max = max(np.linalg.eigvalsh(np.array(A, dtype=float)))
    assert up == pytest.approx(3 * max(0.0, lam_max), rel=1e-9)


def test_every_control_fixed():
    D, lam, A = _instance(p=4)
    fixes = ((0, 1), (1, 1), (2, 0), (3, 0))
    R = _rows(D, 2, fixes=fixes)
    y_eq, y_in = _multipliers(R, 2)
    up = fbound.bound_float(fbound.objective_float(A), R, y_eq, y_in, 2)
    assert Q(up) >= _value(A, (0, 1))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan, 1e308])
def test_a_multiplier_the_floats_cannot_bound_clears_nothing(bad):
    """An infinite or NaN multiplier, or one whose sums overflow: the float bounds
    are +inf and -inf, so neither decides."""
    D, lam, A = _instance()
    R = _rows(D, 2)
    y_eq, y_in = _multipliers(R, 0)
    y_eq = y_eq.copy(); y_eq[3] = bad
    af = fbound.objective_float(A)
    assert fbound.bound_float(af, R, y_eq, y_in, 2) == np.inf
    assert fbound.bound_lower(af, R, y_eq, y_in, 2) == -np.inf


def test_the_size_free_rows():
    """The group screen's rows (m None, trace 1 + p, no face)."""
    D, lam, A = _instance()
    R = Rows(D.c, None)
    y_eq, y_in = _multipliers(R, 6, negative=False)
    af = fbound.objective_float(A)
    up = fbound.bound_float(af, R, y_eq, y_in, R.p)
    assert Q(up) >= sf.group_bound(A, R, y_eq, y_in) - Q(abs(up)) * Q(1, 10**8)
    assert Q(fbound.bound_lower(af, R, y_eq, y_in, R.p)) <= sf.group_bound(A, R, y_eq, y_in)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("which", ["eq", "in"])
def test_multipliers_of_another_shape_are_refused(which):
    D, lam, A = _instance()
    R = _rows(D, 2)
    y_eq, y_in = _multipliers(R, 0)
    if which == "eq":
        y_eq = y_eq[:-1]
    else:
        y_in = np.concatenate([y_in, [1.0]])
    with pytest.raises(MlsynthDataError, match="multipliers"):
        fbound.certificate(A, R, y_eq, y_in, 2)
    with pytest.raises(MlsynthDataError, match="multipliers"):
        fbound.bound_float(fbound.objective_float(A), R, y_eq, y_in, 2)


# ---------------------------------------------------------------- the search


def test_the_search_certifies_the_same_design_with_fewer_exact_bounds(monkeypatch):
    y, X = panel(3, p=7)
    calls = []
    real = fbound.bound
    monkeypatch.setattr(fbound, "bound", lambda *a, **k: calls.append(1) or real(*a, **k))
    fast = certify_slope_free(y, X, restarts=5, solver="clarabel")
    n_float = len(calls)
    monkeypatch.setattr(fbound, "bound_float", lambda *a, **k: np.inf)
    monkeypatch.setattr(fbound, "bound_lower", lambda *a, **k: -np.inf)
    calls.clear()
    slow = certify_slope_free(y, X, restarts=5, solver="clarabel")
    assert fast.certified and slow.certified and fast.controls == slow.controls
    assert n_float < len(calls)


def test_the_workspace_decides_by_the_certificate(monkeypatch):
    """The workspace's chunks are decided by the certificate: with the floats
    forced to defer, every chunk's decision is the exact bound's."""
    from .workspace import Workspaces
    D, lam, A = _instance(p=6)
    R = _rows(D, 3, box=True)
    seen = []
    real = fbound.certificate
    monkeypatch.setattr(fbound, "certificate", lambda *a, **k: seen.append(1) or real(*a, **k))
    sol = Workspaces().solve(D.c, D.G, lam, R)
    assert seen and sol.bound is not None


# ---------------------------------------------------------------- the eigenvalue bounds


@SETTINGS
@given(seed=st.integers(0, 10**6), n=st.integers(2, 6), width=st.sampled_from([0.0, 1e-6, 1e-2, 1.0]))
@example(seed=0, n=4, width=1.0)
def test_property_the_eigenvalue_bounds_hold_for_every_matrix_within_the_error(seed, n, width):
    """_lambda_up is at least, and _lambda_low at most, max(0, lambda_max) of every
    symmetric matrix within E of S: checked at S - E, S + E and random sign
    patterns, whose top eigenvalues reach the ends Weyl's inequality allows."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n)); S = (B + B.T) / 2
    E = np.abs(rng.normal(size=(n, n))) * width; E = (E + E.T) / 2
    up, low = fbound._lambda_up(S, E), fbound._lambda_low(S, E)
    for sign in [np.ones((n, n)), -np.ones((n, n))] + [np.sign(rng.normal(size=(n, n))) for _ in range(4)]:
        sign = np.triu(sign) + np.triu(sign, 1).T
        lam = float(np.linalg.eigvalsh(S + sign * E)[-1])
        assert float(up) >= lam - 1e-12 * (1 + abs(lam))
        assert float(low) <= max(0.0, lam) + 1e-12 * (1 + abs(lam))


def test_the_weyl_term_is_added():
    """With S zero and E the all-ones matrix, a matrix within E has lambda_max up to
    n: the upper bound reaches it only through E's norm."""
    up = fbound._lambda_up(np.zeros((3, 3)), np.ones((3, 3)))
    assert up >= 3


def test_the_rayleigh_bound_subtracts_the_error():
    """With S the identity and E the identity, a matrix within E can be zero: the
    lower bound on max(0, lambda_max) is zero, not one."""
    assert fbound._lambda_low(np.eye(3), np.eye(3)) == 0


def test_a_coefficient_below_the_normal_range_disables_the_floats():
    """A coefficient that rounds into the subnormal range is not within a relative u
    of its double, so the floats bound nothing and the exact certificate decides."""
    c = [Q(1, 2 ** 1070), Q(1), Q(2)]
    R = Rows(c, 1)
    assert not fbound.float_rows(R).exact
    A = objective(c, np.eye(3, dtype=int).astype(object), Q(1))
    y_eq, y_in = _multipliers(R, 0)
    af = fbound.objective_float(A)
    assert fbound.bound_float(af, R, y_eq, y_in, 1) == np.inf
    assert fbound.bound_lower(af, R, y_eq, y_in, 1) == -np.inf


@SETTINGS
@given(seed=st.integers(0, 10**6), width=st.sampled_from([1e-3, 1.0, 1e3]))
@example(seed=0, width=1.0)
def test_property_the_face_encloses_every_slack_within_its_error(seed, width):
    """face_float's EM bounds |fl(T'ST) - T'S*T| for every symmetric S* within E of
    S, not only for S itself: checked in rationals at S + E, S - E and random sign
    patterns, with E wide enough that its propagation |T|'E|T| is most of EM."""
    D, lam, A = _instance(p=6)
    R = _rows(D, 3, fixes=((0, 1), (4, 0)))
    F = fbound.float_rows(R)
    rng = np.random.default_rng(seed)
    n = R.n
    B = rng.normal(size=(n, n)); S = (B + B.T) / 2
    E = np.abs(rng.normal(size=(n, n))) * width; E = (E + E.T) / 2
    M, EM = fbound.face_float(F, S, E)
    T = R.face.integer_basis()
    k = M.shape[0]
    for sign in [np.ones((n, n)), -np.ones((n, n))] + [np.sign(rng.normal(size=(n, n))) for _ in range(3)]:
        sign = np.triu(sign) + np.triu(sign, 1).T
        Sx = np.array([[Q(float(S[i, j])) + int(sign[i, j]) * Q(float(E[i, j])) for j in range(n)] for i in range(n)],
                      dtype=object)
        Mx = T.T.dot(Sx).dot(T)
        assert all(abs(Q(float(M[i, j])) - Mx[i, j]) <= Q(float(EM[i, j])) for i in range(k) for j in range(k))
