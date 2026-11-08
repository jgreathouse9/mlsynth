"""The slope-free relaxation on its face (#965).

Every point of a size's relaxation has the forced null directions v = (-m, 1, ..., 1)
(from sum u = m and its products), e_j for a control fixed out and e_0 - e_j for one
fixed in (Burer 2012, Prop. 8.3, in the Handbook on Semidefinite, Conic and
Polynomial Optimization), so no point is positive definite and an interior-point
solver works on a problem without an interior. On the beds Clarabel returned
optimal_inaccurate on 199 of 222 Shor solves and 254 of 260 full solves of s30_k4.
On the face Z = T W T' (face.py) the forced directions are gone, the rows the face
satisfies identically are dropped, and the certificate bounds the face's slack
T'ST, not S. Both certificates stay exact; the bound is the smaller."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers.certify import Dyadic

from . import face as fc
from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .slopefree import Rows, bound, certify_slope_free, objective, solve
from .treatment import _split

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _designs(p, m, fixes):
    fx = dict(fixes)
    for C in itertools.combinations(range(p), m):
        if all((j in C) == bool(v) for j, v in fx.items()):
            yield C


def _lifted(p, C):
    z = np.zeros(p + 1, dtype=object); z[:] = 0; z[0] = 1
    for j in C:
        z[j + 1] = 1
    return z


def _pinned():
    """panel(2, p=6) at y'y / 2, size 3: the face certificate closes at the face
    solve's multipliers, the full-space one at the same multipliers does not."""
    y, X = panel(2, p=6)
    D = ExactData(y, X)
    lam = slopefree.dyadic_below(D.yy / 2)
    return D, lam


# ---------------------------------------------------------------- smoke


def test_smoke_a_bed_group_certifies_the_same_design_on_the_face():
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    on = certify_slope_free(y, X, restarts=5, face=True)
    off = certify_slope_free(y, X, restarts=5, face=False)
    assert on.certified and off.certified and (on.incumbent, on.controls) == (off.incumbent, off.controls)


def test_smoke_the_shor_relaxation_solves_accurately_on_the_face():
    D, lam = _pinned()
    s = solve(D.c, D.G, lam, Rows(D.c, 3, level="shor", face=True))
    assert s.face and s.status == "optimal"


# ---------------------------------------------------------------- unit: the face


def test_the_face_is_on_by_default():
    assert slopefree.FACE is True


@pytest.mark.parametrize("fixes", [(), ((0, 1),), ((1, 0),), ((0, 1), (3, 0)), ((2, 1), (4, 1))])
def test_every_design_lies_on_the_face(fixes):
    p, m = 6, 3
    F = fc.face_of(p, m, fixes)
    T = F.integer_basis()
    for C in _designs(p, m, fixes):
        z = _lifted(p, C)
        assert np.array_equal(T.dot(z[list(F.index)]), z)


def test_a_point_of_the_relaxation_is_lifted_from_its_restriction():
    """Z, the mean of zz' over the designs, equals T Z_RR T' exactly."""
    p, m, fixes = 6, 3, ((0, 1), (5, 0))
    F = fc.face_of(p, m, fixes)
    T = F.integer_basis()
    zs = [_lifted(p, C) for C in _designs(p, m, fixes)]
    Z = sum(np.outer(z, z) for z in zs) * Q(1, len(zs))
    ix = list(F.index)
    assert np.array_equal(T.dot(Z[np.ix_(ix, ix)]).dot(T.T), Z)


def test_the_basis_is_integer_with_one_dense_row():
    F = fc.face_of(8, 3, ((0, 1), (6, 0)))
    T = F.integer_basis()
    assert all(isinstance(v, int) for v in T.flat)
    dense = [i for i in range(T.shape[0]) if np.count_nonzero(T[i]) > 1]
    assert dense == [F.pivot + 1] and T[F.pivot + 1, 0] == F.q
    assert F.q == 3 - 1 and F.trace == 1 + F.q


def _dyadic_slack(rng, p, e):
    N = rng.integers(-9, 9, (p + 1, p + 1))
    N = np.array(N + N.T, dtype=object)
    return Dyadic(N, e), np.array([[Q(int(v), 1 << e) for v in row] for row in N], dtype=object)


def test_the_slack_is_t_transpose_s_t():
    """T'ST in S's dyadic exponent, from integers: no rational arithmetic."""
    F = fc.face_of(6, 3, ((1, 1), (4, 0)))
    T = F.integer_basis()
    D, S = _dyadic_slack(np.random.default_rng(0), 6, 3)
    M = F.slack(D)
    assert isinstance(M, Dyadic) and M.e == D.e
    assert all(isinstance(v, int) for v in M.N.flat)
    assert np.array_equal(M.N / Q(1 << M.e), T.T.dot(S).dot(T))


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(2, 9), e=st.integers(0, 6), data=st.data())
def test_property_the_slack_is_t_transpose_s_t(seed, p, e, data):
    m = data.draw(st.integers(1, p))
    js = data.draw(st.lists(st.integers(0, p - 1), unique=True, max_size=p))
    ins = data.draw(st.integers(0, min(len(js), m)))
    F = fc.face_of(p, m, [(j, int(i < ins)) for i, j in enumerate(js)])
    if F is None:
        return
    T = F.integer_basis()
    D, S = _dyadic_slack(np.random.default_rng(seed), p, e)
    M = F.slack(D)
    assert M.e == D.e and np.array_equal(M.N / Q(1 << M.e), T.T.dot(S).dot(T))


def test_the_rows_the_face_satisfies_are_dropped():
    """sum u = m becomes m W_00 = m and its products 0 = 0; the fixings and the
    diagonal rows of fixed controls vanish; Z_00 = 1 stays."""
    D, lam = _pinned()
    R = Rows(D.c, 3, fixes=[(0, 1), (2, 0)], level="shor", face=True)
    keep_e, _ = R.face.kept(R)
    assert 0 in keep_e
    size_row = 1 + R.p                            # after Z_00 = 1 and the p diagonal rows
    products = range(size_row + 1, size_row + 1 + R.p)
    fixings = range(size_row + 1 + R.p, R.n_eq)
    assert size_row not in keep_e and not set(products) & set(keep_e) and not set(fixings) & set(keep_e)


def test_the_face_solve_matches_the_full_space_value():
    D, lam = _pinned()
    for level in ("shor", "full"):
        for fixes in ([], [(0, 1)], [(1, 0), (3, 1)]):
            a = solve(D.c, D.G, lam, Rows(D.c, 3, fixes=fixes, level=level, face=True))
            b = solve(D.c, D.G, lam, Rows(D.c, 3, fixes=fixes, level=level))
            assert a.face and not b.face
            assert a.value == pytest.approx(b.value, rel=1e-5, abs=1e-6)
            assert a.Z.shape == b.Z.shape


def test_the_face_bound_closes_where_the_full_space_bound_does_not():
    D, lam = _pinned()
    R = Rows(D.c, 3, level="shor", face=True)
    A = objective(D.c, D.G, lam)
    s = solve(D.c, D.G, lam, R)
    on = bound(A, R, s.y_eq, s.y_in, 3)
    off = bound(A, Rows(D.c, 3, level="shor"), s.y_eq, s.y_in, 3)
    assert on <= 0 < off


def test_the_bound_is_the_smaller_of_the_two():
    """At multipliers far from the face's, (1 + q) lambda_max(T'ST) can exceed the
    full-space bound, T not being orthonormal; the certificate keeps the smaller."""
    D, lam = _pinned()
    A = objective(D.c, D.G, lam)
    R = Rows(D.c, 3, level="shor", face=True)
    rng = np.random.default_rng(3)
    seen_face_larger = False
    for _ in range(20):
        ye = rng.uniform(-1e6, 1e6, R.n_eq); yi = rng.uniform(0, 1e6, R.n_in)
        both = bound(A, R, ye, yi, 3)
        full = bound(A, Rows(D.c, 3, level="shor"), ye, yi, 3)
        face_only = fc.face_bound(A, R, ye, yi)
        assert both == min(full, face_only)
        seen_face_larger |= face_only > full
    assert seen_face_larger


def test_the_trace_on_the_face_is_one_plus_q():
    """tr W = 1 + sum over the free controls but the pivot of u_j <= 1 + q: with S the
    identity on W's coordinates the face bound is k + (1 + q) lambda_max(T'T)."""
    F = fc.face_of(5, 3, ((0, 1),))
    T = F.integer_basis()
    M = T.T.dot(T)
    lam_max = max(np.linalg.eigvalsh(np.array(M, dtype=float)))
    got = fc.trace_term(F, Dyadic(np.eye(6, dtype=int).astype(object), 0))
    assert float(got) >= (1 + F.q) * lam_max * (1 - 1e-12)
    assert float(got) <= (1 + F.q) * lam_max * (1 + 1e-6)


def test_the_screen_reads_the_face():
    D, lam = _pinned()
    R = Rows(D.c, 3, level="shor", face=True)
    A = objective(D.c, D.G, lam)
    s = solve(D.c, D.G, lam, R)
    assert slopefree._screen(A, R, s.y_eq, s.y_in, 3) <= 0
    assert slopefree._screen(A, Rows(D.c, 3, level="shor"), s.y_eq, s.y_in, 3) > 0


def test_without_the_face_the_rows_and_the_solve_are_as_before():
    D, lam = _pinned()
    R = Rows(D.c, 3, level="shor")
    assert R.face is None and not solve(D.c, D.G, lam, R).face


def test_the_size_certificate_builds_its_rows_on_the_face(monkeypatch):
    seen = []
    real = slopefree.Rows.__init__

    def spy(self, *a, face=False, **k):
        seen.append(face)
        real(self, *a, face=face, **k)

    monkeypatch.setattr(slopefree.Rows, "__init__", spy)
    D, lam = _pinned()
    slopefree.size_certificate(D.c, D.G, lam, 3)
    assert seen and all(seen)
    seen.clear()
    slopefree.size_certificate(D.c, D.G, lam, 3, face=False)
    assert seen and not any(seen)


def test_the_face_flag_reaches_every_node(monkeypatch):
    seen = []
    real = slopefree.size_certificate

    def spy(*a, face=True, **k):
        seen.append(face)
        return real(*a, face=face, **k)

    monkeypatch.setattr(slopefree, "size_certificate", spy)
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    certify_slope_free(y, X, restarts=5, face=False)
    assert seen and not any(seen)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 4), nfix=st.integers(0, 2),
       div=st.sampled_from([2, 4, 16]), scale=st.sampled_from([1.0, 1e3, 1e6]))
def test_property_the_face_bound_is_never_below_a_design(seed, m, nfix, div, scale):
    rng = np.random.default_rng(seed)
    y, X = panel(seed % 60, p=6)
    D = ExactData(y, X)
    p = len(D.c)
    js = rng.choice(p, size=nfix, replace=False)
    fixes = [(int(j), int(rng.integers(0, 2))) for j in js]
    if sum(v for _, v in fixes) >= m:
        return
    lam = slopefree.dyadic_below(D.yy / div)
    A = objective(D.c, D.G, lam)
    designs = [C for C in _designs(p, m, fixes) if sum(D.c[j] for j in C) >= 0]
    if not designs:
        return
    best = max(sum(A[i][j] for i in C for j in C) for C in designs)
    for level in ("shor", "full"):
        R = Rows(D.c, m, fixes=fixes, level=level, face=True)
        ye = rng.uniform(-scale, scale, R.n_eq); yi = rng.uniform(0, scale, R.n_in)
        b = fc.face_bound(A, R, ye, yi)
        assert b is None or b >= best


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.sampled_from([5, 6]), data=st.data())
def test_property_the_face_solve_certifies_its_own_value(seed, p, data):
    """Strong duality on the face: the exact bound at the face solve's multipliers,
    returned to the data's units, is the solve's value to the solver's tolerance,
    measured against the objective's scale (Clarabel's tolerances are relative to
    the data). Over 2,400 solves on these panels the gap is at most 3.1e-7 of the
    scale at both levels, `optimal_inaccurate` full solves included, and every Shor
    solve is `optimal`; the full level is not always (its pair rows leave a second
    degeneracy the face does not remove). Multipliers in the wrong units (the rows
    carrying c unscaled, or scaled twice) open the gap by 3% to 600% of the value at
    the full level. A node whose fixings leave no design with c'u >= 0 has an
    infeasible relaxation, reported as such."""
    y, X = panel(seed % 60, p=p)
    D = ExactData(y, X)
    best, _ = positive_optimum(y, X)
    lam = slopefree.dyadic_below(best * 0.99 if best > 0 else D.yy / 2)
    m = data.draw(st.integers(2, p - 2))
    j = data.draw(st.integers(0, p - 1))
    fixes = data.draw(st.sampled_from([(), ((j, 0),), ((j, 1),)]))
    feasible = any(sum(D.c[i] for i in C) >= 0 for C in _designs(p, m, fixes))
    A = objective(D.c, D.G, lam)
    for level in ("shor", "full"):
        R = Rows(D.c, m, fixes=fixes, level=level, face=True)
        s = solve(D.c, D.G, lam, R)
        assert s.face
        if not feasible:
            assert s.status in ("infeasible", "infeasible_inaccurate") and s.Z is None
            continue
        assert s.status == "optimal" if level == "shor" else s.status in ("optimal", "optimal_inaccurate")
        v = s.value * s.scale
        b = float(bound(A, R, s.y_eq, s.y_in, m))
        assert abs(b - v) <= 1e-5 * s.scale, (level, s.status, b, v, s.scale)


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_property_the_face_certifies_the_enumerated_optimum(seed):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=7)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3, face=True)
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edges


def test_no_free_control_has_no_face():
    assert fc.face_of(3, 2, ((0, 1), (1, 1), (2, 0))) is None


def test_every_size_slot_fixed_in_has_no_face():
    assert fc.face_of(5, 2, ((0, 1), (3, 1))) is None        # q = 0: the node is one design


def test_the_group_rows_have_no_face():
    D, _ = _pinned()
    assert Rows(D.c, None, face=True).face is None


def test_one_free_control():
    """With one free control the face is W = [[W_00]], and the size's one design."""
    F = fc.face_of(4, 2, ((0, 1), (1, 0), (2, 0)))
    T = F.integer_basis()
    assert T.shape == (5, 1)
    z = _lifted(4, (0, 3))
    assert np.array_equal(T.dot(z[list(F.index)]), z)


def test_an_infeasible_node_is_reported_as_off_the_face():
    """Fixings that leave no design with c'u >= 0 make the relaxation infeasible.
    Off the face Clarabel reports it with no point; on the face the solve reports
    the same, not an error from lifting a point that does not exist. The search
    closes such a node at its dual tier before any solve (the size certificate is
    the same on and off the face)."""
    y, X = panel(17, p=6)
    D = ExactData(y, X)
    best, _ = positive_optimum(y, X)
    lam = slopefree.dyadic_below(best * 0.99)
    assert not any(sum(D.c[i] for i in C) >= 0 for C in _designs(6, 3, ((0, 0),)))
    for level in ("shor", "full"):
        on = solve(D.c, D.G, lam, Rows(D.c, 3, fixes=((0, 0),), level=level, face=True))
        off = solve(D.c, D.G, lam, Rows(D.c, 3, fixes=((0, 0),), level=level))
        assert off.status == "infeasible" and off.Z is None
        assert on.face and on.status == off.status and on.Z is None and on.value == -np.inf
    on = slopefree.size_certificate(D.c, D.G, lam, 3, fixes=[(0, 0)], face=True)
    off = slopefree.size_certificate(D.c, D.G, lam, 3, fixes=[(0, 0)], face=False)
    assert on.certified and off.certified and on.bound == off.bound


# ---------------------------------------------------------------- failures


def test_more_fixed_in_than_the_size_is_refused():
    with pytest.raises(ValueError, match="fixed in"):
        fc.face_of(5, 2, ((0, 1), (1, 1), (2, 1)))


def test_a_control_fixed_both_ways_is_refused():
    with pytest.raises(ValueError, match="both"):
        fc.face_of(5, 2, ((0, 1), (0, 0)))


def test_a_row_the_face_shows_violated_is_reported(monkeypatch):
    """A structural row that reduces on the face to a constant it does not meet
    means the node holds no design; the solve raises and names the row, never
    returning a value for an empty set."""
    D, lam = _pinned()
    R = Rows(D.c, 3, fixes=[(0, 1)], level="shor", face=True)
    R.ineq.append({(0, 1): 1}); R.bin.append(Q(0)); R.n_in += 1           # u_0 <= 0, with u_0 fixed at 1
    with pytest.raises(fc.FaceError, match=f"row {R.n_in - 1}"):
        solve(D.c, D.G, lam, R)


def test_a_failed_face_solve_is_reported_and_raised(monkeypatch):
    """Clarabel's failure on the face is a solve event with status error, and the
    error reaches the caller (size_certificate decides what an open size means)."""
    import cvxpy as cp
    from .progress import reporting

    def fail(self, *a, **k):
        raise cp.error.SolverError("injected")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    D, lam = _pinned()
    events = []
    with reporting(progress=events.append):
        with pytest.raises(cp.error.SolverError, match="injected"):
            solve(D.c, D.G, lam, Rows(D.c, 3, level="shor", face=True))
    assert [(e.kind, e.fields["status"]) for e in events] == [("solve", "error")]


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_a_malformed_face_flag_is_refused(bad):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="face"):
        certify_slope_free(y, X, restarts=3, face=bad)
