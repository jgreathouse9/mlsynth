"""The joint lift's certificate in floating point with a rigorous rounding bound
(#965, step 3 for the joint lift), and the node's decision with the exact one as
fallback.

Each rounding claim is checked exactly, in rationals: the assembly of S and k, the
Cholesky residual, the eigenvalue bound on matrices of known spectrum, and the
whole bound against an exact upper bound on the dual function it bounds."""
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from .brute import panel
from . import certificate
from .certificate import (U, _coef_error, cholesky_residual, joint_assembly, joint_bound, joint_bound_float,
                          joint_clears, joint_terms, joint_trace, lambda_min_lower_float, qarr)
from .dual import smoothed_lbfgs
from .oracle import Oracle
from .search import unpack

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
UNIT = SimpleNamespace(scale=1.0)


def _hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def _known(n, eig):
    """S = H diag(eig) H' / n, exact in doubles for small integer eig and n a power
    of two, with eigenvalues exactly eig."""
    H = _hadamard(n)
    return (H * np.asarray(eig, dtype=float)) @ H.T / n


def _node(seed=4, p=6, lo=0.4, hi=0.9, m=3, cuts=None, fixes=()):
    """An unscaled oracle and the node solver's multipliers in the data's units."""
    y, X = panel(seed, p=p, n=40)
    O = Oracle(y, X, lo, hi, m, cuts=cuts, fixes=fixes)
    _, yb, _ = smoothed_lbfgs(O, np.zeros(O.dim), maxfun=200)
    O1 = Oracle(y, X, lo, hi, m, cuts=cuts, fixes=fixes, unscaled=True)
    return y, X, O1, yb * O.scale ** 2


def _exact(y, X, O1, yv, lo, hi, m):
    """The exact certificate at the same multipliers: (bound, S, k) in rationals."""
    duals, cuts, fixes = unpack(O1, yv, X.shape[1])
    return joint_bound(UNIT, y, X, lo, hi, m, duals=duals, nogood=cuts or None, fixes=fixes)


def _theta_upper(y, X, O1, yv, lo, hi, m):
    """An exact upper bound on the dual function the certificate bounds below:
    k + T min(0, R) with R the exact Rayleigh quotient of S at the double
    eigenvector, since R >= lambda_min(S)."""
    _, S, k = _exact(y, X, O1, yv, lo, hi, m)
    v = qarr(np.linalg.eigh(np.array(S, dtype=float))[1][:, 0])
    R = (v @ S @ v) / (v @ v)
    return k + min(Q(0), R) * joint_trace(Q(int(m)), Q(float(hi)))


def _multipliers(O, seed, size):
    rng = np.random.default_rng(seed)
    yv = rng.normal(size=O.dim) * size
    yv[O.ineq] = np.abs(yv[O.ineq])
    return yv


# ---------------------------------------------------------------- smoke


def test_smoke_the_float_bound_is_a_finite_float():
    y, X, O1, yv = _node()
    b = joint_bound_float(O1, yv)
    assert isinstance(b, float) and np.isfinite(b)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", range(4))
def test_the_float_bound_is_valid_and_tight(seed):
    """At most the dual function (validity), and within 1e-8 of the exact
    certificate relative to y'y: the rounding bound costs tightness only at the
    scale of eps."""
    y, X, O1, yv = _node(seed)
    fl = joint_bound_float(O1, yv)
    assert Q(fl) <= _theta_upper(y, X, O1, yv, 0.4, 0.9, 3)
    exact, _, _ = _exact(y, X, O1, yv, 0.4, 0.9, 3)
    assert abs(float(exact) - fl) <= 1e-8 * float(y @ y)


def test_the_float_bound_holds_with_cuts_and_fixings():
    y, X, O1, yv = _node(2, cuts=[{0, 1, 2}], fixes=[(3, 1), (4, 0)])
    fl = joint_bound_float(O1, yv)
    assert Q(fl) <= _theta_upper(y, X, O1, yv, 0.4, 0.9, 3)
    exact, _, _ = _exact(y, X, O1, yv, 0.4, 0.9, 3)
    assert abs(float(exact) - fl) <= 1e-8 * float(y @ y)


def test_the_sign_constrained_multipliers_are_clipped():
    """A negative multiplier on an inequality is read as zero, as the exact
    certificate reads it, so the bound stays valid."""
    y, X, O1, yv = _node(1)
    yv = yv.copy(); yv[np.flatnonzero(O1.ineq)[:50]] = -1.0
    assert Q(joint_bound_float(O1, yv)) <= _theta_upper(y, X, O1, yv, 0.4, 0.9, 3)


@pytest.mark.parametrize("n, eig", [(8, [3, 5, 1, 7, 2, 2, 9, 4]), (16, list(range(-8, 8))), (64, [1] * 63 + [-5])])
def test_the_eigenvalue_bound_is_below_and_near_the_smallest(n, eig):
    lam = lambda_min_lower_float(_known(n, eig))
    assert Q(lam) <= min(eig)
    assert min(eig) - lam <= 1e-9 * max(abs(e) for e in eig)


def test_the_unscaled_oracle_keeps_the_data_units():
    y, X = panel(1, p=5, n=30)
    O = Oracle(y, X, 0.3, 0.9, 2, unscaled=True)
    assert O.scale == 1.0 and np.array_equal(O.G, X.T @ X) and np.array_equal(O.c, X.T @ y)
    aX = np.abs(X)
    assert np.array_equal(O.data_moduli[0], aX.T @ aX) and O.nobs == 30


@pytest.mark.parametrize("seed", range(3))
def test_the_clearing_decision_is_the_exact_one(seed):
    """joint_clears prunes when the exact certificate at the same multipliers does,
    on these nodes: the float bound decides when it clears, the exact one otherwise."""
    y, X, O1, yv = _node(seed)
    exact, _, _ = _exact(y, X, O1, yv, 0.4, 0.9, 3)
    for UB in (exact * Q(99, 100), exact, exact * Q(101, 100)):
        got = joint_clears(O1, yv, y, X, 0.4, 0.9, 3, UB, {})
        assert got == (exact >= UB)


def test_the_exact_certificate_is_reached_only_when_the_float_one_falls_short():
    y, X, O1, yv = _node(1)
    exact, _, _ = _exact(y, X, O1, yv, 0.4, 0.9, 3)
    stats = {}
    assert joint_clears(O1, yv, y, X, 0.4, 0.9, 3, exact * Q(9, 10), stats)
    assert stats.get("exact", 0) == 0 and stats["float"] == 1
    assert joint_clears(O1, yv, y, X, 0.4, 0.9, 3, exact, stats)
    assert stats["exact"] == 1 and stats["float"] == 2


def test_the_float_bound_misses_the_incumbent_by_its_rounding_bound_only():
    """Just below the exact certificate the float one still clears: its rounding
    allowance is far below 1e-6 of the bound."""
    y, X, O1, yv = _node(0)
    exact, _, _ = _exact(y, X, O1, yv, 0.4, 0.9, 3)
    stats = {}
    assert joint_clears(O1, yv, y, X, 0.4, 0.9, 3, exact - abs(exact) * Q(1, 10 ** 6), stats)
    assert stats.get("exact", 0) == 0


# ---------------------------------------------------------------- components, exactly


def _assembly_case(seed, size, k, j):
    y, X = panel(seed % 50, p=4, n=20)
    y, X = y * 2.0 ** k, X * 2.0 ** k
    O1 = Oracle(y, X, 0.3, 0.8, 2, cuts=[{0, 1}], fixes=[(2, 1)], unscaled=True)
    return y, X, O1, _multipliers(O1, seed, size) * 2.0 ** j


@SETTINGS
@given(seed=st.integers(0, 10_000), size=st.floats(0.0, 3.0), k=st.integers(-540, 20), j=st.integers(-1074, 2))
@example(seed=0, size=1.0, k=0, j=0)
@example(seed=3, size=1.0, k=-540, j=-1060)      # every product underflows
@example(seed=5, size=2.5, k=-20, j=-30)
def test_property_the_assembly_error_bounds_its_rounding(seed, size, k, j):
    """|fl(S) - S| <= E entrywise and |fl(k) - k| <= e_k, S and k exact in rationals
    from the double data and multipliers: the a priori bound of each sum
    (Bjorck 2015, eq. (1.4.10)), the data's own rounding, and underflow."""
    y, X, O1, yv = _assembly_case(seed, size, k, j)
    S, kk, E, ek = joint_assembly(O1, yv)
    _, Sx, kx = _exact(y, X, O1, yv, 0.3, 0.8, 2)
    assert all(abs(Q(float(S[a, b])) - Sx[a, b]) <= Q(float(E[a, b]))
               for a in range(S.shape[0]) for b in range(S.shape[0]))
    assert abs(Q(kk) - kx) <= Q(ek)


@SETTINGS
@given(seed=st.integers(0, 10_000), n=st.integers(2, 12), k=st.integers(-1070, 500), shift=st.floats(-1e-6, 1.0))
@example(seed=1, n=6, k=-1066, shift=0.5)          # a subnormal matrix: every product rounds to the ETA grid
def test_property_the_residual_error_bounds_its_rounding(seed, n, k, shift):
    """|fl(S - sigma I - L L') - (S - sigma I - L L')| <= E entrywise, the right side
    exact in rationals from the double S, sigma and L."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)); S = (A + A.T) * 2.0 ** k
    lam = np.linalg.eigvalsh(S)[0]
    sigma = lam - shift * abs(lam)
    out = cholesky_residual(S, sigma)
    if out is None:
        return
    L, R, E = out
    Sq, Lq = qarr(S), qarr(L)
    Rx = Sq - Lq @ Lq.T
    for i in range(n):
        Rx[i, i] -= Q(float(sigma))
    assert all(abs(Q(float(R[a, b])) - Rx[a, b]) <= Q(float(E[a, b])) for a in range(n) for b in range(n))


@SETTINGS
@given(seed=st.integers(0, 10_000), n=st.sampled_from([2, 4, 8, 16, 32]), k=st.integers(-1000, 1000))
def test_property_the_eigenvalue_bound_is_below_the_smallest(seed, n, k):
    """On matrices of known spectrum scaled by 2^k, overflow and underflow aside."""
    rng = np.random.default_rng(seed)
    eig = rng.integers(-20, 20, size=n)
    S = _known(n, eig) * 2.0 ** k
    lam = lambda_min_lower_float(S)
    if lam is not None:
        assert Q(lam) <= Q(int(eig.min())) * Q(2) ** k


@SETTINGS
@given(seed=st.integers(0, 10_000), size=st.floats(0.0, 3.0), k=st.integers(-30, 30))
@example(seed=0, size=0.0, k=0)
def test_property_the_float_bound_never_exceeds_the_dual_function(seed, size, k):
    """At any multipliers on data scaled by 2^k the float bound read as a rational
    is at most an exact upper bound on the dual function, so it is a lower bound
    wherever the dual function is."""
    y, X = panel(seed % 50, p=4, n=20)
    y, X = y * 2.0 ** k, X * 2.0 ** k
    O1 = Oracle(y, X, 0.3, 0.8, 2, cuts=[{1, 2}], unscaled=True)
    yv = _multipliers(O1, seed, size) * 4.0 ** k
    assert Q(joint_bound_float(O1, yv)) <= _theta_upper(y, X, O1, yv, 0.3, 0.8, 2)


def test_the_term_count_bounds_every_entry():
    """With every input of modulus one (data, multipliers, coefficients, m), the
    moduli pass counts the terms of each entry: the symmetrised entry is half the
    terms of an entry and its transpose, and k counts its own."""
    y, X = panel(0, p=5, n=10)
    O = Oracle(y, X, 0.3, 0.8, 1, cuts=[{0, 1}, {2, 3}], fixes=[(1, 1), (4, 1)], unscaled=True)
    O.G, O.c, O.yy, O.K = np.ones_like(O.G), np.ones_like(O.c), 1.0, np.ones_like(O.K)
    M, mk = O.S_k_moduli(np.ones(O.dim))
    assert 2 * M.max() <= joint_terms(O) and mk <= joint_terms(O)
    assert mk > 16 * O.p ** 2                       # the count is not loose by a power of p


def test_the_data_rounding_is_bounded_where_the_gram_matrix_cancels():
    """Orthonormal controls and a target orthogonal to them: X'X off its diagonal and
    X'y are rounding noise, so fl(X'X) is wrong by a multiple of itself and only the
    data's own rounding bound (gamma_n |X|'|X|) covers it."""
    rng = np.random.default_rng(7)
    Qm, _ = np.linalg.qr(rng.normal(size=(30, 6)))
    X, y = Qm[:, :5] * 3.0, Qm[:, 5] * 2.0
    O1 = Oracle(y, X, 0.3, 0.8, 2, unscaled=True)
    yv = np.zeros(O1.dim)
    S, kk, E, ek = joint_assembly(O1, yv)
    _, Sx, kx = _exact(y, X, O1, yv, 0.3, 0.8, 2)
    gaps = np.array([[float(abs(Q(float(S[a, b])) - Sx[a, b])) for b in range(S.shape[0])] for a in range(S.shape[0])])
    assert np.all(gaps <= E) and abs(Q(kk) - kx) <= Q(ek)
    assert np.any(gaps > 4 * U * np.abs(S) * joint_terms(O1))   # the case has power: the assembly allowance alone is short


@pytest.mark.parametrize("eig", [[3, -2, 5, 1, 7, -4, 2, 9], [1, 2, 3, 4, 5, 6, 7, 8]])
def test_the_bound_composes_its_parts(monkeypatch, eig):
    """On a planted assembly of known spectrum and a large error E, the bound is
    k - e_k + T min(0, lambda_min - ||E||_inf) to within the eigenvalue bound's
    shift: each part enters once, with the joint lift's trace bound."""
    S = _known(8, eig)
    E = np.full((8, 8), 1e-3); ek = 0.25; k = 10.0
    monkeypatch.setattr(certificate, "joint_assembly", lambda O, yv: (S, k, E, ek))
    O = SimpleNamespace(m=3.0, hi=0.9)
    b = joint_bound_float(O, None)
    T = joint_trace(Q(3), Q(0.9))
    want = Q(k) - Q(ek) + T * min(Q(0), min(eig) - Q(float(E.sum(1).max())))
    assert Q(b) <= want and float(want) - b <= 1e-9 + float(T) * 8e-3 * 2e-6   # inflate's 1e-6 on ||E||


def test_an_infinite_float_bound_falls_back_to_the_exact_one(monkeypatch):
    y, X, O1, yv = _node(0)
    exact, _, _ = _exact(y, X, O1, yv, 0.4, 0.9, 3)
    monkeypatch.setattr(certificate, "joint_bound_float", lambda O, yv: -np.inf)
    stats = {}
    assert joint_clears(O1, yv, y, X, 0.4, 0.9, 3, exact, stats) and stats["exact"] == 1


@SETTINGS
@given(lo=st.floats(0.0, 1.0), width=st.floats(1e-6, 5.0))
@example(lo=2.225073858507203e-309, width=1.0)
@example(lo=1.7558521295436744e-155, width=1.0)
def test_property_the_coefficients_round_once(lo, width):
    """Each coefficient is lo, hi, 1, or a single product of two of them, so it is
    within u of its exact value, unless lo^2 underflows. A product that underflows
    to a subnormal keeps fewer digits, so kappa, the exact relative error, can
    exceed u (lo = 1.76e-155 gives 4.3e-15), and it is infinite above 1/4, where
    the float bound gives up; lo^2 rounding to zero is that case."""
    kappa = _coef_error(lo, lo + width)
    if lo == 0 or lo * lo >= 2.0 ** -1022:
        assert kappa <= U
    elif lo * lo == 0:
        assert kappa == np.inf
    else:
        assert kappa == np.inf or kappa <= 0.25


def test_an_underflowing_coefficient_gives_no_float_bound():
    """lo = 1e-170: lo^2 underflows, so the coefficient table is no longer within
    a relative kappa < 1 of itself and the exact certificate decides."""
    y, X = panel(1, p=4, n=20)
    O1 = Oracle(y, X, 1e-170, 0.8, 2, unscaled=True)
    assert joint_bound_float(O1, np.zeros(O1.dim)) == -np.inf


# ---------------------------------------------------------------- edges and failure


def test_one_candidate_control():
    y, X, O1, yv = _node(3, p=1, m=1, lo=0.2, hi=0.9)
    assert Q(joint_bound_float(O1, yv)) <= _theta_upper(y, X, O1, yv, 0.2, 0.9, 1)


def test_zero_multipliers_give_y_y_less_its_rounding():
    """At y = 0, S is the data block and k = y'y; S has a negative eigenvalue (its
    corner is 0 with -c beside it), so the bound is y'y + T lambda_min."""
    y, X = panel(0, p=4, n=20)
    O1 = Oracle(y, X, 0.3, 0.8, 2, unscaled=True)
    yv = np.zeros(O1.dim)
    fl = joint_bound_float(O1, yv)
    assert Q(fl) <= _theta_upper(y, X, O1, yv, 0.3, 0.8, 2)
    assert fl < float(y @ y)


def test_a_matrix_with_no_factor_gives_no_bound():
    assert lambda_min_lower_float(np.full((4, 4), np.nan)) is None
    assert cholesky_residual(-np.eye(3), 0.0) is None


def test_an_overflow_gives_no_bound():
    y, X, O1, yv = _node()
    assert joint_bound_float(O1, yv * 1e307) == -np.inf
    assert lambda_min_lower_float(np.full((4, 4), 1e308)) is None


def test_a_nonfinite_multiplier_gives_no_bound():
    y, X, O1, yv = _node()
    yv = yv.copy(); yv[0] = np.inf
    assert joint_bound_float(O1, yv) == -np.inf


def test_a_scaled_oracle_is_refused():
    """The certificate is in the data's units: an oracle that divided the data by a
    rounded scale bounds a nearby problem, not this one."""
    y, X = panel(1, p=4, n=20)
    O = Oracle(y * 3.0, X, 0.3, 0.8, 2)
    with pytest.raises(ValueError, match="unscaled"):
        joint_bound_float(O, np.zeros(O.dim))


@pytest.mark.parametrize("seed", range(2))
def test_certify_prunes_on_the_float_certificate(seed):
    """On the small panels every joint-lift prune is decided in doubles, and the
    search still reaches the enumerated optimum."""
    from .brute import positive_optimum
    from .search import certify
    y, X = panel(seed)
    out = certify(y, X, restarts=10)
    best, _ = positive_optimum(y, X)
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9)
    assert out.search.checks["float"] == out.search.pruned > 0
    assert out.search.checks.get("exact", 0) == 0


def test_a_negative_inequality_multiplier_is_read_as_zero():
    y, X, O1, yv = _node(1)
    neg = yv.copy(); idx = np.flatnonzero(O1.ineq)[::7]
    neg[idx] = -np.abs(neg[idx]) - 1.0
    zero = neg.copy(); zero[idx] = 0.0
    assert joint_bound_float(O1, neg) == joint_bound_float(O1, zero)


def test_the_assembly_rounding_is_bounded_where_every_term_is_lost():
    """k = 1 less 24 fixing terms of 0.45 u each: each one rounds away, so fl(k) = 1
    is wrong by 10.8 u, while the data's rounding (one period, y'y = 1) and the
    coefficients (lo, hi dyadic) allow only about u. Only the assembly's own
    gamma_{N+8} covers it."""
    y, X = np.array([1.0]), np.array([[0.5, 0.25, 0.75, 1.0, 0.125, 0.5]])
    fixes = [(j % 6, 1) for j in range(24)]
    O1 = Oracle(y, X, 0.5, 0.75, 2, fixes=fixes, unscaled=True)
    yv = np.zeros(O1.dim); yv[O1.sl["fix"]] = 0.45 * U
    S, kk, E, ek = joint_assembly(O1, yv)
    _, Sx, kx = _exact(y, X, O1, yv, 0.5, 0.75, 2)
    assert kk == 1.0 and 1 - kx == Q(24) * Q(0.45 * U)
    assert abs(Q(kk) - kx) <= Q(ek)
    assert _coef_error(0.5, 0.75) < 1e-300 and float(1 - kx) > 4 * U    # the case has power


def test_an_eigenvalue_estimate_too_high_grows_the_shift(monkeypatch):
    """A first shift above lambda_min leaves S - sigma I indefinite; the shift grows
    until the factor exists, and the bound is still below lambda_min."""
    S = _known(8, [3, -2, 5, 1, 7, -4, 2, 9])
    real = np.linalg.eigvalsh
    monkeypatch.setattr(certificate.np.linalg, "eigvalsh", lambda A: real(A) + 0.5)
    lam = lambda_min_lower_float(S)
    assert lam is not None and Q(lam) <= -4


def test_a_failed_eigenvalue_bound_gives_no_bound(monkeypatch):
    y, X, O1, yv = _node(0)
    monkeypatch.setattr(certificate, "lambda_min_lower_float", lambda S: None)
    assert joint_bound_float(O1, yv) == -np.inf


def test_a_bound_below_the_doubles_gives_no_bound(monkeypatch):
    """k and T lambda each finite, their sum below -1.8e308."""
    S = _known(8, [-1, 0, 0, 0, 0, 0, 0, 0]) * 2.0 ** 1020      # lambda_min = -1.1e307
    monkeypatch.setattr(certificate, "joint_assembly", lambda O, yv: (S, -1.5e308, np.zeros((8, 8)), 0.0))
    assert joint_bound_float(SimpleNamespace(m=3.0, hi=0.9), None) == -np.inf
