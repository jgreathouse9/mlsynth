"""solve_simplex_qp does not depend on the units or the level of its data (#829, #960).

``min ||A - B w||^2`` over the simplex has the same minimiser for ``(s B, s A)``
and for ``(B + c, A + c)``: scaling multiplies the objective by ``s^2``, and on
the simplex ``A - B w = (A - c) - (B - c) w``. The solver used to decide its
stopping test with a floor, ``tol * (1 + max|g|)``, and absolute tolerances,
so the units and the level of the data took the branch. At a scale of 1e-6 or
a level of 1e8 it returned points 37% above the optimum, and at 1e-6 it said
they had converged; at a scale of 1e4 it called a solved problem unconverged
(#829). ``agents_solvers.md``, rules 2, 3 and 5.

``converged`` is now a certificate: the KKT residual of the returned point on
the caller's data, divided by the size of the terms it is computed from.
"""

import warnings

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.solvers.active_set import (
    CERTIFY_TOL,
    SimplexQPUncertifiedWarning,
    simplex_kkt_error,
    solve_simplex_qp,
)


def _problem(seed, m=60, J=30, factors=3, noise=0.3):
    """A centred factor-model design with the target near, not in, the donors' hull."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(m, factors))
    B = F @ rng.normal(size=(factors, J)) + noise * rng.normal(size=(m, J))
    A = 1.3 * F @ rng.normal(size=factors) + noise * rng.normal(size=m)
    return B - B.mean(0), A - A.mean()


def _reference(B, A):
    """CLARABEL on the design divided by its largest entry, at tight tolerances."""
    s = float(np.abs(B).max())
    w = cp.Variable(B.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(A / s - (B / s) @ w)),
               [cp.sum(w) == 1]).solve(solver=cp.CLARABEL, tol_gap_abs=1e-14,
                                      tol_gap_rel=1e-14, tol_feas=1e-12)
    w = np.clip(w.value, 0.0, None)
    return w / w.sum()


def _gap(B, A, w, w_ref):
    f, f_ref = ((A - B @ w) ** 2).sum(), ((A - B @ w_ref) ** 2).sum()
    return (f - f_ref) / f_ref


# --------------------------------------------------------------------------
# Smoke and the two measured failures
# --------------------------------------------------------------------------

def test_smoke_weights_on_the_simplex_and_certified():
    B, A = _problem(0)
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert w.shape == (30,) and np.all(w >= 0) and w.sum() == pytest.approx(1.0)
    assert info["converged"] and info["certified"]
    assert info["kkt_error"] <= CERTIFY_TOL


@pytest.mark.parametrize("scale", [1e-8, 1e-6, 1e-5])
def test_small_scale_reaches_the_optimum(scale):
    """The measured failure: at 1e-6 the worst of these problems stopped 37%
    above the optimum and reported converged."""
    worst = 0.0
    for seed in range(12):
        B, A = _problem(seed)
        w, info = solve_simplex_qp(scale * B, scale * A, return_info=True)
        assert info["converged"]
        worst = max(worst, _gap(B, A, w, _reference(B, A)))
    assert worst <= 1e-9


@pytest.mark.parametrize("level", [1e6, 1e8])
def test_a_common_level_does_not_change_the_answer(level):
    """At a level of 1e8 the solver stopped 37% above the optimum (rule 5). The
    stored data carry a rounding of eps * level per entry, so the objective is
    known to about that, relative to the data's variation of order one."""
    tol = 1e-12 + 1e3 * np.finfo(float).eps * level
    worst = 0.0
    for seed in range(12):
        B, A = _problem(seed)
        w, info = solve_simplex_qp(B + level, A + level, return_info=True)
        assert info["converged"]
        worst = max(worst, _gap(B, A, w, _reference(B, A)))
    assert worst <= tol


@pytest.mark.parametrize("scale", [1.0, 1e2, 1e4, 1e6])
def test_issue_829_solved_problem_reported_converged(scale):
    """#829's shape: two rows, six donors, a zero target inside the hull. On
    main this design converged at 1, 1e2 and 1e6 and not at 1e4."""
    B = np.array([[0.0, -3.0, 2.0, 2.0, 4.0, 3.0], [5.0, 5.0, 4.0, -4.0, -5.0, 0.0]])
    w, info = solve_simplex_qp(scale * B, np.zeros(2), return_info=True)
    assert info["converged"]
    assert np.linalg.norm(B @ w) <= 1e-12 * np.linalg.norm(B)


# --------------------------------------------------------------------------
# Invariants, generatively
# --------------------------------------------------------------------------

@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), m=st.integers(2, 30), J=st.integers(2, 20),
       k=st.integers(-40, 40))
def test_power_of_two_scaling_gives_identical_weights(seed, m, J, k):
    """Scaling by a power of two is exact in floating point, so the solver sees
    the same normalised problem and must return the same bits."""
    rng = np.random.default_rng(seed)
    B, A = rng.normal(size=(m, J)), rng.normal(size=m)
    s = 2.0 ** k
    np.testing.assert_array_equal(solve_simplex_qp(B, A), solve_simplex_qp(s * B, s * A))


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(-20, 20))
def test_power_of_two_scaling_with_a_linear_term(seed, k):
    """The objective scales by s^2, so the linear term scales by s^2 with it."""
    rng = np.random.default_rng(seed)
    B, A = rng.normal(size=(12, 8)), rng.normal(size=12)
    lin = np.abs(rng.normal(size=8))
    s = 2.0 ** k
    np.testing.assert_array_equal(solve_simplex_qp(B, A, linear=lin),
                                  solve_simplex_qp(s * B, s * A, linear=s * s * lin))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(-8, 8))
def test_decimal_scaling_reaches_the_same_optimum(seed, k):
    """Any scale from 1e-8 to 1e8: the same objective as at unit scale, certified."""
    B, A = _problem(seed, m=25, J=12)
    w0 = solve_simplex_qp(B, A)
    w, info = solve_simplex_qp(10.0 ** k * B, 10.0 ** k * A, return_info=True)
    assert info["converged"]
    assert abs(_gap(B, A, w, w0)) <= 1e-10


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), e=st.integers(0, 8))
def test_level_property(seed, e):
    """Adding the same level to B and A leaves the minimiser where it was, to
    the rounding the stored data carry."""
    level = 10.0 ** e
    B, A = _problem(seed, m=25, J=12)
    w0 = solve_simplex_qp(B, A)
    w, info = solve_simplex_qp(B + level, A + level, return_info=True)
    assert info["converged"]
    assert abs(_gap(B, A, w, w0)) <= 1e-12 + 1e3 * np.finfo(float).eps * level


# --------------------------------------------------------------------------
# The certificate (rule 2): it accepts the optimum and refuses a worse point
# --------------------------------------------------------------------------

def test_certificate_accepts_the_reference_optimum():
    B, A = _problem(3)
    assert simplex_kkt_error(B, A, _reference(B, A)) <= CERTIFY_TOL


@pytest.mark.parametrize("step", [1e-6, 1e-4])
def test_certificate_refuses_a_point_moved_off_the_optimum(step):
    """Moved along a direction that keeps the point on the simplex: the KKT
    residual must exceed the threshold, or the certificate proves nothing."""
    B, A = _problem(4)
    w = _reference(B, A)
    on = np.flatnonzero(w > 1e-6)
    d = np.zeros_like(w)
    d[on[0]], d[on[1]] = 1.0, -1.0
    moved = w + step * d
    assert moved.min() >= 0 and moved.sum() == pytest.approx(1.0)
    assert simplex_kkt_error(B, A, moved) > CERTIFY_TOL


def test_certificate_refuses_a_pinned_weight_that_should_be_free():
    """Dual feasibility: the uniform point pins nothing, so only stationarity
    can refuse it; a vertex the optimum does not use fails on its multipliers."""
    B, A = _problem(5)
    w = _reference(B, A)
    unused = int(np.flatnonzero(w <= 1e-9)[0])
    vertex = np.zeros_like(w)
    vertex[unused] = 1.0
    assert simplex_kkt_error(B, A, vertex) > CERTIFY_TOL


def test_certificate_is_scale_free():
    B, A = _problem(6)
    w = _reference(B, A)
    e0 = simplex_kkt_error(B, A, w)
    for k in (-20, -8, 8, 20):
        assert simplex_kkt_error(2.0 ** k * B, 2.0 ** k * A, w) == pytest.approx(e0, rel=1e-12, abs=1e-300)


def test_certificate_with_a_linear_term():
    rng = np.random.default_rng(7)
    B, A = rng.normal(size=(15, 9)), rng.normal(size=15)
    lin = np.abs(rng.normal(size=9))
    w = solve_simplex_qp(B, A, linear=lin)
    assert simplex_kkt_error(B, A, w, linear=lin) <= CERTIFY_TOL
    assert simplex_kkt_error(B, A, w) > CERTIFY_TOL       # the plain program's optimum differs


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_single_donor():
    w, info = solve_simplex_qp(np.ones((4, 1)), np.zeros(4), return_info=True)
    assert w.tolist() == [1.0] and info["converged"]


def test_every_donor_equals_the_target():
    """A constant objective: every point is optimal, and the solver says so."""
    A = np.arange(5.0)
    w, info = solve_simplex_qp(np.tile(A[:, None], (1, 4)), A, return_info=True)
    assert info["converged"] and w.sum() == pytest.approx(1.0)


def test_collinear_donors():
    B, A = _problem(8, m=20, J=6)
    B = np.column_stack([B, B[:, 0], B[:, 1] * (1 + 1e-13)])
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert info["converged"]
    assert _gap(B, A, w, _reference(B, A)) <= 1e-9


def test_more_donors_than_rows():
    B, A = _problem(9, m=5, J=40)
    w, info = solve_simplex_qp(B, A, return_info=True)
    assert info["converged"]


# --------------------------------------------------------------------------
# Rule 1: an uncertified answer is reported, never silent
# --------------------------------------------------------------------------

def test_capped_solve_reports_through_info_and_does_not_warn():
    B, A = _problem(10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, info = solve_simplex_qp(B, A, max_iter=1, accelerate=False, return_info=True)
    assert not info["converged"]
    assert info["kkt_error"] > CERTIFY_TOL


def test_capped_solve_without_info_warns():
    B, A = _problem(10)
    with pytest.warns(SimplexQPUncertifiedWarning, match="KKT"):
        solve_simplex_qp(B, A, max_iter=1, accelerate=False)


def test_clean_solve_is_silent():
    B, A = _problem(11)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_simplex_qp(B, A)


def test_converged_means_certified(monkeypatch):
    """A point the loop accepts but the certificate refuses is not converged:
    ``converged`` is the certificate's verdict, not the pivot loop's last test."""
    import mlsynth.utils.solvers.active_set as active_set
    monkeypatch.setattr(active_set, "simplex_kkt_error", lambda *a, **k: 1.0)
    B, A = _problem(12)
    _, info = active_set.solve_simplex_qp(B, A, return_info=True)
    assert not info["converged"] and not info["certified"]
    with pytest.warns(SimplexQPUncertifiedWarning):
        active_set.solve_simplex_qp(B, A)


# --------------------------------------------------------------------------
# Rows in different units: the Walmart covariate specification (#960)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def walmart_covariate_fit():
    """STACKEDSC on Wiltshire's public panel, matched on eight covariates.

    Its 429 solves have rows from 8.7e3 down to 7.8e-3: population next to
    shares. Run once for the module; about four seconds.
    """
    from benchmarks.cases import wiltshire_walmart as wm
    df = wm._panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", SimplexQPUncertifiedWarning)
        from mlsynth import STACKEDSC
        res = STACKEDSC({"df": df, "outcome": "emps_n10", "treat": "treat",
                         "unitid": "cty_fips", "time": "year",
                         "agg_weights": "pop90", "n_lags": 5, "n_leads": 6,
                         "display_graphs": False, "covariates": wm.COVARIATES,
                         "covariate_windows": wm.COVARIATE_WINDOW}).fit()
    return float(res.fit_diagnostics.rmse_pre)


def test_rows_in_small_units_are_fit(walmart_covariate_fit):
    """The pre-period RMSE of the covariate fit, against the Gram-form loop's
    1.383693. Measured on the other release tests: a threshold on the data's
    norm gave 1.532 (the small rows left unfit, which the benchmark caught),
    and the componentwise test that stops at its rounding bound gave 1.390.
    This release gives 1.383777. The bound, 1.3845, passes it and refuses
    both of the others."""
    assert walmart_covariate_fit <= 1.3845


def test_rows_in_small_units_are_certified(walmart_covariate_fit):
    """Every solve of the fit certified: an uncertified one would have raised
    in the fixture, which turns ``SimplexQPUncertifiedWarning`` into an error."""
    assert np.isfinite(walmart_covariate_fit)


# --------------------------------------------------------------------------
# A release the objective cannot see: the threshold is what resolves it
# --------------------------------------------------------------------------

def _pinned_below_the_certificate(seed, c, J0, m, extra):
    """A solved design plus ``extra`` donors the optimum gives tiny weight.

    Each new donor is the current fit plus a direction orthogonal to the
    residual, moved along the residual so that its reduced gradient at the old
    optimum is ``-c`` times the certificate's size. Releasing one gains about
    ``c^2`` of the objective, below its rounding for ``c`` near 1e-10, so only
    the release threshold can resolve the second and later ones (the first
    checked release always runs).
    """
    rng = np.random.default_rng(seed)
    D0 = rng.normal(size=(m, J0)) + 3.0
    w0 = solve_simplex_qp(D0, np.zeros(m))
    r0 = D0 @ w0
    cols = []
    for _ in range(extra):
        z = rng.normal(size=m)
        cols.append(D0 @ w0 + z - (z @ r0) / (r0 @ r0) * r0)
    D = np.column_stack([D0] + cols)
    size = np.linalg.norm(D) ** 2 * np.linalg.norm(w0)
    D[:, J0:] -= c * size / (r0 @ r0) * r0[:, None]
    return D, np.concatenate([w0, np.zeros(extra)])


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), extra=st.integers(2, 4),
       c_exp=st.floats(-9.8, -9.2), shape=st.sampled_from([(4, 6), (6, 12)]))
def test_a_release_below_the_objectives_rounding_is_still_made(seed, extra, c_exp, shape):
    """Reduced gradients 1.5e-10 to 6e-10 of the certificate's size: the
    certificate refuses the point that keeps them pinned, and the objective
    cannot tell the release from rounding. A release threshold at ``tol`` left
    97 of 405 such designs uncertified; at a tenth of the certificate, none."""
    D, start = _pinned_below_the_certificate(seed, 10.0 ** c_exp, *shape, extra)
    _, info = solve_simplex_qp(D, np.zeros(D.shape[0]), warm_start=start,
                               accelerate=False, return_info=True)
    assert info["certified"], info
