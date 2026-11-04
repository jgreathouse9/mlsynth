"""The capped hull, the hull stage's second tier, as one capped simplex QP.

A slack weight with a zero column, capped at B - A, turns the per-geo cap and
the band A <= sum w <= B into the capped simplex in v = w / B, which the
library's solve_simplex_qp(..., upper=...) solves exactly.
"""
import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from .brute import panel, piece_optimum
from .certificate import ExactData, hull_bound
from .hull import Band, ControlBound
from .search import hull_survivors, incumbent, pieces_to

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _capped_cvxpy(y, X, lo, hi, m):
    """The capped hull by Clarabel at tight tolerances: the reference."""
    s = float(np.linalg.norm(y))
    w = cp.Variable(X.shape[1])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y / s - X @ w / s)),
                      [w >= 0, w <= hi, cp.sum(w) >= m * lo, cp.sum(w) <= m * hi])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return float(np.sum((y - X @ w.value) ** 2))


def _feasible(w, lo, hi, m, tol=1e-9):
    s = float(w.sum())
    return (w.min() >= -tol and w.max() <= hi * (1 + tol) + tol
            and m * lo * (1 - tol) - tol <= s <= m * hi * (1 + tol) + tol)


def _draw(lo, width):
    return lo * width if lo > 0 else width - 1.0


# ---------------------------------------------------------------- smoke


def test_smoke_the_capped_hull_is_finite_at_a_feasible_point():
    y, X = panel(0)
    H = ControlBound(y, X)
    v = H.piece(0.3, 0.9, 3)
    assert np.isfinite(v) and H.point.shape == (6,) and _feasible(H.point, 0.3, 0.9, 3)
    assert v == pytest.approx(float(np.sum((y - X @ H.point) ** 2)), rel=1e-12)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", range(6))
def test_the_capped_hull_is_the_reference_minimum(seed):
    y, X = panel(seed)
    H = ControlBound(y, X)
    for lo, hi, m in ((0.3, 0.9, 3), (0.0, 0.2, 2), (1.5, 2.0, 5), (0.05, 0.08, 6), (0.2, 3.0, 1)):
        assert H.piece(lo, hi, m) == pytest.approx(_capped_cvxpy(y, X, lo, hi, m), rel=1e-7)


@pytest.mark.parametrize("seed", range(6))
def test_the_cap_binds_where_the_band_alone_would_not(seed):
    """A cap below the band's largest weight changes the minimum: the cap is in force."""
    y, X = panel(seed)
    lo, hi, m = 0.05, 0.12, 4
    _, wb = Band(y, X).value(lo, hi, m)
    H = ControlBound(y, X)
    v = H.piece(lo, hi, m)
    assert H.point.max() <= hi * (1 + 1e-9)
    if wb.max() > hi * (1 + 1e-6):
        assert v > Band(y, X).value(lo, hi, m)[0] * (1 + 1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_both_ends_of_the_band_are_reached(seed):
    """A band far above the unconstrained total puts the point on its lower end; one
    far below, on its upper end."""
    y, X = panel(seed)
    H = ControlBound(y, X)
    H.piece(2.0, 3.0, 6)
    assert float(H.point.sum()) == pytest.approx(12.0, rel=1e-9)
    H.piece(0.001, 0.002, 2)
    assert float(H.point.sum()) == pytest.approx(0.004, rel=1e-9)


def test_hull_survivors_keep_the_reference_survivors():
    """Swapping the solver keeps the stage's output: every pair the reference keeps."""
    for seed in range(3):
        y, X = panel(seed)
        D = ExactData(y, X)
        _, C = incumbent(y, X, restarts=20)
        x = X[:, sorted(C)].sum(1); b = float((y @ x) / (x @ x))
        UB = D.rss(C, b)
        pcs = pieces_to(b, D.slope_cap(), 1.25)
        out, _ = hull_survivors(y, X, D, pcs, UB, two_tier=False)
        ref = [(lo, hi, m) for lo, hi in pcs for m in range(1, 7)
               if _capped_cvxpy(y, X, lo, hi, m) < float(UB) * (1 - 1e-6)]
        assert set(ref) <= set(out)
        assert len(out) - len(ref) <= 1        # a pair at the threshold may fall either side


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 6))
def test_the_point_is_feasible_and_certifies_its_own_value(seed, lo, width, m):
    """The first-order bound read at the minimiser equals the minimum, so the point is
    optimal, by a certificate that depends on no solver."""
    y, X = panel(seed)
    hi = _draw(lo, width)
    H = ControlBound(y, X)
    v = H.piece(lo, hi, m)
    assert _feasible(H.point, lo, hi, m)
    assert float(hull_bound(ExactData(y, X), H.point, lo, hi, m)) >= v * (1 - 1e-8)


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 6))
@example(seed=0, lo=0.0, width=1.05, m=6)
@example(seed=0, lo=5e-324, width=1.25, m=6)          # hi rounds to lo: caps that sum to 1 - 1e-16
def test_the_capped_hull_sits_between_the_band_and_the_piece(seed, lo, width, m):
    """The band relaxes the capped hull, which relaxes the piece."""
    y, X = panel(seed)
    hi = _draw(lo, width)
    v = ControlBound(y, X).piece(lo, hi, m)
    assert Band(y, X).value(lo, hi, m)[0] <= v * (1 + 1e-9) + 1e-9
    assert v <= piece_optimum(y, X, lo, hi, m) * (1 + 1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.01, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 5), k=st.integers(-3, 3))
def test_scaling_the_data_scales_the_minimum(seed, lo, width, m, k):
    """y, X -> c y, c X leaves the point and scales the minimum by c^2 (c a power of
    two, so the scaling is exact)."""
    y, X = panel(seed)
    hi = _draw(lo, width)
    c = 2.0 ** k
    a, b = ControlBound(y, X), ControlBound(c * y, c * X)
    va, vb = a.piece(lo, hi, m), b.piece(lo, hi, m)
    assert vb == pytest.approx(c * c * va, rel=1e-9)


# ---------------------------------------------------------------- edges


def test_one_candidate_control():
    y, X = panel(2, p=1)
    H = ControlBound(y, X)
    v = H.piece(0.2, 0.9, 1)
    assert v == pytest.approx(piece_optimum(y, X, 0.2, 0.9, 1), rel=1e-9)
    assert 0.2 - 1e-12 <= float(H.point[0]) <= 0.9 + 1e-12


def test_a_band_from_zero():
    """lo = 0 makes the slack's cap the whole of B."""
    y, X = panel(3)
    v = ControlBound(y, X).piece(0.0, 0.5, 4)
    assert v == pytest.approx(_capped_cvxpy(y, X, 0.0, 0.5, 4), rel=1e-7)


def test_a_point_band_is_the_hull_of_one_slope():
    """lo = hi leaves the slack no room: the total is exactly m hi."""
    y, X = panel(4)
    H = ControlBound(y, X)
    v = H.piece(0.4, 0.4, 3)
    assert float(H.point.sum()) == pytest.approx(1.2, rel=1e-12)
    assert v == pytest.approx(_capped_cvxpy(y, X, 0.4, 0.4, 3), rel=1e-7)


def test_collinear_controls():
    """Duplicated columns make the minimiser a face; the minimum is still the reference's."""
    y, X = panel(5)
    X = np.column_stack([X, X[:, :2]])
    v = ControlBound(y, X).piece(0.1, 0.6, 4)
    assert v == pytest.approx(_capped_cvxpy(y, X, 0.1, 0.6, 4), rel=1e-7)


def test_an_infeasible_piece_contributes_nothing_and_leaves_no_point():
    y, X = panel(1, p=4)
    H = ControlBound(y, X)
    assert H.piece(2.0, 1.0, 4) == np.inf
    assert H.point is None


def test_the_solves_are_counted():
    y, X = panel(0)
    H = ControlBound(y, X)
    H.piece(0.3, 0.9, 3); H.piece(0.3, 0.9, 4)
    assert H.solves == 2


def test_a_zero_slope_leaves_only_the_origin():
    """hi = 0 makes the set the single point w = 0, whose RSS is y'y."""
    y, X = panel(0)
    H = ControlBound(y, X)
    assert H.piece(0.0, 0.0, 3) == pytest.approx(float(y @ y), rel=1e-12)
    assert np.array_equal(H.point, np.zeros(6))


def test_caps_that_round_below_one_are_lifted_by_the_slack():
    """m = p and lo = hi make the set the single point w = hi: the caps sum to one
    exactly, and in floating point 6 caps of 0.1 / 0.6 sum to 1 - 1.1e-16. The slack
    takes the rounding, so the solve sees caps summing to one and returns that point."""
    y, X = panel(0)
    H = ControlBound(y, X)
    v = H.piece(0.1, 0.1, 6)
    assert np.allclose(H.point, 0.1, rtol=1e-12)
    assert v == pytest.approx(float(np.sum((y - X @ np.full(6, 0.1)) ** 2)), rel=1e-12)
