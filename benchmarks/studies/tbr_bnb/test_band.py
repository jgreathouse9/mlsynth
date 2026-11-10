"""The first tier of the hull stage: the band without the per-geo cap, by one NNLS
and simplex QPs at the band's ends, and its certificate."""
import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .brute import panel, piece_optimum
from .certificate import ExactData, hull_bound
from .hull import Band, ControlBound
from .search import hull_survivors, pieces_to

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _band_cvxpy(y, X, A, B):
    s = float(np.linalg.norm(y))
    w = cp.Variable(X.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(y / s - X @ w / s)), [cp.sum(w) >= A, cp.sum(w) <= B]).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return float(np.sum((y - X @ w.value) ** 2))


def test_smoke_the_band_is_finite_with_a_feasible_point():
    y, X = panel(0)
    v, w = Band(y, X).value(0.3, 0.9, 3)
    assert np.isfinite(v) and w.min() >= -1e-12 and 0.9 - 1e-9 <= w.sum() <= 2.7 + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_the_band_is_the_exact_band_minimum(seed):
    y, X = panel(seed)
    band = Band(y, X)
    for lo, hi, m in ((0.3, 0.9, 3), (0.0, 0.2, 2), (1.5, 2.0, 5)):
        assert band.value(lo, hi, m)[0] == pytest.approx(_band_cvxpy(y, X, m * lo, m * hi), rel=1e-7)


def test_the_band_reuses_its_solves_across_pieces():
    """A band's end m * edge is shared by the pieces on either side of an edge."""
    y, X = panel(1)
    band = Band(y, X)
    pcs = pieces_to(0.5, ExactData(y, X).slope_cap(), 1.25)
    for lo, hi in pcs:
        for m in range(1, 7):
            band.value(lo, hi, m)
    assert band.solves < 2 * len(pcs) * 6


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0), m=st.integers(1, 6))
def test_the_band_never_exceeds_the_capped_hull(seed, lo, width, m):
    """It drops the per-geo cap, so it relaxes the capped hull, which relaxes the piece."""
    y, X = panel(seed)
    hi = lo * width if lo > 0 else width - 1.0
    b = Band(y, X).value(lo, hi, m)[0]
    h = ControlBound(y, X).piece(lo, hi, m)
    assert b <= h * (1 + 1e-6) + 1e-9
    assert b <= piece_optimum(y, X, lo, hi, m) * (1 + 1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0), m=st.integers(1, 6),
       shift=st.floats(-1.0, 1.0))
def test_the_band_point_certifies_the_piece(seed, lo, width, m, shift):
    """The capped hull's first-order bound at the band's point, or any other, is
    a bound on the piece."""
    y, X = panel(seed)
    hi = lo * width if lo > 0 else width - 1.0
    _, w = Band(y, X).value(lo, hi, m)
    w = w + shift * np.random.default_rng(seed).normal(size=6)
    b = hull_bound(ExactData(y, X), w, lo, hi, m)
    assert float(b) <= piece_optimum(y, X, lo, hi, m) * (1 + 1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_the_band_point_certifies_at_least_the_band_value(seed):
    """At the band's minimiser the linear minimum over the band is zero, and over
    the smaller capped hull it is no less, so the certificate is at least the band."""
    y, X = panel(seed)
    for lo, hi, m in ((0.3, 0.9, 3), (0.1, 0.25, 2), (0.05, 0.6, 5)):
        v, w = Band(y, X).value(lo, hi, m)
        assert float(hull_bound(ExactData(y, X), w, lo, hi, m)) >= v * (1 - 1e-8)


@pytest.mark.parametrize("seed", range(3))
def test_two_tiers_keep_exactly_the_capped_hulls_survivors(seed):
    """The band only prunes what the capped hull would; the survivors it passes on
    are then judged by the capped hull, so the stage's output is unchanged."""
    y, X = panel(seed)
    D = ExactData(y, X)
    from .search import incumbent
    v, C = incumbent(y, X, restarts=20)
    x = X[:, sorted(C)].sum(1); b = float((y @ x) / (x @ x))
    UB = D.rss(C, b)
    pcs = pieces_to(b, D.slope_cap(), 1.25)
    one, _ = hull_survivors(y, X, D, pcs, UB, two_tier=False)
    two, stats = hull_survivors(y, X, D, pcs, UB, two_tier=True)
    assert one == two
    assert stats["band_pruned"] + stats["capped_solved"] == len(pcs) * 6


def test_a_band_that_holds_the_nnls_point_is_the_nnls_value():
    y, X = panel(4)
    band = Band(y, X)
    s = band.s_nnls
    v, w = band.value(s / 6, s / 2, 3)                       # the band [s/2, 3s/2] holds s
    assert v == pytest.approx(band.f_nnls, rel=1e-12) and band.solves == 0


def test_the_band_prunes_only_on_its_certificate(monkeypatch):
    """A band value at the incumbent is a float; the pair is ruled out only when
    the exact certificate clears the incumbent, and otherwise reaches the capped hull."""
    from . import search
    y, X = panel(5)
    D = ExactData(y, X)
    v, C = search.incumbent(y, X, restarts=20)
    x = X[:, sorted(C)].sum(1); b = float((y @ x) / (x @ x))
    UB = D.rss(C, b)
    pcs = pieces_to(b, D.slope_cap(), 1.25)
    monkeypatch.setattr(search, "hull_clears", lambda *a, **k: False)
    _, stats = hull_survivors(y, X, D, pcs, UB, two_tier=True)
    assert stats["band_pruned"] == 0 and stats["capped_solved"] == len(pcs) * 6
