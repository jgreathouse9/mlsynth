"""The Shor root (#965): a size's root, once the first-order tiers leave it open,
solves the Shor relaxation only and branches on its point.

On the five test beds the full relaxation at a root closed almost nothing it was
asked to: with the root on Shor alone the beds took 30% fewer Clarabel solves for
6% more nodes, and no group certified a different design. At 96 candidate controls
one Shor solve took 39.9 s against 121.5 s for one full solve, and the root with its
triangle rounds 954 s. A size certificate with ``full=False`` stops after Shor."""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .slopefree import SHOR_ROOT, certify_slope_free, size_certificate
from .treatment import _split

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _branching():
    """The first group of the s15_k3 bed: 12 candidate controls, 16 nodes, roots
    the first-order tiers leave open."""
    return _split(geo_panel(1, n=20, p=15), (1, 11, 14))[:2]


def _solves(monkeypatch):
    real = slopefree.solve
    seen = []

    def solve(c, G, lam, R):
        seen.append((R.level, R.n_fixes))
        return real(c, G, lam, R)

    monkeypatch.setattr(slopefree, "solve", solve)
    return seen


# ---------------------------------------------------------------- smoke


def test_smoke_the_shor_root_keeps_the_design_with_fewer_full_solves(monkeypatch):
    """Group 53 of the s15_k3 bed (seed 66, treated 0, 9, 10): seven full solves with
    the full relaxation at the root, three without."""
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    seen = _solves(monkeypatch)
    off = certify_slope_free(y, X, restarts=5, shor_root=False)
    full_off = sum(lv == "full" for lv, _ in seen)
    seen.clear()
    on = certify_slope_free(y, X, restarts=5, shor_root=True)
    assert on.certified and off.certified and (on.incumbent, on.controls) == (off.incumbent, off.controls)
    assert sum(lv == "full" for lv, _ in seen) < full_off


# ---------------------------------------------------------------- unit


def test_the_shor_root_is_the_default():
    assert SHOR_ROOT is True


def test_no_root_solves_the_full_relaxation(monkeypatch):
    y, X = _branching()
    seen = _solves(monkeypatch)
    certify_slope_free(y, X, restarts=5)
    assert any(lv == "shor" and k == 0 for lv, k in seen)
    assert not any(lv == "full" and k == 0 for lv, k in seen)
    assert any(lv == "full" and k > 0 for lv, k in seen)          # deeper nodes still do


def test_without_the_shor_root_the_root_solves_the_full_relaxation(monkeypatch):
    y, X = _branching()
    seen = _solves(monkeypatch)
    certify_slope_free(y, X, restarts=5, shor_root=False)
    assert any(lv == "full" and k == 0 for lv, k in seen)


def test_a_size_without_the_full_relaxation_stops_at_shor_with_its_point(monkeypatch):
    y, X = _branching()
    D = ExactData(y, X)
    seen = _solves(monkeypatch)
    res = size_certificate(D.c, D.G, Q(10) ** -9, 4, full=False)
    assert seen == [("shor", 0)]
    assert not res.certified and res.tier == "shor" and res.u is not None and len(res.u) == len(D.c)
    assert np.all(res.u >= -1e-6) and abs(float(np.sum(res.u)) - 4) < 1e-4    # a point of the size-4 rows


def test_a_shor_certificate_still_closes_without_the_full_relaxation(monkeypatch):
    """A size the Shor relaxation closes is certified by it, full or not."""
    y, X = panel(0, p=6)
    D = ExactData(y, X)
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, None))
    full = size_certificate(D.c, D.G, Q(10) ** 9, 3)
    shor = size_certificate(D.c, D.G, Q(10) ** 9, 3, full=False)
    assert full.certified and full.tier == "shor"
    assert shor.certified and shor.tier == "shor"


def test_the_first_order_tiers_still_run_first(monkeypatch):
    y, X = panel(0, p=6)
    D = ExactData(y, X)
    seen = _solves(monkeypatch)
    res = size_certificate(D.c, D.G, Q(10) ** 9, 3, full=False)
    assert res.certified and res.tier in ("dual", "reuse") and seen == []


def test_a_failed_shor_solve_falls_back_to_the_full_relaxation(monkeypatch):
    y, X = _branching()
    D = ExactData(y, X)
    real = slopefree.solve
    seen = []

    def solve(c, G, lam, R):
        seen.append(R.level)
        if R.level == "shor":
            raise slopefree.cp.error.SolverError("injected")
        return real(c, G, lam, R)

    monkeypatch.setattr(slopefree, "solve", solve)
    res = size_certificate(D.c, D.G, Q(10) ** -9, 4, rounds=0, full=False)
    assert seen == ["shor", "full"] and res.u is not None


def test_only_the_root_skips_the_full_relaxation(monkeypatch):
    y, X = _branching()
    real = slopefree.size_certificate
    calls = []

    def size(c, G, lam, m, cuts=(), fixes=(), **k):
        calls.append((len(fixes), k.get("full", True)))
        return real(c, G, lam, m, cuts=cuts, fixes=fixes, **k)

    monkeypatch.setattr(slopefree, "size_certificate", size)
    certify_slope_free(y, X, restarts=5)
    assert all(full is False for d, full in calls if d == 0)
    assert all(full is True for d, full in calls if d > 0) and any(d > 0 for d, _ in calls)


def test_the_root_branches_on_shors_point(monkeypatch):
    """The root's branching control is the free control nearest a half in Shor's point."""
    y, X = _branching()
    real_branch = slopefree._branch
    real_size = slopefree.size_certificate
    roots = {}

    def size(c, G, lam, m, cuts=(), fixes=(), **k):
        res = real_size(c, G, lam, m, cuts=cuts, fixes=fixes, **k)
        if not fixes and not res.certified:
            roots[m] = res.u
        return res

    first_fix = {}

    def branch(c, G, out, m, cuts, fixes, *a, **k):
        if len(fixes) == 1 and m not in first_fix:
            first_fix[m] = fixes[0][0]
        return real_branch(c, G, out, m, cuts, fixes, *a, **k)

    monkeypatch.setattr(slopefree, "size_certificate", size)
    monkeypatch.setattr(slopefree, "_branch", branch)
    certify_slope_free(y, X, restarts=5)
    checked = [m for m in roots if m in first_fix and roots[m] is not None]
    assert checked
    for m in checked:
        u = roots[m]
        assert first_fix[m] == min(range(len(u)), key=lambda j: abs(u[j] - 0.5))


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), shor_root=st.booleans())
def test_property_the_shor_root_certifies_the_enumerated_optimum(seed, shor_root):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=7)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3, shor_root=shor_root)
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edges


def test_a_shor_root_at_the_depth_limit_is_reported_open():
    y, X = _branching()
    out = certify_slope_free(y, X, restarts=5, max_depth=0)
    assert not out.certified and out.open and out.tiers.get("shor", 0) >= len(out.open)


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_a_malformed_shor_root_is_refused(bad):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="shor_root"):
        certify_slope_free(y, X, restarts=3, shor_root=bad)
