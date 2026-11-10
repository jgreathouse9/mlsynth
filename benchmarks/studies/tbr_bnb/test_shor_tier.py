"""The Shor tier, off by default (#965).

The cascade once ran reuse, Shor, full. The dual tier (``dual_multipliers``) came
after it and bounds the full rows directly, so what the Shor relaxation, a weaker
one, could close, the dual tier closes first. Over every bed group the Shor tier
closed 0 of 756 sizes it was asked to, and on the GeoLift window at k = 4 0 of 3
(results/shor_tier.log); each one went on to the full relaxation. So the Shor
solve is skipped unless asked for (``SHOR_TIER``), and the search's answer is
the same either way."""
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .slopefree import certify_slope_free, size_certificate
from .treatment import _split


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _levels(monkeypatch):
    """The level of every relaxation solved, in order."""
    seen = []
    real = slopefree.solve

    def spy(c, G, lam, R, **k):
        seen.append(R.level)
        return real(c, G, lam, R, **k)

    monkeypatch.setattr(slopefree, "solve", spy)
    return seen


def _open_size():
    """panel(2, p=6) at y'y / 2, size 3: the first-order tiers leave it to the relaxations."""
    y, X = panel(2, p=6)
    D = ExactData(y, X)
    return D, slopefree.dyadic_below(D.yy / 2)


def _bed():
    return _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]


# ---------------------------------------------------------------- smoke


def test_smoke_a_bed_group_certifies_the_same_design_with_and_without_the_shor_tier():
    y, X = _bed()
    off = certify_slope_free(y, X, restarts=5)
    on = certify_slope_free(y, X, restarts=5, shor=True)
    assert off.certified and on.certified and (off.incumbent, off.controls) == (on.incumbent, on.controls)


# ---------------------------------------------------------------- unit


def test_the_shor_tier_is_off_by_default(monkeypatch):
    assert slopefree.SHOR_TIER is False
    seen = _levels(monkeypatch)
    D, lam = _open_size()
    size_certificate(D.c, D.G, lam, 3, schedules=())
    assert seen and "shor" not in seen and seen[0] == "full"


def test_asked_for_the_shor_tier_solves_shor_first(monkeypatch):
    seen = _levels(monkeypatch)
    D, lam = _open_size()
    size_certificate(D.c, D.G, lam, 3, schedules=(), shor=True)
    assert seen[0] == "shor"


def test_without_the_shor_tier_a_size_is_decided_as_with_it():
    """The Shor solve is the only difference: the full tier's answer is the same."""
    D, lam = _open_size()
    off = size_certificate(D.c, D.G, lam, 3, schedules=())
    on = size_certificate(D.c, D.G, lam, 3, schedules=(), shor=True)
    if on.tier != "shor":
        assert (off.certified, off.tier) == (on.certified, on.tier)


def test_the_shor_flag_reaches_every_node(monkeypatch):
    seen = []
    real = slopefree.size_certificate

    def spy(*a, shor=None, **k):
        seen.append(shor)
        return real(*a, shor=shor, **k)

    monkeypatch.setattr(slopefree, "size_certificate", spy)
    y, X = _bed()
    certify_slope_free(y, X, restarts=5, shor=True)
    assert seen and all(s is True for s in seen)
    seen.clear()
    certify_slope_free(y, X, restarts=5)
    assert seen and all(s is False for s in seen)


def test_a_search_without_the_shor_tier_solves_no_shor_relaxation(monkeypatch):
    seen = _levels(monkeypatch)
    y, X = _bed()
    certify_slope_free(y, X, restarts=5)
    assert seen and "shor" not in seen


# ---------------------------------------------------------------- properties


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_property_the_search_certifies_the_enumerated_optimum_without_the_shor_tier(seed):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=7)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3)
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_a_malformed_shor_flag_is_refused(bad):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="shor"):
        certify_slope_free(y, X, restarts=3, shor=bad)
