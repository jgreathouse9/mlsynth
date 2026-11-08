"""The first-order dual's L-BFGS-B memory (#965): five correction pairs, not twenty.

At 96 candidate controls L-BFGS-B's own work per call (scipy's setulb, linear in the
memory times the 18,627 multipliers) was 65% of a first-order schedule's time. With
five pairs the three size-24 nodes of the 100-geo group (4, 27, 33, 58) that the
first-order triangle rounds closed closed in the same round in 64-67 s against
118-121 s; on the five test beds the dual took 24% less time for 8 more nodes of
4,710, and every group certified the same design. The memory only steers the
search for multipliers: what is certified is bound()'s, at whatever multipliers
it finds, so no memory changes what is proved."""
import numpy as np
import pytest
import scipy.optimize
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .slopefree import DUAL_MEMORY, Rows, certify_slope_free, dual_multipliers, group_certificate, objective
from .treatment import _split

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _memories(monkeypatch):
    """Record the memory of every L-BFGS-B call."""
    real = scipy.optimize.minimize
    seen = []

    def spy(*a, **k):
        if k.get("method") == "L-BFGS-B":
            seen.append(k["options"]["maxcor"])
        return real(*a, **k)

    monkeypatch.setattr(scipy.optimize, "minimize", spy)
    return seen


def _open_size():
    """A size of a bed group that the dual's first schedule does not close at a ratio
    just above the incumbent's, so L-BFGS-B runs."""
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    D = ExactData(y, X)
    return D, slopefree.dyadic_below(D.yy) / 4


# ---------------------------------------------------------------- smoke


def test_smoke_a_bed_group_certifies_the_same_design_at_either_memory(monkeypatch):
    """Group 53 of the s15_k3 bed (seed 66, treated 0, 9, 10), at five pairs and at
    twenty: the same certified design."""
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    out = {}
    for mem in (DUAL_MEMORY, 20):
        monkeypatch.setattr(slopefree, "DUAL_MEMORY", mem)
        out[mem] = certify_slope_free(y, X, restarts=5)
    a, b = out[DUAL_MEMORY], out[20]
    assert a.certified and b.certified and (a.incumbent, a.controls) == (b.incumbent, b.controls)


# ---------------------------------------------------------------- unit


def test_the_dual_memory_is_the_measured_one():
    """Five pairs: at 96 controls the three triangle-closed size-24 nodes closed in
    the same round at 64-67 s against 118-121 s for twenty, 90-101 s for ten, and
    three left one of them for a second round (106 s); the beds' dual 24% faster
    (results/dual_memory.log). A change here needs that measurement redone."""
    assert DUAL_MEMORY == 5


def test_the_memory_reaches_l_bfgs_b(monkeypatch):
    D, lam = _open_size()
    seen = _memories(monkeypatch)
    A = objective(D.c, D.G, lam)
    dual_multipliers(A, Rows(D.c, 4), 4, D.c, maxfun=20, stages=2, budget=60)
    assert seen and set(seen) == {DUAL_MEMORY}


def test_the_group_screen_runs_at_the_memory(monkeypatch):
    D, lam = _open_size()
    seen = _memories(monkeypatch)
    group_certificate(D.c, D.G, lam, schedules=((20, 2, 60),))
    assert seen and set(seen) == {DUAL_MEMORY}


def test_the_memory_is_read_when_the_dual_runs(monkeypatch):
    """A memory set after import reaches L-BFGS-B, so the beds and the tests can vary it."""
    D, lam = _open_size()
    monkeypatch.setattr(slopefree, "DUAL_MEMORY", 7)
    seen = _memories(monkeypatch)
    dual_multipliers(objective(D.c, D.G, lam), Rows(D.c, 4), 4, D.c, maxfun=20, stages=2, budget=60)
    assert seen and set(seen) == {7}


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), mem=st.sampled_from([1, 2, 5, 20]))
def test_property_no_memory_changes_what_is_certified(seed, mem):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        mp.setattr(slopefree, "DUAL_MEMORY", mem)
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


# ---------------------------------------------------------------- edges


def test_a_one_pair_memory_still_certifies_the_enumerated_optimum(monkeypatch):
    """With one correction pair L-BFGS-B is close to a projected gradient step and
    the dual closes less; the relaxations close the rest, and the design is the
    enumerated one."""
    monkeypatch.setattr(slopefree, "DUAL_MEMORY", 1)
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    best, C = positive_optimum(y, X)
    out = certify_slope_free(y, X, restarts=5)
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


def test_a_dual_closed_at_its_warm_start_never_calls_l_bfgs_b(monkeypatch):
    """Multipliers that already close the size close it at the start, before the
    memory matters. (From zero they cannot: the certificate's value there is 0.)"""
    y, X = panel(0, p=5)
    D = ExactData(y, X)
    A, R = objective(D.c, D.G, slopefree.dyadic_below(D.yy) * 4), Rows(D.c, 2)
    w, _ = dual_multipliers(A, R, 2, D.c)
    assert w is not None
    seen = _memories(monkeypatch)
    again, info = dual_multipliers(A, R, 2, D.c, warm=w)
    assert again is not None and info["evals"] == 1 and seen == []
    assert np.all(np.isfinite(again.y_eq))
