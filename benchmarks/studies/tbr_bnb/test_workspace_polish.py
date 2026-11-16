"""The first-order polish after a workspace solve's first chunk (#965, workspace.py).

SCS's first chunk of a node leaves multipliers the certificate seldom closes, and
SCS can take many more chunks to close it. After that chunk the workspace runs the
first-order dual (slopefree.dual_multipliers) once, warm from the chunk's
multipliers, and certifies what it returns exactly (fbound.certificate). When the
certificate closes, the node ends "polished"; otherwise SCS goes on as if the polish
had not run. The certificate holds for any multipliers, so the polish can end a node
early and can never make a bound invalid; these tests check both, and that a polish
that fails changes nothing."""
from fractions import Fraction as Q

import numpy as np
import pytest
import scs
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import fbound, slopefree, workspace
from .brute import panel
from .certificate import ExactData
from .progress import reporting
from .slopefree import Rows, bound, objective
from .test_workspace import _best
from .workspace import SizeWorkspace, Workspaces

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _short_chunks(monkeypatch):
    """Chunks of 10 iterations, so SCS takes several chunks on a six-geo node."""
    monkeypatch.setattr(workspace, "CHUNK", 10)


def _node(seed=1, p=6, num=7, m=2, fixes=()):
    """A size-m node of panel(seed, p) at ratio num/10 of y'y."""
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    lam = slopefree.dyadic_below(D.yy * Q(num, 10))
    return D, lam, m, Rows(D.c, m, fixes=list(fixes), face=True, box=True)


def _solve(D, lam, m, R):
    ws = SizeWorkspace(D.c, D.G, lam, m, [], ())
    return ws, ws.solve(R)


def _unpolished(monkeypatch, *node):
    with monkeypatch.context() as mp:
        mp.setattr(workspace, "POLISH", None)
        return _solve(*node)


def _same(a, b):
    assert (a.kind, a.scs_status, a.iters, a.bound) == (b.kind, b.scs_status, b.iters, b.bound)
    for x, y in ((a.y_eq, b.y_eq), (a.y_in, b.y_in), (a.Z, b.Z)):
        assert (x is None) == (y is None)
        if x is not None:
            assert np.asarray(x, dtype=float).tobytes() == np.asarray(y, dtype=float).tobytes()


def _spy_polish(monkeypatch):
    calls, real = [], slopefree.dual_multipliers

    def spy(A, R, m, c, warm=None, **kw):
        out = real(A, R, m, c, warm=warm, **kw)
        calls.append(dict(warm=warm, kw=kw, out=out))
        return out
    monkeypatch.setattr(slopefree, "dual_multipliers", spy)
    return calls


def _spy_chunks(monkeypatch):
    """The multipliers and certificate of every chunk, in order."""
    seen, real = [], fbound.certificate

    def spy(A, R, y_eq, y_in, m, af=None):
        b = real(A, R, y_eq, y_in, m, af)
        seen.append((np.array(y_eq, dtype=float), np.array(y_in, dtype=float), b))
        return b
    monkeypatch.setattr(fbound, "certificate", spy)
    return seen


# A node the certificate does not close at any chunk (its bound is above zero).
OPEN = dict(seed=2, p=6, num=3, m=3)


# ---------------------------------------------------------------- smoke


def test_smoke_the_polish_closes_after_one_chunk_a_node_scs_closes_after_several(monkeypatch):
    node = _node()
    _, plain = _unpolished(monkeypatch, *node)
    assert plain.kind == "certified" and plain.iters > workspace.CHUNK
    D, lam, m, R = node
    ws, o = _solve(*node)
    assert o.kind == "polished" and o.iters <= workspace.CHUNK
    assert o.bound is not None and o.bound <= 0
    assert o.bound == fbound.certificate(ws.A, R, o.y_eq, o.y_in, m)
    assert bound(ws.A, R, o.y_eq, o.y_in, m) <= 0
    assert len(o.y_eq) == R.n_eq and len(o.y_in) == R.n_in


# ---------------------------------------------------------------- unit


def test_the_polish_runs_once_after_the_first_chunk_from_its_multipliers(monkeypatch):
    node = _node(**OPEN)
    calls, chunks = _spy_polish(monkeypatch), _spy_chunks(monkeypatch)
    _, o = _solve(*node)
    assert o.kind != "polished" and len(chunks) > 2
    assert len(calls) == 1
    warm = calls[0]["warm"]
    assert warm.level == "full"
    # the polish's own certificate is not a chunk's: chunks[0] is the first chunk's
    assert np.asarray(warm.y_eq, dtype=float).tobytes() == chunks[0][0].tobytes()
    assert np.asarray(warm.y_in, dtype=float).tobytes() == chunks[0][1].tobytes()
    maxfun, stages = workspace.POLISH
    assert calls[0]["kw"]["maxfun"] == maxfun and calls[0]["kw"]["stages"] == stages


def test_a_failed_polish_leaves_the_solve_as_it_was(monkeypatch):
    node = _node(**OPEN)
    ws_a, plain = _unpolished(monkeypatch, *node)
    calls = _spy_polish(monkeypatch)
    ws_b, o = _solve(*node)
    assert calls and calls[0]["out"][0] is None
    _same(o, plain)
    for x, y in zip(ws_a.last, ws_b.last):
        assert x.tobytes() == y.tobytes()


def test_multipliers_the_certificate_does_not_close_leave_the_solve_to_scs(monkeypatch):
    """The dual's float test only says the polish may have closed: the node ends
    polished only when the certificate at its multipliers closes."""
    node = _node()
    _, plain = _unpolished(monkeypatch, *node)

    def zeros(A, R, m, c, warm=None, **kw):
        return slopefree.Multipliers("full", np.zeros(R.n_eq), np.zeros(R.n_in)), {"evals": 1, "point": None}
    monkeypatch.setattr(slopefree, "dual_multipliers", zeros)
    _, o = _solve(*node)
    _same(o, plain)


def test_a_first_chunk_that_closes_is_not_polished(monkeypatch):
    y, X = panel(2, p=6)
    D = ExactData(y, X)
    lam = slopefree.dyadic_below(D.yy * 4)          # far above every design's ratio
    calls = _spy_polish(monkeypatch)
    _, o = _solve(D, lam, 3, Rows(D.c, 3, face=True, box=True))
    assert o.kind == "certified" and not calls


def test_an_infeasible_node_is_not_polished(monkeypatch):
    real = scs.SCS.solve

    def infeasible(self, *a, **k):
        sol = real(self, *a, **k)
        sol["info"]["status"] = "infeasible"
        return sol
    monkeypatch.setattr(scs.SCS, "solve", infeasible)
    calls = _spy_polish(monkeypatch)
    _, o = _solve(*_node(**OPEN))
    assert o.kind in ("infeasible", "certified") and not calls


def test_the_polish_is_reported(monkeypatch):
    D, lam, m, R = _node()
    events = []
    with reporting(progress=events.append):
        sol = Workspaces().solve(D.c, D.G, lam, R)
    assert sol.status == "polished" and sol.bound <= 0
    assert [e.fields["status"] for e in events if e.kind == "solve"] == ["polished"]


def test_the_polish_reads_its_settings_at_build(monkeypatch):
    monkeypatch.setattr(workspace, "POLISH", (7, 2))
    calls = _spy_polish(monkeypatch)
    _solve(*_node(**OPEN))
    assert calls[0]["kw"]["maxfun"] == 7 and calls[0]["kw"]["stages"] == 2


def test_polished_is_an_outcome():
    assert "polished" in workspace.OUTCOMES


# ---------------------------------------------------------------- edges


def test_the_polish_off(monkeypatch):
    monkeypatch.setattr(workspace, "POLISH", None)
    calls = _spy_polish(monkeypatch)
    _, o = _solve(*_node())
    assert o.kind == "certified" and not calls


def test_a_node_with_fixings_is_polished_and_its_bound_holds(monkeypatch):
    fixes = [(0, 1), (3, 0)]
    D, lam, m, R = _node(seed=1, p=6, num=7, m=3, fixes=fixes)
    ws, o = _solve(D, lam, m, R)
    b = o.bound if o.bound is not None else bound(ws.A, R, o.y_eq, o.y_in, m)
    best = _best(ws.A, 6, m, fixes, D.c)
    assert best is None or b >= best


def test_a_single_chunk_is_still_polished(monkeypatch):
    monkeypatch.setattr(workspace, "STAGES", ((dict(alpha=1.8), 1),))
    _, o = _solve(*_node())
    assert o.kind == "polished"


def test_one_candidate_control(monkeypatch):
    D, lam, m, R = _node(seed=2, p=1, num=5, m=1)
    ws, o = _solve(D, lam, m, R)
    b = o.bound if o.bound is not None else bound(ws.A, R, o.y_eq, o.y_in, m)
    assert b >= _best(ws.A, 1, 1, [], D.c)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("setting", [(0, 3), (150, 0), (1.5, 3), (150, True), (150,), "150", [150, 3]])
def test_a_malformed_polish_setting_is_refused(monkeypatch, setting):
    monkeypatch.setattr(workspace, "POLISH", setting)
    D, lam, m, _ = _node()
    with pytest.raises(MlsynthConfigError, match="POLISH"):
        SizeWorkspace(D.c, D.G, lam, m, [], ())


def test_an_error_in_the_polish_is_reported_and_clarabel_solves(monkeypatch):
    def boom(*a, **k):
        raise np.linalg.LinAlgError("eigh did not converge")
    monkeypatch.setattr(slopefree, "dual_multipliers", boom)
    D, lam, m, R = _node()
    events = []
    with reporting(progress=events.append):
        sol = Workspaces().solve(D.c, D.G, lam, R)
    statuses = [(e.fields["solver"], e.fields["status"]) for e in events if e.kind == "solve"]
    assert statuses[0] == ("workspace", "error") and statuses[-1][0] == "clarabel"
    assert sol.solver == "clarabel"


# ---------------------------------------------------------------- properties


@st.composite
def _nodes(draw):
    p = draw(st.integers(2, 7))
    m = draw(st.integers(1, p))
    geos = draw(st.lists(st.integers(0, p - 1), unique=True, max_size=p - 1))
    fixes = [(j, draw(st.integers(0, 1))) for j in geos]
    return draw(st.integers(0, 10_000)), p, draw(st.integers(1, 12)), m, fixes


@SETTINGS
@given(_nodes())
def test_property_a_polish_ends_a_node_sooner_or_changes_nothing(node):
    """Either the node ends polished, its certificate closed and no later than SCS
    alone would have ended it, or it ends exactly as SCS alone ends it."""
    seed, p, num, m, fixes = node
    if sum(v for _, v in fixes) > m or len(fixes) - sum(v for _, v in fixes) > p - m:
        return
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(workspace, "CHUNK", 10)
        D, lam, m, R = _node(seed, p, num, m, fixes)
        mp.setattr(workspace, "POLISH", None)
        _, plain = _solve(D, lam, m, R)
        mp.setattr(workspace, "POLISH", (150, 3))
        ws, o = _solve(D, lam, m, R)
    finally:
        mp.undo()
    if o.kind == "polished":
        assert o.bound <= 0 and o.iters <= plain.iters
        assert bound(ws.A, R, o.y_eq, o.y_in, m) <= 0
        best = _best(ws.A, p, m, fixes, D.c)
        assert best is None or best <= 0
    else:
        _same(o, plain)
