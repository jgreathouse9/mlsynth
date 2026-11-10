"""The full relaxation on one SCS workspace per size (#965, workspace.py).

The workspace is built once for a size, a cut set and a set of triangle rows; a
node's fixings change only the right-hand side of two box rows per geo, so every
node of the size reuses one factorization. A node runs SCS in chunks, from the
last node's iterate, and stops at the first chunk whose multipliers the exact
certificate closes. The certificate holds for any multipliers, so none of this
can make a bound invalid; the tests below check the bound is valid at every
outcome, and that it is as tight as Clarabel's where Clarabel closes a node,
which is what a wrong mapping of the multipliers would break."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
import scs
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import fbound, slopefree, workspace
from .brute import geo_panel, panel
from .certificate import ExactData
from .progress import reporting
from .slopefree import Rows, bound, certify_slope_free, objective, size_certificate
from .treatment import _split
from .workspace import Workspaces

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _designs(p, m, fixes):
    fx = dict(fixes)
    for C in itertools.combinations(range(p), m):
        if all((j in C) == bool(v) for j, v in fx.items()):
            yield C


def _best(A, p, m, fixes, c):
    """max u'Au over the designs of size m within the fixings with c'u >= 0 (the
    relaxation's rows keep only a nonnegative slope), exactly."""
    vals = [sum((A[i][j] for i in C for j in C), Q(0)) for C in _designs(p, m, fixes)
            if sum((c[j] for j in C), Q(0)) >= 0]
    return max(vals) if vals else None


def _pinned(seed=2, p=6):
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    return D, slopefree.dyadic_below(D.yy / 2)


def _bed(p=15, T=(0, 9, 10), seed=66):
    return _split(geo_panel(seed, n=20, p=p), T)[:2]


def _box_rows(D, m, fixes=(), triangles=(), cuts=()):
    return Rows(D.c, m, cuts=cuts, triangles=triangles, fixes=fixes, face=True, box=True)


# ---------------------------------------------------------------- smoke


def test_smoke_the_workspace_solves_a_size():
    D, lam = _pinned()
    sol = Workspaces().solve(D.c, D.G, lam, _box_rows(D, 3))
    assert sol.solver == "workspace" and sol.Z.shape == (7, 7) and np.isfinite(sol.value)
    assert sol.status in workspace.OUTCOMES


def test_smoke_a_bed_group_certifies_the_same_design_as_clarabel():
    y, X = _bed()
    a = certify_slope_free(y, X, restarts=5, solver="workspace")
    b = certify_slope_free(y, X, restarts=5, solver="clarabel")
    assert a.certified and b.certified and (a.incumbent, a.controls) == (b.incumbent, b.controls)


# ---------------------------------------------------------------- unit: the rows


def test_box_rows_are_two_per_free_geo_at_the_end_of_the_inequalities():
    D, _ = _pinned()
    plain = Rows(D.c, 3, fixes=[(1, 1), (4, 0)])
    boxed = Rows(D.c, 3, fixes=[(1, 1), (4, 0)], box=True)
    assert boxed.n_eq == plain.n_eq and boxed.n_in == plain.n_in + 2 * 4
    assert boxed.ineq[:plain.n_in] == plain.ineq
    free = [j for j in range(6) if j not in (1, 4)]
    assert boxed.ineq[plain.n_in:] == [r for j in free for r in ({(0, j + 1): 1}, {(0, j + 1): -1})]
    assert boxed.bin[plain.n_in:] == [b for _ in free for b in (1, 0)]
    box = boxed.matrices(1.0)[1][plain.n_in:].toarray()           # the rows the workspace reads
    want = np.zeros_like(box)
    for k, j in enumerate(free):
        want[2 * k, boxed.n * (j + 1)], want[2 * k + 1, boxed.n * (j + 1)] = 1.0, -1.0
    assert np.array_equal(box, want)


def test_box_rows_hold_at_every_design():
    D, _ = _pinned()
    R = Rows(D.c, 3, fixes=[(2, 1)], box=True)
    for C in _designs(6, 3, [(2, 1)]):
        if sum((D.c[j] for j in C), Q(0)) < 0:
            continue                             # the rows carrying c exclude it, not the box rows
        u = np.array([Q(1) if j in C else Q(0) for j in range(6)], dtype=object)
        z = np.concatenate([[Q(1)], u])
        assert R.holds(np.outer(z, z))


def test_rows_keep_their_cuts_and_triangles():
    D, _ = _pinned()
    tri = [(0, (0, 1, 2)), (1, (1, 3, 5))]
    R = Rows(D.c, 3, cuts=[{4, 0}, {2}], triangles=tri)
    assert R.cuts == ((0, 4), (2,)) and R.triangles == tuple(tri)


# ---------------------------------------------------------------- unit: one workspace a size


def _count_builds(monkeypatch):
    built = []
    real = scs.SCS

    def spy(*a, **k):
        built.append(k)
        return real(*a, **k)

    monkeypatch.setattr(workspace.scs, "SCS", spy)
    return built


def test_the_nodes_of_a_size_share_one_workspace(monkeypatch):
    built = _count_builds(monkeypatch)
    D, lam = _pinned()
    ws = Workspaces()
    for fixes in ([], [(0, 1)], [(0, 1), (3, 0)], [(2, 0)]):
        ws.solve(D.c, D.G, lam, _box_rows(D, 3, fixes))
    assert 1 <= len(built) <= len(workspace.STAGES)


def test_a_new_size_or_ratio_builds_a_new_workspace(monkeypatch):
    built = _count_builds(monkeypatch)
    D, lam = _pinned()
    ws = Workspaces()
    ws.solve(D.c, D.G, lam, _box_rows(D, 3))
    n = len(built)
    ws.solve(D.c, D.G, lam, _box_rows(D, 4))
    assert len(built) > n
    n = len(built)
    ws.solve(D.c, D.G, slopefree.dyadic_below(lam * Q(3, 4)), _box_rows(D, 4))
    assert len(built) > n


def test_the_first_stage_takes_its_settings(monkeypatch):
    built = _count_builds(monkeypatch)
    D, lam = _pinned()
    Workspaces().solve(D.c, D.G, lam, _box_rows(D, 3))
    for k, v in workspace.STAGES[0][0].items():
        assert built[0][k] == v
    assert built[0]["eps_abs"] == workspace.EPS and built[0]["max_iters"] == workspace.CHUNK


def test_a_node_starts_from_the_last_node_solved(monkeypatch):
    calls = []
    real = scs.SCS.solve
    monkeypatch.setattr(scs.SCS, "solve", lambda self, *a, **k: calls.append(k.get("warm_start")) or real(self, *a, **k))
    D, lam = _pinned()
    ws = Workspaces()
    ws.solve(D.c, D.G, lam, _box_rows(D, 3))
    first = len(calls)
    ws.solve(D.c, D.G, lam, _box_rows(D, 3, [(0, 1)]))
    assert calls[0] is False and all(calls[first:])


def test_a_closed_chunk_ends_the_solve(monkeypatch):
    """A node the certificate closes stops at that chunk: one call to SCS."""
    calls = []
    real = scs.SCS.solve
    monkeypatch.setattr(scs.SCS, "solve", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    D, _ = _pinned()
    A = objective(D.c, D.G, slopefree.dyadic_below(D.yy * 4))   # far above every design's ratio
    sol = Workspaces().solve(D.c, D.G, slopefree.dyadic_below(D.yy * 4), _box_rows(D, 3))
    assert sol.status == "certified" and sol.bound is not None and sol.bound <= 0 and len(calls) == 1
    assert sol.bound == fbound.certificate(A, _box_rows(D, 3), sol.y_eq, sol.y_in, 3)
    assert bound(A, _box_rows(D, 3), sol.y_eq, sol.y_in, 3) <= 0


def test_the_certificate_closes_a_chunk_before_scs_converges(monkeypatch):
    """With a tolerance SCS cannot meet, only the screen and the certificate can end
    the solve: a node they close ends at its first chunk, not at the cap."""
    monkeypatch.setattr(workspace, "EPS", 1e-15)
    monkeypatch.setattr(workspace, "STAGES", ((dict(alpha=1.8), 3),))
    monkeypatch.setattr(workspace, "CHUNK", 100)
    D, _ = _pinned()
    lam = slopefree.dyadic_below(D.yy * 4)
    sol = Workspaces().solve(D.c, D.G, lam, _box_rows(D, 3))
    assert sol.status == "certified" and sol.bound <= 0


def test_the_multipliers_are_on_the_rows_of_r():
    D, lam = _pinned()
    R = _box_rows(D, 3, [(1, 1), (5, 0)])
    sol = Workspaces().solve(D.c, D.G, lam, R)
    assert len(sol.y_eq) == R.n_eq and len(sol.y_in) == R.n_in


@pytest.mark.parametrize("fixes", [[], [(0, 1)], [(2, 0), (4, 1)]])
def test_the_bound_is_as_tight_as_clarabels(fixes):
    """A wrong map from SCS's multipliers to R's rows (a fixing's sign, a box row's
    place, the scaling of the rows carrying c) leaves the bound valid but loose: at
    the relaxation's optimum the two solvers' certificates agree."""
    D, lam = _pinned()
    A = objective(D.c, D.G, lam)
    R = _box_rows(D, 3, fixes)
    ws = Workspaces().solve(D.c, D.G, lam, R)
    cl = slopefree.solve(D.c, D.G, lam, Rows(D.c, 3, fixes=fixes, face=True), solver="clarabel")
    b_ws = ws.bound if ws.bound is not None else bound(A, R, ws.y_eq, ws.y_in, 3)
    b_cl = bound(A, Rows(D.c, 3, fixes=fixes, face=True), cl.y_eq, cl.y_in, 3)
    scale = float(np.abs(np.asarray(A, dtype=float)).max())
    assert abs(float(b_ws) - float(b_cl)) <= 1e-4 * scale


def test_a_binding_fixing_reaches_the_relaxation():
    """At y'y / 4 and size 3 the root is open and the node with the geo the root leans
    on most fixed out closes under Clarabel. The workspace must close it too, after
    solving the root: a fixing left out of the box rows' right-hand side solves the
    root again, whose bound is above zero."""
    D, _ = _pinned()
    lam = slopefree.dyadic_below(D.yy / 4)
    A = objective(D.c, D.G, lam)
    R0 = Rows(D.c, 3, face=True)
    root = slopefree.solve(D.c, D.G, lam, R0, solver="clarabel")
    assert bound(A, R0, root.y_eq, root.y_in, 3) > 0               # the root is open
    fixes = [(int(np.argmax(root.Z[0, 1:])), 0)]
    R1 = Rows(D.c, 3, fixes=fixes, face=True)
    cl = slopefree.solve(D.c, D.G, lam, R1, solver="clarabel")
    assert bound(A, R1, cl.y_eq, cl.y_in, 3) <= 0                  # the node closes
    ws = Workspaces()
    ws.solve(D.c, D.G, lam, _box_rows(D, 3))
    R = _box_rows(D, 3, fixes)
    sol = ws.solve(D.c, D.G, lam, R)
    b = sol.bound if sol.bound is not None else bound(A, R, sol.y_eq, sol.y_in, 3)
    assert b <= 0


def test_a_size_clarabel_closes_the_workspace_closes():
    y, X = _bed()
    D = ExactData(y, X)
    from .search import incumbent
    C = set(incumbent(y, X, restarts=5)[1])
    lam = slopefree._ratio(D.c, D.G, C)[1]
    closed = 0
    for m in range(1, len(D.c) + 1):
        cl = size_certificate(D.c, D.G, lam, m, cuts=[C], solver="clarabel", first_order_rounds=0, schedules=((1, 1, 1),))
        ws = size_certificate(D.c, D.G, lam, m, cuts=[C], solver="workspace", schedules=((1, 1, 1),))
        if cl.certified and cl.tier == "full":
            closed += 1
            assert ws.certified, m
    assert closed > 0


# ---------------------------------------------------------------- unit: the triangle pool


def test_the_pool_holds_the_active_triangles_of_the_last_root():
    y, X = _bed()
    D = ExactData(y, X)
    lam = slopefree.dyadic_below(D.yy / 3)
    ws = Workspaces()
    tri = [(kind, idx) for _, kind, idx in slopefree.separate_triangles(
        np.full(len(D.c), 0.5), np.full((len(D.c), len(D.c)), 0.1), k=60)]
    R = Rows(D.c, 5, triangles=tri, face=True, box=True)
    assert tri
    sol = ws.solve(D.c, D.G, lam, R)
    if sol.status == "certified":
        pytest.skip("closed before the pool was read")
    y_tri = np.asarray(sol.y_in[R.n_in - 2 * len(D.c) - len(tri):R.n_in - 2 * len(D.c)], dtype=float)
    top = y_tri.max()
    assert set(ws.pool) == {t for t, v in zip(tri, y_tri) if top > 0 and v > workspace.ACTIVE * top}
    marker = ((0, (0, 1, 2)),)
    ws.pool = marker
    ws.solve(D.c, D.G, lam, Rows(D.c, 5, triangles=tri, fixes=[(0, 1)], face=True, box=True))
    assert ws.pool == marker                                 # only a root sets the pool


def test_size_certificate_starts_from_the_pool(monkeypatch):
    y, X = _bed()
    D = ExactData(y, X)
    ws = Workspaces()
    ws.pool = ((0, (0, 1, 2)), (1, (3, 4, 5)))
    seen = []
    real = Workspaces.solve
    monkeypatch.setattr(Workspaces, "solve", lambda self, c, G, lam, R: seen.append(R.triangles) or real(self, c, G, lam, R))
    size_certificate(D.c, D.G, slopefree.dyadic_below(D.yy / 3), 5, solver="workspace", workspaces=ws,
                     schedules=((1, 1, 1),))
    assert seen and seen[0][:2] == ((0, (0, 1, 2)), (1, (3, 4, 5)))


def test_one_search_keeps_one_set_of_workspaces(monkeypatch):
    made = []
    real = Workspaces.__init__
    monkeypatch.setattr(Workspaces, "__init__", lambda self: made.append(1) or real(self))
    y, X = _bed()
    r = certify_slope_free(y, X, restarts=5, solver="workspace")
    assert r.certified and r.nodes > 1 and len(made) == 1


def test_two_groups_in_one_process_do_not_share_a_pool():
    """The spike's pool was module state: a group of 14 controls after one of 15 read
    triangles past its matrix. The pool belongs to one certify_slope_free call."""
    for p, T in ((18, (0, 9, 10)), (16, (0, 9))):
        y, X = _bed(p=p, T=T)
        r = certify_slope_free(y, X, restarts=5, solver="workspace")
        assert r.certified


# ---------------------------------------------------------------- unit: schedules, screen, default


def test_the_workspace_takes_one_first_order_schedule_and_no_rounds(monkeypatch):
    monkeypatch.setattr(slopefree, "SIZE_LADDER_FROM", 5)
    assert slopefree.size_schedules(15, "workspace") == slopefree.SCREEN_SCHEDULES[:1]
    assert slopefree.size_schedules(15, "clarabel") == slopefree.SCREEN_SCHEDULES
    rounds = []
    real = slopefree.separate_triangles
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: rounds.append(1) or real(*a, **k))
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, {"evals": 1, "point": np.eye(16)}))
    D = ExactData(*_bed())
    size_certificate(D.c, D.G, slopefree.dyadic_below(D.yy / 3), 5, solver="workspace", relaxations=False)
    size_certificate(D.c, D.G, slopefree.dyadic_below(D.yy / 3), 5, solver="workspace", relaxations=False,
                     schedules=slopefree.SCREEN_SCHEDULES)
    assert rounds == []


def test_the_group_screen_solves_on_clarabel_for_the_workspace(monkeypatch):
    seen = []
    real = slopefree.solve
    monkeypatch.setattr(slopefree, "solve", lambda c, G, lam, R, solver=None: seen.append(solver) or real(c, G, lam, R, solver=solver))
    D = ExactData(*_bed())
    slopefree.group_certificate(D.c, D.G, slopefree.dyadic_below(D.yy / 3), schedules=((1, 1, 1),), solver="workspace")
    assert seen == ["clarabel"]


@pytest.mark.parametrize("p, expected", [(slopefree.WORKSPACE_FROM - 1, "clarabel"), (slopefree.WORKSPACE_FROM, "workspace")])
def test_the_default_solver_is_the_workspace_from_its_threshold(p, expected):
    assert slopefree.default_solver(p) == expected


def test_certify_reads_the_default_solver(monkeypatch):
    seen = []
    real = slopefree._branch
    monkeypatch.setattr(slopefree, "_branch", lambda *a, **k: seen.append(a[10]["solver"]) or real(*a, **k))
    monkeypatch.setattr(slopefree, "WORKSPACE_FROM", 10)
    y, X = _bed()
    certify_slope_free(y, X, restarts=5)
    assert seen and set(seen) == {"workspace"}


# ---------------------------------------------------------------- outcomes


def test_the_cap_is_an_outcome_and_reported(monkeypatch):
    monkeypatch.setattr(workspace, "STAGES", ((dict(alpha=1.8), 1),))
    monkeypatch.setattr(workspace, "CHUNK", 2)
    D, lam = _pinned()
    events = []
    with reporting(progress=events.append):
        sol = Workspaces().solve(D.c, D.G, lam, _box_rows(D, 3))
    assert sol.status == "cap"
    assert [e.fields["status"] for e in events if e.kind == "solve"] == ["cap"]
    A = objective(D.c, D.G, lam)
    assert bound(A, _box_rows(D, 3), sol.y_eq, sol.y_in, 3) >= _best(A, 6, 3, [], D.c)


def test_a_capped_search_still_certifies_only_what_holds(monkeypatch):
    monkeypatch.setattr(workspace, "STAGES", ((dict(alpha=1.8), 1),))
    monkeypatch.setattr(workspace, "CHUNK", 5)
    y, X = _bed()
    a = certify_slope_free(y, X, restarts=5, solver="workspace")
    b = certify_slope_free(y, X, restarts=5, solver="clarabel")
    assert not a.certified or a.incumbent == b.incumbent


def test_an_error_after_a_solve_clears_the_iterate(monkeypatch):
    D, lam = _pinned()
    ws = Workspaces()
    ws.solve(D.c, D.G, lam, _box_rows(D, 3))
    assert all(w.last is not None for w in ws.spaces.values())

    def boom(self, *a, **k):
        raise RuntimeError("scs failed")
    monkeypatch.setattr(scs.SCS, "solve", boom)
    ws.solve(D.c, D.G, lam, _box_rows(D, 3, [(0, 1)]))
    assert all(w.last is None for w in ws.spaces.values())


def test_an_scs_error_is_reported_and_clarabel_solves(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("scs failed")
    monkeypatch.setattr(scs.SCS, "solve", boom)
    D, lam = _pinned()
    events = []
    ws = Workspaces()
    with reporting(progress=events.append):
        sol = ws.solve(D.c, D.G, lam, _box_rows(D, 3))
    statuses = [(e.fields["solver"], e.fields["status"]) for e in events if e.kind == "solve"]
    assert statuses[0] == ("workspace", "error") and statuses[-1][0] == "clarabel"
    assert sol.solver == "clarabel"
    assert all(w.last is None for w in ws.spaces.values())   # no iterate of a failed solve is carried


# ---------------------------------------------------------------- edge cases


def test_one_candidate_control():
    D, lam = _pinned(p=1)
    R = _box_rows(D, 1)
    sol = Workspaces().solve(D.c, D.G, lam, R)
    A = objective(D.c, D.G, lam)
    b = sol.bound if sol.bound is not None else bound(A, R, sol.y_eq, sol.y_in, 1)
    assert b >= _best(A, 1, 1, [], D.c)


def test_a_node_whose_designs_are_all_cut():
    """Size 1 of two geos with geo 0 fixed in, and that design cut: no design is left."""
    D, lam = _pinned(p=2)
    R = Rows(D.c, 1, cuts=[{0}], fixes=[(0, 1)], face=True, box=True)
    sol = Workspaces().solve(D.c, D.G, lam, R)
    assert sol.status in workspace.OUTCOMES


def test_every_geo_fixed_but_one():
    D, lam = _pinned()
    fixes = [(0, 1), (1, 0), (2, 1), (3, 0), (4, 0)]
    R = _box_rows(D, 3, fixes)
    sol = Workspaces().solve(D.c, D.G, lam, R)
    A = objective(D.c, D.G, lam)
    b = sol.bound if sol.bound is not None else bound(A, R, sol.y_eq, sol.y_in, 3)
    assert b >= _best(A, 6, 3, fixes, D.c)


def test_collinear_controls():
    y, X = panel(4, p=5)
    X = np.column_stack([X, X[:, 0]])          # a duplicated control
    D = ExactData(y, X)
    lam = slopefree.dyadic_below(D.yy / 2)
    R = Rows(D.c, 2, face=True, box=True)
    sol = Workspaces().solve(D.c, D.G, lam, R)
    A = objective(D.c, D.G, lam)
    b = sol.bound if sol.bound is not None else bound(A, R, sol.y_eq, sol.y_in, 2)
    assert b >= _best(A, 6, 2, [], D.c)


# ---------------------------------------------------------------- failure


def test_rows_without_box_rows_are_refused():
    D, lam = _pinned()
    with pytest.raises(MlsynthConfigError, match="box"):
        Workspaces().solve(D.c, D.G, lam, Rows(D.c, 3, face=True))


def test_shor_rows_are_refused():
    D, lam = _pinned()
    with pytest.raises(MlsynthConfigError, match="full"):
        Workspaces().solve(D.c, D.G, lam, Rows(D.c, 3, level="shor", box=True))


def test_a_misspelled_solver_is_refused():
    y, X = _bed()
    with pytest.raises(MlsynthConfigError, match="solver"):
        certify_slope_free(y, X, restarts=5, solver="Workspace")


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(2, 7), data=st.data())
def test_property_the_workspace_bound_holds_at_every_node(seed, p, data):
    """Whatever SCS returns, the exact certificate at its multipliers is at least
    every design's value within the node's fixings."""
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    m = data.draw(st.integers(1, p))
    lam = slopefree.dyadic_below(D.yy * Q(data.draw(st.integers(1, 9)), 10))
    geos = data.draw(st.lists(st.integers(0, p - 1), unique=True, max_size=p - 1))
    fixes = [(j, data.draw(st.integers(0, 1))) for j in geos]
    if sum(v for _, v in fixes) > m or len(fixes) - sum(v for _, v in fixes) > p - m:
        return
    A = objective(D.c, D.G, lam)
    R = _box_rows(D, m, fixes)
    sol = Workspaces().solve(D.c, D.G, lam, R)
    b = sol.bound if sol.bound is not None else bound(A, R, sol.y_eq, sol.y_in, m)
    best = _best(A, p, m, fixes, D.c)
    assert b is None or best is None or b >= best


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(3, 7))
def test_property_the_workspace_search_certifies_the_clarabel_design(seed, p):
    y, X = panel(seed, p=p)
    try:
        a = certify_slope_free(y, X, restarts=3, solver="workspace")
    except slopefree.NoDesignLeft:
        return
    b = certify_slope_free(y, X, restarts=3, solver="clarabel")
    assert a.certified == b.certified
    if a.certified:
        assert a.incumbent == b.incumbent


def test_solve_refuses_the_workspace():
    """A node solved through solve() would build and drop a workspace each time."""
    D, lam = _pinned()
    with pytest.raises(MlsynthConfigError, match="Workspaces"):
        slopefree.solve(D.c, D.G, lam, Rows(D.c, 3, box=True), solver="workspace")


def test_a_round_after_an_unsolved_one_starts_cold():
    """After an SCS error the last round's workspace has no iterate (Workspaces.solve
    clears it): the next round's workspace starts from zero, not from a stale or
    missing point."""
    D, lam = _pinned()
    prev = workspace.SizeWorkspace(D.c, D.G, lam, 3, [], ())
    nxt = workspace.SizeWorkspace(D.c, D.G, lam, 3, [], ((1, (0, 1, 2)),))
    nxt.carry_from(prev)
    assert nxt.last is None
    prev.solve(_box_rows(D, 3))
    nxt.carry_from(prev)
    x, y, s = nxt.last
    assert y.size == s.size == nxt.Ascs.shape[0] and x.size == nxt.Ascs.shape[1]
    assert y[nxt.n_eq + prev.n_in0] == 0 and s[nxt.n_eq + prev.n_in0] >= 0
