"""A workspace round ends early when its iterate already shows a violated triangle (#965).

size_certificate's full tier solves a size's relaxation, separates the most violated
triangle inequalities at its point and solves again, up to ``rounds`` times. On the
100-geo groups most rounds ran SCS to its cap although the point had shown, chunks
earlier, the triangles the next round would add. So after SEPARATE_AFTER chunks a
round whose point violates a triangle by more than SEPARATE_TOL ends with the outcome
"separate", and the next round adds the triangles and starts warm. The last round has
no next one, so it runs to its end; the certificate is computed at every chunk and
decides first, so a round that closes still reports "certified"."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
import scs
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree, workspace
from .brute import geo_panel, panel
from .certificate import ExactData
from .progress import reporting
from .slopefree import Rows, Solve, bound, certify_slope_free, dyadic_below, objective, size_certificate
from .test_workspace import _best
from .treatment import _split
from .workspace import Workspaces

SETTINGS = settings(max_examples=15, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])

#: A violated triangle, whatever the point.
VIOLATED = [(0.5, 1, (0, 1, 2))]


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _open(seed=2, p=6, m=3):
    """A size no relaxation closes (lam far below every design's ratio) and its rows."""
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    return D, Q(1, 1 << 20), Rows(D.c, m, face=True, box=True), m


def _closed(seed=2, p=6, m=3):
    """A size the first chunk closes (lam far above every design's ratio)."""
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    return D, dyadic_below(D.yy * 4), Rows(D.c, m, face=True, box=True), m


def _count_chunks(monkeypatch):
    n = []
    real = scs.SCS.solve

    def solve(self, *a, **k):
        n.append(1)
        return real(self, *a, **k)
    monkeypatch.setattr(scs.SCS, "solve", solve)
    return n


def _slow(monkeypatch, stages=((dict(alpha=1.8), 10),), chunk=5):
    """Small chunks and an unreachable tolerance, so SCS never converges."""
    monkeypatch.setattr(workspace, "STAGES", stages)
    monkeypatch.setattr(workspace, "CHUNK", chunk)
    monkeypatch.setattr(workspace, "EPS", 1e-15)


# ---------------------------------------------------------------- smoke


def test_smoke_a_round_ends_on_a_violated_triangle(monkeypatch):
    _slow(monkeypatch)
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    D, lam, R, m = _open()
    sol = Workspaces().solve(D.c, D.G, lam, R, early=True)
    assert sol.status == "separate" and "separate" in workspace.OUTCOMES
    assert sol.Z is not None and sol.Z.shape == (R.n, R.n)
    A = objective(D.c, D.G, lam)
    assert bound(A, R, sol.y_eq, sol.y_in, m) >= _best(A, len(D.c), m, [], D.c)    # still a valid bound


# ---------------------------------------------------------------- unit


def test_without_early_a_round_never_separates(monkeypatch):
    _slow(monkeypatch)
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    D, lam, R, m = _open()
    assert Workspaces().solve(D.c, D.G, lam, R).status == "cap"
    assert Workspaces().solve(D.c, D.G, lam, R, early=False).status == "cap"


def test_the_check_waits_for_separate_after_chunks(monkeypatch):
    _slow(monkeypatch)
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 3)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    chunks = _count_chunks(monkeypatch)
    D, lam, R, m = _open()
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status == "separate"
    assert len(chunks) == 3


def test_the_chunks_count_across_stages(monkeypatch):
    _slow(monkeypatch, stages=((dict(alpha=1.8, scale=0.01), 2), (dict(alpha=1.8), 10)))
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 3)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    chunks = _count_chunks(monkeypatch)
    D, lam, R, m = _open()
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status == "separate"
    assert len(chunks) == 3                       # the second stage's first chunk


def test_the_check_reads_the_tolerance(monkeypatch):
    _slow(monkeypatch)
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    seen = []
    real = slopefree.separate_triangles

    def separate(u, U, k=400, tol=1e-6):
        seen.append((k, tol))
        return real(u, U, k=k, tol=tol)
    monkeypatch.setattr(slopefree, "separate_triangles", separate)
    monkeypatch.setattr(workspace, "SEPARATE_TOL", 1e9)          # nothing is violated by this much
    D, lam, R, m = _open()
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status == "cap"
    assert seen and all(t == 1e9 for _, t in seen) and all(k == 1 for k, _ in seen)


def test_the_certificate_decides_before_the_check(monkeypatch):
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    D, lam, R, m = _closed()
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status == "certified"


def test_a_converged_round_is_not_cut(monkeypatch):
    monkeypatch.setattr(workspace, "STAGES", ((dict(alpha=1.8), 50),))
    monkeypatch.setattr(workspace, "EPS", 1e-2)                 # converges in the first chunk
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    D, lam, R, m = _open()
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status == "converged"


def test_the_outcome_is_reported(monkeypatch):
    _slow(monkeypatch)
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    D, lam, R, m = _open()
    events = []
    with reporting(progress=events.append):
        Workspaces().solve(D.c, D.G, lam, R, early=True)
    assert [e.fields["status"] for e in events if e.kind == "solve"] == ["separate"]


def _fake_round(monkeypatch, n):
    """Every full solve returns a point with a violated triangle and a bound above
    zero, so every round goes on; records each call's ``early``."""
    flags = []
    Z = np.full((n, n), 0.0); Z[0, 0] = 1.0; Z[0, 1:] = Z[1:, 0] = 0.5; np.fill_diagonal(Z[1:, 1:], 0.5)

    def solve(self, c, G, lam, R, early=False):
        flags.append(early)
        return Solve(1.0, 1.0, np.zeros(R.n_eq), np.zeros(R.n_in), Z, "separate" if early else "cap",
                     solver="workspace", bound=Q(1))
    monkeypatch.setattr(Workspaces, "solve", solve)
    return flags


def test_early_is_asked_for_only_while_a_round_remains(monkeypatch):
    D, lam, R, m = _open()
    flags = _fake_round(monkeypatch, R.n)
    fresh = iter(itertools.combinations(range(len(D.c)), 3))
    monkeypatch.setattr(slopefree, "separate_triangles", lambda u, U, k=400, tol=1e-6: [(1.0, 1, next(fresh))])
    res = size_certificate(D.c, D.G, lam, m, rounds=3, per_round=5, solver="workspace", schedules=((5, 1, 5),))
    assert not res.certified and flags == [True, True, True, False]


def test_clarabel_rounds_are_unchanged(monkeypatch):
    """The early end is the workspace's: a Clarabel round is a full solve as before."""
    seen = []
    real = slopefree.solve

    def solve(c, G, lam, R, **k):
        seen.append(k)
        return real(c, G, lam, R, **k)
    monkeypatch.setattr(slopefree, "solve", solve)
    D, lam, R, m = _open()
    size_certificate(D.c, D.G, lam, m, rounds=1, solver="clarabel", schedules=((5, 1, 5),))
    assert seen and all("early" not in k for k in seen)


# ---------------------------------------------------------------- the search


def test_the_search_certifies_the_same_design_with_early_rounds(monkeypatch):
    y, X = _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]
    off = certify_slope_free(y, X, restarts=5, solver="workspace")
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    monkeypatch.setattr(workspace, "SEPARATE_TOL", 1e-6)
    on = certify_slope_free(y, X, restarts=5, solver="workspace")
    assert on.certified == off.certified and on.incumbent == off.incumbent


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(3, 7), m=st.integers(1, 4), f=st.sampled_from([Q(1, 2), Q(1), Q(2)]))
def test_property_a_size_certified_with_early_rounds_holds_no_better_design(seed, p, m, f):
    """Whatever ends a round, a certified size has no design above zero (enumerated)."""
    m = min(m, p)
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    lam = dyadic_below(D.yy * f / 4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        mp.setattr(workspace, "SEPARATE_AFTER", 1)
        mp.setattr(workspace, "SEPARATE_TOL", 1e-6)
        mp.setattr(workspace, "CHUNK", 20)
        res = size_certificate(D.c, D.G, lam, m, solver="workspace", schedules=((5, 1, 5),))
    if res.certified:
        A = objective(D.c, D.G, lam)
        best = _best(A, p, m, [], D.c)
        assert best is None or best <= 0


# ---------------------------------------------------------------- edges


@pytest.mark.parametrize("p", [1, 2])
def test_fewer_than_three_controls_have_no_triangle_to_separate(monkeypatch, p):
    _slow(monkeypatch)
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 1)
    D, lam, R, m = _open(p=p, m=1)
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status in ("cap", "converged", "certified")


def test_separate_after_beyond_the_chunks_never_separates(monkeypatch):
    _slow(monkeypatch, stages=((dict(alpha=1.8), 2),))
    monkeypatch.setattr(workspace, "SEPARATE_AFTER", 3)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda *a, **k: VIOLATED)
    D, lam, R, m = _open()
    assert Workspaces().solve(D.c, D.G, lam, R, early=True).status == "cap"


def test_the_defaults_are_the_measured_ones():
    """K = 5 chunks and a violation above 1e-2: on the two 100-geo groups 176.6 s to
    108.2 s and 167.7 s to 137.2 s, with the same designs (README)."""
    assert workspace.SEPARATE_AFTER == 5 and workspace.SEPARATE_TOL == 1e-2


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_early_must_be_a_bool(bad):
    D, lam, R, m = _open()
    with pytest.raises(MlsynthConfigError, match="early"):
        Workspaces().solve(D.c, D.G, lam, R, early=bad)
