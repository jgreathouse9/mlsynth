"""Progress reporting for the TBR certified search (#965).

The search emits events at four levels: 1, the treatment search, each group, each
gated round, the screen, the hull, each size and each new incumbent; 2, every node;
3, every first-order schedule, triangle round, relaxation solve and priced subtree;
4 adds Clarabel's own iteration log. ``verbose`` prints them to stderr, ``progress``
calls a function with each, and ``log`` writes them to a parquet dataset. Reporting
never changes what the search computes."""
import io

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import progress, slopefree
from .brute import geo_panel, panel, panel_ar
from .gated import certify_gated
from .guided import guided_treatment
from .progress import KINDS, Event, emit, read_log, reporting, tagged
from .slopefree import certify_slope_free
from .treatment import certify_treatment

SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    """Every node is bounded, so the small panels branch."""
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _answer(out):
    return (out.incumbent, out.controls, out.certified, out.nodes, out.rounded)


def _branching():
    """The first group of the s15_k3 bed (seed 1, treated 1, 11, 14): 12 candidate
    controls, 16 nodes with every node bounded, three Clarabel solves."""
    from .treatment import _split
    return _split(geo_panel(1, n=20, p=15), (1, 11, 14))[:2]


def _collect(level=3):
    seen = []
    return seen, dict(progress=seen.append, verbose=level)


# ---------------------------------------------------------------- smoke


def test_smoke_verbose_prints_the_sizes_and_nodes_to_stderr(capsys):
    y, X = panel(3, p=8)
    quiet = certify_slope_free(y, X, restarts=3)
    loud = certify_slope_free(y, X, restarts=3, verbose=2)
    cap = capsys.readouterr()
    assert _answer(loud) == _answer(quiet)
    assert cap.out == ""
    assert " size " in cap.err and " node " in cap.err and " slope-free-done " in cap.err


# ---------------------------------------------------------------- unit


def test_every_event_is_a_declared_kind_at_its_level():
    y, X = panel(3, p=8)
    seen, kw = _collect(3)
    certify_slope_free(y, X, restarts=3, **kw)
    assert seen and all(isinstance(e, Event) for e in seen)
    for e in seen:
        level, fields = KINDS[e.kind]
        assert e.level == level <= 3 and set(e.fields) <= set(fields)


def test_a_level_filters_the_deeper_events():
    y, X = panel(3, p=8)
    for level in (1, 2, 3):
        seen, kw = _collect(level)
        certify_slope_free(y, X, restarts=3, **kw)
        assert max(e.level for e in seen) == level


def test_one_node_event_per_node():
    y, X = panel(3, p=8)
    seen, kw = _collect(2)
    out = certify_slope_free(y, X, restarts=3, **kw)
    nodes = [e for e in seen if e.kind == "node"]
    assert out.nodes > 1 and len(nodes) == out.nodes
    assert {e.fields["outcome"] for e in nodes} <= {"leaf", "priced", "pruned", "open", "branched"}
    sizes = [e for e in seen if e.kind == "size"]
    assert sum(e.fields["nodes"] for e in sizes) == out.nodes
    assert [e.fields["m"] for e in sizes] == list(out.sizes)


def test_a_branched_node_names_its_control_and_its_children_follow(monkeypatch):
    y, X = _branching()
    seen, kw = _collect(2)
    certify_slope_free(y, X, restarts=5, **kw)
    branched = [e for e in seen if e.kind == "node" and e.fields["outcome"] == "branched"]
    assert branched and all(e.fields["control"] is not None for e in branched)
    assert all(e.fields["control"] is None for e in seen if e.kind == "node" and e.fields["outcome"] != "branched")


@pytest.mark.parametrize("face", [True, False])
def test_one_solve_event_per_relaxation_solved(monkeypatch, face):
    """On the face and off it: each is its own Clarabel call with its own event."""
    y, X = _branching()
    real = slopefree.solve
    calls = []

    def spy(c, G, lam, R, solver=None):
        calls.append(R.level)
        return real(c, G, lam, R, solver=solver)

    monkeypatch.setattr(slopefree, "solve", spy)
    seen, kw = _collect(3)
    certify_slope_free(y, X, restarts=5, face=face, **kw)
    solves = [e.fields["relaxation"] for e in seen if e.kind == "solve"]
    assert calls and solves == calls


def test_one_price_event_per_priced_subtree(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 10 ** 6)
    y, X = panel(3, p=8)
    seen, kw = _collect(3)
    out = certify_slope_free(y, X, restarts=3, **kw)
    prices = [e for e in seen if e.kind == "price"]
    assert out.priced_nodes > 0 and len(prices) == out.priced_nodes
    assert sum(e.fields["designs"] for e in prices) == out.priced
    assert sum(e.fields["outcome"] == "priced" for e in seen if e.kind == "node") == out.priced_nodes


def test_each_new_incumbent_is_reported_with_its_source():
    """On the first group of the s15_k3 bed (seed 1, treated 1, 11, 14) rounding finds
    two incumbents; the last incumbent reported is the result's design."""
    from .treatment import _split
    y, X, _ = _split(geo_panel(1, n=20, p=15), (1, 11, 14))
    seen, kw = _collect(1)
    out = certify_slope_free(y, X, restarts=5, **kw)
    inc = [e for e in seen if e.kind == "incumbent"]
    assert out.rounded > 0 and sum(e.fields["source"] == "rounding" for e in inc) == out.rounded
    assert inc[-1].fields["controls"] == out.controls
    assert all(e.fields["source"] in ("rounding", "leaf", "price") for e in inc)


def test_the_schedules_and_rounds_of_a_size_are_reported():
    y, X = panel(3, p=8)
    seen, kw = _collect(3)
    certify_slope_free(y, X, restarts=3, **kw)
    sched = [e for e in seen if e.kind == "schedule"]
    assert sched and all(e.fields["rung"] >= 0 and e.fields["seconds"] >= 0 for e in sched)


def test_the_print_level_shows_only_its_events(capsys):
    y, X = _branching()
    certify_slope_free(y, X, restarts=5, verbose=1)
    err = capsys.readouterr().err
    assert " size " in err and " node " not in err and " solve " not in err
    certify_slope_free(y, X, restarts=5, verbose=3)
    err = capsys.readouterr().err
    assert " node " in err and " solve " in err


@pytest.mark.parametrize("face", [True, False])
def test_clarabel_logs_only_at_four(monkeypatch, face):
    """On the face and off it: each solve passes its own verbose flag."""
    seen = []
    real = slopefree.cp.Problem.solve

    def spy(self, *a, **k):
        seen.append(k.get("verbose", False))
        return real(self, *a, **{**k, "verbose": False})

    monkeypatch.setattr(slopefree.cp.Problem, "solve", spy)
    y, X = _branching()
    certify_slope_free(y, X, restarts=5, verbose=3, face=face)
    assert seen and not any(seen)
    seen.clear()
    certify_slope_free(y, X, restarts=5, verbose=4, face=face)
    assert seen and all(seen)


def test_reporting_restores_the_previous_state_after_an_error():
    before = progress.current()
    with pytest.raises(RuntimeError):
        with reporting(verbose=2, stream=io.StringIO()):
            assert progress.wants(2)
            raise RuntimeError("inside")
    assert progress.current() is before and not progress.wants(1)


def test_an_inner_call_without_settings_keeps_the_outer_reporter():
    seen = []
    with reporting(progress=seen.append, verbose=1):
        with reporting():
            emit("hull", sizes=[1], seconds=0.0)
    assert [e.kind for e in seen] == ["hull"]


def test_a_tag_marks_the_events_inside_it_and_is_removed_after():
    seen = []
    with reporting(progress=seen.append, verbose=1):
        with tagged(treatment=(1, 2)):
            emit("hull", sizes=[1], seconds=0.0)
        emit("hull", sizes=[2], seconds=0.0)
    assert seen[0].tags == {"treatment": (1, 2)} and seen[1].tags == {}


def test_a_progress_function_alone_gets_every_event_to_three():
    seen = []
    y, X = panel(3, p=8)
    certify_slope_free(y, X, restarts=3, progress=seen.append)
    assert max(e.level for e in seen) == 3


def test_the_log_holds_every_event_the_function_saw(tmp_path):
    y, X = panel(3, p=8)
    seen = []
    certify_slope_free(y, X, restarts=3, progress=seen.append, verbose=3, log=tmp_path / "log")
    df = read_log(tmp_path / "log")
    assert len(df) == len(seen)
    assert list(df["kind"]) == [e.kind for e in seen]
    assert {"level", "kind", "t", "pid", "treatment", "m", "depth", "outcome", "tier", "seconds"} <= set(df.columns)
    nodes = df[df["kind"] == "node"]
    assert list(nodes["depth"]) == [e.fields["depth"] for e in seen if e.kind == "node"]


def test_a_log_flushes_in_parts_and_reads_back_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "FLUSH_EVERY", 5)
    y, X = panel(3, p=8)
    seen = []
    certify_slope_free(y, X, restarts=3, progress=seen.append, verbose=3, log=tmp_path / "log")
    parts = sorted((tmp_path / "log").glob("part-*.parquet"))
    assert len(parts) > 1 and len(read_log(tmp_path / "log")) == len(seen)


def test_each_sink_keeps_its_own_level(tmp_path, capsys):
    """Printing at 1 beside a log at 3: the log holds the nodes and solves, the
    printer none of them."""
    y, X = _branching()
    certify_slope_free(y, X, restarts=5, verbose=1, log=tmp_path / "log", log_level=3)
    err = capsys.readouterr().err
    kinds = set(read_log(tmp_path / "log")["kind"])
    assert {"node", "solve"} <= kinds
    assert " size " in err and " node " not in err and " solve " not in err


def test_a_long_run_flushes_on_time(tmp_path, monkeypatch):
    """A log flushes once FLUSH_SECONDS have passed since its last write, so a run
    that is one long group can be read while it runs."""
    monkeypatch.setattr(progress, "FLUSH_SECONDS", 0.0)
    y, X = panel(3, p=8)
    seen = []
    with reporting(progress=seen.append, verbose=1, log=tmp_path / "log", log_level=1):
        certify_slope_free(y, X, restarts=3)
        parts = len(list((tmp_path / "log").glob("part-*.parquet")))
    assert parts == len(seen) > 1


def test_the_flush_clock_restarts_at_each_write(tmp_path, monkeypatch):
    """On a clock that reads 0, 5, 11, 12, 15 and 22 with FLUSH_SECONDS = 10, the log
    writes at 11 and at 22, not at every event after the first ten seconds."""
    import time as _time

    class Clock:
        now = 0.0

        @staticmethod
        def perf_counter():
            return Clock.now

        time_ns = staticmethod(_time.time_ns)

    monkeypatch.setattr(progress, "time", Clock)
    monkeypatch.setattr(progress, "FLUSH_SECONDS", 10.0)
    log = progress.ParquetLog(tmp_path)
    parts = []
    for now in (0, 5, 11, 12, 15, 22):
        Clock.now = float(now)
        log(Event(1, "hull", float(now), 1, {}, {"sizes": [1], "seconds": 0.0}))
        parts.append(len(list(tmp_path.glob("part-*.parquet"))))
    assert parts == [0, 0, 1, 1, 1, 2]
    assert len(read_log(tmp_path)) == 6


def test_the_log_level_is_its_own():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    y, X = panel(3, p=8)
    certify_slope_free(y, X, restarts=3, log=d / "one", log_level=1)
    assert set(read_log(d / "one")["level"]) == {1}


def test_gated_rounds_are_reported():
    y, X = panel_ar(1)
    seen, kw = _collect(1)
    out = certify_gated(y, X, n_test=N_TEST, restarts=5, **kw)
    assert len([e for e in seen if e.kind == "gated-round"]) == len(out.rounds)
    assert sum(e.kind == "slope-free" for e in seen) >= len(out.rounds)


def test_the_treatment_search_reports_each_group_tagged():
    M = geo_panel(5, p=6)
    seen, kw = _collect(1)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, **kw)
    done = [e for e in seen if e.kind == "group-done"]
    assert len(done) == len(out.survivors) + out.retried
    assert all(tuple(e.tags["treatment"]) == tuple(e.fields["treatment"]) for e in done)
    assert seen[0].kind == "treatment" and seen[-1].kind == "treatment-done"
    inner = [e for e in seen if e.kind == "slope-free"]
    assert inner and all("treatment" in e.tags for e in inner)


def test_workers_write_their_own_parts(tmp_path):
    M = geo_panel(5, p=6)
    one = certify_treatment(M, 2, n_test=N_TEST, restarts=5, log=tmp_path / "one", log_level=1)
    two = certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=2, log=tmp_path / "two", log_level=1)
    a, b = read_log(tmp_path / "one"), read_log(tmp_path / "two")
    assert b["pid"].nunique() > 1
    assert (b["kind"] == "treatment").sum() == 1      # a worker does not write the parent's buffered rows
    key = lambda df: sorted(df[df["kind"] == "group-done"]["treatment"])  # noqa: E731
    assert key(a) == key(b) and (one.treatment, one.key) == (two.treatment, two.key)


def test_workers_print_to_the_same_stderr(capfd):
    M = geo_panel(5, p=6)
    certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=2, verbose=1)
    err = capfd.readouterr().err
    assert err.count(" group-done ") >= 2 and " treatment-done " in err


def test_the_guided_search_reports_each_group():
    M = geo_panel(5, p=6)
    seen, kw = _collect(1)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5, **kw)
    assert len([e for e in seen if e.kind == "group-done"]) == out.searched


def test_a_failed_solve_is_reported_and_raised(monkeypatch):
    from .certificate import ExactData
    y, X = _branching()
    D = ExactData(y, X)

    def fail(self, *a, **k):
        raise slopefree.cp.error.SolverError("injected")

    monkeypatch.setattr(slopefree.cp.Problem, "solve", fail)
    seen = []
    with reporting(progress=seen.append, verbose=3):
        with pytest.raises(slopefree.cp.error.SolverError):
            slopefree.solve(D.c, D.G, slopefree.Q(1), slopefree.Rows(D.c, 3, level="shor"))
    assert [(e.kind, e.fields["status"]) for e in seen] == [("solve", "error")]


def test_the_first_order_triangle_rounds_are_reported():
    from .certificate import ExactData
    y, X = _branching()
    D = ExactData(y, X)
    seen = []
    with reporting(progress=seen.append, verbose=3):
        res = slopefree.size_certificate(D.c, D.G, slopefree.Q(10) ** -9, 4, first_order_rounds=2,
                                         schedules=slopefree.SCREEN_SCHEDULES, relaxations=False)
    rounds = [e for e in seen if e.kind == "triangles"]
    assert not res.certified and rounds and [e.fields["round"] for e in rounds] == list(range(1, len(rounds) + 1))
    assert all(not e.fields["closed"] for e in rounds)


def test_an_incumbent_found_by_pricing_is_reported():
    from .certificate import ExactData
    y, X = _branching()
    D = ExactData(y, X)
    out = slopefree.SlopeFreeCertificate(incumbent=slopefree.Q(0), controls=[], ratio=slopefree.Q(0),
                                         sizes_after_hull=[])
    inc = {"lam": slopefree.Q(0), "C": set()}
    seen = []
    with reporting(progress=seen.append, verbose=3):
        slopefree.price_subtree(D.c, D.G, out, 3, [], inc, [])
    found = [e for e in seen if e.kind == "incumbent"]
    assert found and all(e.fields["source"] == "price" for e in found)
    assert found[-1].fields["controls"] == sorted(inc["C"])


def test_an_incumbent_at_a_leaf_is_reported():
    from .certificate import ExactData
    y, X = _branching()
    D = ExactData(y, X)
    out = slopefree.SlopeFreeCertificate(incumbent=slopefree.Q(0), controls=[], ratio=slopefree.Q(0),
                                         sizes_after_hull=[])
    inc = {"lam": slopefree.Q(0), "C": set()}
    seen = []
    with reporting(progress=seen.append, verbose=2):
        slopefree._branch(D.c, D.G, out, 2, [], [(0, 1), (1, 1)], 0, 400, 2, 12, inc, [])
    assert [e.kind for e in seen] == ["incumbent", "node"]
    assert seen[0].fields["source"] == "leaf" and seen[1].fields["outcome"] == "leaf"


def test_a_node_at_the_depth_limit_is_reported_open():
    y, X = _branching()
    seen, kw = _collect(2)
    out = certify_slope_free(y, X, restarts=5, max_depth=0, **kw)
    opened = [e for e in seen if e.kind == "node" and e.fields["outcome"] == "open"]
    assert len(opened) == len(out.open) > 0


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), level=st.integers(0, 3))
def test_property_reporting_never_changes_the_certificate(seed, level):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=7)
        quiet = certify_slope_free(y, X, restarts=3)
        seen = []
        loud = certify_slope_free(y, X, restarts=3, progress=seen.append if level else None, verbose=level)
        assert _answer(loud) == _answer(quiet)
        assert all(e.level <= level for e in seen)
        if level >= 2:
            assert sum(e.kind == "node" for e in seen) == loud.nodes
        if level == 0:
            assert seen == []
    finally:
        mp.undo()


@SETTINGS
@given(t=st.lists(st.integers(0, 50), min_size=1, max_size=4, unique=True))
def test_property_the_printer_writes_one_line_an_event(t):
    buf = io.StringIO()
    with reporting(verbose=3, stream=buf):
        with tagged(treatment=tuple(t)):
            emit("size", m=3, nodes=1, open=0, seconds=0.5)
            emit("node", m=3, depth=2, ones=1, outcome="pruned", tier="dual", control=None, seconds=0.1)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2 and all(",".join(map(str, t)) in ln for ln in lines)
    assert " size " in lines[0] and "m=3" in lines[0] and " node " in lines[1]
    assert lines[1].index(" node ") - lines[0].index(" size ") == 2 * 3   # level 2, depth 2: three steps in


# ---------------------------------------------------------------- edges


def test_without_settings_nothing_is_reported(capsys):
    y, X = panel(3, p=8)
    assert progress.current() is None
    certify_slope_free(y, X, restarts=3)
    assert capsys.readouterr().err == "" and progress.current() is None


def test_a_held_ceiling_reports_the_screen_and_no_size():
    from fractions import Fraction as Q
    from .certificate import ExactData
    y, X = panel(3, p=8)
    D = ExactData(y, X)
    seen, kw = _collect(1)
    out = certify_slope_free(y, X, restarts=3, ceiling=Q(D.yy) * Q(1, 10 ** 6), **kw)
    kinds = [e.kind for e in seen]
    if out.ceiling_held and not out.sizes:
        assert "screen" in kinds and "size" not in kinds and kinds[-1] == "slope-free-done"


def test_a_log_directory_is_made_when_missing(tmp_path):
    y, X = panel(3, p=8)
    certify_slope_free(y, X, restarts=3, log=tmp_path / "a" / "b", log_level=1)
    assert len(read_log(tmp_path / "a" / "b")) > 0


def test_an_empty_log_reads_as_an_empty_frame(tmp_path):
    (tmp_path / "empty").mkdir()
    df = read_log(tmp_path / "empty")
    assert len(df) == 0 and "kind" in df.columns


def test_a_log_without_events_writes_no_file(tmp_path):
    with reporting(log=tmp_path / "quiet"):
        pass
    assert list((tmp_path / "quiet").glob("*")) == [] and len(read_log(tmp_path / "quiet")) == 0


def test_a_tag_without_a_reporter_changes_nothing():
    with tagged(treatment=(1,)):
        emit("hull", sizes=[1], seconds=0.0)
    assert progress.current() is None


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("verbose", [-1, 5, 1.5, True, "2"])
def test_a_malformed_verbose_is_refused(verbose):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="verbose"):
        certify_slope_free(y, X, restarts=3, verbose=verbose)


def test_a_progress_that_is_not_a_function_is_refused():
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="progress"):
        certify_slope_free(y, X, restarts=3, progress="print")


@pytest.mark.parametrize("level", [0, 4, 1.0, True])
def test_a_malformed_log_level_is_refused(level, tmp_path):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="log_level"):
        certify_slope_free(y, X, restarts=3, log=tmp_path, log_level=level)


def test_a_log_path_that_is_a_file_is_refused(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="log"):
        certify_slope_free(y, X, restarts=3, log=f)


def test_an_undeclared_kind_or_field_is_refused():
    with reporting(progress=lambda e: None, verbose=3):
        with pytest.raises(ValueError, match="kind"):
            emit("nonsense", x=1)
        with pytest.raises(ValueError, match="field"):
            emit("hull", sizes=[1], seconds=0.0, typo=3)


def test_a_failing_progress_function_reaches_the_caller():
    def boom(e):
        raise RuntimeError("from the callback")

    y, X = panel(0, p=4)
    with pytest.raises(RuntimeError, match="from the callback"):
        certify_slope_free(y, X, restarts=3, progress=boom, verbose=1)
    assert progress.current() is None
