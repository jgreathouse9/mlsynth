"""The test beds every change to the TBR search is measured on first (#965).

A bed is a fixed list of treatment groups on simulated panels whose certificates
branch when every node is bounded. ``run_bed`` certifies each group, counts nodes and
solves, and, where the controls are few enough to enumerate, checks the design
against the enumerated optimum."""
import json

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import beds, slopefree
from .beds import BEDS, ENUMERATION_LIMIT, BedRun, compare, counting, every_node_bounded, run_bed, run_group
from .brute import positive_optimum

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------- smoke


def test_smoke_two_groups_of_the_smallest_bed():
    out = run_bed("s15_k3", groups=[0, 1])
    assert isinstance(out, BedRun) and len(out.runs) == 2
    for r in out.runs:
        assert r.certified and r.optimum is True and r.nodes > 0 and r.full >= 0 and r.seconds >= 0
    t = out.totals()
    assert t["groups"] == 2 and t["nodes"] == sum(r.nodes for r in out.runs)
    assert t["unsound"] == 0 and out.unsound() == []


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("name", sorted(BEDS))
def test_every_bed_group_names_its_treated_units(name):
    bed = BEDS[name]
    assert bed.groups, name
    for seed, T, periods in bed.groups:
        assert len(T) == bed.treated and len(set(T)) == len(T)
        assert all(0 <= j < bed.units for j in T) and list(T) == sorted(T)
        assert periods > 0 and isinstance(seed, int)


def test_the_beds_listed():
    assert set(BEDS) == {"s15_k3", "s20_k3", "s20_k4", "s30_k10", "s30_k4"}
    assert [len(BEDS[b].groups) for b in ("s15_k3", "s20_k3", "s20_k4", "s30_k10", "s30_k4")] == [62, 45, 21, 13, 25]


def test_a_group_panel_is_the_treated_sum_and_the_other_units():
    bed = BEDS["s15_k3"]
    seed, T, periods = bed.groups[0]
    y, X = bed.panel(0)
    from .brute import geo_panel
    M = geo_panel(seed, n=periods, p=bed.units)
    assert np.array_equal(y, M[:, list(T)].sum(axis=1))
    assert np.array_equal(X, M[:, [j for j in range(bed.units) if j not in T]])


def test_counting_counts_each_level_and_restores_the_solver(monkeypatch):
    real = slopefree.solve
    with counting() as n:
        y, X = BEDS["s15_k3"].panel(0)
        with every_node_bounded():
            slopefree.certify_slope_free(y, X, restarts=5, shor=True)
    assert slopefree.solve is real
    assert n["full"] > 0 and n["shor"] > 0 and set(n) == {"full", "shor"}


def test_counting_counts_the_workspaces_full_solves(monkeypatch):
    """Under the workspace a size's full relaxations go through Workspaces.solve, not
    slopefree.solve: a bed run would report none."""
    from .workspace import Workspaces
    real = Workspaces.solve
    with counting() as n:
        y, X = BEDS["s15_k3"].panel(0)
        with every_node_bounded():
            slopefree.certify_slope_free(y, X, restarts=5, solver="workspace")
    assert Workspaces.solve is real
    assert n["full"] > 0


def test_counting_restores_the_solver_after_an_error():
    real = slopefree.solve
    with pytest.raises(RuntimeError):
        with counting():
            raise RuntimeError("inside")
    assert slopefree.solve is real


def test_every_node_bounded_sets_and_restores_the_price_limit(monkeypatch):
    """A known limit going in, so a run earlier in the process that left 0 behind
    cannot pass for a restore."""
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 12345)
    with every_node_bounded():
        assert slopefree.PRICE_LIMIT == 0
    assert slopefree.PRICE_LIMIT == 12345
    with pytest.raises(RuntimeError):
        with every_node_bounded():
            raise RuntimeError("inside")
    assert slopefree.PRICE_LIMIT == 12345


def test_a_run_counts_what_the_search_did():
    r = run_group("s15_k3", 0)
    y, X = BEDS["s15_k3"].panel(0)
    with counting() as n, every_node_bounded():
        out = slopefree.certify_slope_free(y, X, restarts=beds.BED_RESTARTS)
    assert (r.nodes, r.full, r.shor, r.rounded, r.open) == (out.nodes, n["full"], n["shor"], out.rounded, len(out.open))
    assert r.controls == sorted(out.controls) and r.certified == out.certified


def test_the_search_settings_reach_the_search():
    off = run_group("s15_k3", 0, rounding=False)
    assert off.rounded == 0 and run_group("s15_k3", 0).rounded > 0


def test_a_design_that_misses_the_optimum_is_reported(monkeypatch):
    """A certified design other than the enumerated optimum is unsound, and the bed
    run lists it."""
    real = slopefree.certify_slope_free

    def wrong(y, X, **k):
        out = real(y, X, **k)
        best, C = positive_optimum(y, X)
        other = next(j for j in range(X.shape[1]) if j not in C)
        out.controls = sorted(C | {other})
        return out

    monkeypatch.setattr(slopefree, "certify_slope_free", wrong)
    out = run_bed("s15_k3", groups=[0])
    assert out.runs[0].certified and out.runs[0].optimum is False
    assert out.unsound() == [out.runs[0]] and out.totals()["unsound"] == 1


def test_an_open_group_that_missed_the_optimum_is_not_unsound(monkeypatch):
    real = slopefree.certify_slope_free

    def open_wrong(y, X, **k):
        out = real(y, X, **k)
        best, C = positive_optimum(y, X)
        out.controls = sorted(C | {next(j for j in range(X.shape[1]) if j not in C)})
        out.certified = False
        return out

    monkeypatch.setattr(slopefree, "certify_slope_free", open_wrong)
    out = run_bed("s15_k3", groups=[0])
    assert out.runs[0].optimum is False and out.unsound() == []


def test_compare_reports_each_total_against_the_baseline():
    out = run_bed("s15_k3", groups=[0])
    base = {"s15_k3": {"groups": [dict(seed=r.seed, treated=list(r.treated), periods=r.periods, nodes=r.nodes + 3,
                                       full=r.full - 1, shor=r.shor, open=r.open, certified=r.certified)
                                  for r in out.runs]}}
    d = compare(out, base)
    assert d["nodes"] == (out.totals()["nodes"] + 3, out.totals()["nodes"], -3)
    assert d["full"] == (out.totals()["full"] - 1, out.totals()["full"], 1)
    assert d["shor"][2] == 0


def test_the_committed_baseline_covers_every_group_of_every_bed():
    base = json.loads(beds.BASELINE.read_text())
    assert set(base) - {"recorded_on"} == set(BEDS)
    for name, bed in BEDS.items():
        rows = base[name]["groups"]
        assert [(r["seed"], tuple(r["treated"]), r["periods"]) for r in rows] == list(bed.groups)
        assert all(r["certified"] or r["open"] > 0 for r in rows)
        if bed.units - bed.treated <= ENUMERATION_LIMIT:     # every certified design is the optimum
            assert all(r["optimum"] is True for r in rows if r["certified"])
        else:
            assert all(r["optimum"] is None for r in rows)


def test_workers_give_the_same_runs():
    one = run_bed("s15_k3", groups=[0, 1])
    two = run_bed("s15_k3", groups=[0, 1], workers=2)
    key = lambda r: (r.seed, r.treated, r.nodes, r.full, r.shor, r.controls, r.certified)  # noqa: E731
    assert [key(r) for r in one.runs] == [key(r) for r in two.runs]


def test_the_cli_prints_the_bed_and_its_change(capsys):
    assert beds.main(["--bed", "s15_k3", "--groups", "0", "1"]) == 0
    out = capsys.readouterr().out
    assert "s15_k3" in out and "nodes" in out and "baseline" in out


def test_the_cli_takes_a_solver(monkeypatch, capsys):
    seen = []
    real = slopefree.certify_slope_free

    def spy(y, X, **k):
        seen.append(k.get("solver"))
        return real(y, X, **k)

    monkeypatch.setattr(slopefree, "certify_slope_free", spy)
    assert beds.main(["--bed", "s15_k3", "--groups", "0", "--solver", "scs"]) == 0
    assert seen == ["scs"] and " full " in capsys.readouterr().out


def test_the_cli_refuses_an_unknown_solver(capsys):
    with pytest.raises(SystemExit):
        beds.main(["--bed", "s15_k3", "--groups", "0", "--solver", "gurobi"])


@pytest.fixture
def tiny(monkeypatch):
    """A two-group bed, so a whole-bed run is cheap."""
    monkeypatch.setitem(beds.BEDS, "tiny", beds.Bed("tiny", 15, 3, BEDS["s15_k3"].groups[:2]))
    return "tiny"


def test_a_whole_bed_runs_every_group_in_order(tiny):
    out = run_bed(tiny)
    assert [(r.seed, r.treated, r.periods) for r in out.runs] == list(BEDS[tiny].groups)


def test_record_writes_every_run_as_the_baseline(tiny, tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(beds, "BASELINE", path)
    monkeypatch.setattr(beds, "_head", lambda: "abc1234")
    assert beds.main(["--bed", tiny, "--record"]) == 0
    base = json.loads(path.read_text())
    assert base["recorded_on"] == "abc1234" and set(base) == {"recorded_on", tiny}
    assert [(r["seed"], tuple(r["treated"]), r["periods"]) for r in base[tiny]["groups"]] == list(BEDS[tiny].groups)
    assert base[tiny]["totals"]["groups"] == 2
    again = run_bed(tiny)
    assert compare(again, base)["nodes"][2] == 0              # a recorded run reproduces


def test_the_cli_fails_on_an_unsound_group(monkeypatch, capsys):
    real = slopefree.certify_slope_free

    def wrong(y, X, **k):
        out = real(y, X, **k)
        best, C = positive_optimum(y, X)
        out.controls = sorted(C | {next(j for j in range(X.shape[1]) if j not in C)})
        return out

    monkeypatch.setattr(slopefree, "certify_slope_free", wrong)
    assert beds.main(["--bed", "s15_k3", "--groups", "0"]) == 1
    assert "UNSOUND" in capsys.readouterr().out


# ---------------------------------------------------------------- properties


@SETTINGS
@given(data=st.data())
def test_property_every_group_panel_has_the_bed_shape(data):
    name = data.draw(st.sampled_from(sorted(BEDS)))
    bed = BEDS[name]
    i = data.draw(st.integers(0, len(bed.groups) - 1))
    seed, T, periods = bed.groups[i]
    y, X = bed.panel(i)
    assert y.shape == (periods,) and X.shape == (periods, bed.units - bed.treated)
    y2, X2 = bed.panel(i)
    assert np.array_equal(y, y2) and np.array_equal(X, X2)


# ---------------------------------------------------------------- edges


def test_no_groups_is_an_empty_run():
    out = run_bed("s15_k3", groups=[])
    assert out.runs == [] and out.totals() == dict(groups=0, nodes=0, full=0, shor=0, open=0, uncertified=0,
                                                    rounded=0, optimum_missed=0, unsound=0, seconds=0.0)


def test_at_the_enumeration_limit_the_optimum_is_checked():
    bed = BEDS["s20_k3"]
    assert bed.units - bed.treated == ENUMERATION_LIMIT
    assert run_group("s20_k3", 0).optimum is not None


def test_the_beds_run_at_the_recorded_restarts(monkeypatch):
    real = slopefree.certify_slope_free
    seen = []

    def spy(y, X, **k):
        seen.append(k.get("restarts"))
        return real(y, X, **k)

    monkeypatch.setattr(slopefree, "certify_slope_free", spy)
    run_group("s15_k3", 0)
    run_group("s15_k3", 0, restarts=3)
    assert seen == [beds.BED_RESTARTS, 3] and beds.BED_RESTARTS == 5


def test_above_the_enumeration_limit_the_optimum_is_not_checked():
    bed = BEDS["s30_k10"]
    assert bed.units - bed.treated > ENUMERATION_LIMIT
    assert run_group("s30_k10", 4).optimum is None


# ---------------------------------------------------------------- failures


def test_an_unknown_bed_is_refused():
    with pytest.raises(MlsynthConfigError, match="bed"):
        run_bed("s99_k1")


@pytest.mark.parametrize("groups", [[62], [-1], ["0"], [True], "0"])
def test_a_group_outside_the_bed_is_refused(groups):
    with pytest.raises(MlsynthConfigError, match="group"):
        run_bed("s15_k3", groups=groups)


@pytest.mark.parametrize("workers", [0, -1, 1.5, "2", True])
def test_a_malformed_worker_count_is_refused(workers):
    with pytest.raises(MlsynthConfigError, match="workers"):
        run_bed("s15_k3", groups=[0], workers=workers)
