"""Checks of the #847 runner: each command writes one row naming its instance and
solver, the solvers are dispatched as named, unknown ones are refused, and the
BiqCrunch rows need the program."""
import json

import numpy as np
import pytest

from . import run


@pytest.fixture
def rows(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "RESULTS", tmp_path)

    def read(kind):
        return [json.loads(l) for l in (tmp_path / f"{kind}.jsonl").read_text().splitlines()]

    return read


def test_each_syndes_solver_writes_its_row(monkeypatch, rows):
    calls = []
    monkeypatch.setattr(run, "ours", lambda n, N, K: calls.append(("ours", n, N, K)) or {"solver": "ours"})
    monkeypatch.setattr(run, "scip", lambda n, N, K, t: calls.append(("scip", t)) or {"solver": "scip"})
    monkeypatch.setattr(run, "biqcrunch",
                        lambda n, N, K, s, t: calls.append((s, t)) or {"solver": s})
    for solver in run.SOLVERS:
        assert run.main(["syndes", "noise", "40", "8", solver, "60"]) == 0
    got = rows("syndes")
    assert [r["solver"] for r in got] == list(run.SOLVERS)
    assert all((r["panel"], r["N"], r["K"], r["time_limit"]) == ("noise", 40, 8, 60.0)
               for r in got)
    assert calls[0] == ("ours", "noise", 40, 8) and ("scip", 60.0) in calls
    assert ("seeded", 60.0) in calls and ("root", 60.0) in calls


def test_each_marex_backend_writes_its_row(monkeypatch, rows):
    monkeypatch.setattr(run, "marex", lambda n, m, b, t: {"backend": b, "limit": t})
    for backend in run.BACKENDS:
        assert run.main(["marex", "walmart", "6", backend]) == 0
    got = rows("marex")
    assert [(r["backend"], r["panel"], r["m"], r["limit"]) for r in got] == [
        ("exact", "walmart", 6, 1800.0), ("mip", "walmart", 6, 1800.0)]


@pytest.mark.parametrize("argv", [["syndes", "noise", "40", "8", "gurobi"],
                                  ["marex", "walmart", "6", "cplex"],
                                  ["mip", "walmart"]])
def test_an_unknown_choice_is_refused(argv, rows):
    with pytest.raises(ValueError):
        run.main(argv)


def test_biqcrunch_rows_need_the_program(monkeypatch):
    monkeypatch.delenv("BIQCRUNCH", raising=False)
    with pytest.raises(SystemExit, match="BIQCRUNCH"):
        run.biqcrunch("noise", 12, 4, "biqcrunch", 60)


def test_the_core_is_sigma_r_sigma():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    R = a + a.T
    sigma = -np.ones(6)
    sigma[[1, 4]] = 1.0
    assert run._core(R, [1, 4]) == pytest.approx(sigma @ R @ sigma)
