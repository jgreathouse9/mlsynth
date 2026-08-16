"""One broken case does not take the suite down with it.

``run_one`` caught only ``BenchmarkSkipped``, so any other exception a case
raised propagated through the list comprehension in ``main`` and ended the
process. The daily run of 2026-08-13 shows what that costs: shard 3 of 4 died on
``proximal_panic1907``, 45 of its 46 cases never ran, ``REPORT.md`` was never
written so the upload step found no artifact, and the failing case's name
appeared nowhere in the log -- ``run_one`` prints its ``[PASS]``/``[FAIL]`` line
only after ``mod.run()`` returns, so a case that raises is anonymous. The case
had to be identified by reconstructing shard membership.

The suite is a survey instrument. A case that raises is a result about that
case, and the other 188 answers are still worth collecting and still worth
reporting. So the error is caught, attributed to its case, and counted against
the exit code -- which is reporting, not swallowing.
"""
from __future__ import annotations

import sys
import types

import pytest

from benchmarks import run_benchmarks
from benchmarks.compare import BenchmarkSkipped


def _case(name, run):
    """Register a synthetic case module under ``name`` and return the name."""
    mod = types.ModuleType(name)
    mod.run = run
    mod.EXPECTED = {"x": (1.0, 0.1)}
    mod.__doc__ = f"synthetic case {name}"
    sys.modules[f"benchmarks.cases.{name}"] = mod
    return name


@pytest.fixture
def registered(monkeypatch):
    """Three cases: one passes, one raises, one passes after the raiser."""
    from benchmarks import registry

    names = {
        "ok_before": _case("ok_before", lambda: {"x": 1.0}),
        "boom": _case("boom", lambda: (_ for _ in ()).throw(
            RuntimeError("the treatment bridge did not solve its moment"))),
        "ok_after": _case("ok_after", lambda: {"x": 1.0}),
    }
    monkeypatch.setattr(
        registry, "CASES",
        {n: f"benchmarks.cases.{n}" for n in names}, raising=False)
    yield list(names)
    for n in names:
        sys.modules.pop(f"benchmarks.cases.{n}", None)


class TestARaisingCaseIsContained:
    def test_it_is_reported_as_error_not_propagated(self, registered):
        got = run_benchmarks.run_one("boom")
        assert got["status"] == "error"

    def test_the_case_is_named_in_its_own_result(self, registered):
        assert run_benchmarks.run_one("boom")["name"] == "boom"

    def test_the_exception_message_is_kept(self, registered):
        got = run_benchmarks.run_one("boom")
        assert "did not solve its moment" in got["reason"]

    def test_the_exception_type_is_kept(self, registered):
        assert "RuntimeError" in run_benchmarks.run_one("boom")["reason"]

    def test_it_prints_the_case_name(self, registered, capsys):
        run_benchmarks.run_one("boom")
        assert "boom" in capsys.readouterr().out

    def test_cases_after_it_still_run(self, registered):
        results = [run_benchmarks.run_one(n) for n in registered]
        assert [r["status"] for r in results] == ["pass", "error", "pass"]


class TestTheExitCodeStillFails:
    def test_an_error_is_not_a_pass(self, registered, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--all"])
        assert run_benchmarks.main() != 0

    def test_a_clean_suite_still_returns_zero(self, monkeypatch):
        from benchmarks import registry
        name = _case("only_ok", lambda: {"x": 1.0})
        monkeypatch.setattr(registry, "CASES",
                            {name: f"benchmarks.cases.{name}"}, raising=False)
        monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--all"])
        try:
            assert run_benchmarks.main() == 0
        finally:
            sys.modules.pop(f"benchmarks.cases.{name}", None)


class TestSkipIsStillDistinctFromError:
    def test_a_skip_is_not_an_error(self, monkeypatch):
        from benchmarks import registry
        name = _case("skipper", lambda: (_ for _ in ()).throw(
            BenchmarkSkipped("reference clone unavailable")))
        monkeypatch.setattr(registry, "CASES",
                            {name: f"benchmarks.cases.{name}"}, raising=False)
        try:
            assert run_benchmarks.run_one(name)["status"] == "skip"
        finally:
            sys.modules.pop(f"benchmarks.cases.{name}", None)

    def test_a_skip_does_not_fail_the_suite(self, monkeypatch):
        from benchmarks import registry
        name = _case("skipper2", lambda: (_ for _ in ()).throw(
            BenchmarkSkipped("reference clone unavailable")))
        monkeypatch.setattr(registry, "CASES",
                            {name: f"benchmarks.cases.{name}"}, raising=False)
        monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--all"])
        try:
            assert run_benchmarks.main() == 0
        finally:
            sys.modules.pop(f"benchmarks.cases.{name}", None)


class TestAnUnimportableCaseIsAlsoContained:
    def test_a_case_that_will_not_import_is_an_error(self, monkeypatch):
        from benchmarks import registry
        monkeypatch.setattr(
            registry, "CASES",
            {"no_such_case": "benchmarks.cases.no_such_case_at_all"},
            raising=False)
        got = run_benchmarks.run_one("no_such_case")
        assert got["status"] == "error"


class TestTheReportSurvivesAnError:
    """The report is the artifact CI uploads, so it has to be written even when
    a case raised -- that is the whole point of containing the error. An
    ``error`` row must be legible in it and counted in its totals."""

    def _write(self, tmp_path, results):
        run_benchmarks.write_report(results, tmp_path)
        return (tmp_path / "REPORT.md").read_text()

    def _results(self):
        return [{"name": "ok", "status": "pass", "seconds": 1.0, "metrics": []},
                {"name": "boom", "status": "error", "seconds": 0.1,
                 "reason": "RuntimeError: bridge did not solve", "metrics": []}]

    def test_the_report_file_is_written(self, tmp_path):
        self._write(tmp_path, self._results())
        assert (tmp_path / "REPORT.md").exists()

    def test_the_json_is_written(self, tmp_path):
        self._write(tmp_path, self._results())
        assert (tmp_path / "report.json").exists()

    def test_the_errored_case_is_named_in_the_table(self, tmp_path):
        assert "`boom`" in self._write(tmp_path, self._results())

    def test_the_error_status_is_shown(self, tmp_path):
        assert "ERROR" in self._write(tmp_path, self._results())

    def test_errors_are_counted_in_the_totals(self, tmp_path):
        assert "1 errored" in self._write(tmp_path, self._results())
