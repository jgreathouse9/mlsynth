"""Tests for the semantic mutation harness in ``tools/mutation``.

The harness is code that decides whether other tests are strong enough, so a
defect in it is worse than a defect in a test: it reports confidence that was
never measured. Two failure modes matter most, and both are asserted here.

A mutant that never applied must never be reported as a survivor. That is not
hypothetical -- during the session this harness was written, a `sed` range
address silently failed to match and the run looked like a clean survivor,
which would have been recorded as "the tests cannot see this defect" when in
fact the defect was never introduced. The harness reports such a mutant as an
error instead.

The module under mutation must be restored whatever happens -- a passing run,
a failing run, or an exception out of the test command. A harness that leaves
a mutated file behind corrupts every subsequent run and, worse, the working
tree.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_HARNESS = (Path(__file__).resolve().parents[2]
            / "tools" / "mutation" / "run_mutants.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("mutation_harness", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation_harness"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway module plus a test that pins its behaviour."""
    (tmp_path / "subject.py").write_text(textwrap.dedent("""
        def total(values):
            return sum(values)
    """).lstrip())
    (tmp_path / "test_subject.py").write_text(textwrap.dedent("""
        from subject import total

        def test_total():
            assert total([1, 2, 3]) == 6
    """).lstrip())
    return tmp_path


def _target(harness, workspace, mutants):
    return harness.Target(
        name="subject",
        module_path=Path("subject.py"),
        test_command=f"{sys.executable} -m pytest test_subject.py -q -p no:cacheprovider",
        timeout=120.0,
        mutants=[harness.Mutant(**m) for m in mutants],
    )


# ---------------------------------------------------------------------------
# Applying a mutant
# ---------------------------------------------------------------------------

def test_a_mutant_that_changes_nothing_is_an_error_not_a_survivor(harness, workspace):
    """The failure mode that motivated this harness."""
    target = _target(harness, workspace, [
        {"id": "absent", "find": "this text is not in the module",
         "replace": "irrelevant", "models": "a pattern that no longer matches"},
    ])
    results = harness.run_target(target, root=workspace)
    assert [r.outcome for r in results] == [harness.NOT_APPLIED]
    assert "not found" in results[0].detail.lower()


def test_an_ambiguous_pattern_is_an_error(harness, workspace):
    """A ``find`` matching twice would mutate an arbitrary one of the sites."""
    (workspace / "subject.py").write_text("a = sum([1])\nb = sum([2])\n")
    (workspace / "test_subject.py").write_text("def test_ok():\n    assert True\n")
    target = _target(harness, workspace, [
        {"id": "ambiguous", "find": "sum(", "replace": "max(",
         "models": "a pattern that matches more than one site"},
    ])
    results = harness.run_target(target, root=workspace)
    assert results[0].outcome == harness.NOT_APPLIED
    assert "twice" in results[0].detail or "2" in results[0].detail


# ---------------------------------------------------------------------------
# Killed and survived
# ---------------------------------------------------------------------------

def test_a_mutant_the_tests_catch_is_reported_killed(harness, workspace):
    target = _target(harness, workspace, [
        {"id": "sum-to-max", "find": "return sum(values)",
         "replace": "return max(values)", "models": "a total that returns the maximum"},
    ])
    results = harness.run_target(target, root=workspace)
    assert results[0].outcome == harness.KILLED


def test_a_mutant_the_tests_miss_is_reported_survived(harness, workspace):
    """The tests pin only ``[1, 2, 3]``, where sorting changes nothing."""
    target = _target(harness, workspace, [
        {"id": "sorts-first", "find": "return sum(values)",
         "replace": "return sum(sorted(values))",
         "models": "a total that sorts first, which no test can distinguish"},
    ])
    results = harness.run_target(target, root=workspace)
    assert results[0].outcome == harness.SURVIVED


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("replacement,expected", [
    ("return max(values)", "KILLED"),
    ("return sum(sorted(values))", "SURVIVED"),
])
def test_the_module_is_restored_after_every_outcome(harness, workspace,
                                                    replacement, expected):
    original = (workspace / "subject.py").read_text()
    target = _target(harness, workspace, [
        {"id": "m", "find": "return sum(values)", "replace": replacement,
         "models": "either outcome"},
    ])
    harness.run_target(target, root=workspace)
    assert (workspace / "subject.py").read_text() == original


def test_the_module_is_restored_when_the_test_command_cannot_run(harness, workspace):
    """An unrunnable command must not leave the mutant on disk."""
    original = (workspace / "subject.py").read_text()
    target = harness.Target(
        name="subject", module_path=Path("subject.py"),
        test_command="a-command-that-does-not-exist --please",
        timeout=30.0,
        mutants=[harness.Mutant(id="m", find="return sum(values)",
                                replace="return max(values)", models="x")],
    )
    harness.run_target(target, root=workspace)
    assert (workspace / "subject.py").read_text() == original


def test_a_timeout_counts_as_killed_and_restores_the_module(harness, workspace):
    """A mutant that hangs the suite is caught, not silently tolerated.

    The hang has to come from the mutant, not the command: a command that
    times out on the unmutated module fails the baseline check instead, which
    is a different and correct outcome.
    """
    original = (workspace / "subject.py").read_text()
    target = harness.Target(
        name="subject", module_path=Path("subject.py"),
        test_command=f"{sys.executable} -m pytest test_subject.py -q -p no:cacheprovider",
        timeout=5.0,
        mutants=[harness.Mutant(
            id="slow", find="return sum(values)",
            replace="import time; time.sleep(120); return sum(values)",
            models="a mutant that hangs the suite")],
    )
    results = harness.run_target(target, root=workspace)
    assert results[0].outcome == harness.KILLED
    assert "timed out" in results[0].detail.lower()
    assert (workspace / "subject.py").read_text() == original


# ---------------------------------------------------------------------------
# Line endings
# ---------------------------------------------------------------------------

def test_a_crlf_module_is_restored_byte_for_byte(harness, tmp_path):
    """Restoring must not rewrite line endings.

    ``mlsynth/utils/datautils.py`` is CRLF throughout. A text-mode read
    translates line endings on the way in but a text-mode write does not put
    them back, so a round trip rewrites all 1032 of them and the "restored"
    file differs from the original on every line. The mutant is gone but the
    diff is not.
    """
    module = tmp_path / "subject.py"
    module.write_bytes(b"def total(values):\r\n    return sum(values)\r\n")
    (tmp_path / "test_subject.py").write_text(
        "from subject import total\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n")
    original = module.read_bytes()

    target = _target(harness, tmp_path, [
        {"id": "m", "find": "return sum(values)", "replace": "return max(values)",
         "models": "x"},
    ])
    results = harness.run_target(target, root=tmp_path)

    assert results[0].outcome == harness.KILLED
    assert module.read_bytes() == original
    assert b"\r\n" in module.read_bytes()


def test_a_pattern_written_with_unix_newlines_matches_a_crlf_module(harness, tmp_path):
    """Multi-line patterns in the catalogue are written with ``\n``."""
    module = tmp_path / "subject.py"
    module.write_bytes(b"def total(values):\r\n    x = 1\r\n    return sum(values)\r\n")
    (tmp_path / "test_subject.py").write_text(
        "from subject import total\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n")

    target = _target(harness, tmp_path, [
        {"id": "multiline", "find": "x = 1\n    return sum(values)",
         "replace": "x = 1\n    return max(values)", "models": "x"},
    ])
    results = harness.run_target(target, root=tmp_path)
    assert results[0].outcome == harness.KILLED
    assert b"\r\n" in module.read_bytes()


# ---------------------------------------------------------------------------
# Stale bytecode
# ---------------------------------------------------------------------------

def test_stale_bytecode_is_purged_before_a_run(harness, workspace):
    """A same-length mutant must not execute the previous run's bytecode.

    CPython validates a ``.pyc`` on the source's size and its mtime in whole
    seconds, so ``sum`` -> ``max`` matches a cache written a moment earlier by
    the baseline run and the mutant is never compiled. The result is a false
    survivor whose appearance depends on which side of a second boundary the
    writes fall -- nondeterministic, and in the direction that reports
    confidence nobody measured.
    """
    cache = workspace / "__pycache__"
    cache.mkdir()
    stale = cache / "subject.cpython-311.pyc"
    stale.write_bytes(b"not real bytecode")

    harness.purge_bytecode(workspace / "subject.py")
    assert not stale.exists()


def test_purging_bytecode_is_safe_when_there_is_no_cache(harness, workspace):
    harness.purge_bytecode(workspace / "subject.py")          # must not raise


def test_a_same_length_mutant_is_still_killed(harness, workspace):
    """End to end, on the exact pair that triggered the false survivor."""
    target = _target(harness, workspace, [
        {"id": "same-length", "find": "return sum(values)",
         "replace": "return max(values)",
         "models": "a mutant the same length as the code it replaces"},
    ])
    for _ in range(3):          # the race is timing dependent; repeat it
        results = harness.run_target(target, root=workspace)
        assert results[0].outcome == harness.KILLED, results[0].detail


# ---------------------------------------------------------------------------
# Baseline verification
# ---------------------------------------------------------------------------

def test_a_target_whose_suite_is_already_red_scores_nothing(harness, workspace):
    """The other way to report confidence that was never measured.

    If the test command fails before anything is mutated -- a renamed file, a
    missing dependency, a typo -- then every mutant "fails" it too and the run
    reports a perfect score while measuring nothing. The unmutated suite is
    checked first, and the target is reported unusable instead.
    """
    (workspace / "test_subject.py").write_text(
        "from subject import total\n\ndef test_total():\n    assert total([1]) == 999\n")
    target = _target(harness, workspace, [
        {"id": "sum-to-max", "find": "return sum(values)",
         "replace": "return max(values)", "models": "x"},
    ])
    results = harness.run_target(target, root=workspace)
    assert [r.outcome for r in results] == [harness.NOT_APPLIED]
    assert "unmutated" in results[0].detail


def test_a_target_whose_suite_is_red_leaves_the_module_untouched(harness, workspace):
    original = (workspace / "subject.py").read_text()
    (workspace / "test_subject.py").write_text("def test_x():\n    assert False\n")
    target = _target(harness, workspace, [
        {"id": "m", "find": "return sum(values)", "replace": "return max(values)",
         "models": "x"},
    ])
    harness.run_target(target, root=workspace)
    assert (workspace / "subject.py").read_text() == original


# ---------------------------------------------------------------------------
# The catalogue itself
# ---------------------------------------------------------------------------

def test_every_catalogued_target_points_at_test_files_that_exist(harness):
    """A target naming a file that is not on this branch cannot be scored.

    Without this the failure is silent-ish: the baseline check reports the
    target unusable at run time, but the catalogue would keep claiming
    coverage it does not have.
    """
    root = Path(__file__).resolve().parents[2]
    targets = harness.load_targets(root / "tools" / "mutation" / "targets.toml")
    for target in targets:
        for token in target.test_command.split():
            if token.endswith(".py"):
                assert (root / token).is_file(), (
                    f"{target.name} names {token}, which does not exist here"
                )

def test_the_shipped_catalogue_parses_and_is_well_formed(harness):
    """Every declared target points at files that exist and is non-empty."""
    root = Path(__file__).resolve().parents[2]
    targets = harness.load_targets(root / "tools" / "mutation" / "targets.toml")
    assert targets, "no targets declared"

    seen_ids = set()
    for target in targets:
        assert (root / target.module_path).is_file(), target.module_path
        assert target.mutants, f"{target.name} declares no mutants"
        assert target.test_command.strip()
        for mutant in target.mutants:
            key = (target.name, mutant.id)
            assert key not in seen_ids, f"duplicate mutant id {key}"
            seen_ids.add(key)
            assert mutant.find and mutant.replace
            assert mutant.find != mutant.replace
            assert mutant.models.strip(), f"{mutant.id} says nothing about what it models"


def test_every_catalogued_mutant_still_matches_its_module(harness):
    """The catalogue rots silently otherwise.

    A ``find`` that stops matching after a refactor turns into a permanent
    not-applied result, which is the same failure this harness exists to
    prevent -- so it is checked as a unit test, not only at run time.
    """
    root = Path(__file__).resolve().parents[2]
    targets = harness.load_targets(root / "tools" / "mutation" / "targets.toml")
    for target in targets:
        source = (root / target.module_path).read_text()
        for mutant in target.mutants:
            count = source.count(mutant.find)
            assert count == 1, (
                f"{target.name}/{mutant.id}: pattern occurs {count} times in "
                f"{target.module_path}; it must match exactly once"
            )
