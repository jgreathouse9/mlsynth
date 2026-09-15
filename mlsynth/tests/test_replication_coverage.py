"""The coverage table on ``docs/replications.rst`` is generated, and these tests
are what stop it drifting from the library again.

The table previously said "Thirty-seven of the thirty-eight estimators" and
"Of mlsynth's 36 estimators, 35 (97%)" while the library exported 78. Nothing
failed, because nothing was checking. The counts now come from
``tools/estimator_families.toml``; these tests assert that the file describes
exactly the exported estimators, that every family it references is defined,
and that the committed table is what the generator produces from it.

A new estimator therefore lands red until it is given a family, which is the
point: the classification is a decision someone makes, not one the generator
guesses.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import mlsynth
from tools.estimator_families import (
    FAMILY_ORDER,
    coverage_rows,
    load_families,
    render_table,
)

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "replications.rst"


def exported_estimators() -> set[str]:
    """The public estimator classes: exported names carrying a ``fit()``."""
    return {
        name
        for name in mlsynth.__all__
        if inspect.isclass(getattr(mlsynth, name))
        and hasattr(getattr(mlsynth, name), "fit")
    }


@pytest.fixture(scope="module")
def data():
    return load_families()


class TestTheFileDescribesTheLibrary:
    """Smoke and unit level: the data file and ``mlsynth.__all__`` agree."""

    def test_it_loads(self, data):
        assert data.families, "no families defined"
        assert data.assignments, "no estimators assigned"

    def test_every_exported_estimator_has_a_family(self, data):
        missing = exported_estimators() - set(data.assignments)
        assert not missing, (
            "exported estimators with no family in "
            "tools/estimator_families.toml: %s" % sorted(missing)
        )

    def test_no_assignment_names_a_nonexistent_estimator(self, data):
        extra = set(data.assignments) - exported_estimators()
        assert not extra, (
            "tools/estimator_families.toml names estimators mlsynth does not "
            "export: %s" % sorted(extra)
        )

    def test_every_assignment_names_a_defined_family(self, data):
        defined = {f.key for f in data.families}
        used = set(data.assignments.values())
        assert not (used - defined), "undefined families: %s" % sorted(used - defined)

    def test_every_family_has_at_least_one_estimator(self, data):
        used = set(data.assignments.values())
        empty = [f.key for f in data.families if f.key not in used]
        assert not empty, "families with no estimators: %s" % empty

    def test_family_order_covers_every_family(self, data):
        assert [f.key for f in data.families] == FAMILY_ORDER


class TestTheVerifiedColumn:
    """The claim the page makes in prose has to match the claim in the table."""

    def test_unverified_estimators_are_assigned(self, data):
        unplaced = set(data.unverified) - set(data.assignments)
        assert not unplaced, sorted(unplaced)

    def test_verified_never_exceeds_the_family_size(self, data):
        for row in coverage_rows(data):
            assert 0 <= row.verified <= row.in_family

    def test_the_totals_add_up(self, data):
        rows = coverage_rows(data)
        assert sum(r.in_family for r in rows) == len(data.assignments)
        assert sum(r.verified for r in rows) == len(data.assignments) - len(
            data.unverified
        )

    def test_iscm_is_the_only_unverified_estimator(self, data):
        # The page's prose says so in two places. If a second estimator ever
        # lands without a replication, this fails and the prose gets revisited.
        assert sorted(data.unverified) == ["ISCM"]

    def test_each_unverified_estimator_states_a_reason(self, data):
        for name, reason in data.unverified.items():
            assert reason.strip(), "%s is unverified with no reason given" % name


class TestTheCommittedTable:
    """The rendered table in the docs is the generator's output, not a copy."""

    def test_markers_are_present_and_ordered(self):
        text = DOCS.read_text()
        start = text.find(".. coverage-table-start")
        end = text.find(".. coverage-table-end")
        assert start != -1, "start marker missing from docs/replications.rst"
        assert end != -1, "end marker missing from docs/replications.rst"
        assert start < end

    def test_the_committed_table_matches_the_generator(self, data):
        text = DOCS.read_text()
        start = text.index(".. coverage-table-start")
        end = text.index(".. coverage-table-end")
        committed = text[start:end]
        expected = ".. coverage-table-start\n\n" + render_table(data) + "\n"
        assert committed == expected, (
            "docs/replications.rst is out of date with "
            "tools/estimator_families.toml -- run "
            "`python tools/gen_coverage_table.py`"
        )

    def test_the_generator_is_idempotent(self):
        before = DOCS.read_text()
        subprocess.run(
            [sys.executable, str(REPO / "tools" / "gen_coverage_table.py")],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
        assert DOCS.read_text() == before


class TestFailuresAreReported:
    """Edge and failure level: a malformed file is refused, not worked around."""

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_families(tmp_path / "nope.toml")

    def test_an_unknown_family_raises(self, tmp_path):
        p = tmp_path / "f.toml"
        p.write_text(
            '[[family]]\nkey = "canonical"\ntitle = "Canonical"\n\n'
            '[estimators]\nFDID = "canonical"\nTSSC = "nosuchfamily"\n'
        )
        with pytest.raises(ValueError, match="nosuchfamily"):
            load_families(p)

    def test_a_duplicate_family_key_raises(self, tmp_path):
        p = tmp_path / "f.toml"
        p.write_text(
            '[[family]]\nkey = "canonical"\ntitle = "Canonical"\n'
            '[[family]]\nkey = "canonical"\ntitle = "Again"\n\n'
            '[estimators]\nFDID = "canonical"\n'
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_families(p)

    def test_an_unverified_name_outside_the_assignments_raises(self, tmp_path):
        p = tmp_path / "f.toml"
        p.write_text(
            '[[family]]\nkey = "canonical"\ntitle = "Canonical"\n\n'
            '[estimators]\nFDID = "canonical"\n\n'
            '[unverified]\nGHOST = "no such estimator"\n'
        )
        with pytest.raises(ValueError, match="GHOST"):
            load_families(p)
