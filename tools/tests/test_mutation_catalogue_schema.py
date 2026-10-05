"""A catalogue key the harness does not read has to fail, not be ignored.

``run_mutants.py`` reads ``targets.toml`` with ``dict.get``, so until this guard
existed a misspelled key was not an error. It was an instruction that silently
did not run, while the entry read as though it did.

That reddened the weekly job. ``active-set-linear-ray/returned-point-is-not-checked``
is an accepted survivor -- the guard it removes is equivalent to the original
while the ray branch above it is correct -- and the acceptance was recorded as
``equivalent`` / ``equivalent-reason``. The harness reads ``expected`` and
``accepted-because``. So ``expected`` defaulted to ``killed``, the mutant
survived, the job exited 1, and the justification sat in the file being read
by nobody. The failure mode is the one the catalogue is least able to
survive: the run was red for a reason already answered in the file.

``extra="forbid"`` on every estimator config is the same rule for the library's
inputs. This is it for the catalogue that scores the library.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MUTATION = Path(__file__).resolve().parents[1] / "mutation"
if str(_MUTATION) not in sys.path:
    sys.path.insert(0, str(_MUTATION))

from run_mutants import (  # noqa: E402
    KILLED,
    MUTANT_KEYS,
    SURVIVED,
    TARGET_KEYS,
    _mutant,
    _reject_unknown,
    load_targets,
    tomllib,            # whichever reader the harness resolved: 3.11+ stdlib, else tomli
)

_CATALOGUE = _MUTATION / "targets.toml"


def _entry(**over):
    base = {"id": "m", "find": "a", "replace": "b", "models": "why"}
    base.update(over)
    return base


class TestUnknownKeysAreRefused:
    def test_an_unknown_mutant_key_raises(self):
        with pytest.raises(ValueError, match="unknown key"):
            _mutant("t", _entry(equivalent=True))

    def test_the_message_names_the_offending_key(self):
        with pytest.raises(ValueError, match="'equivalent-reason'"):
            _mutant("t", _entry(**{"equivalent-reason": "because"}))

    def test_the_message_names_the_keys_that_are_read(self):
        """So the near-miss spelling is fixable from the message alone."""
        with pytest.raises(ValueError, match="accepted-because"):
            _mutant("t", _entry(**{"accepted_because": "underscored, not hyphenated"}))

    def test_every_allowed_mutant_key_together_is_accepted(self):
        m = _mutant("t", _entry(expected=SURVIVED,
                                **{"accepted-because": "equivalent"}))
        assert m.expected == SURVIVED

    def test_a_target_level_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown key"):
            _reject_unknown("target 'x'", {"name": "x", "module_path": "a.py"},
                            TARGET_KEYS)

    def test_a_clean_entry_passes(self):
        _reject_unknown("target 'x'", {"name": "x"}, TARGET_KEYS)


class TestTheAcceptanceContractStillHolds:
    """The pre-existing rules, now that an adjacent guard runs before them."""

    def test_expected_survived_without_a_reason_raises(self):
        with pytest.raises(ValueError, match="accepted-because"):
            _mutant("t", _entry(expected=SURVIVED))

    def test_a_reason_without_expected_survived_raises(self):
        with pytest.raises(ValueError, match="does not act on"):
            _mutant("t", _entry(**{"accepted-because": "stated but inert"}))

    def test_an_unrecognised_expected_value_raises(self):
        with pytest.raises(ValueError, match="expected must be"):
            _mutant("t", _entry(expected="maybe"))

    def test_the_default_is_killed(self):
        assert _mutant("t", _entry()).expected == KILLED


class TestTheShippedCatalogue:
    def test_it_parses(self):
        targets = load_targets(_CATALOGUE)
        assert targets, "the catalogue is empty"
        assert sum(len(t.mutants) for t in targets) > 0

    def test_no_entry_carries_a_key_the_harness_ignores(self):
        """The regression itself: parsing is what refuses it, so this restates
        the property over the real file so a hand-edit cannot reintroduce it.

        Read through the harness's own ``tomllib`` name rather than importing
        the stdlib one: ``run_mutants`` falls back to ``tomli`` below 3.11, and
        the suite runs on 3.10.
        """
        data = tomllib.loads(_CATALOGUE.read_text())
        for entry in data["target"]:
            assert not set(entry) - TARGET_KEYS, entry.get("name")
            for m in entry.get("mutant", []):
                extra = set(m) - MUTANT_KEYS
                assert not extra, f"{entry['name']}/{m.get('id')}: {sorted(extra)}"

    def test_every_accepted_survivor_states_its_reason(self):
        for t in load_targets(_CATALOGUE):
            for m in t.mutants:
                if m.expected == SURVIVED:
                    assert m.accepted_because.strip(), f"{t.name}/{m.id}"
